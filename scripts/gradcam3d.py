r"""3D Grad-CAM for cnn3d: where in the volume is the model looking?

Generates MIP overlays of the Grad-CAM heatmap for a set of scans.
Useful for checking whether FN/FP cases have attention on the putamen or elsewhere.

Usage (pod):
    # show top confident FNs and FPs automatically
    python scripts/gradcam3d.py \
        --run output/runs/cnn3d_m4_baseline_s0 \
        --fold 0 \
        --mode auto

    # or pass specific UIDs
    python scripts/gradcam3d.py \
        --run output/runs/cnn3d_m4_baseline_s0 \
        --fold 0 \
        --uids uid1 uid2 uid3
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.config import AnalysisConfig
from training.config import TrainConfig
from training.models.cnn3d import Cnn3d
from training.view_volume import VolumeView

ACFG = AnalysisConfig()
TCFG = TrainConfig()


class GradCam3d:
    def __init__(self, model, target_block_idx=-1):
        self.model = model
        self._acts = None
        self._grads = None
        block = model.features[target_block_idx]
        block.register_forward_hook(self._fwd_hook)

    def _fwd_hook(self, _m, _i, out):
        self._acts = out
        out.register_hook(lambda g: setattr(self, "_grads", g))

    def __call__(self, x):
        """Returns (pred_prob, cam_volume) where cam_volume has same spatial dims as x."""
        self.model.zero_grad()
        logit = self.model(x[None])
        logit.backward()
        # global-average-pool gradients over spatial dims → channel weights
        w = self._grads.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((w * self._acts).sum(dim=1, keepdim=True))   # (1,1,D,H,W)
        cam = F.interpolate(cam, size=x.shape[-3:], mode="trilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam / (cam.max() + 1e-9)
        return float(torch.sigmoid(logit).item()), cam


def plot_scan(uid, vol, cam, pred, label, out_path):
    """Axial + coronal + sagittal MIP of intensity and cam overlay."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    views = [
        ("axial MIP",   vol.max(axis=2),   cam.max(axis=2)),
        ("coronal MIP", vol.max(axis=1),   cam.max(axis=1)),
        ("sagittal MIP",vol.max(axis=0),   cam.max(axis=0)),
    ]
    label_str = "ABN" if label else "NRM"
    pred_str  = f"{'ABN' if pred > 0.5 else 'NRM'} {pred:.3f}"
    for col, (title, mip_v, mip_c) in enumerate(views):
        for row, (img, cmap, alpha) in enumerate([
            (np.rot90(mip_v), "hot", 1.0),
            (np.rot90(mip_c), "jet", 0.6),
        ]):
            ax = axes[row, col]
            if row == 0:
                ax.imshow(img, cmap="hot")
                ax.set_title(f"{title}", fontsize=9)
            else:
                ax.imshow(np.rot90(mip_v), cmap="hot")
                ax.imshow(np.rot90(mip_c), cmap="jet", alpha=0.55)
                ax.set_title(f"{title} + GradCAM", fontsize=9)
            ax.axis("off")
    fig.suptitle(f"{uid}  label={label_str}  pred={pred_str}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--mode", default="auto",
                    choices=["auto", "fn", "fp", "manual"],
                    help="auto=top FN+FP, fn=top FN only, fp=top FP only, manual=use --uids")
    ap.add_argument("--uids", nargs="+", default=[])
    ap.add_argument("--n", type=int, default=8, help="number of cases per group")
    ap.add_argument("--block", type=int, default=-1,
                    help="which block to hook (-1=last=block3)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folds-csv", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    folds_path = Path(args.folds_csv) if args.folds_csv else TCFG.folds_csv
    folds = pd.read_csv(folds_path)
    crops_dir = TCFG.prepared_dir / "crops_m4"
    if not crops_dir.exists():
        crops_dir = TCFG.crops_dir

    vp = pd.read_csv(run_dir / f"fold{args.fold}" / "val_preds.csv")
    vp["confidence"] = (vp["pred"] - 0.5).abs()
    vp["pred_label"] = (vp["pred"] > 0.5).astype(int)

    if args.mode == "manual":
        uids = args.uids
    else:
        fn = vp[(vp["pred_label"] == 0) & (vp["is_pathologic"] == 1)]
        fp = vp[(vp["pred_label"] == 1) & (vp["is_pathologic"] == 0)]
        fn_top = fn.nlargest(args.n, "confidence")["uid"].tolist()
        fp_top = fp.nlargest(args.n, "confidence")["uid"].tolist()
        if args.mode == "fn":
            uids = fn_top
        elif args.mode == "fp":
            uids = fp_top
        else:
            uids = fn_top + fp_top

    model = Cnn3d(pretrained=False, dropout=0.3)
    ckpt = run_dir / f"fold{args.fold}" / "best_ema.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(args.device).eval()
    cam_engine = GradCam3d(model, target_block_idx=args.block)

    view = VolumeView()
    out_dir = ACFG.output_dir / "error_analysis" / "gradcam3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    vp_idx = vp.set_index("uid")
    folds_idx = folds.set_index("uid")

    for uid in uids:
        vol = np.load(crops_dir / f"{uid}.npy").astype(np.float32)
        x = view(vol).to(args.device)
        pred, cam = cam_engine(x)
        label = int(folds_idx.loc[uid, "is_pathologic"])
        quad = {(0,0):"TN",(0,1):"FN",(1,0):"FP",(1,1):"TP"}
        q = quad[(int(pred > 0.5), label)]
        out_path = out_dir / f"{q}_{uid}_fold{args.fold}.png"
        plot_scan(uid, vol, cam, pred, label, out_path)
        print(f"  {q}  {uid}  pred={pred:.3f}  -> {out_path.name}")

    print(f"\nall plots in {out_dir}/")


if __name__ == "__main__":
    main()
