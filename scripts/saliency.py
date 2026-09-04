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
from training.models.registry import ModelRegistry
from training.view_mip import TriaxialMip

ACFG = AnalysisConfig()
TCFG = TrainConfig()
CHANNEL_NAMES = ("axial MIP", "coronal MIP", "sagittal MIP")


class GradCam:
    """Grad-CAM on the last conv feature map of a timm CNN."""

    def __init__(self, model, target_layer):
        self.model = model
        self.acts, self.grads = None, None
        target_layer.register_forward_hook(self._fwd)

    def _fwd(self, _m, _i, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x):
        self.model.zero_grad()
        logit = self.model(x[None])
        logit.backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        return float(torch.sigmoid(logit)), cam / (cam.max() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("uids", nargs="+")
    ap.add_argument("--run", default="effb0_mip_ref40")
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    folds = pd.read_csv(TCFG.folds_csv).set_index("uid")
    view = TriaxialMip(TCFG.input_size)
    out_dir = ACFG.output_dir / "saliency"
    out_dir.mkdir(exist_ok=True)

    for uid in args.uids:
        fold = int(folds.loc[uid, "fold"])
        label = folds.loc[uid, "is_pathologic"]
        model = ModelRegistry.get(args.model)(pretrained=False)
        state = torch.load(TCFG.runs_dir / args.run / f"fold{fold}" / "best_ema.pt",
                           map_location=args.device)
        model.load_state_dict(state)
        model.to(args.device).eval()
        cam_engine = GradCam(model, model.net.conv_head)

        crop = np.load(TCFG.crops_dir / f"{uid}.npy").astype(np.float32)
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 7))
        for row, (tag, vol) in enumerate((("orig", crop), ("LR-flipped", crop[::-1].copy()))):
            x = view(vol).to(args.device)
            prob, cam = cam_engine(x)
            for c in range(3):
                ax = axes[row, c]
                ax.imshow(np.rot90(x[c].cpu().numpy()), cmap="gray")
                ax.imshow(np.rot90(cam), cmap="jet", alpha=0.35)
                ax.set_title(f"{CHANNEL_NAMES[c]} ({tag})\npred {prob:.3f}", fontsize=9)
                ax.axis("off")
        fig.suptitle(f"{uid} — label {label:.0f}, fold {fold}, {args.run} "
                     "(Grad-CAM overlay, red = evidence used)")
        fig.tight_layout()
        path = out_dir / f"{uid}_{args.run}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"{uid}: label {label:.0f} -> {path}")


if __name__ == "__main__":
    main()
