import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.config import TrainConfig
from training.models.registry import ModelRegistry
from training.view_mip import TriaxialMip
from training.view_volume import VolumeView

CFG = TrainConfig()
SHIFTS = (-3, -2, -1, 1, 2, 3)  # voxels (2 mm each), per axis


def main():
    ap = argparse.ArgumentParser(description="Prediction stability under translations")
    ap.add_argument("--run", default="cnn3d_crop")
    ap.add_argument("--model", default="cnn3d")
    ap.add_argument("--view", default="volume3d", choices=["volume3d", "mip"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    folds = pd.read_csv(CFG.folds_csv)
    val0 = folds[folds["fold"] == 0].sample(args.n, random_state=17)
    view = VolumeView() if args.view == "volume3d" else TriaxialMip(CFG.input_size)
    model = ModelRegistry.get(args.model)(pretrained=False)
    model.load_state_dict(torch.load(CFG.runs_dir / args.run / "fold0" / "best_ema.pt",
                                     map_location=args.device))
    model.to(args.device).eval()

    @torch.no_grad()
    def predict(vols):
        x = torch.stack([view(v) for v in vols]).to(args.device)
        with torch.autocast("cuda", enabled=args.device.startswith("cuda")):
            return torch.sigmoid(model(x).float()).cpu().numpy()

    rows = []
    for uid in val0["uid"]:
        vol = np.load(CFG.crops_dir / f"{uid}.npy").astype(np.float32)
        variants, tags = [vol], [(0, 0)]
        for ax in range(3):
            for s in SHIFTS:
                off = [0, 0, 0]; off[ax] = s
                variants.append(ndimage.shift(vol, off, order=1, cval=0.0))
                tags.append((ax, s))
        preds = predict(variants)
        base = preds[0]
        for (ax, s), p in zip(tags[1:], preds[1:]):
            rows.append({"uid": uid, "axis": "xyz"[ax], "shift_vox": s,
                         "base": float(base), "pred": float(p),
                         "delta": float(abs(p - base))})
    df = pd.DataFrame(rows)
    out = CFG.repo_dir / "output" / f"shift_test_{args.run}.csv"
    df.to_csv(out, index=False)

    print(f"{args.run} ({args.model}/{args.view}), n={args.n} fold-0 val scans:")
    for lim in (1, 2, 3):
        sub = df[df.shift_vox.abs() <= lim]
        per_scan = sub.groupby("uid")["delta"].max()
        print(f"  shifts <= {lim} vox ({2*lim} mm): median max|dp| {per_scan.median():.3f}, "
              f"p95 {per_scan.quantile(.95):.3f}, scans with max|dp|>0.2: "
              f"{(per_scan > 0.2).sum()}/{args.n}")
    by_axis = df[df.shift_vox.abs() <= 2].groupby("axis")["delta"].median()
    print("  median |dp| by axis (<=2 vox):", {k: round(v, 3) for k, v in by_axis.items()})
    print(f"  detail: {out}")


if __name__ == "__main__":
    main()
