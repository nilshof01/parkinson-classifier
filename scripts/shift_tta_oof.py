import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.config import TrainConfig
from training.models.registry import ModelRegistry
from training.view_mip import TriaxialMip
from training.view_mip_asym import MipWithAsymmetry
from training.view_volume import VolumeView

CFG = TrainConfig()
SHIFTS = [(0, 0, 0)] + [tuple(s if a == ax else 0 for a in range(3))
                        for ax in range(3) for s in (-1, 1)]  # 7 variants
VIEWS = {"mip": TriaxialMip(224), "mipasym": MipWithAsymmetry(224),
         "volume3d": VolumeView()}


def main():
    ap = argparse.ArgumentParser(description="Recompute a run's OOF with shift+flip TTA")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    folds = pd.read_csv(CFG.folds_csv)
    for run in args.runs:
        rargs = json.loads((CFG.runs_dir / run / "args.json").read_text())
        view = VIEWS[rargs["view"]]
        crops_dir = Path(rargs.get("crops_dir") or CFG.crops_dir)
        import inspect
        cls = ModelRegistry.get(rargs["model"])
        kw = {"pretrained": False}
        for k in ("pool", "head"):
            if k in inspect.signature(cls.__init__).parameters and rargs.get(k):
                kw[k] = rargs[k]

        rows = []
        for k in sorted(folds["fold"].unique()):
            model = cls(**kw)
            model.load_state_dict(torch.load(
                CFG.runs_dir / run / f"fold{k}" / "best_ema.pt",
                map_location=args.device))
            model.to(args.device).eval()
            va = folds[folds["fold"] == k]
            with torch.no_grad():
                for uid, label in zip(va["uid"], va["is_pathologic"]):
                    vol = np.load(crops_dir / f"{uid}.npy").astype(np.float32)
                    variants = []
                    for s in SHIFTS:
                        v = ndimage.shift(vol, s, order=1, cval=0.0) if any(s) else vol
                        variants += [view(v), view(v[::-1].copy())]
                    x = torch.stack(variants).to(args.device)
                    with torch.autocast("cuda", enabled=args.device.startswith("cuda")):
                        p = torch.sigmoid(model(x).float()).mean().item()
                    rows.append({"uid": uid, "is_pathologic": label, "pred": p})

        oof = pd.DataFrame(rows)
        out_dir = CFG.runs_dir / f"{run}_stta"
        out_dir.mkdir(exist_ok=True)
        oof.to_csv(out_dir / "oof.csv", index=False)
        (out_dir / "args.json").write_text(json.dumps(
            {**rargs, "note": f"shift+flip TTA ({len(SHIFTS)}x2) over {run}"}))
        base = pd.read_csv(CFG.runs_dir / run / "oof.csv")
        for name, df in ((run, base), (f"{run}_stta", oof)):
            y, p = df["is_pathologic"], df["pred"].clip(1e-6, 1 - 1e-6)
            print(f"  {name}: auroc {roc_auc_score(y, p):.4f}  raw ll {log_loss(y, p):.4f}")


if __name__ == "__main__":
    main()
