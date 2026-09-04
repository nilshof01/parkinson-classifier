import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.config import AnalysisConfig
from training.config import TrainConfig

ACFG, TCFG = AnalysisConfig(), TrainConfig()
AXES = (("axial", 2), ("coronal", 1), ("sagittal", 0))


def pick_errors(n):
    e = pd.read_csv(ACFG.output_dir / "ensemble" / "oof_ensemble.csv")
    f = pd.read_csv(ACFG.output_dir / "features.csv")[["uid", "sbr_putamen_min"]]
    e = e.merge(f, on="uid")
    y, p = e.is_pathologic, e.ensemble_pred.clip(1e-6, 1 - 1e-6)
    e["ll"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    fp = e[(y == 0) & (p > 0.5)].nlargest(n, "ll")
    fn = e[(y == 1) & (p < 0.5)].nlargest(n, "ll")
    return {"false_positives": fp, "false_negatives": fn}


def render(group_name, df, mode, vmax):
    fig, axes = plt.subplots(3, len(df), figsize=(2.8 * len(df), 8.5))
    for col, (_, r) in enumerate(df.iterrows()):
        vol = np.load(TCFG.crops_dir / f"{r.uid}.npy").astype(np.float32)
        for row, (ax_name, ax_idx) in enumerate(AXES):
            proj = vol.max(axis=ax_idx) if mode == "max" else vol.mean(axis=ax_idx)
            ax = axes[row, col]
            # shared scale across all scans and both groups: brightness is
            # comparable between images, unlike per-image autoscaling
            ax.imshow(np.rot90(proj), cmap="hot", vmin=0,
                      vmax=vmax if mode == "max" else vmax / 2)
            if row == 0:
                ax.set_title(f"{r.uid}\npred {r.ensemble_pred:.2f} SBR "
                             f"{r.sbr_putamen_min:.2f}", fontsize=8)
            if col == 0:
                ax.set_ylabel(ax_name, fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                ax.axis("off")
    proj_word = "maximum" if mode == "max" else "average"
    fig.suptitle(f"{group_name.replace('_', ' ')} — {proj_word} intensity projections "
                 "(rows: axial / coronal / sagittal)")
    fig.tight_layout()
    out = ACFG.figures_dir.parent / "galleries" / f"{group_name}_{mode}proj.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    groups = pick_errors(args.n)
    all_uids = pd.concat(groups.values())["uid"]
    vmax = float(np.percentile(np.stack(
        [np.load(TCFG.crops_dir / f"{u}.npy").astype(np.float32).max()
         for u in all_uids]), 90))
    for name, df in groups.items():
        for mode in ("max", "mean"):
            render(name, df, mode, vmax)


if __name__ == "__main__":
    main()
