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

ACFG = AnalysisConfig()
TCFG = TrainConfig()


def main():
    ap = argparse.ArgumentParser(description="Render crop galleries (axial MIP)")
    ap.add_argument("--label", type=float, choices=[0.0, 1.0], default=None,
                    help="0 = normal, 1 = abnormal")
    ap.add_argument("--cluster", type=int, default=None,
                    help="hard-case cluster id (from hard_clusters.csv)")
    ap.add_argument("--uids", nargs="*", default=None, help="explicit uid list")
    ap.add_argument("--sort", choices=["sbr", "sbr_desc", "random"], default="sbr",
                    help="sbr = ascending worst-putamen SBR (most severe first); "
                         "sbr_desc = descending (healthiest first)")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--out", default=None, help="output filename stem")
    args = ap.parse_args()

    df = pd.read_csv(TCFG.folds_csv)
    feats = pd.read_csv(ACFG.output_dir / "features.csv")
    df = df.merge(feats[["uid", "sbr_putamen_min"]], on="uid", how="left")
    parts = []
    if args.uids:
        df = df[df["uid"].isin(args.uids)]
        parts.append("uids")
    if args.cluster is not None:
        hc = pd.read_csv(ACFG.output_dir / "error_analysis" / "hard_clusters.csv")
        df = df[df["uid"].isin(hc.loc[hc["cluster"] == args.cluster, "uid"])]
        parts.append(f"cluster{args.cluster}")
    if args.label is not None:
        df = df[df["is_pathologic"] == args.label]
        parts.append("abnormal" if args.label == 1 else "normal")
    if df.empty:
        raise SystemExit("no scans match the given filters")

    if args.sort == "sbr":
        df = df.sort_values("sbr_putamen_min")
    elif args.sort == "sbr_desc":
        df = df.sort_values("sbr_putamen_min", ascending=False)
    else:
        df = df.sample(frac=1, random_state=17)
    df = df.head(args.n)

    cols = 8
    rows = int(np.ceil(len(df) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 3.0 * rows))
    for ax, (_, r) in zip(np.ravel(axes), df.iterrows()):
        crop = np.load(TCFG.crops_dir / f"{r.uid}.npy").astype(np.float32)
        ax.imshow(np.rot90(crop.max(axis=2)), cmap="hot")
        ax.set_title(f"{r.uid}\nlabel {r.is_pathologic:.0f}  SBR {r.sbr_putamen_min:.2f}",
                     fontsize=7)
        ax.axis("off")
    for ax in np.ravel(axes)[len(df):]:
        ax.axis("off")
    out_dir = ACFG.output_dir / "galleries"
    out_dir.mkdir(exist_ok=True)
    name = args.out or "_".join(parts or ["all"])
    path = out_dir / f"{name}.png"
    fig.suptitle(f"{name} (n={len(df)}, sorted by {args.sort})")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"gallery ({len(df)} scans): {path}")


if __name__ == "__main__":
    main()
