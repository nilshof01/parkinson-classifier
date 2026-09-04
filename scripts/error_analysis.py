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
BORDER = 0.15  # |pred-0.5| below this = borderline, above 0.30 = confident


def load_run_preds(run_names):
    preds = {}
    for name in run_names:
        oof = pd.read_csv(TCFG.runs_dir / name / "oof.csv")
        preds[name] = oof.set_index("uid")["pred"]
    xgb_path = ACFG.output_dir / "crop_xgb_oof_pool2.csv"
    if xgb_path.exists():
        xgb = pd.read_csv(xgb_path)
        preds["xgb_crop"] = xgb.set_index("uid")["oof_pred"]
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=None,
                    help="run names under output/runs (default: efficientnet_b0_mip)")
    ap.add_argument("--gallery-n", type=int, default=24)
    args = ap.parse_args()
    run_names = args.runs or ["efficientnet_b0_mip"]

    folds = pd.read_csv(TCFG.folds_csv).set_index("uid")
    df = folds[["is_pathologic"]].copy()
    preds = load_run_preds(run_names)
    for name, s in preds.items():
        df[f"pred_{name}"] = s
        y, p = df["is_pathologic"], df[f"pred_{name}"].clip(1e-6, 1 - 1e-6)
        df[f"ll_{name}"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        df[f"wrong_{name}"] = (df[f"pred_{name}"] > 0.5) != (y > 0.5)

    feats = pd.read_csv(ACFG.output_dir / "features.csv").set_index("uid")
    qc = pd.read_csv(ACFG.output_dir / "crop_qc.csv").set_index("uid")
    hdr = pd.read_csv(ACFG.output_dir / "header_stats.csv").set_index("uid")
    df["sbr_putamen_min"] = feats["sbr_putamen_min"]
    df["contrast"] = qc["contrast"]
    df["spacing_group"] = hdr["spacing_x"].round(1)

    names = list(preds)
    print("error counts (threshold 0.5):")
    for n in names:
        wrong = df[f"wrong_{n}"]
        dist = (df.loc[wrong, f"pred_{n}"] - 0.5).abs()
        fn = int((wrong & (df["is_pathologic"] == 1)).sum())
        print(f"  {n}: {int(wrong.sum())}/{len(df)} "
              f"({100 * wrong.mean():.1f}%) — {fn} missed abnormal, "
              f"{int(wrong.sum()) - fn} false alarms; "
              f"borderline (|p-0.5|<{BORDER}) {int((dist < BORDER).sum())}, "
              f"confident (>0.30) {int((dist > 0.30).sum())}")

    if len(names) > 1:
        print("\npairwise overlap of wrong scans (count, Jaccard):")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                wa, wb = df[f"wrong_{a}"], df[f"wrong_{b}"]
                inter, union = int((wa & wb).sum()), int((wa | wb).sum())
                print(f"  {a} & {b}: {inter} shared / {union} union "
                      f"(J={inter / max(union, 1):.2f})")
        all_wrong = np.logical_and.reduce([df[f"wrong_{n}"] for n in names])
        print(f"  wrong in ALL models: {int(all_wrong.sum())}")

    main_run = names[0]
    print(f"\n{main_run}: error rate by spacing group:")
    g = df.groupby("spacing_group").agg(
        n=("is_pathologic", "size"), err=(f"wrong_{main_run}", "mean"))
    print(g[g["n"] >= 30].to_string(float_format=lambda x: f"{x:.3f}"))

    out_dir = ACFG.output_dir / "error_analysis"
    out_dir.mkdir(exist_ok=True)
    df["mean_ll"] = df[[f"ll_{n}" for n in names]].mean(axis=1)
    hard = df[np.logical_or.reduce([df[f"wrong_{n}"] for n in names])].copy()
    hard = hard.sort_values("mean_ll", ascending=False)
    hard.to_csv(out_dir / "hard_cases.csv")
    print(f"\nhard cases table ({len(hard)} scans): {out_dir / 'hard_cases.csv'}")

    top = hard.head(args.gallery_n)
    cols = 6
    rows = int(np.ceil(len(top) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.6 * rows))
    for ax, (uid, r) in zip(np.ravel(axes), top.iterrows()):
        crop = np.load(TCFG.crops_dir / f"{uid}.npy").astype(np.float32)
        ax.imshow(np.rot90(crop.max(axis=2)), cmap="hot")
        ax.set_title(f"{uid}\nlabel {r.is_pathologic:.0f}  pred "
                     f"{r[f'pred_{main_run}']:.2f}  SBRmin {r.sbr_putamen_min:.1f}",
                     fontsize=8)
        ax.axis("off")
    for ax in np.ravel(axes)[len(top):]:
        ax.axis("off")
    fig.suptitle(f"Hardest scans (axial MIP), sorted by mean OOF log loss — {main_run}")
    fig.tight_layout()
    gal = out_dir / "gallery_hardest.png"
    fig.savefig(gal, dpi=140)
    print(f"gallery: {gal}")


if __name__ == "__main__":
    main()
