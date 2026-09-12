"""Prediction analysis: confidence vs correctness vs asymmetry.

Breaks OOF predictions into four quadrants and generates galleries for each,
plus a scatter plot of putamen asymmetry index vs model confidence.

Usage:
    export SCAN_REPO=/workspace/parkinson-classifier
    python scripts/analyse_predictions.py cnn3d_m4_baseline_s0

Output (output/error_analysis/):
    pred_analysis_<run>.png   — scatter + confidence histogram
    gallery_fp_asym.png       — false positives with highest asymmetry
    gallery_fn_bilateral.png  — missed abnormals with lowest asymmetry
    gallery_uncertain_correct.png — borderline but right
    gallery_confident_wrong.png   — confident but wrong
"""

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


def load_df(run_name):
    oof = pd.read_csv(TCFG.runs_dir / run_name / "oof.csv")
    feats = pd.read_csv(ACFG.output_dir / "features.csv")
    df = oof.merge(feats[["uid", "asym_putamen", "asym_caudate",
                           "sbr_putamen_min", "sbr_putamen_l", "sbr_putamen_r"]],
                   on="uid", how="left")
    df["correct"] = (df["pred"] > 0.5) == (df["is_pathologic"] > 0.5)
    df["confidence"] = (df["pred"] - 0.5).abs()
    df["pred_label"] = (df["pred"] > 0.5).astype(int)
    return df


def gallery(df, crops_dir, title, path, n=24, sort_col=None, ascending=True):
    if df.empty:
        print(f"  skipping {path.name} — no matching scans")
        return
    if sort_col:
        df = df.sort_values(sort_col, ascending=ascending)
    df = df.head(n)
    cols = 6
    rows = int(np.ceil(len(df) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.8 * rows))
    for ax, (_, r) in zip(np.ravel(axes), df.iterrows()):
        try:
            crop = np.load(crops_dir / f"{r.uid}.npy").astype(np.float32)
            ax.imshow(np.rot90(crop.max(axis=2)), cmap="hot")
        except FileNotFoundError:
            ax.set_facecolor("black")
        label_str = "ABN" if r.is_pathologic else "NRM"
        pred_str = f"{'ABN' if r.pred > 0.5 else 'NRM'} {r.pred:.2f}"
        ax.set_title(
            f"{r.uid}\n{label_str} → {pred_str}\n"
            f"AI={r.asym_putamen:.3f}  SBR={r.sbr_putamen_min:.2f}",
            fontsize=7)
        ax.axis("off")
    for ax in np.ravel(axes)[len(df):]:
        ax.axis("off")
    fig.suptitle(f"{title} (n={len(df)})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  gallery ({len(df)}): {path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run name under output/runs")
    args = ap.parse_args()

    df = load_df(args.run)
    out = ACFG.output_dir / "error_analysis"
    out.mkdir(exist_ok=True)

    # try crops_m4 first, fall back to crops
    crops_dir = TCFG.prepared_dir / "crops_m4"
    if not crops_dir.exists():
        crops_dir = TCFG.crops_dir

    # ── Summary table ──────────────────────────────────────────────────────────
    total = len(df)
    fp = df[(df["pred_label"] == 1) & (df["is_pathologic"] == 0)]
    fn = df[(df["pred_label"] == 0) & (df["is_pathologic"] == 1)]
    tp = df[(df["pred_label"] == 1) & (df["is_pathologic"] == 1)]
    tn = df[(df["pred_label"] == 0) & (df["is_pathologic"] == 0)]

    print(f"\n{'='*55}")
    print(f"  {args.run}  (n={total})")
    print(f"{'='*55}")
    print(f"  TP {len(tp):4d}  FP {len(fp):4d}")
    print(f"  FN {len(fn):4d}  TN {len(tn):4d}")
    print()

    for name, sub in [("FP (false alarms)", fp), ("FN (missed abnormal)", fn),
                      ("TP (correct abnormal)", tp), ("TN (correct normal)", tn)]:
        conf = sub["confidence"]
        asym = sub["asym_putamen"]
        print(f"  {name} (n={len(sub)}):")
        print(f"    confidence  mean {conf.mean():.3f}  median {conf.median():.3f}")
        print(f"    asym_put    mean {asym.mean():.3f}  median {asym.median():.3f}")
        print(f"    uncertain (<0.15): {int((conf < 0.15).sum())}  "
              f"confident (>0.30): {int((conf > 0.30).sum())}")
        print()

    # ── Scatter plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for mask, color, label in [
        (df["correct"] & (df["is_pathologic"] == 1), "steelblue", "TP"),
        (df["correct"] & (df["is_pathologic"] == 0), "lightblue", "TN"),
        (~df["correct"] & (df["is_pathologic"] == 1), "crimson", "FN"),
        (~df["correct"] & (df["is_pathologic"] == 0), "orange", "FP"),
    ]:
        sub = df[mask]
        ax.scatter(sub["asym_putamen"], sub["pred"], alpha=0.4, s=10,
                   color=color, label=f"{label} (n={len(sub)})")
    ax.axhline(0.5, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Putamen asymmetry index (|L-R|/mean)")
    ax.set_ylabel("Model prediction")
    ax.set_title("Prediction vs asymmetry — all scans")
    ax.legend(fontsize=8)

    ax = axes[1]
    bins = np.linspace(0, 1, 30)
    for mask, color, label in [
        (df["correct"], "steelblue", "correct"),
        (~df["correct"], "crimson", "wrong"),
    ]:
        ax.hist(df.loc[mask, "pred"], bins=bins, alpha=0.5,
                color=color, label=label, density=True)
    ax.axvline(0.5, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Model prediction")
    ax.set_ylabel("Density")
    ax.set_title("Prediction distribution: correct vs wrong")
    ax.legend()

    fig.suptitle(args.run)
    fig.tight_layout()
    scatter_path = out / f"pred_analysis_{args.run}.png"
    fig.savefig(scatter_path, dpi=140)
    plt.close(fig)
    print(f"  scatter: {scatter_path.name}")

    # ── Galleries ──────────────────────────────────────────────────────────────
    # FP sorted by asymmetry descending — healthy but model thinks diseased
    gallery(fp.copy(), crops_dir,
            "False positives — normal scans called abnormal (sorted by asymmetry)",
            out / "gallery_fp_asym.png",
            sort_col="asym_putamen", ascending=False)

    # FN sorted by asymmetry ascending — missed abnormals with low asymmetry (bilateral)
    gallery(fn.copy(), crops_dir,
            "False negatives — missed abnormals (sorted by asymmetry, most bilateral first)",
            out / "gallery_fn_bilateral.png",
            sort_col="asym_putamen", ascending=True)

    # borderline correct (uncertain but right)
    uncertain_correct = df[df["correct"] & (df["confidence"] < 0.15)].copy()
    gallery(uncertain_correct, crops_dir,
            "Uncertain but correct (|pred-0.5|<0.15)",
            out / "gallery_uncertain_correct.png",
            sort_col="confidence", ascending=True)

    # confident wrong
    confident_wrong = df[~df["correct"] & (df["confidence"] > 0.30)].copy()
    gallery(confident_wrong, crops_dir,
            "Confident but wrong (|pred-0.5|>0.30) — the hard core",
            out / "gallery_confident_wrong.png",
            sort_col="confidence", ascending=False)

    print(f"\nall outputs in {out}/")


if __name__ == "__main__":
    main()
