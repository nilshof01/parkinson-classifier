"""Fold-level analysis: why do some folds validate much better than others?

Loads per-fold val_preds.csv, folds.csv, features.csv, and header_stats.csv,
then characterises each fold's validation set and scores to find what differs.

Usage (Windows):
    $env:SCAN_REPO = "C:\Users\nilsh\Projects\DaT Parkinson's Challenge\parkinson-classifier"
    python scripts/fold_analysis.py `
        --run "C:\Users\nilsh\Projects\DaT Parkinson's Challenge\baseline\cnn3d_m4_baseline_s0" `
        --folds-csv "C:\Users\nilsh\Projects\DaT Parkinson's Challenge\prepared\folds.csv"

Usage (pod):
    export SCAN_REPO=/workspace/parkinson-classifier
    python scripts/fold_analysis.py --run $SCAN_REPO/output/runs/cnn3d_m4_baseline_s0
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.config import AnalysisConfig
from training.config import TrainConfig

ACFG = AnalysisConfig()
TCFG = TrainConfig()


def calibrated_ll(y, p):
    from scipy.optimize import minimize_scalar
    lo = logit(np.clip(p, 1e-6, 1 - 1e-6))
    t = minimize_scalar(
        lambda t: log_loss(y, np.clip(expit(lo / t), 1e-6, 1 - 1e-6)),
        bounds=(0.25, 10.0), method="bounded").x
    return float(log_loss(y, np.clip(expit(lo / t), 0.02, 0.98)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="path to run directory")
    ap.add_argument("--folds-csv", default=None,
                    help="path to folds.csv (default: CFG.folds_csv)")
    args = ap.parse_args()

    run_dir = Path(args.run)
    folds_path = Path(args.folds_csv) if args.folds_csv else TCFG.folds_csv
    folds = pd.read_csv(folds_path)

    feats = pd.read_csv(ACFG.output_dir / "features.csv")
    hdr = pd.read_csv(ACFG.output_dir / "header_stats.csv")

    # merge everything by uid
    meta = folds.merge(feats[["uid", "sbr_putamen_min", "sbr_putamen_l",
                               "sbr_putamen_r", "asym_putamen"]], on="uid", how="left")
    meta = meta.merge(hdr[["uid", "spacing_x"]], on="uid", how="left")
    meta["spacing_group"] = meta["spacing_x"].round(1)

    # ── Per-fold scores and val-set characterisation ───────────────────────────
    fold_ids = sorted(folds["fold"].unique())
    rows = []
    preds_by_fold = {}

    for k in fold_ids:
        pred_path = run_dir / f"fold{k}" / "val_preds.csv"
        if not pred_path.exists():
            print(f"  fold {k}: val_preds.csv missing, skipping")
            continue
        vp = pd.read_csv(pred_path)
        y = vp["is_pathologic"].values
        p = vp["pred"].values
        auc = roc_auc_score(y, p)
        ll = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
        cal_ll = calibrated_ll(y, p)

        val_meta = meta[meta["fold"] == k]
        rows.append({
            "fold": k,
            "n_val": len(vp),
            "prevalence": float(y.mean()),
            "auc": auc,
            "val_ll": ll,
            "cal_ll": cal_ll,
            "sbr_min_median": float(val_meta.loc[val_meta["is_pathologic"] == 1,
                                                  "sbr_putamen_min"].median()),
            "sbr_min_mean": float(val_meta.loc[val_meta["is_pathologic"] == 1,
                                               "sbr_putamen_min"].mean()),
            "asym_median": float(val_meta["asym_putamen"].median()),
            "pct_1p5mm": float((val_meta["spacing_group"] == 1.5).mean()),
            "pct_3p3mm": float((val_meta["spacing_group"] == 3.3).mean()),
            "pct_3p9mm": float((val_meta["spacing_group"] == 3.9).mean()),
            "pct_2p5mm": float((val_meta["spacing_group"] == 2.5).mean()),
        })
        preds_by_fold[k] = vp.merge(
            meta[["uid", "sbr_putamen_min", "asym_putamen", "spacing_group"]], on="uid")

    df = pd.DataFrame(rows).set_index("fold")

    print("\n" + "="*70)
    print("  Per-fold scores and validation set characteristics")
    print("="*70)
    print(df[["n_val", "prevalence", "auc", "val_ll", "cal_ll"]].round(4).to_string())
    print()
    print("  Validation set composition:")
    print(df[["sbr_min_median", "sbr_min_mean", "asym_median",
              "pct_1p5mm", "pct_3p3mm", "pct_3p9mm", "pct_2p5mm"]].round(3).to_string())

    # ── Which scans drive the fold difference ──────────────────────────────────
    # Compare SBR distribution of abnormal val scans across folds
    print("\n  Abnormal val scans — SBR putamen min percentiles:")
    print(f"  {'fold':>5}  {'p10':>6}  {'p25':>6}  {'p50':>6}  {'p75':>6}  {'p90':>6}")
    for k, vp in preds_by_fold.items():
        abn = vp[vp["is_pathologic"] == 1]["sbr_putamen_min"].dropna()
        pcts = np.percentile(abn, [10, 25, 50, 75, 90])
        print(f"  {k:>5}  " + "  ".join(f"{v:6.3f}" for v in pcts))

    # ── Plots ──────────────────────────────────────────────────────────────────
    out = ACFG.output_dir / "error_analysis"
    out.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # AUC and ll per fold
    ax = axes[0, 0]
    ax.bar(df.index, df["auc"], color=["steelblue" if v > 0.96 else "salmon"
                                        for v in df["auc"]])
    ax.axhline(df["auc"].mean(), ls="--", color="black", lw=0.8)
    ax.set_title("Val AUROC per fold")
    ax.set_ylim(0.92, 1.0)
    ax.set_xlabel("fold")

    ax = axes[0, 1]
    ax.bar(df.index, df["cal_ll"], color=["steelblue" if v < df["cal_ll"].mean()
                                           else "salmon" for v in df["cal_ll"]])
    ax.axhline(df["cal_ll"].mean(), ls="--", color="black", lw=0.8)
    ax.set_title("Val log-loss (calibrated) per fold")
    ax.set_xlabel("fold")

    # SBR distribution of abnormals per fold
    ax = axes[0, 2]
    for k, vp in preds_by_fold.items():
        abn = vp[vp["is_pathologic"] == 1]["sbr_putamen_min"].dropna()
        ax.hist(abn, bins=20, alpha=0.4, label=f"fold {k}", density=True)
    ax.set_title("SBR dist of abnormal val scans per fold")
    ax.set_xlabel("sbr_putamen_min")
    ax.legend(fontsize=7)

    # Spacing group distribution per fold
    ax = axes[1, 0]
    spacing_cols = ["pct_1p5mm", "pct_3p3mm", "pct_3p9mm", "pct_2p5mm"]
    spacing_labels = ["1.5mm", "3.3mm", "3.9mm", "2.5mm"]
    x = np.arange(len(fold_ids))
    width = 0.2
    for i, (col, lab) in enumerate(zip(spacing_cols, spacing_labels)):
        ax.bar(x + i * width, df[col], width, label=lab)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"fold {k}" for k in fold_ids])
    ax.set_title("Spacing group % per fold")
    ax.legend(fontsize=7)

    # Prediction distribution per fold for errors only
    ax = axes[1, 1]
    for k, vp in preds_by_fold.items():
        wrong = vp[(vp["pred"] > 0.5) != (vp["is_pathologic"] > 0.5)]
        if len(wrong):
            ax.hist(wrong["pred"], bins=15, alpha=0.4,
                    label=f"fold {k} ({len(wrong)} err)", density=True)
    ax.axvline(0.5, color="black", lw=0.8, ls="--")
    ax.set_title("Prediction dist of wrong scans per fold")
    ax.set_xlabel("prediction")
    ax.legend(fontsize=7)

    # Asymmetry distribution per fold (abnormal val scans)
    ax = axes[1, 2]
    for k, vp in preds_by_fold.items():
        abn = vp[vp["is_pathologic"] == 1]["asym_putamen"].dropna()
        ax.hist(abn, bins=20, alpha=0.4, label=f"fold {k}", density=True)
    ax.set_title("Putamen asymmetry dist — abnormal val scans")
    ax.set_xlabel("asym_putamen")
    ax.legend(fontsize=7)

    fig.suptitle(f"Fold analysis — {run_dir.name}", fontsize=12)
    fig.tight_layout()
    plot_path = out / f"fold_analysis_{run_dir.name}.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    print(f"\n  plot: {plot_path}")


if __name__ == "__main__":
    main()
