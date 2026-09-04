import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.config import AnalysisConfig
from training.config import TrainConfig

ACFG = AnalysisConfig()
TCFG = TrainConfig()


def crop_stats(crop):
    v = crop[crop > 0]
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "var": float(v.var()),
        "p95": float(np.percentile(v, 95)),
    }


def outcome(row):
    wrong = (row["pred"] > 0.5) != (row["is_pathologic"] > 0.5)
    conf = abs(row["pred"] - 0.5) > 0.30
    base = ("F" if wrong else "T") + ("P" if row["pred"] > 0.5 else "N")
    return base + ("_conf" if conf else "_border")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="efficientnet_b0_mip")
    args = ap.parse_args()
    out_dir = ACFG.output_dir / "signal_stats"
    out_dir.mkdir(exist_ok=True)

    folds = pd.read_csv(TCFG.folds_csv)
    print(f"loading {len(folds)} crops ...")
    crops, rows = {}, []
    for uid in folds["uid"]:
        c = np.load(TCFG.crops_dir / f"{uid}.npy").astype(np.float32)
        crops[uid] = c
        rows.append({"uid": uid, **crop_stats(c)})
    df = pd.DataFrame(rows).merge(folds, on="uid")
    df.to_csv(out_dir / "per_crop_stats.csv", index=False)

    print("\nper-crop signal statistics by expert label "
          "(intensities are in units of the scan's whole-head mean):")
    for stat in ("var", "mean", "median"):
        a = df.loc[df.is_pathologic == 1, stat]
        b = df.loc[df.is_pathologic == 0, stat]
        _, p = stats.mannwhitneyu(a, b)
        print(f"  {stat:>6}: normal median {b.median():.3f} (IQR {b.quantile(.25):.3f}-"
              f"{b.quantile(.75):.3f}), abnormal median {a.median():.3f} "
              f"(IQR {a.quantile(.25):.3f}-{a.quantile(.75):.3f}), MW p={p:.1e}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, stat in zip(axes, ("var", "mean", "median")):
        data = [df.loc[df.is_pathologic == v, stat] for v in (0.0, 1.0)]
        parts = ax.violinplot(data, showmedians=True)
        for body, color in zip(parts["bodies"], ("#4878a8", "#c1554e")):
            body.set_facecolor(color)
        ax.set_xticks([1, 2], ["normal", "abnormal"])
        ax.set_title(f"crop {stat}")
    fig.suptitle("Crop signal statistics by expert label (normalized intensity)")
    fig.tight_layout()
    fig.savefig(out_dir / "group_stats.png", dpi=140)
    plt.close(fig)

    stack = np.stack([crops[u] for u in folds["uid"]])
    labels = folds["is_pathologic"].values
    fig, axes = plt.subplots(3, 5, figsize=(15, 8.5))
    zc = stack.shape[3] // 2
    for r, (name, sel) in enumerate(
        (("all scans", slice(None)), ("normal", labels == 0), ("abnormal", labels == 1))
    ):
        std_map = stack[sel].std(axis=0)
        for i, dz in enumerate((-6, -3, 0, 3, 6)):
            im = axes[r, i].imshow(np.rot90(std_map[:, :, zc + dz]), cmap="magma")
            axes[r, i].set_title(f"{name}, z {dz:+d}", fontsize=9)
            axes[r, i].axis("off")
        fig.colorbar(im, ax=axes[r, -1], fraction=0.046)
    fig.suptitle("Across-crop variability: voxel-wise std over scans (axial slices)")
    fig.tight_layout()
    fig.savefig(out_dir / "voxelwise_std.png", dpi=140)
    plt.close(fig)

    oof = pd.read_csv(TCFG.runs_dir / args.run / "oof.csv")
    m = df.merge(oof[["uid", "pred"]], on="uid")
    m["outcome"] = m.apply(outcome, axis=1)
    feats = pd.read_csv(ACFG.output_dir / "features.csv")
    qc = pd.read_csv(ACFG.output_dir / "crop_qc.csv")
    m = m.merge(feats[["uid", "sbr_putamen_min", "asym_putamen"]], on="uid")
    m = m.merge(qc[["uid", "contrast", "noise_cv"]], on="uid")

    print(f"\nsignal patterns by outcome ({args.run}, threshold 0.5, "
          "conf = |pred-0.5|>0.30):")
    table = m.groupby("outcome").agg(
        n=("uid", "size"), crop_var=("var", "median"), crop_mean=("mean", "median"),
        sbr_put_min=("sbr_putamen_min", "median"), asym=("asym_putamen", "median"),
        contrast=("contrast", "median"), noise_cv=("noise_cv", "median"),
    ).sort_index()
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))
    table.to_csv(out_dir / "outcome_signal_patterns.csv")

    order = [o for o in ("TN_conf", "TN_border", "FP_border", "FP_conf",
                         "FN_conf", "FN_border", "TP_border", "TP_conf")
             if o in set(m["outcome"])]
    fig, axes = plt.subplots(1, len(order), figsize=(2.6 * len(order), 3.2))
    for ax, o in zip(axes, order):
        uids = m.loc[m["outcome"] == o, "uid"]
        mean_crop = np.mean([crops[u] for u in uids], axis=0)
        ax.imshow(np.rot90(mean_crop.max(axis=2)), cmap="hot")
        ax.set_title(f"{o}\nn={len(uids)}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Mean crop (axial MIP) per prediction outcome")
    fig.tight_layout()
    fig.savefig(out_dir / "outcome_mean_crops.png", dpi=140)
    print(f"\nfigures in {out_dir}: group_stats.png, voxelwise_std.png, "
          "outcome_mean_crops.png")


if __name__ == "__main__":
    main()
