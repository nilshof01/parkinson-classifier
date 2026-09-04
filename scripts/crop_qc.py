import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.aligner import Aligner
from analysis.config import AnalysisConfig
from analysis.crop_features import CropFeatures
from analysis.crop_qc import CropQc
from analysis.dataset_index import DatasetIndex
from analysis.frame_cache import FrameCache
from analysis.preprocessor import Preprocessor
from training.config import TrainConfig

ACFG = AnalysisConfig()
TCFG = TrainConfig()
_W = {}


def _init_worker():
    _W["cache"] = FrameCache(ACFG)
    _W["pre"] = Preprocessor(ACFG)
    _W["aligner"] = Aligner(ACFG)
    _W["mean"] = np.load(ACFG.output_dir / "cohort_mean_frame.npy")
    _W["qc"] = CropQc(ACFG)
    _W["index"] = DatasetIndex(ACFG)


def _one(uid):
    try:
        if _W["cache"].has(uid):
            frame = _W["cache"].load(uid)
        else:
            frame, _info = _W["pre"].process(_W["index"].path_for(uid))
        m = frame[frame > 0].mean()
        frame = frame / m
        for _ in range(2):
            frame, _s = _W["aligner"].align(frame, _W["mean"])
        return uid, _W["qc"].assess(frame)
    except Exception as e:
        return uid, {"error": repr(e)}


def main():
    folds = pd.read_csv(TCFG.folds_csv)
    print(f"assessing crops of {len(folds)} scans ...")
    rows = []
    with ProcessPoolExecutor(12, initializer=_init_worker) as pool:
        for i, (uid, res) in enumerate(pool.map(_one, folds["uid"], chunksize=8)):
            rows.append({"uid": uid, **res})
            if (i + 1) % 300 == 0:
                print(f"  {i + 1}/{len(folds)}")
    df = pd.DataFrame(rows).merge(folds[["uid", "is_pathologic"]], on="uid")
    headers = pd.read_csv(ACFG.output_dir / "header_stats.csv")
    df = df.merge(headers[["uid", "spacing_x"]], on="uid")
    df["spacing_group"] = df["spacing_x"].round(1)
    df.to_csv(ACFG.output_dir / "crop_qc.csv", index=False)

    n = len(df)
    blob = df[df["clear_blob"]]
    print(f"\nclear striatal blob (contrast >= {CropQc.LOW_CONTRAST}): "
          f"{len(blob)}/{n} ({100 * len(blob) / n:.1f}%)")
    for name, sub in (("clear-blob scans", blob), ("low-contrast scans", df[~df["clear_blob"]])):
        inside = sub["peak_in_crop"].sum()
        print(f"{name}: peak inside crop {inside}/{len(sub)} "
              f"({100 * inside / max(len(sub), 1):.1f}%), "
              f"margin p5 {sub['margin_vox'].quantile(0.05):.1f} vox, "
              f"min {sub['margin_vox'].min():.1f} vox")
    tight = df[(df["margin_vox"] < 3) & df["clear_blob"]]
    print(f"clear-blob peaks within 3 vox of crop edge or outside: {len(tight)}/{len(blob)}")
    print(f"boundary_ratio: median {df['boundary_ratio'].median():.3f}, "
          f"p95 {df['boundary_ratio'].quantile(0.95):.3f}, "
          f">0.9: {(df['boundary_ratio'] > 0.9).sum()}/{n}")
    print(f"zero_frac: mean {df['zero_frac'].mean():.4f}, max {df['zero_frac'].max():.3f}")
    print("\nnoise by spacing group (noise_cv = high-freq residual std / crop mean):")
    g = df.groupby("spacing_group").agg(
        n=("uid", "size"), noise_cv=("noise_cv", "median"), snr_peak=("snr_peak", "median")
    )
    print(g[g["n"] >= 30].to_string(float_format=lambda x: f"{x:.3f}"))

    cf = CropFeatures()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(df["margin_vox"], bins=40, color="#4878a8")
    axes[0, 0].axvline(0, color="k", ls="--", lw=0.8)
    axes[0, 0].set_xlabel("peak distance to crop edge (vox, <0 = outside)")
    axes[0, 1].hist(df["boundary_ratio"].dropna(), bins=40, color="#4878a8")
    axes[0, 1].set_xlabel("boundary max / interior max")
    for label, color in ((0.0, "#4878a8"), (1.0, "#c1554e")):
        s = df[df["is_pathologic"] == label]
        axes[1, 0].scatter(s["peak_x"], s["peak_y"], s=4, alpha=0.3, color=color,
                           label="normal" if label == 0 else "abnormal")
    axes[1, 0].add_patch(plt.Rectangle((cf.X.start, cf.Y.start),
                                       cf.X.stop - cf.X.start, cf.Y.stop - cf.Y.start,
                                       fill=False, color="k"))
    axes[1, 0].set_xlabel("peak x (vox)"); axes[1, 0].set_ylabel("peak y (vox)")
    axes[1, 0].legend()
    df.boxplot(column="noise_cv", by="spacing_group", ax=axes[1, 1])
    axes[1, 1].set_xlabel("native spacing group (mm)"); axes[1, 1].set_ylabel("noise_cv")
    fig.suptitle("Crop QC")
    fig.tight_layout()
    out = ACFG.figures_dir / "crop_qc.png"
    fig.savefig(out, dpi=140)
    print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()
