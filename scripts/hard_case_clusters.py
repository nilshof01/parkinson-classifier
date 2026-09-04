import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.config import AnalysisConfig
from training.config import TrainConfig

ACFG = AnalysisConfig()
TCFG = TrainConfig()
RUNS = ("effb0_mip_ref40", "effb0_mip_chimera25")
CLUSTER_FEATURES = [
    "sbr_putamen_min", "sbr_caudate_min", "asym_putamen", "asym_caudate",
    "put_caud_ratio_min", "var", "mean", "contrast",
]


def main():
    folds = pd.read_csv(TCFG.folds_csv)
    df = folds.copy()
    for r in RUNS:
        oof = pd.read_csv(TCFG.runs_dir / r / "oof.csv")[["uid", "pred"]]
        df = df.merge(oof.rename(columns={"pred": f"pred_{r}"}), on="uid")
    feats = pd.read_csv(ACFG.output_dir / "features.csv")
    stats = pd.read_csv(ACFG.output_dir / "signal_stats" / "per_crop_stats.csv")
    qc = pd.read_csv(ACFG.output_dir / "crop_qc.csv")
    hdr = pd.read_csv(ACFG.output_dir / "header_stats.csv")
    df = (df.merge(feats.drop(columns=["is_pathologic"]), on="uid")
            .merge(stats[["uid", "var", "mean", "median"]], on="uid")
            .merge(qc[["uid", "contrast", "noise_cv"]], on="uid")
            .merge(hdr[["uid", "spacing_x"]], on="uid"))
    df["spacing_group"] = df["spacing_x"].round(1)

    ref = f"pred_{RUNS[0]}"
    wrong_any = np.zeros(len(df), dtype=bool)
    for r in RUNS:
        wrong_any |= (df[f"pred_{r}"] > 0.5) != (df["is_pathologic"] > 0.5)
    borderline = (df[ref] - 0.5).abs() < 0.15
    df["hard"] = wrong_any | borderline
    hard = df[df["hard"]].dropna(subset=CLUSTER_FEATURES).copy()
    print(f"overlap-zone scans: {len(hard)}/{len(df)} "
          f"({100 * len(hard) / len(df):.1f}%) — wrong in a run or |pred-0.5|<0.15")

    X = StandardScaler().fit_transform(hard[CLUSTER_FEATURES])
    best = None
    for k in range(2, 7):
        km = KMeans(k, n_init=10, random_state=17).fit(X)
        s = silhouette_score(X, km.labels_)
        print(f"  k={k}: silhouette {s:.3f}")
        if best is None or s > best[1]:
            best = (k, s, km.labels_)
    k, sil, labels = best
    hard["cluster"] = labels
    print(f"chosen k={k} (silhouette {sil:.3f}; <0.25 = weak structure, "
          ">0.5 = clear clusters)")

    pca = PCA(n_components=2).fit(X)
    print("PCA explained variance:", np.round(pca.explained_variance_ratio_, 2),
          "| PC1 loadings:",
          dict(zip(CLUSTER_FEATURES, np.round(pca.components_[0], 2))))

    print("\ncluster profiles:")
    prof = hard.groupby("cluster").agg(
        n=("uid", "size"),
        frac_abnormal=("is_pathologic", "mean"),
        pred=(ref, "median"),
        sbr_put_min=("sbr_putamen_min", "median"),
        asym_put=("asym_putamen", "median"),
        crop_var=("var", "median"),
        contrast=("contrast", "median"),
    )
    print(prof.to_string(float_format=lambda x: f"{x:.3f}"))
    hard.to_csv(ACFG.output_dir / "error_analysis" / "hard_clusters.csv", index=False)

    proj = pca.transform(X)
    fig, axes = plt.subplots(1, k + 1, figsize=(3.2 * (k + 1), 3.4))
    for c in range(k):
        uids = hard.loc[hard["cluster"] == c, "uid"]
        mean_crop = np.mean(
            [np.load(TCFG.crops_dir / f"{u}.npy").astype(np.float32) for u in uids], axis=0)
        axes[c].imshow(np.rot90(mean_crop.max(axis=2)), cmap="hot")
        axes[c].set_title(f"cluster {c} (n={len(uids)}, "
                          f"{100 * hard.loc[hard.cluster == c, 'is_pathologic'].mean():.0f}% abn)",
                          fontsize=9)
        axes[c].axis("off")
    sc = axes[k].scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", s=10)
    axes[k].set_xlabel("PC1"); axes[k].set_ylabel("PC2"); axes[k].set_title("PCA of hard scans")
    fig.suptitle("Overlap-zone scans: mean crop per cluster (axial MIP)")
    fig.tight_layout()
    out = ACFG.output_dir / "error_analysis" / "hard_clusters.png"
    fig.savefig(out, dpi=140)
    print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()
