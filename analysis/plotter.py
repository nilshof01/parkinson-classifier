import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """Writes all figures to output/figures (nothing shown interactively)."""

    def __init__(self, config):
        self.config = config

    def _save(self, fig, name):
        path = self.config.figures_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path

    def geometry(self, header_df):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        axes[0].hist(header_df["spacing_x"], bins=40, color="#4878a8")
        axes[0].set_xlabel("in-plane voxel spacing (mm)")
        axes[0].set_ylabel("scans")
        axes[1].hist(header_df["dim_x"], bins=40, color="#4878a8")
        axes[1].set_xlabel("matrix size (x)")
        axes[2].hist(header_df["fov_z_mm"], bins=40, color="#4878a8")
        axes[2].set_xlabel("axial field of view (mm)")
        fig.suptitle("Acquisition geometry across the training set")
        return self._save(fig, "geometry.png")

    def qc(self, qc_df):
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for label, color, name in ((0.0, "#4878a8", "normal"), (1.0, "#c1554e", "abnormal")):
            sub = qc_df[qc_df["is_pathologic"] == label]["mask_volume_ml"]
            ax.hist(sub, bins=50, alpha=0.6, color=color, label=name)
        lo, hi = self.config.mask_volume_ml_bounds
        for v in (lo, hi):
            ax.axvline(v, color="k", ls="--", lw=0.8)
        ax.set_xlabel("head-mask volume (mL)")
        ax.set_ylabel("scans")
        ax.legend()
        ax.set_title("Head-mask volume QC (dashed = accepted range)")
        return self._save(fig, "qc_mask_volume.png")

    def mean_image(self, mean_frame, roi_masks):
        cz = self.config.frame_center[2]
        roi_union = np.zeros(self.config.frame_shape, dtype=bool)
        for m in roi_masks.masks.values():
            roi_union |= m
        offsets = (-6, -3, 0, 3, 6)
        fig, axes = plt.subplots(1, len(offsets), figsize=(3 * len(offsets), 3.4))
        for ax, dz in zip(axes, offsets):
            sl = np.rot90(mean_frame[:, :, cz + dz])
            ax.imshow(sl, cmap="hot")
            ax.contour(np.rot90(roi_union[:, :, cz + dz]), colors="cyan", linewidths=0.8)
            ax.set_title(f"z = center{dz:+d}")
            ax.axis("off")
        fig.suptitle("Cohort mean image (axial) with calibrated ROIs (cyan)")
        return self._save(fig, "mean_image_rois.png")

    def feature_distributions(self, df, features):
        n = len(features)
        cols = 3
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows))
        for ax, f in zip(axes.ravel(), features):
            data = [
                df.loc[(df["is_pathologic"] == v) & np.isfinite(df[f]), f]
                for v in (0.0, 1.0)
            ]
            parts = ax.violinplot(data, showmedians=True)
            for body, color in zip(parts["bodies"], ("#4878a8", "#c1554e")):
                body.set_facecolor(color)
            ax.set_xticks([1, 2], ["normal", "abnormal"])
            ax.set_title(f, fontsize=10)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.suptitle("ROI feature distributions by expert label")
        return self._save(fig, "feature_distributions.png")

    def roc(self, per_feature_df, cv_result):
        from sklearn.metrics import roc_curve

        fig, ax = plt.subplots(figsize=(5.2, 5))
        fpr, tpr, _ = roc_curve(cv_result["y"], cv_result["oof"])
        ax.plot(fpr, tpr, color="#c1554e", lw=2,
                label=f"logistic (all features), AUROC={cv_result['oof_auroc']:.3f}")
        top = per_feature_df.iloc[0]
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_title(f"Out-of-fold ROC (best single feature: {top['feature']}, "
                     f"AUROC={top['auroc']:.3f})")
        ax.legend(loc="lower right")
        return self._save(fig, "roc.png")
