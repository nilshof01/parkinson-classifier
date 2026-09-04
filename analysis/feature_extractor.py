import numpy as np


class FeatureExtractor:
    """Semi-quantitative ROI features per scan: specific binding ratios (SBR),
    putamen/caudate ratios and left/right asymmetry indices."""

    FEATURES = [
        "sbr_caudate_l", "sbr_caudate_r", "sbr_putamen_l", "sbr_putamen_r",
        "sbr_caudate_min", "sbr_putamen_min", "sbr_striatum_mean",
        "put_caud_ratio_l", "put_caud_ratio_r", "put_caud_ratio_min",
        "asym_caudate", "asym_putamen",
    ]

    def __init__(self, config, roi_masks):
        self.config = config
        self.roi_masks = roi_masks

    def extract(self, frame):
        occ = self._occipital_reference(frame)
        if not np.isfinite(occ) or occ <= 0:
            return {f: np.nan for f in self.FEATURES}
        sbr = {
            name: (self.roi_masks.mean_counts(frame, name) - occ) / occ
            for name in ("caudate_l", "caudate_r", "putamen_l", "putamen_r")
        }
        eps = 1e-6
        pc_l = sbr["putamen_l"] / max(sbr["caudate_l"], eps)
        pc_r = sbr["putamen_r"] / max(sbr["caudate_r"], eps)
        return {
            "sbr_caudate_l": sbr["caudate_l"],
            "sbr_caudate_r": sbr["caudate_r"],
            "sbr_putamen_l": sbr["putamen_l"],
            "sbr_putamen_r": sbr["putamen_r"],
            "sbr_caudate_min": min(sbr["caudate_l"], sbr["caudate_r"]),
            "sbr_putamen_min": min(sbr["putamen_l"], sbr["putamen_r"]),
            "sbr_striatum_mean": float(np.mean(list(sbr.values()))),
            "put_caud_ratio_l": pc_l,
            "put_caud_ratio_r": pc_r,
            "put_caud_ratio_min": min(pc_l, pc_r),
            "asym_caudate": self._asym(sbr["caudate_l"], sbr["caudate_r"]),
            "asym_putamen": self._asym(sbr["putamen_l"], sbr["putamen_r"]),
        }

    def _occipital_reference(self, frame):
        # placed per scan off the head mask's own posterior extent, so it stays
        # inside the head regardless of head size or centroid shift
        cx, _, cz = self.config.frame_center
        nz = frame > 0
        ys = np.where(nz.any(axis=(0, 2)))[0]
        if ys.size == 0:
            return np.nan
        band = frame[cx - 15 : cx + 15, ys.min() + 2 : ys.min() + 9, cz - 4 : cz + 8]
        vals = band[band > 0]
        return float(vals.mean()) if vals.size else np.nan

    @staticmethod
    def _asym(left, right):
        denom = abs(left) + abs(right)
        return abs(left - right) / denom if denom > 1e-6 else np.nan
