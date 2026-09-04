import numpy as np
from scipy import ndimage

from analysis.crop_features import CropFeatures


class CropQc:
    """Per-scan crop quality metrics on the aligned, normalized frame:
    does the hottest central structure lie inside the crop (and how far from
    its edge), does bright signal touch the crop boundary (possible cut), and
    how noisy is the crop content."""

    SEARCH_MARGIN = 10  # voxels beyond the crop box for the hotspot search
    LOW_CONTRAST = 1.5  # peak/occipital ratio below this = no clear blob

    def __init__(self, config):
        self.config = config
        cf = CropFeatures()
        self.crop = (cf.X, cf.Y, cf.Z)

    def assess(self, frame):
        smooth = ndimage.gaussian_filter(frame, 1.0)
        (xs, ys, zs) = self.crop
        lo = np.array([xs.start, ys.start, zs.start])
        hi = np.array([xs.stop, ys.stop, zs.stop])

        s_lo = np.maximum(lo - self.SEARCH_MARGIN, 0)
        s_hi = np.minimum(hi + self.SEARCH_MARGIN, frame.shape)
        search = smooth[s_lo[0]:s_hi[0], s_lo[1]:s_hi[1], s_lo[2]:s_hi[2]]
        peak = np.array(np.unravel_index(np.argmax(search), search.shape)) + s_lo
        peak_val = float(smooth[tuple(peak)])

        occ = self._occipital(frame)
        contrast = peak_val / occ if occ and occ > 0 else np.nan
        margin = float(np.minimum(peak - lo, hi - 1 - peak).min())

        crop_s = smooth[xs, ys, zs]
        interior_max = float(crop_s[3:-3, 3:-3, 3:-3].max())
        shell = crop_s.copy()
        shell[1:-1, 1:-1, 1:-1] = 0
        boundary_ratio = float(shell.max() / interior_max) if interior_max > 0 else np.nan

        crop_raw = frame[xs, ys, zs]
        inside = crop_raw > 0
        resid = crop_raw - ndimage.gaussian_filter(crop_raw, 1.0)
        noise_std = float(resid[inside].std()) if inside.any() else np.nan
        mean_sig = float(crop_raw[inside].mean()) if inside.any() else np.nan

        return {
            "peak_x": int(peak[0]), "peak_y": int(peak[1]), "peak_z": int(peak[2]),
            "peak_val": peak_val,
            "contrast": float(contrast),
            "clear_blob": bool(np.isfinite(contrast) and contrast >= self.LOW_CONTRAST),
            "peak_in_crop": bool(margin >= 0),
            "margin_vox": margin,
            "boundary_ratio": boundary_ratio,
            "noise_std": noise_std,
            "noise_cv": noise_std / mean_sig if mean_sig else np.nan,
            "snr_peak": peak_val / noise_std if noise_std else np.nan,
            "zero_frac": float((~inside).mean()),
        }

    def _occipital(self, frame):
        cx, _, cz = self.config.frame_center
        nz = frame > 0
        ys = np.where(nz.any(axis=(0, 2)))[0]
        if ys.size == 0:
            return np.nan
        band = frame[cx - 15 : cx + 15, ys.min() + 2 : ys.min() + 9, cz - 4 : cz + 8]
        vals = band[band > 0]
        return float(vals.mean()) if vals.size else np.nan
