import numpy as np
from scipy import ndimage


class RoiCalibrator:
    """Derives caudate/putamen/occipital ROI centers ONCE from the cohort mean
    image, so per-scan ROI placement never depends on that scan's striatal
    signal (which can be nearly absent in abnormal scans)."""

    def __init__(self, config):
        self.config = config

    def calibrate(self, mean_frame):
        cx, cy, cz = self.config.frame_center
        smooth = ndimage.gaussian_filter(mean_frame, 1.0)

        search = np.zeros_like(smooth, dtype=bool)
        search[cx - 22 : cx + 22, cy - 8 : cy + 30, cz - 8 : cz + 20] = True
        vals = smooth[search]
        thr = np.percentile(vals[vals > 0], 97.0)
        hot = (smooth > thr) & search

        left = self._hemisphere_blob(hot, side_slice=np.s_[:cx])
        right = self._hemisphere_blob(hot, side_slice=np.s_[cx:], x_offset=cx)
        rois = {}
        for name, blob in (("l", left), ("r", right)):
            caud, put = self._split_blob(blob, smooth)
            rois[f"caudate_{name}"] = caud
            rois[f"putamen_{name}"] = put

        brain = smooth > np.percentile(smooth[smooth > 0], 25)
        ys = np.where(brain.any(axis=(0, 2)))[0]
        occ_y = int(ys.min() + 4)
        rois["occipital"] = ("box", (cx - 15, occ_y, cz - 4), (cx + 15, occ_y + 8, cz + 8))
        return rois

    def _hemisphere_blob(self, hot, side_slice, x_offset=0):
        half = np.zeros_like(hot)
        half[side_slice] = hot[side_slice]
        labeled, n = ndimage.label(half)
        if n == 0:
            raise RuntimeError("no striatal blob found in mean image")
        sizes = ndimage.sum(half, labeled, range(1, n + 1))
        return labeled == (int(np.argmax(sizes)) + 1)

    def _split_blob(self, blob, smooth):
        coords = np.array(np.where(blob)).T.astype(float)
        centered = coords - coords.mean(axis=0)
        # principal axis of the striatal blob runs caudate head (anterior-medial)
        # to posterior putamen; take its endpoints, not a plain y-split
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]
        if axis[1] < 0:
            axis = -axis
        proj = centered @ axis
        caud_pts = coords[proj >= np.percentile(proj, 72)].astype(int)
        put_pts = coords[proj <= np.percentile(proj, 40)].astype(int)
        caud = self._weighted_center(caud_pts, smooth)
        put = self._weighted_center(put_pts, smooth)
        r_c, r_p = self.config.caudate_radius_vox, self.config.putamen_radius_vox
        return ("sphere", caud, r_c), ("sphere", put, r_p)

    @staticmethod
    def _weighted_center(points, smooth):
        w = smooth[points[:, 0], points[:, 1], points[:, 2]]
        return tuple(np.round((points * w[:, None]).sum(axis=0) / w.sum()).astype(int))
