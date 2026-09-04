import nibabel as nib
import numpy as np
from scipy import ndimage


class Preprocessor:
    """Reorients to RAS, resamples to isotropic spacing, masks the head,
    and extracts a fixed-size frame centered on the mask centroid."""

    def __init__(self, config, affine_correct=False):
        self.config = config
        # affine_correct=True resamples through the header's full affine, applying
        # the recorded head rotations (~30% of scans carry >2 deg) instead of only
        # snapping to the nearest axis orientation
        self.affine_correct = affine_correct

    def process(self, path):
        img = nib.load(path)
        if self.affine_correct:
            from nibabel.processing import resample_to_output
            img = resample_to_output(img, voxel_sizes=self.config.target_spacing_mm,
                                     order=1, cval=0.0)
            vol = np.asarray(img.dataobj, dtype=np.float32)
            if vol.ndim == 4:
                vol = vol[..., 0]
        else:
            img = nib.as_closest_canonical(img)
            vol = np.asarray(img.dataobj, dtype=np.float32)
            if vol.ndim == 4:
                vol = vol[..., 0]
            zooms = np.array(img.header.get_zooms()[:3], dtype=float)
            factors = zooms / self.config.target_spacing_mm
            vol = ndimage.zoom(vol, factors, order=1)

        mask = self._head_mask(vol)
        voxel_ml = (self.config.target_spacing_mm ** 3) / 1000.0
        mask_ml = float(mask.sum() * voxel_ml)
        if mask.sum() == 0:
            return None, {"mask_volume_ml": 0.0, "qc_pass": False}

        # axial FOV varies from ~160 to ~340 mm across scanners; restrict the
        # centroid to the top 140 mm of the mask so included neck/shoulders
        # cannot drag the anchor away from the brain
        head_mask = mask.copy()
        z_top = int(np.where(mask.any(axis=(0, 1)))[0].max())
        head_vox = int(140.0 / self.config.target_spacing_mm)
        head_mask[:, :, : max(0, z_top - head_vox)] = False
        centroid = np.array(ndimage.center_of_mass(head_mask))
        frame = self._extract_frame(vol * mask, centroid)
        lo, hi = self.config.mask_volume_ml_bounds
        info = {
            "mask_volume_ml": mask_ml,
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_z": float(centroid[2]),
            "qc_pass": bool(lo <= mask_ml <= hi),
        }
        return frame, info

    def _head_mask(self, vol):
        sigma_vox = self.config.smooth_sigma_mm / self.config.target_spacing_mm
        smooth = ndimage.gaussian_filter(vol, sigma_vox)
        thr = self._otsu(smooth[smooth > 0])
        mask = smooth > thr
        labeled, n = ndimage.label(mask)
        if n == 0:
            return mask
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        mask = labeled == (int(np.argmax(sizes)) + 1)
        mask = ndimage.binary_fill_holes(mask)
        mask = ndimage.binary_erosion(mask, iterations=self.config.mask_erode_voxels)
        return mask

    @staticmethod
    def _otsu(values, bins=256):
        hist, edges = np.histogram(values, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        w0 = np.cumsum(hist)
        w1 = w0[-1] - w0
        m0 = np.cumsum(hist * centers)
        mu0 = np.divide(m0, w0, out=np.zeros_like(m0), where=w0 > 0)
        mu1 = np.divide(m0[-1] - m0, w1, out=np.zeros_like(m0), where=w1 > 0)
        between = w0 * w1 * (mu0 - mu1) ** 2
        return centers[int(np.argmax(between))]

    def _extract_frame(self, vol, centroid):
        shape = self.config.frame_shape
        frame = np.zeros(shape, dtype=np.float32)
        start = np.round(centroid).astype(int) - np.array(shape) // 2
        src_lo = np.maximum(start, 0)
        src_hi = np.minimum(start + shape, vol.shape)
        dst_lo = src_lo - start
        dst_hi = dst_lo + (src_hi - src_lo)
        if np.any(src_hi <= src_lo):
            return frame
        frame[
            dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]
        ] = vol[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
        return frame
