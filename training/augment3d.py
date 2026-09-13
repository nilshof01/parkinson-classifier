import numpy as np
from scipy import ndimage


class Augment3D:
    """Per-epoch stochastic augmentation on the 3D striatal crop, per
    augmentations.md. Order: flip -> geometric -> resolution -> multiplicative
    field -> global scaling -> noise. Crops are in-head-mean normalized, so
    intensities are ~O(1)."""

    def __init__(
        self,
        p_flip=0.5,
        p_geom=0.7, rot_deg=8.0, shift_vox=2.0,
        p_zoom=0.5, zoom_range=(0.9, 1.1),
        p_aniso=0.0, aniso_range=(0.9, 1.1),
        p_res=0.5, sigma_vox=(0.25, 0.75),
        p_field=0.5, field_amp=0.1,
        p_scale=0.8, scale_range=(0.85, 1.15),
        p_noise=0.5, noise_frac=(0.05, 0.15),
        p_gamma=0.0, gamma_range=(0.75, 1.35),
    ):
        self.p_flip = p_flip
        self.p_geom, self.rot_deg, self.shift_vox = p_geom, rot_deg, shift_vox
        self.p_zoom, self.zoom_range = p_zoom, zoom_range
        self.p_aniso, self.aniso_range = p_aniso, aniso_range
        self.p_res, self.sigma_vox = p_res, sigma_vox
        self.p_field, self.field_amp = p_field, field_amp
        self.p_scale, self.scale_range = p_scale, scale_range
        self.p_noise, self.noise_frac = p_noise, noise_frac
        self.p_gamma, self.gamma_range = p_gamma, gamma_range

    def __call__(self, vol):
        r = np.random
        if r.random() < self.p_flip:
            vol = vol[::-1].copy()
        if r.random() < self.p_geom:
            angle = r.uniform(-self.rot_deg, self.rot_deg)
            vol = ndimage.rotate(vol, angle, axes=(0, 1), reshape=False, order=1)
            vol = ndimage.shift(vol, r.uniform(-self.shift_vox, self.shift_vox, 3), order=1)
        if r.random() < self.p_zoom:
            s = r.uniform(*self.zoom_range)
            c = (np.array(vol.shape) - 1) / 2
            vol = ndimage.affine_transform(
                vol, np.eye(3) / s, offset=c * (1 - 1 / s), order=1
            )
        if r.random() < self.p_aniso:
            # independent scale per axis — simulates scanner axis-specific
            # resolution differences and slice-thickness variation
            s = r.uniform(*self.aniso_range, size=3)
            c = (np.array(vol.shape) - 1) / 2
            vol = ndimage.affine_transform(
                vol, np.diag(1 / s), offset=c * (1 - 1 / s), order=1
            )
        if r.random() < self.p_res:
            vol = ndimage.gaussian_filter(vol, r.uniform(*self.sigma_vox))
        if r.random() < self.p_field:
            grid = 1.0 + r.uniform(-self.field_amp, self.field_amp, (2, 2, 2))
            field = ndimage.zoom(grid, [s / 2 for s in vol.shape], order=1)
            vol = vol * field
        if r.random() < self.p_gamma:
            # monotonic contrast warp, simulating cross-site reconstruction
            # differences; moderate range because uptake ratios carry the label
            vol = np.maximum(vol, 0) ** r.uniform(*self.gamma_range)
        if r.random() < self.p_scale:
            vol = vol * r.uniform(*self.scale_range)
        if r.random() < self.p_noise:
            c = r.uniform(*self.noise_frac)
            vol = vol + r.normal(0.0, 1.0, vol.shape) * c * np.sqrt(np.maximum(vol, 0))
        return vol.astype(np.float32)
