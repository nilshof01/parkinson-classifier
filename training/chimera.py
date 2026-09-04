import numpy as np


class ChimeraMixer:
    """Normal+normal hemisphere chimeras (augmentations.md): left hemisphere of
    one healthy crop, right hemisphere of another, intensity-matched and blended
    over a few voxels at the midline. Label stays normal. Site matching is not
    possible (no site metadata), which augmentations.md lists as preferable."""

    def __init__(self, normal_crops, blend_vox=2):
        self.normal_crops = normal_crops  # list of float16/32 arrays
        self.blend_vox = blend_vox

    def mix(self, vol):
        other = self.normal_crops[np.random.randint(len(self.normal_crops))].astype(np.float32)
        m_v, m_o = vol[vol > 0].mean(), other[other > 0].mean()
        if m_o > 0:
            other = other * (m_v / m_o)
        mid = vol.shape[0] // 2
        w = np.zeros(vol.shape[0], dtype=np.float32)
        w[mid + self.blend_vox :] = 1.0
        ramp = np.linspace(0.0, 1.0, 2 * self.blend_vox + 1, dtype=np.float32)
        w[mid - self.blend_vox : mid + self.blend_vox + 1] = ramp
        w = w[:, None, None]
        return vol * (1 - w) + other * w
