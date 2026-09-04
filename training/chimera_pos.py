import numpy as np


class PositiveChimeraMixer:
    """Abnormal+abnormal hemisphere chimeras that avoid the healthy+healthy trap:
    the host keeps its WORSE hemisphere (lower per-side putamen SBR) and the
    donor's worse hemisphere is mirrored onto the other side, so every synthetic
    scan contains two diseased hemispheres and the abnormal label stays valid."""

    def __init__(self, donors, blend_vox=2):
        self.donors = donors  # list of (crop array, worse_side "L"/"R")
        self.blend_vox = blend_vox

    def mix(self, vol, host_worse_side):
        d_vol, d_side = self.donors[np.random.randint(len(self.donors))]
        d = d_vol.astype(np.float32)
        m_v, m_d = vol[vol > 0].mean(), d[d > 0].mean()
        if m_d > 0:
            d = d * (m_v / m_d)
        host_keeps_left = host_worse_side == "L"
        # donor's diseased half must land on the side the host does NOT keep
        if (d_side == "L") == host_keeps_left:
            d = d[::-1].copy()
        n = vol.shape[0]
        ramp = np.zeros(n, dtype=np.float32)
        ramp[n // 2 + self.blend_vox:] = 1.0
        ramp[n // 2 - self.blend_vox: n // 2 + self.blend_vox + 1] = np.linspace(
            0.0, 1.0, 2 * self.blend_vox + 1, dtype=np.float32)
        w_donor = ramp if host_keeps_left else 1.0 - ramp
        w_donor = w_donor[:, None, None]
        return vol * (1.0 - w_donor) + d * w_donor
