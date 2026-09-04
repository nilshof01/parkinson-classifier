import numpy as np
import torch


class VolumeWithAsymmetry:
    """3D twin of the mipasym idea: channel 1 = standardized volume, channel 2 =
    signed left-minus-mirrored-right difference volume (symmetric anatomy cancels
    to ~0, unilateral differences stand at full contrast in 3D)."""

    name = "volume3dasym"
    channels = 2

    def __call__(self, vol):
        v = np.ascontiguousarray(vol, dtype=np.float32)
        asym = v - v[::-1]
        t = torch.from_numpy(v)
        inside = t > 0
        if inside.any():
            t = (t - t[inside].mean()) / (t[inside].std() + 1e-6)
        a = torch.from_numpy(np.ascontiguousarray(asym))
        a = a / (a.abs().std() + 1e-6)
        return torch.stack([t, a])
