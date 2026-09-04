import numpy as np
import torch
import torch.nn.functional as F


class AdjacentSlices:
    """training.md Approach B (2.5D): three axial slices at [center-k, center,
    center+k] of the crop, stacked as channels. Center = geometric center, which
    stays valid when striatal uptake is nearly absent."""

    name = "slices25d"

    def __init__(self, size=224, k=2):
        self.size = size
        self.k = k

    def __call__(self, vol):
        c = vol.shape[2] // 2
        idx = (c - self.k, c, c + self.k)
        chans = []
        for z in idx:
            s = vol[:, :, int(np.clip(z, 0, vol.shape[2] - 1))]
            t = torch.from_numpy(np.ascontiguousarray(s, dtype=np.float32))[None, None]
            t = F.interpolate(t, size=(self.size, self.size), mode="bilinear",
                              align_corners=False)
            chans.append(t[0, 0])
        x = torch.stack(chans)
        return (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-6)
