import numpy as np
import torch
import torch.nn.functional as F


class TriaxialMip:
    """training.md Approach A: axial/coronal/sagittal maximum-intensity
    projections of the 3D crop, each resized and stacked as 3 channels."""

    name = "mip"

    def __init__(self, size=224):
        self.size = size

    def __call__(self, vol):
        mips = (vol.max(axis=2), vol.max(axis=1), vol.max(axis=0))
        chans = []
        for m in mips:
            t = torch.from_numpy(np.ascontiguousarray(m, dtype=np.float32))[None, None]
            t = F.interpolate(t, size=(self.size, self.size), mode="bilinear",
                              align_corners=False)
            chans.append(t[0, 0])
        x = torch.stack(chans)
        return (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-6)
