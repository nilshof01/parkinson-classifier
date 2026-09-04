import numpy as np
import torch
import torch.nn.functional as F


class MipWithAsymmetry:
    """Contrastive-input variant of the MIP view: axial MIP, coronal MIP, and a
    signed left-right asymmetry map (axial MIP minus its own mirror). Symmetric
    anatomy cancels toward zero in channel 3, so subtle unilateral differences
    appear at full contrast instead of being a small residual on a bright image."""

    name = "mipasym"

    def __init__(self, size=224):
        self.size = size

    def __call__(self, vol):
        axial = vol.max(axis=2)  # (x, y); x is the left-right axis
        coronal = vol.max(axis=1)
        asym = axial - axial[::-1]
        chans = []
        for m, standardize in ((axial, True), (coronal, True), (asym, False)):
            t = torch.from_numpy(np.ascontiguousarray(m, dtype=np.float32))[None, None]
            t = F.interpolate(t, size=(self.size, self.size), mode="bilinear",
                              align_corners=False)[0, 0]
            if standardize:
                t = (t - t.mean()) / (t.std() + 1e-6)
            else:
                t = t / (t.abs().std() + 1e-6)  # keep the sign; zero stays zero
            chans.append(t)
        return torch.stack(chans)
