import numpy as np
import torch
import torch.nn.functional as F


class MipWithApGradient:
    """MIP view with an explicit anterior/posterior gradient channel.

    Channels:
      1. Axial MIP (standardized)
      2. Coronal MIP (standardized)
      3. Axial MIP minus its Y-flipped mirror (posterior - anterior signed diff)

    The crop frame has Y axis running P→A, so small-Y = posterior, large-Y = anterior.
    In early DaT loss the posterior putamen fades first, appearing bilaterally — a signal
    invisible to the L-R asymmetry channel (mipasym) but directly encoded here as a
    negative residual in the posterior half of channel 3."""

    name = "mipapgrad"
    channels = 3

    def __init__(self, size=224):
        self.size = size

    def __call__(self, vol):
        axial = vol.max(axis=2)    # (X, Y): L-R × P-A
        coronal = vol.max(axis=1)  # (X, Z): L-R × I-S
        apgrad = axial - axial[:, ::-1]  # posterior - anterior signed diff
        chans = []
        for m, standardize in ((axial, True), (coronal, True), (apgrad, False)):
            t = torch.from_numpy(np.ascontiguousarray(m, dtype=np.float32))[None, None]
            t = F.interpolate(t, size=(self.size, self.size), mode="bilinear",
                              align_corners=False)[0, 0]
            if standardize:
                t = (t - t.mean()) / (t.std() + 1e-6)
            else:
                t = t / (t.abs().std() + 1e-6)
            chans.append(t)
        return torch.stack(chans)
