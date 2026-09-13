import numpy as np
import torch


class VolumeView:
    """Passes the augmented 3D volume through as a (1, X, Y, Z) tensor.

    norm='zscore'       — subtract mean, divide by std over non-zero voxels (default)
    norm='percentile'   — divide by the Nth percentile of non-zero voxels (n=99 default);
                          preserves inter-subject relative intensity better than zscore
    """

    name = "volume3d"

    def __init__(self, norm="zscore", norm_percentile=99.0):
        self.norm = norm
        self.norm_percentile = norm_percentile

    def __call__(self, vol):
        t = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))
        inside = t > 0
        if inside.any():
            if self.norm == "percentile":
                pct = float(np.percentile(t[inside].numpy(), self.norm_percentile))
                t = t / (pct + 1e-6)
            else:
                t = (t - t[inside].mean()) / (t[inside].std() + 1e-6)
        return t[None]
