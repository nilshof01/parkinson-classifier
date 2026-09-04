import numpy as np
import torch


class VolumeView:
    """Passes the augmented 3D volume through as a (1, X, Y, Z) tensor,
    standardized over in-head voxels — for 3D models and slice-voting models."""

    name = "volume3d"

    def __call__(self, vol):
        t = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))
        inside = t > 0
        if inside.any():
            t = (t - t[inside].mean()) / (t[inside].std() + 1e-6)
        return t[None]
