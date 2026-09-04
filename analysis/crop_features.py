import numpy as np


class CropFeatures:
    """Deterministic striatal-region crop from the aligned common frame,
    mean-pooled to a coarse voxel grid and flattened to a feature vector."""

    # generous box around both striata in the 96x112x96 frame (2 mm voxels)
    X = slice(26, 70)
    Y = slice(48, 86)
    Z = slice(36, 62)
    POOL = 2

    def extract(self, frame):
        crop = frame[self.X, self.Y, self.Z]
        head = frame[frame > 0]
        if head.size == 0 or head.mean() <= 0:
            return None
        crop = crop / head.mean()
        p = self.POOL
        sx, sy, sz = (s - s % p for s in crop.shape)
        crop = crop[:sx, :sy, :sz]
        pooled = crop.reshape(sx // p, p, sy // p, p, sz // p, p).mean(axis=(1, 3, 5))
        return pooled.ravel().astype(np.float32)

    @property
    def n_features(self):
        p = self.POOL
        dims = [
            ((s.stop - s.start) - (s.stop - s.start) % p) // p
            for s in (self.X, self.Y, self.Z)
        ]
        return int(np.prod(dims))
