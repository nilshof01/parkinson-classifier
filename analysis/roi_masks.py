import numpy as np


class RoiMasks:
    """Materializes calibrated ROI definitions as boolean masks in the common frame."""

    def __init__(self, config, roi_defs):
        self.config = config
        self.masks = {name: self._build(d) for name, d in roi_defs.items()}

    def _build(self, definition):
        kind = definition[0]
        shape = self.config.frame_shape
        if kind == "sphere":
            _, center, radius = definition
            grid = np.ogrid[: shape[0], : shape[1], : shape[2]]
            dist2 = sum((g - c) ** 2 for g, c in zip(grid, center))
            return dist2 <= radius ** 2
        if kind == "box":
            _, lo, hi = definition
            mask = np.zeros(shape, dtype=bool)
            mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = True
            return mask
        raise ValueError(kind)

    def mean_counts(self, frame, name):
        m = self.masks[name]
        vals = frame[m]
        vals = vals[vals > 0]  # frame is head-masked; zeros are outside the head
        return float(vals.mean()) if vals.size else np.nan
