import numpy as np


class PosteriorPutamenReduction:
    """Attenuates the posterior portion of the striatal crop with a half-Gaussian
    ramp along the P→A axis to simulate early DaT loss (posterior putamen fades
    first). Apply to normal crops and flip the label to abnormal.

    Crop convention: axis 1 is Y (posterior→anterior), so Y=0 is the most
    posterior voxel. The attenuation field is:
        f(y) = factor + (1 - factor) * (1 - exp(-y² / (2σ²)))
    At y=0 the field equals `factor`; it recovers to ≈1 toward the anterior end.

    Usage:
        reducer = PosteriorPutamenReduction()
        synthetic = reducer(normal_vol, unilateral=False)   # bilateral  → label 1
        synthetic = reducer(normal_vol, unilateral=True)    # one-sided  → label 1
    """

    def __init__(self, sigma_y: float = 5.0,
                 min_factor: float = 0.30,
                 max_factor: float = 0.75):
        self.sigma_y = sigma_y        # Gaussian width in voxels (~2 mm/vox)
        self.min_factor = min_factor
        self.max_factor = max_factor

    def _field(self, n_y: int, factor: float) -> np.ndarray:
        y = np.arange(n_y, dtype=np.float32)
        return factor + (1.0 - factor) * (1.0 - np.exp(-y ** 2 / (2.0 * self.sigma_y ** 2)))

    def __call__(self, vol: np.ndarray, unilateral: bool = False) -> np.ndarray:
        factor = np.random.uniform(self.min_factor, self.max_factor)
        field = self._field(vol.shape[1], factor)[np.newaxis, :, np.newaxis]  # (1,Y,1)
        result = vol.copy()
        if not unilateral:
            result = result * field
        else:
            mid = result.shape[0] // 2
            if np.random.random() < 0.5:
                result[:mid] = result[:mid] * field
            else:
                result[mid:] = result[mid:] * field
        return result.astype(np.float32)
