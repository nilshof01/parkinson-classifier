import numpy as np


class AsymmetryJitter:
    """Applies a small random L-R scale difference to a crop, keeping the
    resulting asymmetry index (AI) within the healthy-control range.

    AI = |L - R| / ((L + R) / 2).  For a hemisphere scale factor f close to 1:
        AI ≈ |f - 1|
    so drawing f ∈ [1 - max_ai, 1 + max_ai] keeps AI ≤ max_ai.

    Label is unchanged — this teaches all models that small hemispheric
    intensity differences are a normal finding, not a disease sign.
    """

    def __init__(self, max_ai: float = 0.10):
        self.max_ai = max_ai

    def __call__(self, vol: np.ndarray) -> np.ndarray:
        target_ai = np.random.uniform(0.0, self.max_ai)
        f = 1.0 + np.random.choice([-1.0, 1.0]) * target_ai
        mid = vol.shape[0] // 2
        result = vol.copy()
        if np.random.random() < 0.5:
            result[:mid] = result[:mid] * f
        else:
            result[mid:] = result[mid:] * f
        return result.astype(np.float32)
