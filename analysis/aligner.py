import numpy as np
from scipy import ndimage


class Aligner:
    """Translation-only registration of a frame to a reference (cohort mean)
    via FFT phase correlation. Uses the whole head's intensity pattern, never
    striatal signal alone."""

    def __init__(self, config, max_shift_vox=14):
        self.config = config
        self.max_shift = max_shift_vox

    def estimate_shift(self, frame, reference):
        f = np.fft.rfftn(frame)
        r = np.fft.rfftn(reference)
        cross = r * np.conj(f)
        cross /= np.abs(cross) + 1e-9
        corr = np.fft.irfftn(cross, s=frame.shape, axes=(0, 1, 2))
        corr = np.fft.fftshift(corr)
        center = np.array([s // 2 for s in frame.shape])
        m = self.max_shift
        window = corr[
            center[0] - m : center[0] + m + 1,
            center[1] - m : center[1] + m + 1,
            center[2] - m : center[2] + m + 1,
        ]
        peak = np.unravel_index(np.argmax(window), window.shape)
        return tuple(int(p - m) for p in peak)

    @staticmethod
    def apply_shift(frame, shift):
        if not any(shift):
            return frame
        return ndimage.shift(frame, shift, order=1, mode="constant", cval=0.0)

    def align(self, frame, reference):
        shift = self.estimate_shift(frame, reference)
        return self.apply_shift(frame, shift), shift
