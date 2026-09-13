import numpy as np
import torch
from torch.utils.data import Dataset


class DatScanDataset(Dataset):
    """Serves (view_tensor, label) pairs from preloaded 3D crops.
    Augmentation and chimera mixing are training-only; force_flip mirrors the
    volume left-right before the view (used for test-time augmentation).

    For normal samples (label=0, frames=None) exactly one of the following fires
    per item, chosen by a single random draw (mutually exclusive):
      1. Bilateral posterior reduction → relabeled abnormal  (rate=posterior_frac)
      2. Unilateral posterior reduction → relabeled abnormal  (rate=posterior_uni_frac)
      3. Normal+normal chimera, label unchanged               (rate=chimera_frac)
    Then, independently (only when none of the above fired):
      4. L-R asymmetry jitter, label unchanged               (rate=asym_jitter_frac)

    If posterior_frac + posterior_uni_frac + chimera_frac > 1 the excess options
    are silently suppressed (they can never be reached by the uniform draw).
    """

    def __init__(self, crops, records, view, augment=None,
                 chimera=None, chimera_frac=0.0, force_flip=False,
                 frames=None, frame_augment=None,
                 pos_chimera=None, pos_frac=0.0, worse_sides=None,
                 posterior_reducer=None, posterior_frac=0.0, posterior_uni_frac=0.0,
                 asym_jitter=None, asym_jitter_frac=0.0,
                 sample_weights=None,
                 chimera_weights=None):
        self.sample_weights = sample_weights  # dict uid -> float, or None
        self.chimera_weights = chimera_weights  # dict uid -> chimera prob override, or None
        self.pos_chimera = pos_chimera
        self.pos_frac = pos_frac
        self.worse_sides = worse_sides
        self.crops = crops  # dict uid -> float16 array
        self.records = records  # list of (uid, label)
        self.view = view
        self.augment = augment
        self.chimera = chimera
        self.chimera_frac = chimera_frac
        self.force_flip = force_flip
        # fusion mode: also serve the whole-head frame; LR flip is drawn once
        # and applied to both streams so laterality stays consistent
        self.frames = frames
        self.frame_augment = frame_augment
        self.posterior_reducer = posterior_reducer
        self.posterior_frac = posterior_frac
        self.posterior_uni_frac = posterior_uni_frac
        self.asym_jitter = asym_jitter
        self.asym_jitter_frac = asym_jitter_frac

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        uid, label = self.records[i]
        vol = self.crops[uid].astype(np.float32)
        y = torch.tensor(label, dtype=torch.float32)

        if (
            self.frames is None
            and self.pos_chimera is not None
            and label == 1.0
            and np.random.random() < self.pos_frac
        ):
            vol = self.pos_chimera.mix(vol, self.worse_sides[uid])

        if self.frames is None and label == 0.0:
            r = np.random.random()
            cum = 0.0
            applied = False

            cum += self.posterior_frac
            if not applied and self.posterior_reducer is not None and r < cum:
                vol = self.posterior_reducer(vol, unilateral=False)
                y = torch.tensor(1.0, dtype=torch.float32)
                applied = True

            cum += self.posterior_uni_frac
            if not applied and self.posterior_reducer is not None and r < cum:
                vol = self.posterior_reducer(vol, unilateral=True)
                y = torch.tensor(1.0, dtype=torch.float32)
                applied = True

            chi_prob = (self.chimera_weights.get(uid, self.chimera_frac)
                        if self.chimera_weights else self.chimera_frac)
            cum += chi_prob
            if not applied and self.chimera is not None and r < cum:
                vol = self.chimera.mix(vol)
                applied = True

            if not applied and self.asym_jitter is not None and np.random.random() < self.asym_jitter_frac:
                vol = self.asym_jitter(vol)

        if self.frames is None:
            if self.augment is not None:
                vol = self.augment(vol)
            if self.force_flip:
                vol = vol[::-1].copy()
            if self.sample_weights is not None:
                w = torch.tensor(self.sample_weights.get(uid, 1.0), dtype=torch.float32)
                return self.view(vol), y, w
            return self.view(vol), y

        frame = self.frames[uid].astype(np.float32)
        if self.augment is not None and np.random.random() < 0.5:
            vol, frame = vol[::-1].copy(), frame[::-1].copy()
        if self.augment is not None:
            vol = self.augment(vol)
            frame = self.frame_augment(frame)
        if self.force_flip:
            vol, frame = vol[::-1].copy(), frame[::-1].copy()
        return (self.view(vol), self.view(frame)), y
