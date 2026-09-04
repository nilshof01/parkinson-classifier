import numpy as np
import torch
from torch.utils.data import Dataset


class DatScanDataset(Dataset):
    """Serves (view_tensor, label) pairs from preloaded 3D crops.
    Augmentation and chimera mixing are training-only; force_flip mirrors the
    volume left-right before the view (used for test-time augmentation)."""

    def __init__(self, crops, records, view, augment=None,
                 chimera=None, chimera_frac=0.0, force_flip=False,
                 frames=None, frame_augment=None,
                 pos_chimera=None, pos_frac=0.0, worse_sides=None):
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

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        uid, label = self.records[i]
        vol = self.crops[uid].astype(np.float32)
        y = torch.tensor(label, dtype=torch.float32)
        if (
            self.frames is None
            and self.chimera is not None
            and label == 0.0
            and np.random.random() < self.chimera_frac
        ):
            vol = self.chimera.mix(vol)
        if (
            self.frames is None
            and self.pos_chimera is not None
            and label == 1.0
            and np.random.random() < self.pos_frac
        ):
            vol = self.pos_chimera.mix(vol, self.worse_sides[uid])
        if self.frames is None:
            if self.augment is not None:
                vol = self.augment(vol)
            if self.force_flip:
                vol = vol[::-1].copy()
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
