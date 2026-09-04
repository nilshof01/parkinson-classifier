import json

import numpy as np


class FrameCache:
    """Per-scan common-frame volumes (float16 .npy) plus QC info (.json)."""

    def __init__(self, config):
        self.config = config

    def path(self, uid):
        return self.config.cache_dir / f"{uid}.npy"

    def info_path(self, uid):
        return self.config.cache_dir / f"{uid}.json"

    def save(self, uid, frame, info):
        np.save(self.path(uid), frame.astype(np.float16))
        self.info_path(uid).write_text(json.dumps(info))

    def load(self, uid):
        return np.load(self.path(uid)).astype(np.float32)

    def load_info(self, uid):
        return json.loads(self.info_path(uid).read_text())

    def has(self, uid):
        return self.path(uid).exists() and self.info_path(uid).exists()
