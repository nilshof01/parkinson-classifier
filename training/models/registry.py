from training.models.cnn3d import Cnn3d
from training.models.convnext_tiny import ConvNextTiny
from training.models.dinov2_small import DinoV2Small
from training.models.fusion3d import Fusion3d
from training.models.efficientnet_b0 import EfficientNetB0
from training.models.r3d18 import R3d18
from training.models.slice_voter import SliceVoter
from training.models.vit_small import VitSmall


class ModelRegistry:
    """Add a new model by dropping a class file next to the others (must expose
    `name` and `__init__(pretrained, pool, head)`) and listing it here."""

    _models = {m.name: m for m in
               (EfficientNetB0, ConvNextTiny, VitSmall, Cnn3d, SliceVoter, R3d18,
                Fusion3d, DinoV2Small)}

    @classmethod
    def get(cls, name):
        if name not in cls._models:
            raise KeyError(f"unknown model '{name}', available: {cls.names()}")
        return cls._models[name]

    @classmethod
    def names(cls):
        return sorted(cls._models)
