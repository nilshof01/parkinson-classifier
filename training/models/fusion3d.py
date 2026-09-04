import torch
import torch.nn as nn

from training.models.cnn3d import _block


def _trunk(chs, in_chans=1):
    layers = [_block(in_chans, chs[0], 1)]
    for cin, cout in zip(chs, chs[1:]):
        layers.append(_block(cin, cout, 2))
    return nn.Sequential(*layers)


class Fusion3d(nn.Module):
    """Two-stream 3D model: fine striatal crop (2 mm) and whole-head frame
    (4 mm) encoded separately, embeddings concatenated before the classifier —
    so context can condition the interpretation of the crop end-to-end."""

    name = "fusion3d"

    def __init__(self, pretrained=True, pool=None, head="linear", dropout=0.3):
        super().__init__()
        self.crop_trunk = _trunk((32, 64, 128, 256))
        self.frame_trunk = _trunk((16, 32, 64, 128))
        dim = 256 + 128
        if head == "mlp":
            self.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(dim, 128), nn.SiLU(),
                nn.Dropout(dropout), nn.Linear(128, 1))
        else:
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 1))

    def forward(self, x):  # x = (crop (B,1,44,38,26), frame (B,1,48,56,48))
        crop, frame = x
        e = torch.cat([self.crop_trunk(crop).mean(dim=(2, 3, 4)),
                       self.frame_trunk(frame).mean(dim=(2, 3, 4))], dim=1)
        return self.classifier(e).squeeze(-1)
