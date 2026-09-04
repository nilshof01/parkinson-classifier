import torch
import torch.nn as nn


def _block(cin, cout, stride):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm3d(cout), nn.SiLU(),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm3d(cout), nn.SiLU(),
    )


class Cnn3d(nn.Module):
    """Small from-scratch 3D CNN over the volume — sees true slice-to-slice
    morphology that the MIP views project away. No pretraining exists for this,
    so `pretrained` is ignored. `head` follows the 2D models: 'linear' or 'mlp'."""

    name = "cnn3d"

    def __init__(self, pretrained=True, pool="avg", head="linear",
                 in_chans=1, width_mult=1.0, dropout=0.3):
        super().__init__()
        pool = pool or "avg"
        chs = [max(8, int(round(c * width_mult))) for c in (32, 64, 128, 256)]
        layers = [_block(in_chans, chs[0], 1)]
        for cin, cout in zip(chs, chs[1:]):
            layers.append(_block(cin, cout, 2))
        self.features = nn.Sequential(*layers)
        self.pool_kind = pool
        dim = chs[-1] * (2 if pool == "catavgmax" else 1)
        # leading Flatten is a no-op on the pooled 2D tensor; it keeps state-dict
        # keys identical to earlier cnn3d checkpoints (classifier.2.*)
        if head == "mlp":
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Dropout(dropout), nn.Linear(dim, 256), nn.SiLU(),
                nn.Dropout(dropout), nn.Linear(256, 1))
        else:
            self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                            nn.Linear(dim, 1))

    def forward(self, x):
        f = self.features(x)
        avg = f.mean(dim=(2, 3, 4))
        if self.pool_kind == "catavgmax":
            pooled = torch.cat([avg, f.amax(dim=(2, 3, 4))], dim=1)
        elif self.pool_kind == "max":
            pooled = f.amax(dim=(2, 3, 4))
        else:
            pooled = avg
        return self.classifier(pooled).squeeze(-1)
