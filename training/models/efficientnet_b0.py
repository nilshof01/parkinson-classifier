import timm
import torch.nn as nn


def build_timm(arch, pretrained, pool, head):
    """head='linear': timm's own single-linear classifier (checkpoint-compatible
    with earlier runs). head='mlp': pooled features -> 256 -> 1, so the model can
    learn interactions between pooled statistics (e.g. avg vs max halves)."""
    kw = {"global_pool": pool} if pool else {}
    if head == "linear":
        return timm.create_model(arch, pretrained=pretrained, num_classes=1,
                                 in_chans=3, **kw), None
    backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0,
                                 in_chans=3, **kw)
    dim = backbone.num_features * (2 if pool == "catavgmax" else 1)
    mlp = nn.Sequential(nn.Linear(dim, 256), nn.SiLU(), nn.Dropout(0.3),
                        nn.Linear(256, 1))
    return backbone, mlp


class EfficientNetB0(nn.Module):
    name = "efficientnet_b0"

    def __init__(self, pretrained=True, pool=None, head="linear"):
        super().__init__()
        self.net, self.mlp = build_timm("efficientnet_b0", pretrained, pool, head)

    def forward(self, x):
        out = self.net(x)
        if self.mlp is not None:
            out = self.mlp(out)
        return out.squeeze(-1)
