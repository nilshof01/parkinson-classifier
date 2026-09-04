import torch.nn as nn

from training.models.efficientnet_b0 import build_timm


class ConvNextTiny(nn.Module):
    name = "convnext_tiny"

    def __init__(self, pretrained=True, pool=None, head="linear"):
        super().__init__()
        self.net, self.mlp = build_timm("convnext_tiny.fb_in22k", pretrained, pool, head)

    def forward(self, x):
        out = self.net(x)
        if self.mlp is not None:
            out = self.mlp(out)
        return out.squeeze(-1)
