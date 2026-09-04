import timm
import torch.nn as nn


class VitSmall(nn.Module):
    name = "vit_small"

    def __init__(self, pretrained=True, pool=None, head="linear"):
        # pool is a CNN concept (avg/max/catavgmax); ViT aggregates via its
        # CLS token attention, so pool/head args are accepted but ignored here
        super().__init__()
        self.net = timm.create_model(
            "vit_small_patch16_224.augreg_in21k_ft_in1k",
            pretrained=pretrained, num_classes=1, in_chans=3,
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
