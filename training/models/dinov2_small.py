import timm
import torch.nn as nn


class DinoV2Small(nn.Module):
    """DINOv2 ViT-S/14 — self-supervised pretraining (no ImageNet labels), so its
    features are a genuinely different prior from the supervised members."""

    name = "dinov2_small"

    def __init__(self, pretrained=True, pool=None, head="linear"):
        super().__init__()
        self.net = timm.create_model(
            "vit_small_patch14_reg4_dinov2.lvd142m",
            pretrained=pretrained, num_classes=1, in_chans=3, img_size=224,
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
