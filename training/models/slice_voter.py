import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class SliceVoter(nn.Module):
    """The 'doctor scrolls slices' model: every axial slice is scored by a shared
    pretrained 2D backbone and the final logit is the mean of the slice votes.
    During training a random subset of slices is used per step (cheaper, and acts
    as augmentation); at eval time all slices vote."""

    name = "slicevoter"

    def __init__(self, pretrained=True, pool=None, head="linear", train_slices=8):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=pretrained,
                                          num_classes=0, in_chans=3)
        self.slice_logit = nn.Linear(self.backbone.num_features, 1)
        self.train_slices = train_slices

    def forward(self, x):  # x: (B, 1, X, Y, Z)
        b = x.shape[0]
        slices = x[:, 0].permute(0, 3, 1, 2)  # (B, Z, X, Y)
        if self.training and slices.shape[1] > self.train_slices:
            idx = torch.randperm(slices.shape[1], device=x.device)[: self.train_slices]
            slices = slices[:, idx]
        z = slices.shape[1]
        imgs = slices.reshape(b * z, 1, *slices.shape[2:])
        imgs = F.interpolate(imgs, size=(224, 224), mode="bilinear", align_corners=False)
        imgs = (imgs - imgs.mean(dim=(2, 3), keepdim=True)) / (
            imgs.std(dim=(2, 3), keepdim=True) + 1e-6)
        feats = self.backbone(imgs.repeat(1, 3, 1, 1))
        return self.slice_logit(feats).reshape(b, z).mean(dim=1)
