import torch.nn as nn
import torch.nn.functional as F


def _deblock(cin, cout):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm3d(cout), nn.SiLU(),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm3d(cout), nn.SiLU(),
    )


class Decoder3d(nn.Module):
    """Mirror decoder for Cnn3d. Takes the spatial feature map (before global
    pooling) and reconstructs the input volume via trilinear upsampling +
    conv blocks. The final interpolate snaps to the exact input size."""

    def __init__(self, out_chans=1, width_mult=1.0):
        super().__init__()
        chs = [max(8, int(round(c * width_mult))) for c in (32, 64, 128, 256)]
        self.up1 = _deblock(chs[3], chs[2])
        self.up2 = _deblock(chs[2], chs[1])
        self.up3 = _deblock(chs[1], chs[0])
        self.out = nn.Conv3d(chs[0], out_chans, 1)

    def forward(self, feat, target_size):
        x = self.up1(feat)
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = self.up2(x)
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = self.up3(x)
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = self.out(x)
        return F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
