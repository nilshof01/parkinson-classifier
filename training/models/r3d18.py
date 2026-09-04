import torch.nn as nn
import torch.nn.functional as F


class R3d18(nn.Module):
    """Pretrained 3D ResNet-18 (Kinetics-400 video weights) — fills the
    pretrained-3D quadrant. The crop's z axis is treated as the video time
    axis; slices are resized to the 112x112 the pretrain expects."""

    name = "r3d18"

    def __init__(self, pretrained=True, pool=None, head="linear"):
        super().__init__()
        from torchvision.models.video import R3D_18_Weights, r3d_18
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        self.net = r3d_18(weights=weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, 1)

    def forward(self, x):  # (B, 1, X, Y, Z)
        v = x[:, 0].permute(0, 3, 1, 2)  # (B, Z, X, Y): z -> frames
        b, t, hh, ww = v.shape
        v = F.interpolate(v.reshape(b * t, 1, hh, ww), size=(112, 112),
                          mode="bilinear", align_corners=False)
        v = v.reshape(b, t, 1, 112, 112).permute(0, 2, 1, 3, 4).repeat(1, 3, 1, 1, 1)
        return self.net(v).squeeze(-1)
