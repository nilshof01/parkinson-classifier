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
    so `pretrained` is ignored. `head` follows the 2D models: 'linear' or 'mlp'.

    pool='axisaware': instead of global avg, separately averages over (L-R, I-S)
    to keep the P-A profile and over (P-A, I-S) to keep the L-R profile, then
    concatenates max and min along each retained axis with global avg → 5C.
    This makes the representation more robust to small crop misalignments and
    explicitly encodes the posterior-anterior gradient and L-R asymmetry."""

    name = "cnn3d"

    def __init__(self, pretrained=True, pool="avg", head="linear",
                 in_chans=1, width_mult=1.0, dropout=0.3, aux_weight=0.0,
                 bottleneck_dim=256):
        super().__init__()
        pool = pool or "avg"
        chs = [max(8, int(round(c * width_mult))) for c in (32, 64, 128, 256)]
        layers = [_block(in_chans, chs[0], 1)]
        for cin, cout in zip(chs, chs[1:]):
            layers.append(_block(cin, cout, 2))
        self.features = nn.Sequential(*layers)
        self.pool_kind = pool
        self.aux_weight = aux_weight

        if pool == "axisaware":
            # global(1C) + pa_max(1C) + pa_min(1C) + lr_max(1C) + lr_min(1C) = 5C
            dim = chs[-1] * 5
            if aux_weight > 0:
                self.aux_pa = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(chs[-1] * 2, 1))
                self.aux_lr = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(chs[-1] * 2, 1))
        elif pool == "axisaware_split":
            # global(1C) + pa_post(2C) + pa_ant(2C) + lr(2C) = 7C
            # pa split at midpoint: posterior=putamen region, anterior=caudate region
            dim = chs[-1] * 7
            if aux_weight > 0:
                self.aux_pa_post = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(chs[-1] * 2, 1))
                self.aux_pa_ant = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(chs[-1] * 2, 1))
                self.aux_lr = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(chs[-1] * 2, 1))
        elif pool == "catavgmax":
            dim = chs[-1] * 2
        else:
            dim = chs[-1]

        # axisaware variants always use a bottleneck — direct linear from 5-7C is too wide
        if head == "mlp" or pool in ("axisaware", "axisaware_split"):
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Dropout(dropout),
                nn.Linear(dim, bottleneck_dim), nn.SiLU(),
                nn.Dropout(dropout), nn.Linear(bottleneck_dim, 1))
        else:
            # leading Flatten is a no-op on the pooled tensor; keeps state-dict
            # keys identical to earlier cnn3d checkpoints (classifier.2.*)
            self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                            nn.Linear(dim, 1))

    def forward(self, x):
        f = self.features(x)                    # (B, C, X, Y, Z)
        if self.pool_kind == "axisaware":
            # input axes: 2=X(L-R), 3=Y(P-A), 4=Z(I-S)
            global_avg = f.mean(dim=(2, 3, 4))  # (B, C)
            pa = f.mean(dim=(2, 4))             # (B, C, Y) — P-A profile
            lr = f.mean(dim=(3, 4))             # (B, C, X) — L-R profile
            pa_vec = torch.cat([pa.amax(dim=2), pa.amin(dim=2)], dim=1)  # (B, 2C)
            lr_vec = torch.cat([lr.amax(dim=2), lr.amin(dim=2)], dim=1)  # (B, 2C)
            pooled = torch.cat([global_avg, pa_vec, lr_vec], dim=1)      # (B, 5C)
            if self.training and self.aux_weight > 0:
                self._aux_pa_logit = self.aux_pa(pa_vec).squeeze(-1)
                self._aux_lr_logit = self.aux_lr(lr_vec).squeeze(-1)
            else:
                self._aux_pa_logit = None
                self._aux_lr_logit = None
            self._aux_pa2_logit = None
        elif self.pool_kind == "axisaware_split":
            # input axes: 2=X(L-R), 3=Y(P-A), 4=Z(I-S)
            global_avg = f.mean(dim=(2, 3, 4))  # (B, C)
            pa = f.mean(dim=(2, 4))             # (B, C, Y) — full P-A profile
            lr = f.mean(dim=(3, 4))             # (B, C, X) — L-R profile
            mid = pa.shape[2] // 2
            pa_post = pa[:, :, :mid]            # posterior half — putamen depletes first
            pa_ant  = pa[:, :, mid:]            # anterior half — caudate more preserved
            pa_post_vec = torch.cat([pa_post.amax(2), pa_post.amin(2)], dim=1)  # (B, 2C)
            pa_ant_vec  = torch.cat([pa_ant.amax(2),  pa_ant.amin(2)],  dim=1)  # (B, 2C)
            lr_vec = torch.cat([lr.amax(2), lr.amin(2)], dim=1)                 # (B, 2C)
            pooled = torch.cat([global_avg, pa_post_vec, pa_ant_vec, lr_vec], dim=1)  # (B, 7C)
            if self.training and self.aux_weight > 0:
                self._aux_pa_logit  = self.aux_pa_post(pa_post_vec).squeeze(-1)
                self._aux_pa2_logit = self.aux_pa_ant(pa_ant_vec).squeeze(-1)
                self._aux_lr_logit  = self.aux_lr(lr_vec).squeeze(-1)
            else:
                self._aux_pa_logit  = None
                self._aux_pa2_logit = None
                self._aux_lr_logit  = None
        elif self.pool_kind == "catavgmax":
            avg = f.mean(dim=(2, 3, 4))
            pooled = torch.cat([avg, f.amax(dim=(2, 3, 4))], dim=1)
        elif self.pool_kind == "max":
            pooled = f.amax(dim=(2, 3, 4))
        else:
            pooled = f.mean(dim=(2, 3, 4))
        return self.classifier(pooled).squeeze(-1)
