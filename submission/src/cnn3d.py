"""Self-contained cnn3d model architecture for inference.
Copy of training/models/cnn3d.py — no dependency on the training package."""

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
    def __init__(self, pool="avg", head="linear", in_chans=1,
                 width_mult=1.0, dropout=0.0, aux_weight=0.0, bottleneck_dim=256):
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
            dim = chs[-1] * 5
            if aux_weight > 0:
                self.aux_pa = nn.Sequential(nn.Dropout(dropout), nn.Linear(chs[-1] * 2, 1))
                self.aux_lr = nn.Sequential(nn.Dropout(dropout), nn.Linear(chs[-1] * 2, 1))
        elif pool == "axisaware_split":
            dim = chs[-1] * 7
            if aux_weight > 0:
                self.aux_pa_post = nn.Sequential(nn.Dropout(dropout), nn.Linear(chs[-1] * 2, 1))
                self.aux_pa_ant  = nn.Sequential(nn.Dropout(dropout), nn.Linear(chs[-1] * 2, 1))
                self.aux_lr      = nn.Sequential(nn.Dropout(dropout), nn.Linear(chs[-1] * 2, 1))
        elif pool == "axisaware_bins":
            dim = chs[-1] * 12
            if aux_weight > 0:
                self.aux_pp  = nn.Sequential(nn.Dropout(dropout), nn.Linear(chs[-1] * 2, 1))
                self.aux_ap  = nn.Sequential(nn.Dropout(dropout), nn.Linear(chs[-1] * 2, 1))
                self.aux_cau = nn.Sequential(nn.Dropout(dropout), nn.Linear(chs[-1] * 2, 1))
                self.aux_lr  = nn.Sequential(nn.Dropout(dropout), nn.Linear(chs[-1] * 2, 1))
        elif pool == "catavgmax":
            dim = chs[-1] * 2
        else:
            dim = chs[-1]

        if head == "mlp" or pool in ("axisaware", "axisaware_split", "axisaware_bins"):
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Dropout(dropout),
                nn.Linear(dim, bottleneck_dim), nn.SiLU(),
                nn.Dropout(dropout), nn.Linear(bottleneck_dim, 1))
        else:
            self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                            nn.Linear(dim, 1))

    def forward(self, x):
        f = self.features(x)
        if self.pool_kind == "axisaware":
            global_avg = f.mean(dim=(2, 3, 4))
            pa = f.mean(dim=(2, 4))
            lr = f.mean(dim=(3, 4))
            pa_vec = torch.cat([pa.amax(dim=2), pa.amin(dim=2)], dim=1)
            lr_vec = torch.cat([lr.amax(dim=2), lr.amin(dim=2)], dim=1)
            pooled = torch.cat([global_avg, pa_vec, lr_vec], dim=1)
        elif self.pool_kind == "axisaware_split":
            global_avg = f.mean(dim=(2, 3, 4))
            pa = f.mean(dim=(2, 4))
            lr = f.mean(dim=(3, 4))
            mid = pa.shape[2] // 2
            pa_post = pa[:, :, :mid]
            pa_ant  = pa[:, :, mid:]
            pa_post_vec = torch.cat([pa_post.amax(2), pa_post.amin(2)], dim=1)
            pa_ant_vec  = torch.cat([pa_ant.amax(2),  pa_ant.amin(2)],  dim=1)
            lr_vec = torch.cat([lr.amax(2), lr.amin(2)], dim=1)
            pooled = torch.cat([global_avg, pa_post_vec, pa_ant_vec, lr_vec], dim=1)
        elif self.pool_kind == "axisaware_bins":
            global_avg = f.mean(dim=(2, 3, 4))
            pa = f.mean(dim=(2, 4))
            lr = f.mean(dim=(3, 4))
            Y = pa.shape[2]
            t1, t2 = Y // 3, 2 * Y // 3
            pa_pp  = pa[:, :, :t1]
            pa_ap  = pa[:, :, t1:t2]
            pa_cau = pa[:, :, t2:]
            pp_vec  = torch.cat([pa_pp.amax(2),  pa_pp.amin(2)],  dim=1)
            ap_vec  = torch.cat([pa_ap.amax(2),  pa_ap.amin(2)],  dim=1)
            cau_vec = torch.cat([pa_cau.amax(2), pa_cau.amin(2)], dim=1)
            lr_vec  = torch.cat([lr.amax(2),     lr.amin(2)],     dim=1)
            pp_mean  = pa_pp.mean(2)
            ap_mean  = pa_ap.mean(2)
            cau_mean = pa_cau.mean(2)
            gradient = pp_mean - ap_mean
            pc_diff  = ((pp_mean - cau_mean) / (
                pp_mean.abs() + cau_mean.abs() + 1e-4)).clamp(-1, 1)
            mid_x = lr.shape[2] // 2
            asym = ((lr[:, :, :mid_x].mean(2) - lr[:, :, mid_x:].mean(2)) / (
                lr[:, :, :mid_x].mean(2).abs() + lr[:, :, mid_x:].mean(2).abs()
                + 1e-4)).clamp(-1, 1)
            pooled = torch.cat(
                [global_avg, pp_vec, ap_vec, cau_vec, lr_vec,
                 gradient, pc_diff, asym], dim=1)
        elif self.pool_kind == "catavgmax":
            avg = f.mean(dim=(2, 3, 4))
            pooled = torch.cat([avg, f.amax(dim=(2, 3, 4))], dim=1)
        elif self.pool_kind == "max":
            pooled = f.amax(dim=(2, 3, 4))
        else:
            pooled = f.mean(dim=(2, 3, 4))
        return self.classifier(pooled).squeeze(-1)
