"""Self-supervised pretraining for the cnn3d encoder via masked-volume reconstruction.

Randomly masks cubic patches of each 3D crop and trains an encoder+decoder to
reconstruct only the masked voxels. The encoder learns scan anatomy without labels.
Fine-tune afterwards with train.py --pretrained-encoder <run>/encoder_best.pt.

Usage:
    export SCAN_REPO=/workspace/parkinson-classifier
    python scripts/pretrain_ssl.py \
        --crops-dir $SCAN_REPO/prepared/crops_m4 \
        --epochs 80 \
        --run-name ssl_m4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.config import TrainConfig
from training.models.cnn3d import Cnn3d
from training.models.decoder3d import Decoder3d

CFG = TrainConfig()


class MaskedCropDataset(Dataset):
    def __init__(self, crop_paths, patch_size=6, mask_frac=0.40):
        self.paths = crop_paths
        self.patch_size = patch_size
        self.mask_frac = mask_frac

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        vol = np.load(self.paths[i]).astype(np.float32)  # (X, Y, Z)
        p = self.patch_size
        X, Y, Z = vol.shape
        patches = [
            (x, y, z)
            for x in range(0, X - p + 1, p)
            for y in range(0, Y - p + 1, p)
            for z in range(0, Z - p + 1, p)
        ]
        n_mask = max(1, int(len(patches) * self.mask_frac))
        chosen = np.random.choice(len(patches), n_mask, replace=False)

        masked = vol.copy()
        mask = np.zeros_like(vol)
        for idx in chosen:
            x, y, z = patches[idx]
            masked[x:x+p, y:y+p, z:z+p] = 0.0
            mask[x:x+p, y:y+p, z:z+p] = 1.0

        return (
            torch.from_numpy(masked[None]),   # (1, X, Y, Z) — network input
            torch.from_numpy(vol[None]),       # (1, X, Y, Z) — reconstruction target
            torch.from_numpy(mask[None]),      # (1, X, Y, Z) — where to compute loss
        )


class MaskedAutoencoder(nn.Module):
    def __init__(self, width_mult=1.0):
        super().__init__()
        self.encoder = Cnn3d(width_mult=width_mult)
        self.decoder = Decoder3d(out_chans=1, width_mult=width_mult)

    def forward(self, x):
        feat = self.encoder.features(x)
        return self.decoder(feat, target_size=x.shape[2:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops-dir", default=None,
                    help="directory of .npy crop files (default: CFG.crops_dir)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patch-size", type=int, default=6,
                    help="side length (voxels) of each masked cubic patch")
    ap.add_argument("--mask-frac", type=float, default=0.40,
                    help="fraction of patches to mask per scan")
    ap.add_argument("--width-mult", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--run-name", default="ssl_pretrain")
    args = ap.parse_args()

    crops_dir = Path(args.crops_dir) if args.crops_dir else CFG.crops_dir
    out_dir = CFG.runs_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    crop_paths = sorted(crops_dir.glob("*.npy"))
    if not crop_paths:
        raise SystemExit(f"no .npy files found in {crops_dir}")
    print(f"found {len(crop_paths)} crops in {crops_dir}")
    print(f"masking {args.mask_frac*100:.0f}% of {args.patch_size}^3-voxel patches")

    ds = MaskedCropDataset(crop_paths, patch_size=args.patch_size,
                           mask_frac=args.mask_frac)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True,
                        worker_init_fn=lambda _: np.random.seed(
                            torch.initial_seed() % 2**32))

    device = torch.device(args.device)
    model = MaskedAutoencoder(width_mult=args.width_mult).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                            eta_min=args.lr * 0.01)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_scans = 0.0, 0
        for masked, orig, mask in loader:
            masked = masked.to(device)
            orig = orig.to(device)
            mask = mask.to(device)
            pred = model(masked)
            # MSE only on masked voxels
            loss = ((pred - orig) ** 2 * mask).sum() / (mask.sum() + 1e-6)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(masked)
            n_scans += len(masked)
        scheduler.step()

        avg = total_loss / n_scans
        if avg < best_loss:
            best_loss = avg
            torch.save(model.encoder.state_dict(), out_dir / "encoder_best.pt")
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}/{args.epochs}  "
                  f"recon_loss {avg:.5f}  best {best_loss:.5f}")

    torch.save(model.encoder.state_dict(), out_dir / "encoder_final.pt")
    print(f"\ndone. encoder weights -> {out_dir}/encoder_best.pt")
    print(f"fine-tune with:")
    print(f"  python scripts/train.py --model cnn3d --view volume3d "
          f"--pretrained-encoder {out_dir}/encoder_best.pt "
          f"--crops-dir {crops_dir} --run-name {args.run_name}_ft")


if __name__ == "__main__":
    main()
