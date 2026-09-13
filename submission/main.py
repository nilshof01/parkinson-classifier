"""Submission entrypoint for the DrivenData DaT-SPECT challenge.

Directory layout (as unzipped into /code_execution/src/):
  main.py          ← this file
  config.py        ← user-editable settings (norm, arch, TTA, calibration)
  models/          ← drop best_ema.pt checkpoint files here; all are averaged
  assets/
    mean_frame.npy       ← cohort mean frame for alignment
    template_crop.npy    ← template crop for placement QC
  src/
    preprocess.py        ← NIfTI → aligned crop
    cnn3d.py             ← model architecture

Reads:   /code_execution/data/niftis/<uid>.nii.gz
         /code_execution/data/submission_format.csv
Writes:  /code_execution/submission.csv
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

import config as cfg
from src.preprocess import preprocess
from src.cnn3d import Cnn3d

DATA = Path("/code_execution/data")
OUT  = Path("/code_execution/submission.csv")


# ── Assets ────────────────────────────────────────────────────────────────────

def load_assets():
    mean_frame    = np.load(HERE / "assets" / "mean_frame.npy").astype(np.float32)
    template_crop = np.load(HERE / "assets" / "template_crop.npy").ravel().astype(np.float32)
    return mean_frame, template_crop


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize(vol: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))
    inside = t > 0
    if not inside.any():
        return t[None]
    if cfg.NORM == "percentile":
        pct = float(np.percentile(t[inside].numpy(), cfg.NORM_PERCENTILE))
        t = t / (pct + 1e-6)
    else:
        t = (t - t[inside].mean()) / (t[inside].std() + 1e-6)
    return t[None]   # (1, X, Y, Z)


# ── TTA variants ──────────────────────────────────────────────────────────────

def tta_volumes(vol: np.ndarray):
    """Yield (weight, tensor) pairs for all TTA variants."""
    base = normalize(vol)
    yield 1.0, base
    if cfg.TTA_FLIP:
        yield 1.0, normalize(vol[::-1].copy())
    for angle in cfg.TTA_ROTATIONS:
        rot = ndimage.rotate(vol, angle, axes=(0, 1), reshape=False, order=1)
        yield 1.0, normalize(rot)
        if cfg.TTA_FLIP:
            yield 1.0, normalize(rot[::-1].copy())


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(device):
    checkpoints = sorted((HERE / "models").glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No .pt files found in {HERE / 'models'}. "
            "Drop your best_ema.pt checkpoint(s) there.")
    models = []
    for ckpt in checkpoints:
        model = Cnn3d(
            pool=cfg.MODEL_POOL or "avg",
            width_mult=cfg.MODEL_WIDTH_MULT,
            dropout=cfg.MODEL_DROPOUT,
        )
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state)
        model.eval().to(device)
        models.append(model)
    print(f"loaded {len(models)} checkpoint(s): {[c.name for c in checkpoints]}")
    return models


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_one(vol: np.ndarray, models, device) -> float:
    """Average predictions across all models and all TTA variants."""
    total, count = 0.0, 0
    variants = list(tta_volumes(vol))
    batch = torch.stack([t for _, t in variants]).to(device)  # (N, 1, X, Y, Z)
    weights = torch.tensor([w for w, _ in variants], dtype=torch.float32, device=device)
    for model in models:
        logits = model(batch).float()          # (N,)
        probs  = torch.sigmoid(logits)         # (N,)
        total += (probs * weights).sum().item()
        count += weights.sum().item()
    raw_p = total / count

    # temperature scaling
    logit = np.log(max(raw_p, 1e-7) / max(1 - raw_p, 1e-7))
    p = 1.0 / (1.0 + np.exp(-logit / cfg.TEMPERATURE))
    return float(np.clip(p, cfg.CLIP_LO, cfg.CLIP_HI))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  |  norm: {cfg.NORM}  |  crop_margin: {cfg.CROP_MARGIN}")
    print(f"tta_flip: {cfg.TTA_FLIP}  |  tta_rotations: {cfg.TTA_ROTATIONS}")

    mean_frame, template_flat = load_assets()
    models = load_models(device)

    fmt = pd.read_csv(DATA / "submission_format.csv")
    preds, n_fallback = [], 0

    for i, uid in enumerate(fmt["uid"].astype(str)):
        try:
            path = DATA / "niftis" / f"{uid}.nii.gz"
            crop = preprocess(path, mean_frame, template_flat, margin=cfg.CROP_MARGIN)
            p = predict_one(crop, models, device)
        except Exception as e:
            print(f"fallback ({type(e).__name__})")
            p = cfg.FALLBACK_P
            n_fallback += 1
        preds.append(p)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(fmt)} done")

    pd.DataFrame({"uid": fmt["uid"], "is_pathologic": preds}).to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(preds)} rows, {n_fallback} fallbacks)")


if __name__ == "__main__":
    main()
