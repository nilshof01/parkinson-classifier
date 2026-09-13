"""Submission entrypoint for the DrivenData DaT-SPECT challenge.

Directory layout (as unzipped into /code_execution/src/):
  main.py          <- this file
  config.py        <- user-editable settings (norm, arch, TTA, calibration)
  models/          <- drop best_ema.pt checkpoint files here; all are averaged
  assets/
    mean_frame.npy       <- cohort mean frame for alignment
    template_crop.npy    <- template crop for placement QC
    calibrator.npz       <- optional: isotonic calibrator from scripts/calibrate.py
  src/
    preprocess.py        <- NIfTI -> aligned crop
    cnn3d.py             <- model architecture

Reads:   /code_execution/data/niftis/<uid>.nii.gz
         /code_execution/data/submission_format.csv
Writes:  /code_execution/submission.csv
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def load_isotonic():
    path = HERE / "assets" / "calibrator.npz"
    if not path.exists():
        raise FileNotFoundError(
            "CALIBRATION='isotonic' but assets/calibrator.npz not found. "
            "Run: python scripts/calibrate.py --method isotonic --save-calibrator ...")
    d = np.load(path)
    return d["x"], d["y"]   # breakpoints for np.interp


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
    base_w = getattr(cfg, "TTA_BASE_WEIGHT", 1.0)
    flip_w = getattr(cfg, "TTA_FLIP_WEIGHT", 1.0)
    rot_w  = getattr(cfg, "TTA_ROT_WEIGHT",  1.0)

    base = normalize(vol)
    yield base_w, base
    if cfg.TTA_FLIP:
        yield flip_w, normalize(vol[::-1].copy())
    for angle in cfg.TTA_ROTATIONS:
        rot = ndimage.rotate(vol, angle, axes=(0, 1), reshape=False, order=1)
        yield rot_w, normalize(rot)
        if cfg.TTA_FLIP:
            yield rot_w, normalize(rot[::-1].copy())


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(device):
    checkpoint_cfgs = getattr(cfg, "CHECKPOINTS", None)
    if checkpoint_cfgs:
        entries = [(HERE / c["path"], c) for c in checkpoint_cfgs]
    else:
        entries = [(p, {}) for p in sorted((HERE / "models").glob("*.pt"))]

    if not entries:
        raise FileNotFoundError(
            f"No checkpoints found. Add .pt files to models/ or set CHECKPOINTS in config.py.")

    models = []
    for ckpt, overrides in entries:
        pool       = overrides.get("pool",       cfg.MODEL_POOL or "avg")
        width_mult = overrides.get("width_mult", cfg.MODEL_WIDTH_MULT)
        dropout    = overrides.get("dropout",    cfg.MODEL_DROPOUT)
        aux_weight = overrides.get("aux_weight", getattr(cfg, "MODEL_AUX_WEIGHT", 0.0))
        model = Cnn3d(pool=pool, width_mult=width_mult,
                      dropout=dropout, aux_weight=aux_weight)
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state)
        model.eval().to(device)
        models.append(model)
    print(f"loaded {len(models)} checkpoint(s)")
    return models


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_one(vol: np.ndarray, models, device,
                iso_x=None, iso_y=None) -> float:
    """Average logits across all models and TTA variants, then calibrate."""
    variants = list(tta_volumes(vol))
    batch   = torch.stack([t for _, t in variants]).to(device)  # (N, 1, X, Y, Z)
    weights = torch.tensor([w for w, _ in variants],
                            dtype=torch.float32, device=device)

    total, count = 0.0, 0.0
    for model in models:
        logits = model(batch).float()                  # (N,)
        total += (logits * weights).sum().item()
        count += weights.sum().item()

    mean_logit = total / count                         # average in logit space

    calibration = getattr(cfg, "CALIBRATION", "temperature")

    if calibration == "isotonic":
        raw_p = float(torch.sigmoid(torch.tensor(mean_logit)).item())
        p = float(np.interp(raw_p, iso_x, iso_y))
    elif calibration == "temperature":
        t = getattr(cfg, "TEMPERATURE", 1.0)
        p = float(1.0 / (1.0 + np.exp(-mean_logit / t)))
    else:
        p = float(1.0 / (1.0 + np.exp(-mean_logit)))

    return float(np.clip(p, cfg.CLIP_LO, cfg.CLIP_HI))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  |  norm: {cfg.NORM}  |  crop_margin: {cfg.CROP_MARGIN}")
    calibration = getattr(cfg, "CALIBRATION", "temperature")
    print(f"tta_flip: {cfg.TTA_FLIP}  |  tta_rotations: {cfg.TTA_ROTATIONS}  "
          f"|  calibration: {calibration}")

    mean_frame, template_flat = load_assets()
    models = load_models(device)

    iso_x, iso_y = None, None
    if calibration == "isotonic":
        iso_x, iso_y = load_isotonic()
        print(f"isotonic calibrator: {len(iso_x)} breakpoints")

    fmt = pd.read_csv(DATA / "submission_format.csv")
    uids = fmt["uid"].astype(str).tolist()

    # Preprocess all scans in parallel (CPU-bound; releases GIL via numpy/scipy C code).
    # Workers saturate CPUs while GPU runs inference sequentially on the results.
    n_workers = min(8, len(uids))
    margin = cfg.CROP_MARGIN

    def _preprocess(uid):
        path = DATA / "niftis" / f"{uid}.nii.gz"
        return uid, preprocess(path, mean_frame, template_flat, margin=margin)

    crops = {}   # uid -> crop array, preserved for ordering below
    n_fallback = 0
    print(f"preprocessing {len(uids)} scans with {n_workers} workers...")
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_preprocess, uid): uid for uid in uids}
        for i, fut in enumerate(as_completed(futures)):
            uid = futures[fut]
            try:
                _, crop = fut.result()
                crops[uid] = crop
            except Exception as e:
                print(f"preprocess fallback ({type(e).__name__}): {uid}")
                crops[uid] = None
            if (i + 1) % 50 == 0:
                print(f"  preprocessed {i + 1}/{len(uids)}")

    print("running inference...")
    preds = []
    for i, uid in enumerate(uids):
        crop = crops[uid]
        if crop is None:
            p = cfg.FALLBACK_P
            n_fallback += 1
        else:
            p = predict_one(crop, models, device, iso_x, iso_y)
        preds.append(p)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(uids)} done")

    pd.DataFrame({"uid": fmt["uid"], "is_pathologic": preds}).to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(preds)} rows, {n_fallback} fallbacks)")


if __name__ == "__main__":
    main()
