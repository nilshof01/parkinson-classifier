"""Evaluate rotation TTA on an existing trained run.

Loads each fold's best_ema.pt, re-runs inference on the val set with
several rotation angles (and optional L-R flip), averages the predictions,
then reports OOF log loss for each TTA strategy.

Usage:
    python scripts/eval_rotation_tta.py <run_dir> --crops-dir <path>

Example:
    python scripts/eval_rotation_tta.py \
        "C:/Users/nilsh/Projects/DaT Parkinson's Challenge/baseline/cnn3d_m4_baseline_s0" \
        --crops-dir "C:/Users/nilsh/Projects/DaT Parkinson's Challenge/prepared/crops_m4"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.config import TrainConfig
from training.models.cnn3d import Cnn3d
from training.view_volume import VolumeView

CFG = TrainConfig()


def load_model(checkpoint_path, args_dict):
    width_mult = args_dict.get("width_mult", 1.0)
    dropout = args_dict.get("dropout3d", 0.3)
    pool = args_dict.get("pool", None)
    head = args_dict.get("head", "linear")
    model = Cnn3d(pretrained=False, pool=pool, head=head,
                  width_mult=width_mult, dropout=dropout)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def rotate_vol(vol, angle_deg):
    """Rotate volume in the axial plane (axes 0,1 = L-R / P-A), same as training aug."""
    if angle_deg == 0:
        return vol
    return ndimage.rotate(vol, angle_deg, axes=(0, 1), reshape=False, order=1)


def predict_single(model, vol, view, device, flip=False):
    """Normalize and run one volume through the model. Returns scalar probability."""
    if flip:
        vol = vol[::-1].copy()
    t = view(vol).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(t)
    return torch.sigmoid(logit.float()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="path to the run directory (contains fold0/, fold1/, …)")
    ap.add_argument("--crops-dir", default=None)
    ap.add_argument("--angles", default="-15,-10,-5,0,5,10,15",
                    help="comma-separated rotation angles to test (default: -15…15 in 5° steps)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--norm", default="zscore", choices=["zscore", "percentile"])
    ap.add_argument("--norm-percentile", type=float, default=99.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    args_path = run_dir / "args.json"
    if not args_path.exists():
        sys.exit(f"args.json not found in {run_dir}")

    run_args = json.loads(args_path.read_text())
    angles = [float(a) for a in args.angles.split(",")]

    crops_dir = Path(args.crops_dir) if args.crops_dir else \
                CFG.prepared_dir / "crops_m4"
    if not crops_dir.exists():
        sys.exit(f"crops directory not found: {crops_dir}")

    folds_csv = run_dir.parent.parent / "prepared" / "folds.csv"
    if not folds_csv.exists():
        folds_csv = Path(run_args.get("folds_csv", ""))
    if not folds_csv.exists():
        folds_csv = CFG.prepared_dir / "folds.csv"
    if not folds_csv.exists():
        sys.exit(f"folds.csv not found — pass it via run_args or CFG")

    device = torch.device(args.device)
    view = VolumeView(norm=args.norm, norm_percentile=args.norm_percentile)

    all_preds = {angle: [] for angle in angles}
    all_preds["flip"] = []
    all_preds["no_tta"] = []
    all_labels = []

    fold_dirs = sorted(run_dir.glob("fold*"))
    if not fold_dirs:
        sys.exit(f"no fold* directories found in {run_dir}")

    for fold_dir in fold_dirs:
        k = int(fold_dir.name.replace("fold", ""))
        ckpt = fold_dir / "best_ema.pt"
        if not ckpt.exists():
            print(f"  skipping fold {k} — no best_ema.pt")
            continue

        # use the val_preds.csv that was written during training — these are
        # the exact UIDs/labels the model was validated on, regardless of which
        # folds.csv is available locally
        val_preds_csv = fold_dir / "val_preds.csv"
        if not val_preds_csv.exists():
            print(f"  skipping fold {k} — no val_preds.csv")
            continue
        vp = pd.read_csv(val_preds_csv)
        val_uids = vp["uid"].values
        val_labels = vp["is_pathologic"].values

        model = load_model(ckpt, run_args).to(device)
        print(f"fold {k}: {len(val_uids)} val scans, angles {angles}")

        fold_preds = {angle: [] for angle in angles}
        fold_preds["flip"] = []
        fold_preds["no_tta"] = []

        for uid in val_uids:
            crop_path = crops_dir / f"{uid}.npy"
            if not crop_path.exists():
                print(f"  missing crop: {uid}")
                for key in fold_preds:
                    fold_preds[key].append(0.5)
                continue

            vol = np.load(crop_path).astype(np.float32)

            fold_preds["no_tta"].append(predict_single(model, vol, view, device))
            fold_preds["flip"].append(
                (predict_single(model, vol, view, device) +
                 predict_single(model, vol, view, device, flip=True)) / 2
            )
            for angle in angles:
                rot_vol = rotate_vol(vol, angle)
                fold_preds[angle].append(predict_single(model, rot_vol, view, device))

        for key in fold_preds:
            all_preds[key].extend(fold_preds[key])
        all_labels.extend(val_labels.tolist())

    all_labels = np.array(all_labels)
    eps = 1e-6

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{'Strategy':<45} {'OOF ll':>8} {'AUROC':>7}")
    print("-" * 63)

    def report(name, probs):
        p = np.clip(np.array(probs), eps, 1 - eps)
        ll = log_loss(all_labels, p)
        auc = roc_auc_score(all_labels, p)
        print(f"{name:<45} {ll:>8.4f} {auc:>7.4f}")
        return ll

    report("no_tta", all_preds["no_tta"])
    report("flip_tta (current)", all_preds["flip"])

    print()
    # individual rotation angles
    for angle in angles:
        report(f"rot {angle:+.0f}°", all_preds[angle])

    print()
    # averaged over all angles (pure rotation TTA, no flip)
    avg_rot = np.mean([all_preds[a] for a in angles], axis=0)
    report(f"avg all {len(angles)} rotations", avg_rot)

    # averaged over all angles + flip of each
    print("\nBuilding rotation+flip combos (computing flip of each rotation)…")
    angle_subsets = [
        ("±5° (3 angles)", [-5, 0, 5]),
        ("±10° (5 angles)", [-10, -5, 0, 5, 10]),
        ("±15° (7 angles)", [-15, -10, -5, 0, 5, 10, 15]),
    ]
    # we need flip of each rotated version — recompute
    # (store per-fold crops isn't feasible here; approximate by flipping then rotating)
    # instead report averaged subsets of existing angles
    print()
    for label, subset in angle_subsets:
        if not all(a in all_preds for a in subset):
            continue
        avg = np.mean([all_preds[a] for a in subset], axis=0)
        report(f"avg {label}", avg)

    # best combo: rotation avg + flip_tta
    avg_rot_flip = (avg_rot + np.array(all_preds["flip"])) / 2
    report("avg all rotations + flip", avg_rot_flip)


if __name__ == "__main__":
    main()
