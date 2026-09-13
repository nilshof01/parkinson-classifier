"""Visualize suspicious samples from exclude_mislabels.csv.

For each scan, saves a PNG with three MIP projections (axial, coronal, sagittal)
and annotates with label, OOF prediction, SBR, and the reason for flagging.

Usage (on the pod):
    python scripts/visualize_samples.py
    python scripts/visualize_samples.py \
        --exclude-csv prepared/exclude_mislabels.csv \
        --crops-dir prepared/crops_m4 \
        --out-dir output/mislabel_review
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.config import TrainConfig

CFG = TrainConfig()


def percentile_norm(vol, pct=99.0):
    inside = vol > 0
    if not inside.any():
        return vol
    p = float(np.percentile(vol[inside], pct))
    return vol / (p + 1e-6)


def make_panel(vol):
    """Return (axial, coronal, sagittal) MIP arrays, each display-ready (Y up)."""
    # crop axes: 0=X(L-R), 1=Y(P→A, low=posterior), 2=Z(I→S)
    axial   = vol.max(axis=2)        # (X, Y) — looking from above
    coronal = vol.max(axis=1)        # (X, Z) — looking from front
    sagittal = vol.max(axis=0)       # (Y, Z) — looking from side

    # transpose so rows = second axis, flip Z so superior is up
    axial    = axial.T               # (Y, X): posterior at top (row 0)
    coronal  = coronal.T[::-1]       # (Z, X): superior at top
    sagittal = sagittal.T[::-1]      # (Z, Y): superior at top, posterior left
    return axial, coronal, sagittal


def save_figure(uid, vol, label, pred, sbr, reason, out_path):
    vol_norm = percentile_norm(vol.astype(np.float32))
    axial, coronal, sagittal = make_panel(vol_norm)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("#1a1a1a")

    vmax = np.percentile(vol_norm[vol_norm > 0], 99) if (vol_norm > 0).any() else 1.0

    for ax, img, title in zip(
        axes,
        [axial, coronal, sagittal],
        ["Axial MIP\n(looking down, I→S)", "Coronal MIP\n(looking front, P→A)", "Sagittal MIP\n(looking side, L→R)"],
    ):
        ax.imshow(img, cmap="hot", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(title, color="white", fontsize=9)
        ax.axis("off")

    label_str = "PD (1)" if label == 1 else "Healthy (0)"
    color = "#ff6b6b" if label == 1 else "#6bcb77"
    pred_str = f"{pred:.3f}"
    title = (
        f"{uid}   label={label_str}   pred={pred_str}   SBR_put_min={sbr:.2f}\n"
        f"reason: {reason}"
    )
    fig.suptitle(title, color=color, fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-csv", default=None,
                    help="CSV with uid, true_label, oof_pred, sbr_putamen_min, reason "
                         "(default: prepared/exclude_mislabels.csv)")
    ap.add_argument("--crops-dir", default=None,
                    help="directory with {uid}.npy crops (default: prepared/crops_m4)")
    ap.add_argument("--out-dir", default=None,
                    help="output directory for PNGs (default: output/mislabel_review)")
    ap.add_argument("--uid", default=None,
                    help="visualize a single UID instead of all flagged cases")
    args = ap.parse_args()

    exc_path = Path(args.exclude_csv) if args.exclude_csv \
        else CFG.prepared_dir / "exclude_mislabels.csv"
    crops_dir = Path(args.crops_dir) if args.crops_dir \
        else CFG.prepared_dir / "crops_m4"
    out_dir = Path(args.out_dir) if args.out_dir \
        else CFG.repo_dir / "output" / "mislabel_review"
    out_dir.mkdir(parents=True, exist_ok=True)

    exc = pd.read_csv(exc_path)

    if args.uid:
        exc = exc[exc["uid"] == args.uid]
        if exc.empty:
            # allow visualizing any UID not in the exclude list
            exc = pd.DataFrame([{"uid": args.uid, "true_label": -1,
                                  "oof_pred": float("nan"),
                                  "sbr_putamen_min": float("nan"),
                                  "reason": "manual_query"}])

    missing = 0
    for _, row in exc.iterrows():
        uid = row["uid"]
        crop_path = crops_dir / f"{uid}.npy"
        if not crop_path.exists():
            print(f"  skip {uid} — crop not found")
            missing += 1
            continue

        vol = np.load(crop_path).astype(np.float32)
        label  = int(row.get("true_label", -1))
        pred   = float(row.get("oof_pred", float("nan")))
        sbr    = float(row.get("sbr_putamen_min", float("nan")))
        reason = str(row.get("reason", ""))

        out_path = out_dir / f"{uid}_label{label}.png"
        save_figure(uid, vol, label, pred, sbr, reason, out_path)
        print(f"  saved {out_path.name}")

    print(f"\ndone — {len(exc) - missing} images in {out_dir}  ({missing} crops missing)")


if __name__ == "__main__":
    main()
