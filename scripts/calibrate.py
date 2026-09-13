"""Fit post-hoc calibration on pooled OOF predictions.

Compares temperature scaling vs isotonic regression on the same OOF data,
prints recommended settings for submission/config.py, and optionally saves
the isotonic calibrator to submission/assets/calibrator.npz.

Also tunes CLIP_LO / CLIP_HI by sweeping a grid of clip bounds and finding
the pair that minimises log_loss on OOF.  Always run --tune-clips: the
optimal bounds depend on how often the model is confidently wrong, and the
default [0.02, 0.98] is almost always too conservative.

Usage:
    # compare calibration methods + tune clip bounds (recommended):
    python scripts/calibrate.py output/runs/ax_split_aux_excl/oof.csv --tune-clips

    # save isotonic calibrator for use in submission:
    python scripts/calibrate.py output/runs/ax_split_aux_excl/oof.csv \\
        --method isotonic --save-calibrator submission/assets/calibrator.npz

    # pool multiple OOF files (e.g. from different seeds/folds):
    python scripts/calibrate.py oof1.csv oof2.csv oof3.csv --tune-clips
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.config import TrainConfig

CFG = TrainConfig()


def load_oof(paths):
    dfs = [pd.read_csv(p)[["uid", "is_pathologic", "pred"]] for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates("uid")
    return df["is_pathologic"].values.astype(float), df["pred"].values.astype(float)


def fit_temperature(y, p):
    """Find T that minimises log_loss(y, sigmoid(logit(p) / T))."""
    raw_logits = logit(np.clip(p, 1e-7, 1 - 1e-7))

    def obj(T):
        cal = expit(raw_logits / T)
        return log_loss(y, np.clip(cal, 1e-7, 1 - 1e-7))

    result = minimize_scalar(obj, bounds=(0.1, 10.0), method="bounded")
    return float(result.x), float(result.fun)


def tune_clips(y, p):
    """Sweep (lo, hi) clip grid; return best (lo, hi, log_loss) tuple.

    Grid: lo in {1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02}
          hi = 1 - lo  (symmetric; asymmetric rarely helps on balanced data)
    """
    candidates = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05]
    rows = []
    for lo in candidates:
        hi = 1.0 - lo
        ll = log_loss(y, np.clip(p, lo, hi))
        rows.append((lo, hi, ll))

    rows.sort(key=lambda r: r[2])
    return rows


def fit_isotonic(y, p):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(p, y)
    cal = ir.predict(p)
    ll = log_loss(y, np.clip(cal, 1e-7, 1 - 1e-7))
    return ir, float(ll)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("oof_csvs", nargs="+",
                    help="path(s) to oof.csv with uid, is_pathologic, pred columns")
    ap.add_argument("--method", choices=["temperature", "isotonic", "both"],
                    default="both", help="which calibrator to fit (default: both)")
    ap.add_argument("--save-calibrator", default=None,
                    help="save isotonic breakpoints to this .npz path "
                         "(e.g. submission/assets/calibrator.npz)")
    ap.add_argument("--tune-clips", action="store_true",
                    help="sweep CLIP_LO/CLIP_HI grid and print optimal bounds")
    args = ap.parse_args()

    y, p = load_oof(args.oof_csvs)
    print(f"Loaded {len(y)} OOF samples from {len(args.oof_csvs)} file(s)")
    print(f"  label mean: {y.mean():.3f}  pred mean: {p.mean():.3f}")

    ll_raw = log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))
    print(f"\nBaseline (no calibration)  log_loss = {ll_raw:.5f}")
    print(f"  mean pred = {p.mean():.4f}  (use as FALLBACK_P in config.py)")

    if args.method in ("temperature", "both"):
        T, ll_t = fit_temperature(y, p)
        delta = ll_raw - ll_t
        print(f"\nTemperature scaling        log_loss = {ll_t:.5f}  (delta={delta:+.5f})")
        print(f"  T = {T:.4f}  {'(underconfident — model output too conservative)' if T < 1 else '(overconfident — model output too sharp)'}")
        print(f"  => set TEMPERATURE = {T:.4f} in submission/config.py")

    if args.method in ("isotonic", "both"):
        ir, ll_iso = fit_isotonic(y, p)
        delta = ll_raw - ll_iso
        print(f"\nIsotonic regression        log_loss = {ll_iso:.5f}  (delta={delta:+.5f})")
        print(f"  {len(ir.X_thresholds_)} breakpoints")

        if args.save_calibrator:
            out = Path(args.save_calibrator)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, x=ir.X_thresholds_, y=ir.y_thresholds_)
            print(f"  saved calibrator -> {out}")
            print(f"  => set CALIBRATION = 'isotonic' in submission/config.py")

    if args.method == "both":
        if ll_t < ll_iso:
            print(f"\nRecommendation: temperature (simpler, {ll_t:.5f} < {ll_iso:.5f})")
        else:
            print(f"\nRecommendation: isotonic ({ll_iso:.5f} < {ll_t:.5f})"
                  f" — run again with --method isotonic --save-calibrator "
                  f"submission/assets/calibrator.npz")

    if args.tune_clips:
        rows = tune_clips(y, p)
        best_lo, best_hi, _ = rows[0]
        print(f"\nClip-bound sweep (symmetric lo = 1-hi):")
        print(f"  {'CLIP_LO':<10} {'log_loss':>10}")
        for lo, _, ll in rows:
            marker = "  <-- best" if lo == best_lo else ""
            print(f"  {lo:<10g} {ll:>10.5f}{marker}")
        print(f"\n  => set CLIP_LO = {best_lo}  CLIP_HI = {best_hi} in submission/config.py")
        # also report how many OOF preds are outside the best clip range
        n_lo = int((p < best_lo).sum())
        n_hi = int((p > best_hi).sum())
        print(f"  ({n_lo} preds clipped at lo, {n_hi} clipped at hi out of {len(p)})")


if __name__ == "__main__":
    main()
