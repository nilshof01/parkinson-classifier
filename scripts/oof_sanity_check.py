"""Compare two OOF runs on an equal, clean subset.

Usage:
    python scripts/oof_sanity_check.py \
        --run-all   output/runs/cnn3d_volume3d_axbins      \
        --run-clean output/runs/cnn3d_volume3d_axbins_excl \
        --exclude-csv prepared/exclude_mislabels.csv

Prints a side-by-side table of:
  - full OOF (including suspicious samples where present)
  - OOF restricted to the clean subset (same samples for both models)

The "clean subset" comparison is the apples-to-apples check:
  - If run-clean beats run-all on the same clean subset → model genuinely improved
  - If they tie → the reported improvement was only measurement artifact
    (dirty val folds inflated run-all's loss but not run-clean's)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.config import TrainConfig

CFG = TrainConfig()


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": len(y),
        "log_loss": float(log_loss(y, p)),
        "auroc": float(roc_auc_score(y, p)),
    }


def load_oof(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "oof.csv"
    if not path.exists():
        raise FileNotFoundError(f"oof.csv not found in {run_dir}")
    return pd.read_csv(path)


def fmt(m):
    return f"ll={m['log_loss']:.4f}  AUC={m['auroc']:.4f}  n={m['n']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-all", required=True,
                    help="run trained on ALL data (no --exclude-csv)")
    ap.add_argument("--run-clean", required=True,
                    help="run trained with --exclude-csv")
    ap.add_argument("--exclude-csv", default=None,
                    help="exclude list CSV with 'uid' column "
                         "(default: prepared/exclude_mislabels.csv)")
    args = ap.parse_args()

    excl_path = Path(args.exclude_csv) if args.exclude_csv \
        else CFG.prepared_dir / "exclude_mislabels.csv"
    excl_uids = set(pd.read_csv(excl_path)["uid"])

    run_all   = Path(args.run_all)
    run_clean = Path(args.run_clean)

    oof_all   = load_oof(run_all)
    oof_clean = load_oof(run_clean)

    # Subset of run_all that excludes suspicious samples
    oof_all_filtered = oof_all[~oof_all["uid"].isin(excl_uids)].copy()

    # Make sure we compare on exactly the same UIDs
    common_uids = set(oof_all_filtered["uid"]) & set(oof_clean["uid"])
    oof_all_common   = oof_all_filtered[oof_all_filtered["uid"].isin(common_uids)]
    oof_clean_common = oof_clean[oof_clean["uid"].isin(common_uids)]

    # Sort both the same way so we can sanity-check label alignment
    oof_all_common   = oof_all_common.sort_values("uid").reset_index(drop=True)
    oof_clean_common = oof_clean_common.sort_values("uid").reset_index(drop=True)

    assert (oof_all_common["is_pathologic"].values ==
            oof_clean_common["is_pathologic"].values).all(), \
        "Label mismatch between the two OOF files for common UIDs"

    print(f"\nExclude list: {len(excl_uids)} UIDs from {excl_path.name}")
    print(f"Run ALL  ({run_all.name}):   {len(oof_all)} samples in oof.csv")
    print(f"Run CLEAN({run_clean.name}): {len(oof_clean)} samples in oof.csv")
    print(f"Common clean subset: {len(common_uids)} samples\n")

    m_all_full     = metrics(oof_all["is_pathologic"].values,   oof_all["pred"].values)
    m_all_clean    = metrics(oof_all_common["is_pathologic"].values,
                             oof_all_common["pred"].values)
    m_clean_full   = metrics(oof_clean["is_pathologic"].values, oof_clean["pred"].values)
    m_clean_common = metrics(oof_clean_common["is_pathologic"].values,
                             oof_clean_common["pred"].values)

    print("=" * 65)
    print(f"{'Subset':<28} {'run-all':>18}   {'run-clean':>18}")
    print("-" * 65)
    print(f"{'Full OOF (own samples)':<28} {fmt(m_all_full):>18}   {fmt(m_clean_full):>18}")
    print(f"{'Clean subset only':<28} {fmt(m_all_clean):>18}   {fmt(m_clean_common):>18}")
    print("=" * 65)

    delta_ll  = m_all_clean["log_loss"]  - m_clean_common["log_loss"]
    delta_auc = m_clean_common["auroc"]  - m_all_clean["auroc"]
    print(f"\nClean-subset improvement  Δlog_loss={delta_ll:+.4f}  ΔAUC={delta_auc:+.4f}")

    if delta_ll > 0.005:
        print("=> Model genuinely improved on clean data (not just measurement artifact)")
    elif delta_ll < -0.005:
        print("=> run-clean is WORSE on clean subset — exclusion may have removed useful signal")
    else:
        print("=> Near-identical on clean subset: improvement was mostly measurement artifact "
              "(contaminated val folds inflated run-all's reported loss)")


if __name__ == "__main__":
    main()
