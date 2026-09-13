"""Iterative mislabel exclusion coordinator.

Two sub-commands:

  prepare  — pick the N most suspect samples from a consensus CSV, optionally
             filtered by SBR thresholds, write an exclusion CSV, and print the
             training command to run on the pod.

  evaluate — compare the baseline OOF (trained on all data) against the exclusion
             OOF (trained with --exclude-csv) on the same clean subset.

Typical workflow
----------------
Round 0  (already done)
  Baseline OOF lives at, e.g., baseline/oof.csv
  Seed-consensus CSV: output/seed_consensus.csv

Round 1
  python scripts/exclusion_iter.py prepare \\
      --consensus output/seed_consensus.csv \\
      --top-n 10 \\
      --sbr-csv output/features.csv \\
      --sbr-neg-max 1.5 \\
      --sbr-pos-min 2.5 \\
      --out prepared/excl_r1.csv \\
      --run-name cnn3d_excl_r1
  => prints training command; run it on the pod

  python scripts/exclusion_iter.py evaluate \\
      --baseline-oof baseline/oof.csv \\
      --excl-oof output/runs/cnn3d_excl_r1/oof.csv \\
      --exclude-csv prepared/excl_r1.csv
  => prints clean-subset comparison

Round 2  (extend or shrink based on result)
  python scripts/exclusion_iter.py prepare \\
      --consensus output/seed_consensus.csv \\
      --top-n 15 \\              # or --top-n 8 if r1 hurt
      ...
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.config import TrainConfig

CFG = TrainConfig()


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def cmd_prepare(args):
    consensus = pd.read_csv(args.consensus)
    print(f"Consensus file: {len(consensus)} samples, "
          f"{consensus['suspect'].sum()} flagged as suspect")

    suspects = consensus[consensus["suspect"]].copy()
    suspects = suspects.sort_values("gap", ascending=False)

    # Optional SBR-based filter to keep only samples where the SBR
    # independently corroborates the model's suspicion.
    if args.sbr_csv:
        feats = pd.read_csv(args.sbr_csv).set_index("uid")["sbr_putamen_min"].dropna()
        suspects["sbr_put_min"] = suspects["uid"].map(feats)

        keep_mask = pd.Series([True] * len(suspects), index=suspects.index)
        if args.sbr_neg_max is not None:
            # negatives: keep only if SBR is low (scan looks pathological)
            neg_mask = (suspects["is_pathologic"] == 0) & \
                       (suspects["sbr_put_min"] > args.sbr_neg_max)
            keep_mask = keep_mask & ~neg_mask  # drop negs with high SBR
        if args.sbr_pos_min is not None:
            # positives: keep only if SBR is high (likely DIP, not true PD)
            pos_mask_not_dip = (suspects["is_pathologic"] == 1) & \
                               (suspects["sbr_put_min"] < args.sbr_pos_min) & \
                               ~suspects["sbr_put_min"].isna()
            keep_mask = keep_mask & ~pos_mask_not_dip  # drop pos with low SBR (genuine PD)

        # also flag samples without SBR as ambiguous — skip them
        keep_mask = keep_mask & suspects["sbr_put_min"].notna()
        filtered_out = (~keep_mask).sum()
        suspects = suspects[keep_mask].copy()
        print(f"After SBR filter: {filtered_out} suspects removed (ambiguous / genuine hard cases)")
        print(f"  (neg with SBR >= {args.sbr_neg_max} removed; pos with SBR < {args.sbr_pos_min} removed)")

    # Top-N by gap
    if args.top_n and len(suspects) > args.top_n:
        suspects = suspects.head(args.top_n)

    out = Path(args.out) if args.out else CFG.prepared_dir / "excl_round.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    suspects[["uid"]].to_csv(out, index=False)

    print(f"\nExclusion list ({len(suspects)} samples) -> {out}")
    print(suspects[["uid", "is_pathologic", "mean_pred", "gap",
                     *([c for c in suspects.columns if c == "sbr_put_min"])]].to_string(index=False))

    run = args.run_name or "cnn3d_excl"
    folds_arg = f"--folds-csv {args.folds_csv}" if args.folds_csv else ""
    print(f"""
Training command (run on the pod):
  python scripts/train.py \\
      --run-name {run} \\
      --exclude-csv {out} \\
      {folds_arg}
      [other flags same as your baseline run]

After training, evaluate with:
  python scripts/exclusion_iter.py evaluate \\
      --baseline-oof <path/to/baseline/oof.csv> \\
      --excl-oof output/runs/{run}/oof.csv \\
      --exclude-csv {out}
""")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": len(y),
        "log_loss": float(log_loss(y, p)),
        "auroc": float(roc_auc_score(y, p)),
    }


def fmt(m):
    return f"ll={m['log_loss']:.4f}  AUC={m['auroc']:.4f}  n={m['n']}"


def cmd_evaluate(args):
    excl_path = Path(args.exclude_csv)
    excl_uids = set(pd.read_csv(excl_path)["uid"])

    oof_all   = pd.read_csv(args.baseline_oof)
    oof_clean = pd.read_csv(args.excl_oof)

    # Clean-subset: samples not in exclusion list, present in both OOFs
    oof_all_filt = oof_all[~oof_all["uid"].isin(excl_uids)].copy()
    common_uids  = set(oof_all_filt["uid"]) & set(oof_clean["uid"])
    oof_a = oof_all_filt[oof_all_filt["uid"].isin(common_uids)].sort_values("uid").reset_index(drop=True)
    oof_c = oof_clean[oof_clean["uid"].isin(common_uids)].sort_values("uid").reset_index(drop=True)

    assert (oof_a["is_pathologic"].values == oof_c["is_pathologic"].values).all(), \
        "Label mismatch between OOF files for common UIDs"

    m_all_full     = metrics(oof_all["is_pathologic"].values,   oof_all["pred"].values)
    m_all_clean    = metrics(oof_a["is_pathologic"].values,     oof_a["pred"].values)
    m_clean_full   = metrics(oof_clean["is_pathologic"].values, oof_clean["pred"].values)
    m_clean_common = metrics(oof_c["is_pathologic"].values,     oof_c["pred"].values)

    bline = Path(args.baseline_oof).parent.name
    excl  = Path(args.excl_oof).parent.name

    print(f"\nExclude list: {len(excl_uids)} UIDs  ({excl_path.name})")
    print(f"Baseline OOF  ({bline}): {len(oof_all)} samples")
    print(f"Exclusion OOF ({excl}):  {len(oof_clean)} samples")
    print(f"Common clean subset: {len(common_uids)} samples\n")

    w = 32
    print("=" * 72)
    print(f"{'Subset':<{w}} {'baseline':>19}   {'excl':>19}")
    print("-" * 72)
    print(f"{'Full OOF (own samples)':<{w}} {fmt(m_all_full):>19}   {fmt(m_clean_full):>19}")
    print(f"{'Clean subset only':<{w}} {fmt(m_all_clean):>19}   {fmt(m_clean_common):>19}")
    print("=" * 72)

    delta_ll  = m_all_clean["log_loss"]  - m_clean_common["log_loss"]
    delta_auc = m_clean_common["auroc"]  - m_all_clean["auroc"]
    print(f"\nClean-subset  Δlog_loss={delta_ll:+.4f}  ΔAUC={delta_auc:+.4f}")

    if delta_ll > 0.005:
        print("=> GENUINE improvement on clean data  → try adding more suspects next round")
    elif delta_ll < -0.005:
        print("=> Exclusion HURT the model on clean data  → shrink the list next round")
    else:
        print("=> No real change on clean subset  → improvement was measurement artifact; "
              "list may be too aggressive or too conservative")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # prepare
    pp = sub.add_parser("prepare", help="generate exclusion CSV for a round")
    pp.add_argument("--consensus", required=True,
                    help="path to output/seed_consensus.csv")
    pp.add_argument("--top-n", type=int, default=None,
                    help="keep at most top-N suspects (sorted by gap)")
    pp.add_argument("--sbr-csv", default=None,
                    help="features.csv with sbr_putamen_min column for corroboration filter")
    pp.add_argument("--sbr-neg-max", type=float, default=None,
                    help="negs with SBR above this threshold are kept (not mislabels)")
    pp.add_argument("--sbr-pos-min", type=float, default=None,
                    help="pos with SBR below this threshold are kept (genuine PD, not DIP)")
    pp.add_argument("--out", default=None,
                    help="output CSV path (default: prepared/excl_round.csv)")
    pp.add_argument("--run-name", default=None,
                    help="run name to embed in the printed training command")
    pp.add_argument("--folds-csv", default=None,
                    help="custom folds CSV to embed in the training command")

    # evaluate
    ep = sub.add_parser("evaluate", help="compare baseline vs exclusion OOF on clean subset")
    ep.add_argument("--baseline-oof", required=True,
                    help="oof.csv from the baseline run (all samples)")
    ep.add_argument("--excl-oof", required=True,
                    help="oof.csv from the exclusion run")
    ep.add_argument("--exclude-csv", required=True,
                    help="exclusion CSV used for the excl run (with 'uid' column)")

    args = ap.parse_args()
    if args.cmd == "prepare":
        cmd_prepare(args)
    else:
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
