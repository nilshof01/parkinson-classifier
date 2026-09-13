"""Multi-seed OOF consensus — identify samples with persistent model disagreement.

Train the same config with different --seed-offset values, then run:

    python scripts/seed_consensus.py \\
        output/runs/run_seed0/oof.csv \\
        output/runs/run_seed1/oof.csv \\
        output/runs/run_seed2/oof.csv \\
        --out output/seed_consensus.csv

Samples where the mean OOF prediction consistently disagrees with the label
across all independent seeds are the strongest evidence of a bad label —
independent of SBR, domain knowledge, or a single model's opinion.

Interpretation of output columns:
  mean_pred   mean prediction across all seeds
  std_pred    standard deviation (high = model is uncertain / inconsistent)
  n_seeds     how many seeds contributed a prediction for this UID
  gap         |mean_pred - label|  (1 = perfect disagreement, 0 = agreement)
  suspect     True if gap > --gap-thresh in all seeds (consensus disagreement)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.config import TrainConfig

CFG = TrainConfig()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oof_csvs", nargs="+",
                    help="paths to oof.csv files from different seed runs")
    ap.add_argument("--out", default=None,
                    help="output CSV path (default: output/seed_consensus.csv)")
    ap.add_argument("--gap-thresh", type=float, default=0.5,
                    help="prediction distance from label to flag a seed as disagreeing "
                         "(default 0.5; pred>0.5 for label=0, pred<0.5 for label=1)")
    ap.add_argument("--min-seeds", type=int, default=None,
                    help="minimum seeds required to flag a sample as suspect "
                         "(default: all seeds must disagree)")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out \
        else CFG.repo_dir / "output" / "seed_consensus.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    min_seeds = args.min_seeds  # None means all

    dfs = []
    for i, p in enumerate(args.oof_csvs):
        df = pd.read_csv(p)[["uid", "is_pathologic", "pred"]].copy()
        df = df.rename(columns={"pred": f"pred_{i}"})
        dfs.append(df)

    merged = dfs[0][["uid", "is_pathologic"]].copy()
    for df in dfs:
        pred_col = [c for c in df.columns if c.startswith("pred_")][0]
        merged = merged.merge(df[["uid", pred_col]], on="uid", how="outer")

    pred_cols = [c for c in merged.columns if c.startswith("pred_")]
    pred_mat = merged[pred_cols].values  # (N, n_seeds)

    merged["mean_pred"] = np.nanmean(pred_mat, axis=1)
    merged["std_pred"]  = np.nanstd(pred_mat,  axis=1)
    merged["n_seeds"]   = (~np.isnan(pred_mat)).sum(axis=1)
    merged["gap"]       = (merged["mean_pred"] - merged["is_pathologic"]).abs()

    # Per-seed disagreement: pred > gap_thresh away from label
    label = merged["is_pathologic"].values[:, None]
    disagreed = np.where(
        label == 0,
        pred_mat > args.gap_thresh,      # label=0 but pred says PD
        pred_mat < (1 - args.gap_thresh), # label=1 but pred says healthy
    )  # (N, n_seeds), NaN slots treated as agreement
    disagreed = np.where(np.isnan(pred_mat), False, disagreed)
    n_disagree = disagreed.sum(axis=1)

    req = min_seeds if min_seeds is not None else pred_mat.shape[1]
    merged["suspect"] = n_disagree >= req
    merged["n_disagree"] = n_disagree

    # Sort by severity: suspects first, then by gap
    result = merged[["uid", "is_pathologic", "mean_pred", "std_pred",
                      "n_seeds", "n_disagree", "gap", "suspect"] + pred_cols]
    result = result.sort_values(["suspect", "gap"], ascending=[False, False])

    result.to_csv(out_path, index=False)

    suspects = result[result["suspect"]]
    neg_suspects = suspects[suspects["is_pathologic"] == 0]
    pos_suspects = suspects[suspects["is_pathologic"] == 1]

    print(f"\n{len(args.oof_csvs)} seeds, {len(merged)} samples")
    print(f"Consensus suspects (disagreed in {req}/{len(pred_cols)} seeds): "
          f"{len(suspects)} total  "
          f"({len(neg_suspects)} neg / {len(pos_suspects)} pos)")
    print(f"\nTop 20 suspects:")
    print(suspects[["uid", "is_pathologic", "mean_pred", "std_pred",
                     "n_disagree", "gap"]].head(20).to_string(index=False))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
