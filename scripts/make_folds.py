"""Generate an N-fold stratified split from an existing folds.csv.

Reads UIDs and labels from a reference CSV (default: prepared/folds.csv),
applies StratifiedKFold(n_folds), and writes prepared/folds{n}.csv.
The crops are unchanged — only the fold-assignment column changes.

Usage (on the pod):
    python scripts/make_folds.py --n-folds 10
    python scripts/make_folds.py --n-folds 10 --seed 17
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.config import TrainConfig

CFG = TrainConfig()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-folds", type=int, required=True,
                    help="number of folds to create (e.g. 10)")
    ap.add_argument("--seed", type=int, default=CFG.seed,
                    help=f"random seed (default: {CFG.seed})")
    ap.add_argument("--source-csv", default=None,
                    help="source folds CSV with uid + is_pathologic columns "
                         "(default: prepared/folds.csv)")
    ap.add_argument("--out-csv", default=None,
                    help="output path (default: prepared/folds{n}.csv)")
    args = ap.parse_args()

    src = Path(args.source_csv) if args.source_csv else CFG.folds_csv
    if not src.exists():
        raise SystemExit(f"source CSV not found: {src}")

    df = pd.read_csv(src)[["uid", "is_pathologic"]].copy()
    df = df.sort_values("uid").reset_index(drop=True)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    df["fold"] = -1
    for k, (_, val_idx) in enumerate(skf.split(df, df["is_pathologic"])):
        df.loc[val_idx, "fold"] = k

    if args.out_csv:
        out = Path(args.out_csv)
    elif args.seed == CFG.seed:
        out = src.parent / f"folds{args.n_folds}.csv"        # backward-compatible default
    else:
        out = src.parent / f"folds{args.n_folds}_s{args.seed}.csv"  # seed-tagged
    df.to_csv(out, index=False)

    counts = df.groupby("fold")["is_pathologic"].agg(["sum", "count"])
    counts.columns = ["n_pos", "n_total"]
    counts["n_neg"] = counts["n_total"] - counts["n_pos"]
    counts["pos_pct"] = (counts["n_pos"] / counts["n_total"] * 100).round(1)
    print(f"wrote {out}  ({len(df)} samples, {args.n_folds} folds)")
    print(counts.to_string())


if __name__ == "__main__":
    main()
