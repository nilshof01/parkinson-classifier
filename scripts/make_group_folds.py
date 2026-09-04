import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.config import TrainConfig

MIN_GROUP = 40

CFG = TrainConfig()


def main():
    folds = pd.read_csv(CFG.folds_csv)
    headers = pd.read_csv(CFG.repo_dir / "output" / "header_stats.csv")
    df = folds.merge(headers[["uid", "spacing_x"]], on="uid")
    df["group"] = df["spacing_x"].round(1).astype(str)
    counts = df["group"].value_counts()
    df.loc[df["group"].map(counts) < MIN_GROUP, "group"] = "other"

    group_ids = {g: i for i, g in enumerate(sorted(df["group"].unique()))}
    df["fold"] = df["group"].map(group_ids)
    out = CFG.prepared_dir / "folds_group.csv"
    df[["uid", "is_pathologic", "fold", "group"]].to_csv(out, index=False)
    print(f"written {out}")
    summary = df.groupby(["fold", "group"])["is_pathologic"].agg(["count", "mean"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
