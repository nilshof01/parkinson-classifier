import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.config import TrainConfig

CFG = TrainConfig()


def main():
    run_dir = CFG.runs_dir / sys.argv[1]
    oof = pd.read_csv(run_dir / "oof.csv")
    groups = pd.read_csv(CFG.prepared_dir / "folds_group.csv")
    df = oof.merge(groups[["uid", "group"]], on="uid")

    rows = []
    for g, sub in df.groupby("group"):
        y, p = sub["is_pathologic"].values, sub["pred"].values
        train_prev = df.loc[df["group"] != g, "is_pathologic"].mean()
        rows.append(
            {
                "group": g,
                "n": len(sub),
                "prevalence": float(y.mean()),
                "auroc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else np.nan,
                "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
                "log_loss_prevalence_baseline": float(
                    log_loss(y, np.full_like(p, train_prev))
                ),
            }
        )
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    out.to_csv(run_dir / "per_group_metrics.csv", index=False)


if __name__ == "__main__":
    main()
