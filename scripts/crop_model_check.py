import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.aligner import Aligner
from analysis.config import AnalysisConfig
from analysis.crop_features import CropFeatures
from analysis.frame_cache import FrameCache

CFG = AnalysisConfig()


def main():
    shared = CFG.repo_dir / "prepared" / "folds.csv"
    if shared.exists():
        ok = pd.read_csv(shared)  # shared CV folds -> OOF comparable across models
    else:
        qc = pd.read_csv(CFG.output_dir / "qc_per_scan.csv")
        ok = qc[qc["qc_pass"].fillna(False)][["uid", "is_pathologic"]]
    cache = FrameCache(CFG)
    mean_frame = np.load(CFG.output_dir / "cohort_mean_frame.npy")
    aligner = Aligner(CFG)
    cf = CropFeatures()
    if len(sys.argv) > 1:
        cf.POOL = int(sys.argv[1])

    print(f"building crop features for {len(ok)} scans ({cf.n_features} features) ...")
    X, y, uids = [], [], []
    for i, (uid, label) in enumerate(zip(ok["uid"], ok["is_pathologic"])):
        f = cache.load(uid)
        m = f[f > 0].mean()
        if m <= 0:
            continue
        f = f / m
        for _ in range(2):
            f, _s = aligner.align(f, mean_frame)
        feats = cf.extract(f)
        if feats is None:
            continue
        X.append(feats)
        y.append(label)
        uids.append(uid)
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(ok)}")
    X = np.stack(X)
    y = np.array(y)
    print(f"feature matrix: {X.shape}")

    model_params = dict(
        n_estimators=600, learning_rate=0.05, max_depth=4, subsample=0.8,
        colsample_bytree=0.5, reg_lambda=1.0, tree_method="hist",
        eval_metric="logloss", random_state=CFG.seed, n_jobs=16,
    )
    oof = np.full(len(y), np.nan)
    if "fold" in ok.columns:
        fold_of = ok.set_index("uid")["fold"]
        fold_ids = np.array([fold_of[u] for u in uids])
        splits = [(np.where(fold_ids != k)[0], np.where(fold_ids == k)[0])
                  for k in sorted(set(fold_ids))]
    else:
        skf = StratifiedKFold(CFG.cv_folds, shuffle=True, random_state=CFG.seed)
        splits = list(skf.split(X, y))
    for k, (tr, te) in enumerate(splits):
        model = XGBClassifier(**model_params)
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
        print(f"  fold {k + 1}: auroc {roc_auc_score(y[te], oof[te]):.4f}")

    result = {
        "n": int(len(y)),
        "n_features": int(X.shape[1]),
        "oof_auroc": float(roc_auc_score(y, oof)),
        "oof_log_loss": float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))),
        "oof_log_loss_clipped_02_98": float(log_loss(y, np.clip(oof, 0.02, 0.98))),
    }
    print(json.dumps(result, indent=2))
    tag = f"pool{cf.POOL}"
    pd.DataFrame({"uid": uids, "is_pathologic": y, "oof_pred": oof}).to_csv(
        CFG.output_dir / f"crop_xgb_oof_{tag}.csv", index=False
    )
    (CFG.output_dir / f"crop_xgb_result_{tag}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
