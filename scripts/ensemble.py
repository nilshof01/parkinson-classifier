import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.config import AnalysisConfig
from training.config import TrainConfig

ACFG = AnalysisConfig()
TCFG = TrainConfig()


def calibrated_ll(y, p):
    lo = logit(np.clip(p, 1e-6, 1 - 1e-6))
    t = minimize_scalar(
        lambda t: log_loss(y, np.clip(expit(lo / t), 1e-6, 1 - 1e-6)),
        bounds=(0.25, 10.0), method="bounded",
    ).x
    p_cal = np.clip(expit(lo / t), 0.02, 0.98)
    return float(t), float(log_loss(y, p_cal))


def main():
    ap = argparse.ArgumentParser(description="Fit ensemble weights on shared-fold OOF")
    ap.add_argument("runs", nargs="+", help="training run names under output/runs")
    ap.add_argument("--xgb-csv", default=str(ACFG.output_dir / "crop_xgb_oof_pool2.csv"),
                    help="XGBoost OOF csv ('' to exclude)")
    args = ap.parse_args()

    folds = pd.read_csv(TCFG.folds_csv)
    df = folds[["uid", "is_pathologic"]].copy()
    members = []
    for r in args.runs:
        oof = pd.read_csv(TCFG.runs_dir / r / "oof.csv")[["uid", "pred"]]
        df = df.merge(oof.rename(columns={"pred": r}), on="uid")
        members.append(r)
    if args.xgb_csv:
        xgb = pd.read_csv(args.xgb_csv).rename(columns={"oof_pred": "xgb"})
        df = df.merge(xgb[["uid", "xgb"]], on="uid")
        members.append("xgb")

    y = df["is_pathologic"].values
    P = df[members].values.clip(1e-6, 1 - 1e-6)
    fold = df.merge(folds[["uid", "fold"]], on="uid")["fold"].values

    print(f"members (n={len(df)} scans):")
    for i, m in enumerate(members):
        t, ll = calibrated_ll(y, P[:, i])
        print(f"  {m}: auroc {roc_auc_score(y, P[:, i]):.4f}  cal+clip ll {ll:.4f}")

    mean_p = P.mean(axis=1)
    t, ll_mean = calibrated_ll(y, mean_p)
    print(f"plain mean: auroc {roc_auc_score(y, mean_p):.4f}  cal+clip ll {ll_mean:.4f}")

    def fit_w(Pt, yt):
        def loss(w):
            w = np.abs(w) / np.abs(w).sum()
            return log_loss(yt, (Pt * w).sum(axis=1))
        res = minimize(loss, np.full(Pt.shape[1], 1 / Pt.shape[1]), method="Nelder-Mead")
        return np.abs(res.x) / np.abs(res.x).sum()

    # nested: weights+temperature never see the fold they are scored on, so the
    # mean-vs-weighted decision is not flattered by fitting optimism
    nested = {}
    for mode in ("mean", "weighted"):
        oof = np.zeros(len(y))
        for k in sorted(set(fold)):
            tr, te = fold != k, fold == k
            w = fit_w(P[tr], y[tr]) if mode == "weighted" else np.full(len(members), 1 / len(members))
            lo_tr = logit(np.clip((P[tr] * w).sum(axis=1), 1e-6, 1 - 1e-6))
            t = minimize_scalar(
                lambda t: log_loss(y[tr], np.clip(expit(lo_tr / t), 1e-6, 1 - 1e-6)),
                bounds=(0.25, 10.0), method="bounded").x
            oof[te] = expit(logit(np.clip((P[te] * w).sum(axis=1), 1e-6, 1 - 1e-6)) / t)
        nested[mode] = float(log_loss(y, np.clip(oof, 0.02, 0.98)))
        print(f"nested {mode}: cal+clip ll {nested[mode]:.4f}")

    use_weights = nested["weighted"] < nested["mean"]
    w = fit_w(P, y)
    print("full-data weights (used only if nested prefers weighted):",
          {m: round(float(x), 3) for m, x in zip(members, w)})
    final_w = w if use_weights else np.full(len(members), 1 / len(members))
    final_p = (P * final_w).sum(axis=1)
    temp, ll_final = calibrated_ll(y, final_p)
    out_dir = ACFG.output_dir / "ensemble"
    out_dir.mkdir(exist_ok=True)
    spec = {
        "members": members,
        "weights": [float(x) for x in final_w],
        "weighted_beats_mean_nested": bool(use_weights),
        "temperature": temp,
        "clip": [0.02, 0.98],
        "oof_auroc": float(roc_auc_score(y, final_p)),
        "oof_log_loss_calibrated_clipped": ll_final,
        "nested_log_loss": nested,
    }
    (out_dir / "spec.json").write_text(json.dumps(spec, indent=2))
    df["ensemble_pred"] = final_p
    df.to_csv(out_dir / "oof_ensemble.csv", index=False)
    print(f"\nfinal ({'weighted' if use_weights else 'mean'}): "
          f"cal+clip ll {ll_final:.4f} — spec saved to {out_dir / 'spec.json'}")


if __name__ == "__main__":
    main()
