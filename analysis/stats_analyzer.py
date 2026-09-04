import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class StatsAnalyzer:
    """Group comparisons (normal vs abnormal), per-feature discrimination,
    and a cross-validated multivariable check against the competition metric."""

    def __init__(self, config):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def per_feature(self, df, features):
        rows = []
        y = df["is_pathologic"].values
        for f in features:
            x = df[f].values
            ok = np.isfinite(x)
            xo, yo = x[ok], y[ok]
            a, b = xo[yo == 1], xo[yo == 0]
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            rank_biserial = 1 - 2 * u / (len(a) * len(b))
            auc = roc_auc_score(yo, xo)
            auc = max(auc, 1 - auc)
            lo, hi = self._bootstrap_auc(xo, yo)
            rows.append(
                {
                    "feature": f,
                    "n": int(ok.sum()),
                    "normal_median": float(np.median(b)),
                    "normal_iqr": float(np.subtract(*np.percentile(b, [75, 25]))),
                    "abnormal_median": float(np.median(a)),
                    "abnormal_iqr": float(np.subtract(*np.percentile(a, [75, 25]))),
                    "mannwhitney_p": float(p),
                    "rank_biserial": float(rank_biserial),
                    "auroc": float(auc),
                    "auroc_ci_lo": lo,
                    "auroc_ci_hi": hi,
                }
            )
        out = pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)
        out["p_holm"] = self._holm(out["mannwhitney_p"].values)
        return out

    def _bootstrap_auc(self, x, y):
        n = len(y)
        aucs = []
        for _ in range(self.config.n_bootstrap):
            idx = self.rng.integers(0, n, n)
            ys = y[idx]
            if ys.min() == ys.max():
                continue
            a = roc_auc_score(ys, x[idx])
            aucs.append(max(a, 1 - a))
        return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

    @staticmethod
    def _holm(pvals):
        order = np.argsort(pvals)
        m = len(pvals)
        adj = np.empty(m)
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, (m - rank) * pvals[i])
            adj[i] = min(running, 1.0)
        return adj

    def multivariable_cv(self, df, features):
        ok = np.isfinite(df[features].values).all(axis=1)
        X = df.loc[ok, features].values
        y = df.loc[ok, "is_pathologic"].values
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        skf = StratifiedKFold(self.config.cv_folds, shuffle=True, random_state=self.config.seed)
        oof = np.full(len(y), np.nan)
        for tr, te in skf.split(X, y):
            model.fit(X[tr], y[tr])
            oof[te] = model.predict_proba(X[te])[:, 1]
        prevalence = float(y.mean())
        return {
            "n": int(len(y)),
            "n_excluded_nan": int((~ok).sum()),
            "oof_auroc": float(roc_auc_score(y, oof)),
            "oof_log_loss": float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))),
            "baseline_log_loss_prevalence": float(
                log_loss(y, np.full_like(oof, prevalence))
            ),
            "oof": oof,
            "y": y,
            "uids": df.loc[ok, "uid"].tolist(),
        }
