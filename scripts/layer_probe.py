r"""Per-layer probing of cnn3d: fit a logistic regression head at each block
and compare predictions for TP/TN/FP/FN cases.

Usage (pod):
    python scripts/layer_probe.py \
        --run output/runs/cnn3d_m4_baseline_s0 \
        --fold 0
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.config import AnalysisConfig
from training.config import TrainConfig
from training.models.cnn3d import Cnn3d
from training.view_volume import VolumeView

ACFG = AnalysisConfig()
TCFG = TrainConfig()


def extract_layer_embeddings(model, uids, crops_dir, device, batch=32):
    """Return dict layer_idx -> (N, C) numpy array of gap-pooled activations."""
    view = VolumeView()
    acts = {i: [] for i in range(len(model.features))}

    hooks = []
    buffers = {i: [] for i in range(len(model.features))}

    def make_hook(idx):
        def hook(_, __, output):
            pooled = output.mean(dim=(2, 3, 4)).cpu().numpy()
            buffers[idx].append(pooled)
        return hook

    for i, block in enumerate(model.features):
        hooks.append(block.register_forward_hook(make_hook(i)))

    model.eval()
    with torch.no_grad():
        for start in range(0, len(uids), batch):
            batch_uids = uids[start:start + batch]
            vols = []
            for u in batch_uids:
                npy = np.load(crops_dir / f"{u}.npy").astype(np.float32)
                vols.append(view(npy))
            x = torch.stack(vols).to(device)
            model(x)

    for h in hooks:
        h.remove()

    return {i: np.concatenate(buffers[i], axis=0) for i in buffers}


def probe_layer(E_tr, y_tr, E_va, y_va):
    scaler = StandardScaler().fit(E_tr)
    clf = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
    clf.fit(scaler.transform(E_tr), y_tr)
    p = clf.predict_proba(scaler.transform(E_va))[:, 1]
    auc = roc_auc_score(y_va, p)
    ll = log_loss(y_va, np.clip(p, 1e-6, 1 - 1e-6))
    return p, auc, ll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folds-csv", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    folds_path = Path(args.folds_csv) if args.folds_csv else TCFG.folds_csv
    folds = pd.read_csv(folds_path)
    crops_dir = TCFG.prepared_dir / "crops_m4"
    if not crops_dir.exists():
        crops_dir = TCFG.crops_dir

    tr_df = folds[folds["fold"] != args.fold].reset_index(drop=True)
    va_df = folds[folds["fold"] == args.fold].reset_index(drop=True)
    y_tr = tr_df["is_pathologic"].values
    y_va = va_df["is_pathologic"].values

    # load val predictions to identify TP/TN/FP/FN
    vp = pd.read_csv(run_dir / f"fold{args.fold}" / "val_preds.csv")
    vp = vp.set_index("uid")
    va_df["model_pred"] = va_df["uid"].map(vp["pred"])
    va_df["pred_label"] = (va_df["model_pred"] > 0.5).astype(int)
    quad = {
        "TP": (va_df["pred_label"] == 1) & (va_df["is_pathologic"] == 1),
        "TN": (va_df["pred_label"] == 0) & (va_df["is_pathologic"] == 0),
        "FP": (va_df["pred_label"] == 1) & (va_df["is_pathologic"] == 0),
        "FN": (va_df["pred_label"] == 0) & (va_df["is_pathologic"] == 1),
    }

    model = Cnn3d(pretrained=False, dropout=0.3)
    ckpt = run_dir / f"fold{args.fold}" / "best_ema.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(args.device)

    print(f"\nExtracting layer activations — fold {args.fold} "
          f"(train n={len(tr_df)}, val n={len(va_df)})")
    E_tr = extract_layer_embeddings(model, list(tr_df["uid"]), crops_dir, args.device)
    E_va = extract_layer_embeddings(model, list(va_df["uid"]), crops_dir, args.device)

    n_layers = len(E_tr)
    layer_names = [f"block{i} ({ch}ch)"
                   for i, ch in enumerate([32, 64, 128, 256])]

    print(f"\n{'Layer':<18}  {'AUROC':>6}  {'LogLoss':>8}  "
          f"{'TP mean':>8}  {'TN mean':>8}  {'FP mean':>8}  {'FN mean':>8}")
    print("-" * 72)

    results = []
    for i in range(n_layers):
        p, auc, ll = probe_layer(E_tr[i], y_tr, E_va[i], y_va)
        row = {"layer": layer_names[i], "auc": auc, "ll": ll}
        for name, mask in quad.items():
            row[f"mean_{name}"] = float(p[mask].mean()) if mask.any() else float("nan")
        results.append(row)
        print(f"{layer_names[i]:<18}  {auc:>6.4f}  {ll:>8.4f}  "
              f"{row['mean_TP']:>8.3f}  {row['mean_TN']:>8.3f}  "
              f"{row['mean_FP']:>8.3f}  {row['mean_FN']:>8.3f}")

    # also report final model head for comparison
    p_final = va_df["model_pred"].values
    auc_f = roc_auc_score(y_va, p_final)
    ll_f = log_loss(y_va, np.clip(p_final, 1e-6, 1 - 1e-6))
    row_f = {"layer": "model head", "auc": auc_f, "ll": ll_f}
    for name, mask in quad.items():
        row_f[f"mean_{name}"] = float(p_final[mask].mean()) if mask.any() else float("nan")
    results.append(row_f)
    print(f"{'model head':<18}  {auc_f:>6.4f}  {ll_f:>8.4f}  "
          f"{row_f['mean_TP']:>8.3f}  {row_f['mean_TN']:>8.3f}  "
          f"{row_f['mean_FP']:>8.3f}  {row_f['mean_FN']:>8.3f}")

    df = pd.DataFrame(results)

    # ── Plot ───────────────────────────────────────────────────────────────────
    out = ACFG.output_dir / "error_analysis"
    out.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = [r["layer"] for r in results]
    x = np.arange(len(labels))

    ax = axes[0]
    ax.bar(x, df["auc"], color="steelblue")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_title("Probe AUROC per layer"); ax.set_ylim(0.5, 1.0)

    ax = axes[1]
    ax.bar(x, df["ll"], color="salmon")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_title("Probe log-loss per layer")

    ax = axes[2]
    for col, color, label in [("mean_TP", "steelblue", "TP"),
                               ("mean_TN", "lightblue", "TN"),
                               ("mean_FP", "orange", "FP"),
                               ("mean_FN", "crimson", "FN")]:
        ax.plot(x, df[col], marker="o", label=label, color=color)
    ax.axhline(0.5, ls="--", color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_title("Mean probe prediction per quadrant")
    ax.set_ylabel("mean predicted probability")
    ax.legend(fontsize=8)

    run_name = run_dir.name
    fig.suptitle(f"Layer probe — {run_name} fold {args.fold}", fontsize=11)
    fig.tight_layout()
    plot_path = out / f"layer_probe_{run_name}_fold{args.fold}.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    print(f"\nplot: {plot_path}")


if __name__ == "__main__":
    main()
