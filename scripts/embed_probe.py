import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.config import TrainConfig
from training.models.registry import ModelRegistry
from training.view_volume import VolumeView

CFG = TrainConfig()


def main():
    ap = argparse.ArgumentParser(description="Is the embedding too small or too big?")
    ap.add_argument("--run", default="cnn3d_crop")
    ap.add_argument("--model", default="cnn3d")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    folds = pd.read_csv(CFG.folds_csv)
    model = ModelRegistry.get(args.model)(pretrained=False)
    model.load_state_dict(torch.load(
        CFG.runs_dir / args.run / f"fold{args.fold}" / "best_ema.pt",
        map_location=args.device))
    model.to(args.device).eval()
    view = VolumeView()

    @torch.no_grad()
    def embed(uids):
        out = []
        for i in range(0, len(uids), 64):
            x = torch.stack([view(np.load(CFG.crops_dir / f"{u}.npy").astype(np.float32))
                             for u in uids[i:i+64]]).to(args.device)
            f = model.features(x).mean(dim=(2, 3, 4))
            out.append(f.cpu().numpy())
        return np.concatenate(out)

    tr = folds[folds["fold"] != args.fold]
    va = folds[folds["fold"] == args.fold]
    E_tr, E_va = embed(list(tr["uid"])), embed(list(va["uid"]))
    y_tr, y_va = tr["is_pathologic"].values, va["is_pathologic"].values
    d = E_tr.shape[1]

    stds = E_tr.std(axis=0)
    dead = (stds < 1e-4 * stds.max()).sum()
    pca = PCA().fit(E_tr)
    cum = np.cumsum(pca.explained_variance_ratio_)
    r95, r99 = int(np.searchsorted(cum, 0.95) + 1), int(np.searchsorted(cum, 0.99) + 1)
    print(f"{args.run}: embedding dim {d}; dead dims {dead}; "
          f"effective rank: {r95} dims for 95% var, {r99} for 99%")

    print("linear probe on top-k PCA components (fold-{} val):".format(args.fold))
    Z_tr, Z_va = pca.transform(E_tr), pca.transform(E_va)
    for k in (2, 4, 8, 16, 32, 64, 128, d):
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(Z_tr[:, :k], y_tr)
        p = clf.predict_proba(Z_va[:, :k])[:, 1]
        print(f"  k={k:>3}: auroc {roc_auc_score(y_va, p):.4f}  "
              f"log_loss {log_loss(y_va, np.clip(p, 1e-6, 1-1e-6)):.4f}")


if __name__ == "__main__":
    main()
