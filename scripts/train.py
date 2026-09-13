import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.augment3d import Augment3D
from training.chimera import ChimeraMixer
from training.config import TrainConfig
from training.dataset import DatScanDataset
from training.models.registry import ModelRegistry
from training.trainer import Trainer
from training.view_mip import TriaxialMip
from training.view_mip_asym import MipWithAsymmetry
from training.view_mip_apgrad import MipWithApGradient
from training.view_volume import VolumeView
from training.view_volume_asym import VolumeWithAsymmetry
from training.view_slices import AdjacentSlices

CFG = TrainConfig()


def parse_args():
    p = argparse.ArgumentParser(description="Train a model on prepared DaT crops")
    p.add_argument("--model", default="efficientnet_b0", choices=ModelRegistry.names())
    p.add_argument("--view", default="mip",
                   choices=["mip", "slices25d", "mipasym", "mipapgrad", "volume3d",
                            "volume3dasym", "fusion3d"])
    p.add_argument("--width-mult", type=float, default=1.0, help="cnn3d width multiplier")
    p.add_argument("--dropout3d", type=float, default=0.5, help="cnn3d head dropout")
    p.add_argument("--seed-offset", type=int, default=0,
                   help="offsets the training seed (for seed-ensembling); folds unchanged")
    p.add_argument("--fold", default="all", help="fold index 0..4 or 'all'")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--chimera-frac", type=float, default=0.0,
                   help="fraction of normal samples replaced by normal+normal chimeras")
    p.add_argument("--chimera-pos-frac", type=float, default=0.0,
                   help="fraction of abnormal samples replaced by worst-side "
                        "abnormal+abnormal chimeras")
    p.add_argument("--posterior-frac", type=float, default=0.0,
                   help="fraction of normal samples converted to bilateral posterior-gradient "
                        "synthetic abnormals (label flipped to 1)")
    p.add_argument("--posterior-uni-frac", type=float, default=0.0,
                   help="fraction of normal samples converted to unilateral posterior-gradient "
                        "synthetic abnormals (label flipped to 1)")
    p.add_argument("--posterior-sigma", type=float, default=5.0,
                   help="Gaussian sigma (voxels, ~2 mm/vox) for posterior reduction ramp")
    p.add_argument("--posterior-min-factor", type=float, default=0.30,
                   help="minimum attenuation factor at the posterior edge (severity ceiling)")
    p.add_argument("--posterior-max-factor", type=float, default=0.75,
                   help="maximum attenuation factor at the posterior edge (severity floor)")
    p.add_argument("--asym-jitter-frac", type=float, default=0.0,
                   help="fraction of normal samples to receive a random L-R scale jitter "
                        "within the healthy asymmetry range (label unchanged)")
    p.add_argument("--asym-jitter-max-ai", type=float, default=0.10,
                   help="upper bound on the asymmetry index applied by asym-jitter")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="BCE target smoothing, e.g. 0.05 -> targets 0.05/0.95")
    p.add_argument("--aux-weight", type=float, default=0.0,
                   help="weight for axisaware auxiliary heads (0=disabled, try 0.3)")
    p.add_argument("--bottleneck-dim", type=int, default=256,
                   help="hidden dim of the MLP bottleneck head for axisaware pool (default 256)")
    p.add_argument("--pool", default=None, choices=["avg", "max", "catavgmax", "axisaware", "axisaware_split", "axisaware_bins"],
                   help="global pooling of CNN backbones (default: model's own, avg); "
                        "catavgmax = concatenated avg+max")
    p.add_argument("--head", default="linear", choices=["linear", "mlp"],
                   help="classifier head: single linear (default) or pooled->256->1 MLP")
    p.add_argument("--norm", default="zscore", choices=["zscore", "percentile"],
                   help="input normalization for volume3d view: zscore (default) or "
                        "percentile (divide by Nth percentile of in-head voxels)")
    p.add_argument("--norm-percentile", type=float, default=99.0,
                   help="which percentile to use when --norm percentile (default 99)")
    p.add_argument("--loss", default="bce", choices=["bce", "focal"])
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--aug", default="all",
                   help="which augmentations to enable: 'all', 'none', or a comma-"
                        "separated subset of flip,geom,res,field,scale,noise")
    p.add_argument("--slice-k", type=int, default=2, help="slice offset for slices25d view")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-tta", action="store_true", help="disable flip TTA on val predictions")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--run-name", default=None)
    p.add_argument("--folds-csv", default=None,
                   help="alternative fold assignment (e.g. prepared/folds_group.csv "
                        "for leave-one-spacing-group-out CV)")
    p.add_argument("--crops-dir", default=None,
                   help="alternative crop directory (e.g. prepared/crops_h3.9 "
                        "for harmonized crops)")
    p.add_argument("--overwrite", action="store_true",
                   help="allow reusing a run directory that already contains results")
    p.add_argument("--pretrained-encoder", default=None,
                   help="path to encoder_best.pt from pretrain_ssl.py; loads weights "
                        "into the cnn3d encoder before fine-tuning (cnn3d only)")
    p.add_argument("--sbr-weight", action="store_true",
                   help="up-weight mild positive samples by 1/sbr_putamen_min so the "
                        "model pays more attention to hard borderline cases")
    p.add_argument("--aug-rot-deg", type=float, default=15.0,
                   help="max rotation angle (degrees) for geom augmentation (default 15)") ## sweep confirmed this choice
    p.add_argument("--aug-shift-vox", type=float, default=5.0,
                   help="max shift (voxels) for geom augmentation (default 5)") ## sweep confirmed this choice
    p.add_argument("--aug-aniso-range", type=float, nargs=2, default=[0.9, 1.1],
                   metavar=("LO", "HI"),
                   help="per-axis stretch range for aniso augmentation (default 0.9 1.1)")
    p.add_argument("--hard-chimera", action="store_true",
                   help="online hard mining: after every 5 epochs re-score normal training "
                        "samples and focus chimera mixing on the ones the model most confuses")
    return p.parse_args()


def build_view(args):
    if args.view == "mip":
        return TriaxialMip(CFG.input_size)
    if args.view == "mipasym":
        return MipWithAsymmetry(CFG.input_size)
    if args.view == "mipapgrad":
        return MipWithApGradient(CFG.input_size)
    if args.view in ("volume3d", "fusion3d"):
        return VolumeView(norm=getattr(args, "norm", "zscore"),
                          norm_percentile=getattr(args, "norm_percentile", 99.0))
    if args.view == "volume3dasym":
        return VolumeWithAsymmetry()
    return AdjacentSlices(CFG.input_size, k=args.slice_k)


AUG_NAMES = ("flip", "geom", "zoom", "aniso", "res", "field", "scale", "noise", "gamma")


AUG_DEFAULT_OFF = {"gamma": 0.4}  # p when explicitly enabled; 0 in Augment3D default


def build_augment(spec, rot_deg=15.0, shift_vox=5.0, aniso_range=(0.9, 1.1)):
    if spec == "none":
        return None
    chosen = set(AUG_NAMES) if spec == "all" else set(spec.split(","))
    unknown = chosen - set(AUG_NAMES)
    if unknown:
        raise SystemExit(f"unknown augmentations {sorted(unknown)}, valid: {AUG_NAMES}")
    kw = {f"p_{name}": 0.0 for name in set(AUG_NAMES) - chosen}
    for name, p in AUG_DEFAULT_OFF.items():
        if name in chosen:
            kw[f"p_{name}"] = p
    kw["rot_deg"] = rot_deg
    kw["shift_vox"] = shift_vox
    kw["aniso_range"] = tuple(aniso_range)
    return Augment3D(**kw)


def load_crops(folds, crops_dir=None):
    d = Path(crops_dir) if crops_dir else CFG.crops_dir
    return {uid: np.load(d / f"{uid}.npy") for uid in folds["uid"]}


def records(df):
    return list(zip(df["uid"], df["is_pathologic"].astype(np.float32)))


def worker_init(_):
    np.random.seed(torch.initial_seed() % 2**32)


def fit_temperature(y, probs):
    lo = logit(np.clip(probs, 1e-6, 1 - 1e-6))
    res = minimize_scalar(
        lambda t: log_loss(y, np.clip(expit(lo / t), 1e-6, 1 - 1e-6)),
        bounds=(0.25, 10.0), method="bounded",
    )
    return float(res.x)


def build_model(args, view):
    import inspect

    cls = ModelRegistry.get(args.model)
    kwargs = {"pretrained": not args.no_pretrained, "pool": args.pool, "head": args.head}
    extra = {"in_chans": getattr(view, "channels", None), "width_mult": args.width_mult,
             "dropout": args.dropout3d, "aux_weight": getattr(args, "aux_weight", 0.0),
             "bottleneck_dim": getattr(args, "bottleneck_dim", 256)}
    accepted = inspect.signature(cls.__init__).parameters
    for k, v in extra.items():
        if k in accepted and v is not None:
            kwargs[k] = v
    model = cls(**kwargs)
    if args.pretrained_encoder:
        enc_path = Path(args.pretrained_encoder)
        if not hasattr(model, "features"):
            raise SystemExit("--pretrained-encoder only works with cnn3d (has .features)")
        state = torch.load(enc_path, map_location="cpu")
        # strip classifier keys — keep only encoder (features.*)
        enc_state = {k: v for k, v in state.items() if k.startswith("features.")}
        missing, unexpected = model.load_state_dict(enc_state, strict=False)
        print(f"loaded pretrained encoder from {enc_path} "
              f"({len(enc_state)} tensors, {len(missing)} missing, {len(unexpected)} unexpected)")
    return model


def _make_hard_chimera_cb(normal_uids, crops, view, dataset, base_frac, device,
                          update_every=5, warmup=5):
    """Returns an epoch callback that re-scores normal training samples every
    `update_every` epochs and increases chimera probability for the ones the
    model currently predicts as pathological (hard negatives)."""
    @torch.no_grad()
    def callback(epoch, model):
        if epoch < warmup or epoch % update_every != 0:
            return
        model.eval()
        preds = {}
        for uid in normal_uids:
            x = view(crops[uid].astype(np.float32))[None].to(device)
            preds[uid] = float(torch.sigmoid(model(x)).item())
        vals = np.array([preds[u] for u in normal_uids])
        # Scale chimera probability: clearly normal (pred≈0) → base_frac/3,
        # hard negative (pred≈1) → base_frac*3. Centred at pred=0.5 → base_frac.
        lo, hi = base_frac / 3.0, base_frac * 3.0
        scaled = lo + (hi - lo) * vals
        dataset.chimera_weights = {u: float(p) for u, p in zip(normal_uids, scaled)}
        hard = sum(1 for p in vals if p > 0.3)
        print(f"  [hard-chimera] epoch {epoch}: {hard}/{len(normal_uids)} "
              f"normals with pred>0.3 (max {vals.max():.3f})")
    return callback


def run_fold(k, folds, crops, view, args, run_dir, frames=None):
    torch.manual_seed(CFG.seed + k + 1000 * args.seed_offset)
    np.random.seed(CFG.seed + k + 1000 * args.seed_offset)
    tr, va = folds[folds["fold"] != k], folds[folds["fold"] == k]
    chimera = None
    if args.chimera_frac > 0 and frames is None:
        normals = [crops[u] for u in tr.loc[tr["is_pathologic"] == 0.0, "uid"]]
        chimera = ChimeraMixer(normals)
    aug = build_augment(args.aug, rot_deg=args.aug_rot_deg, shift_vox=args.aug_shift_vox,
                        aniso_range=args.aug_aniso_range)
    frame_aug = None
    if frames is not None and aug is not None:
        frame_aug = build_augment(args.aug, rot_deg=args.aug_rot_deg, shift_vox=args.aug_shift_vox,
                                  aniso_range=args.aug_aniso_range)
        frame_aug.p_flip = 0.0  # the shared flip is drawn once in the dataset
        aug.p_flip = 0.0
    pos_chimera, sides = None, None
    if args.chimera_pos_frac > 0 and frames is None:
        from training.chimera_pos import PositiveChimeraMixer
        feats = pd.read_csv(CFG.repo_dir / "output" / "features.csv").set_index("uid")
        sides = {u: ("L" if feats.loc[u, "sbr_putamen_l"] < feats.loc[u, "sbr_putamen_r"]
                     else "R") if u in feats.index else "L"
                 for u in folds["uid"]}
        # Only use donors whose worse side is known from features.csv
        abn = tr.loc[(tr["is_pathologic"] == 1.0) & tr["uid"].isin(feats.index), "uid"]
        pos_chimera = PositiveChimeraMixer([(crops[u], sides[u]) for u in abn])
    posterior_reducer = None
    if (args.posterior_frac > 0 or args.posterior_uni_frac > 0) and frames is None:
        from training.augment_posterior import PosteriorPutamenReduction
        posterior_reducer = PosteriorPutamenReduction(
            sigma_y=args.posterior_sigma,
            min_factor=args.posterior_min_factor,
            max_factor=args.posterior_max_factor,
        )
    asym_jitter = None
    if args.asym_jitter_frac > 0 and frames is None:
        from training.augment_asymmetry import AsymmetryJitter
        asym_jitter = AsymmetryJitter(max_ai=args.asym_jitter_max_ai)
    sample_weights = None
    if getattr(args, "sbr_weight", False) and frames is None:
        feats_path = CFG.repo_dir / "output" / "features.csv"
        if feats_path.exists():
            sbr = pd.read_csv(feats_path).set_index("uid")["sbr_putamen_min"]
            sample_weights = {}
            for uid, label in records(tr):
                if label == 1.0 and uid in sbr.index:
                    sample_weights[uid] = float(1.0 / max(sbr[uid], 0.1))
                else:
                    sample_weights[uid] = 1.0
            # normalise so mean weight stays ~1
            mean_w = np.mean(list(sample_weights.values()))
            sample_weights = {u: w / mean_w for u, w in sample_weights.items()}
    train_ds = DatScanDataset(crops, records(tr), view, augment=aug,
                              chimera=chimera, chimera_frac=args.chimera_frac,
                              frames=frames, frame_augment=frame_aug,
                              pos_chimera=pos_chimera, pos_frac=args.chimera_pos_frac,
                              worse_sides=sides,
                              posterior_reducer=posterior_reducer,
                              posterior_frac=args.posterior_frac,
                              posterior_uni_frac=args.posterior_uni_frac,
                              asym_jitter=asym_jitter,
                              asym_jitter_frac=args.asym_jitter_frac,
                              sample_weights=sample_weights)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers,
                     pin_memory=True, worker_init_fn=worker_init)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(DatScanDataset(crops, records(va), view, frames=frames),
                            **loader_kw)

    model = build_model(args, view)
    trainer = Trainer(model, args.device, epochs=args.epochs, lr=args.lr,
                      weight_decay=args.weight_decay, ema_decay=args.ema_decay,
                      label_smoothing=args.label_smoothing,
                      loss=args.loss, focal_gamma=args.focal_gamma)
    epoch_cb = None
    if getattr(args, "hard_chimera", False) and args.chimera_frac > 0 and frames is None:
        normal_uids = list(tr.loc[tr["is_pathologic"] == 0.0, "uid"])
        epoch_cb = _make_hard_chimera_cb(
            normal_uids, crops, view, train_ds, args.chimera_frac, args.device)
    print(f"fold {k}: train {len(tr)}, val {len(va)}")
    best, history = trainer.fit(train_loader, val_loader, epoch_callback=epoch_cb)

    fold_dir = run_dir / f"fold{k}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(fold_dir / "metrics.csv", index=False)
    torch.save(best["state"], fold_dir / "best_ema.pt")

    model.load_state_dict(best["state"])
    probs, y = trainer.predict(val_loader)
    if not args.no_tta:
        flip_loader = DataLoader(
            DatScanDataset(crops, records(va), view, force_flip=True, frames=frames),
            **loader_kw
        )
        probs_f, _ = trainer.predict(flip_loader)
        probs = (probs + probs_f) / 2
    preds = pd.DataFrame({"uid": va["uid"].values, "is_pathologic": y, "pred": probs})
    preds.to_csv(fold_dir / "val_preds.csv", index=False)
    print(f"fold {k} best epoch {best['epoch']}: "
          f"val_ll {log_loss(y, np.clip(probs, 1e-6, 1 - 1e-6)):.4f}  "
          f"val_auc {roc_auc_score(y, probs):.4f} (with TTA: {not args.no_tta})")
    return preds


def main():
    args = parse_args()
    folds_path = Path(args.folds_csv or CFG.folds_csv)
    if folds_path.is_dir() or not folds_path.exists():
        raise SystemExit(
            f"--folds-csv must be a fold assignment CSV (got {folds_path}). "
            "For harmonized/alternative crops use --crops-dir <directory> instead; "
            f"the default folds file is {CFG.folds_csv}."
        )
    folds = pd.read_csv(folds_path)
    if args.crops_dir and not Path(args.crops_dir).is_dir():
        raise SystemExit(f"--crops-dir {args.crops_dir} does not exist — "
                         "run scripts/prepare_dataset.py (with --harmonize-to) first.")
    crops_d = Path(args.crops_dir) if args.crops_dir else CFG.crops_dir
    available = {uid for uid in folds["uid"] if (crops_d / f"{uid}.npy").exists()}
    if len(available) < len(folds):
        print(f"warning: {len(folds) - len(available)} scans in folds.csv have no crop "
              f"in {crops_d} — skipping them (prepared from a subset of the full cohort)")
        folds = folds[folds["uid"].isin(available)].reset_index(drop=True)
    view = build_view(args)
    crops = load_crops(folds, args.crops_dir)
    frames = None
    if args.view == "fusion3d":
        fdir = CFG.prepared_dir / "frames"
        frames = {uid: np.load(fdir / f"{uid}.npy") for uid in folds["uid"]}
    run_name = args.run_name or f"{args.model}_{args.view}"
    run_dir = CFG.runs_dir / run_name
    has_results = run_dir.exists() and any(run_dir.glob("fold*/val_preds.csv"))
    if has_results and not args.overwrite:
        raise SystemExit(
            f"run directory {run_dir} already contains results — "
            "choose a different --run-name or pass --overwrite to replace them."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    fold_ids = (
        sorted(folds["fold"].unique()) if args.fold == "all" else [int(args.fold)]
    )
    all_preds = [run_fold(k, folds, crops, view, args, run_dir, frames=frames)
                 for k in fold_ids]

    if args.fold == "all":
        oof = pd.concat(all_preds, ignore_index=True)
        oof.to_csv(run_dir / "oof.csv", index=False)
        y, p = oof["is_pathologic"].values, oof["pred"].values
        temp = fit_temperature(y, p)
        p_cal = expit(logit(np.clip(p, 1e-6, 1 - 1e-6)) / temp)
        summary = {
            "n": int(len(y)),
            "oof_auroc": float(roc_auc_score(y, p)),
            "oof_log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
            "temperature": temp,
            "oof_log_loss_calibrated": float(log_loss(y, np.clip(p_cal, 1e-6, 1 - 1e-6))),
            "oof_log_loss_calibrated_clipped": float(log_loss(y, np.clip(p_cal, 0.02, 0.98))),
        }
        (run_dir / "oof_summary.json").write_text(json.dumps(summary, indent=2))
        print("OOF:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
