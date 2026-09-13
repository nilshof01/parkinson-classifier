"""Pre-submit sanity check for submission.zip.

Usage (from repo root on the pod):
    python scripts/check_submission.py
    python scripts/check_submission.py --zip path/to/submission.zip
"""

import argparse
import importlib.util
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch


def check(condition, msg):
    if not condition:
        print(f"FAIL  {msg}")
        sys.exit(1)
    print(f"OK    {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="submission.zip")
    args = ap.parse_args()

    z = zipfile.ZipFile(args.zip)
    names = z.namelist()

    # ── zip structure ──────────────────────────────────────────────────────────
    check("main.py"                   in names, "main.py at zip root")
    check("config.py"                 in names, "config.py present")
    check("assets/calibrator.npz"    in names, "assets/calibrator.npz present")
    check("assets/mean_frame.npy"    in names, "assets/mean_frame.npy present")
    check("assets/template_crop.npy" in names, "assets/template_crop.npy present")
    check("src/cnn3d.py"             in names, "src/cnn3d.py present")
    check("src/preprocess.py"        in names, "src/preprocess.py present")

    pts = [n for n in names if n.startswith("models/") and n.endswith(".pt")]
    check(len(pts) > 0,  f"at least one .pt checkpoint ({len(pts)} found)")
    check(not any("asets" in n for n in names), "no stale asets/ typo folder")

    # ── config values ──────────────────────────────────────────────────────────
    cfg_src = z.read("config.py").decode()
    check("MODEL_POOL"        in cfg_src, "MODEL_POOL defined in config.py")
    check('"axisaware"'       in cfg_src, 'MODEL_POOL contains "axisaware"')
    check("CALIBRATION"       in cfg_src, "CALIBRATION defined")
    check('"isotonic"'        in cfg_src, 'CALIBRATION = "isotonic"')
    check("ThreadPoolExecutor" in z.read("main.py").decode(),
          "main.py has parallel preprocessing (ThreadPoolExecutor)")

    # ── architecture vs checkpoint — mirrors load_models exactly ──────────────
    with tempfile.TemporaryDirectory() as tmp:
        z.extractall(tmp)

        spec = importlib.util.spec_from_file_location("cfg", f"{tmp}/config.py")
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)

        spec2 = importlib.util.spec_from_file_location("cnn3d", f"{tmp}/src/cnn3d.py")
        mod = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(mod)

        checkpoint_cfgs = getattr(cfg, "CHECKPOINTS", None)
        if checkpoint_cfgs:
            entries = [(Path(tmp) / c["path"], c) for c in checkpoint_cfgs]
            check(True, f"CHECKPOINTS list with {len(entries)} entries")
        else:
            entries = [(p, {}) for p in sorted(Path(tmp).glob("models/*.pt"))]
            check(True, f"CHECKPOINTS=None, loading {len(entries)} .pt files with MODEL_POOL={cfg.MODEL_POOL}")

        for ckpt, overrides in entries:
            pool       = overrides.get("pool",       cfg.MODEL_POOL or "avg")
            width_mult = overrides.get("width_mult", cfg.MODEL_WIDTH_MULT)
            dropout    = overrides.get("dropout",    cfg.MODEL_DROPOUT)
            aux_weight = overrides.get("aux_weight", getattr(cfg, "MODEL_AUX_WEIGHT", 0.0))
            model      = mod.Cnn3d(pool=pool, width_mult=width_mult,
                                   dropout=dropout, aux_weight=aux_weight)
            model_keys = set(model.state_dict().keys())
            ckpt_keys  = set(torch.load(ckpt, map_location="cpu",
                                        weights_only=True).keys())
            missing    = model_keys - ckpt_keys
            unexpected = ckpt_keys  - model_keys
            check(not missing,    f"{ckpt.name} (pool={pool}): no missing keys")
            check(not unexpected, f"{ckpt.name} (pool={pool}): no unexpected keys")

        cal = np.load(f"{tmp}/assets/calibrator.npz")
        check("x" in cal and "y" in cal, "calibrator.npz has x and y arrays")
        check(len(cal["x"]) > 0, f"calibrator has {len(cal['x'])} breakpoints")

    print(f"\n{'='*40}")
    print(f"All checks passed — safe to submit.")
    print(f"  {len(pts)} checkpoint(s): {pts}")


if __name__ == "__main__":
    main()
