import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.config import TrainConfig
from training.models.registry import ModelRegistry

CFG = TrainConfig()
BUILD = CFG.repo_dir / "submission_build"

MEMBERS = [
    {"name": "cnn3d_seedavg", "view": "volume3d", "input": "crop", "tta": "flip",
     "runs": ["cnn3d_crop", "cnn3d_seed1", "cnn3d_do05", "cnn3d_lr5e4_e60",
              "cnn3d_gamma"]},
    {"name": "cnn3d_focal2", "view": "volume3d", "input": "crop", "tta": "flip",
     "runs": ["cnn3d_focal2"]},
    {"name": "fusion", "view": "volume3d", "input": "fusion", "tta": "flip",
     "runs": ["fusion3d_crop_frame"]},
    {"name": "r3d18", "view": "volume3d", "input": "crop", "tta": "flip",
     "runs": ["r3d18_crop"]},
    {"name": "slicevoter", "view": "volume3d", "input": "crop", "tta": "shiftflip",
     "runs": ["slicevoter_crop"]},
    {"name": "ref40", "view": "mip", "input": "crop", "tta": "shiftflip",
     "runs": ["effb0_mip_ref40"]},
    {"name": "catavgmax_mlp", "view": "mip", "input": "crop", "tta": "shiftflip",
     "runs": ["effb0_mip_catavgmax_mlp"]},
    {"name": "mipasym", "view": "mipasym", "input": "crop", "tta": "shiftflip",
     "runs": ["effb0_mipasym"]},
    {"name": "frame", "view": "mip", "input": "frame", "tta": "shiftflip",
     "runs": ["effb0_frame_mip"]},
]
DUMMIES = {"volume3d": (2, 1, 44, 38, 26), "mip": (2, 3, 224, 224),
           "mipasym": (2, 3, 224, 224)}


class FusionWrapper(torch.nn.Module):
    """Trace-friendly wrapper: forward(crop, frame) instead of forward(tuple)."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, crop, frame):
        return self.inner((crop, frame))


def load_model(run, fold):
    args = json.loads((CFG.runs_dir / run / "args.json").read_text())
    cls = ModelRegistry.get(args["model"])
    import inspect
    kw = {"pretrained": False}
    accepted = inspect.signature(cls.__init__).parameters
    for k in ("pool", "head"):
        if k in accepted and args.get(k):
            kw[k] = args[k]
    model = cls(**kw)
    model.load_state_dict(torch.load(CFG.runs_dir / run / f"fold{fold}" / "best_ema.pt",
                                     map_location="cpu"))
    return model.eval()


def main():
    if BUILD.exists():
        shutil.rmtree(BUILD)
    (BUILD / "assets").mkdir(parents=True)

    ens_spec = json.loads((CFG.repo_dir / "output" / "ensemble" / "spec.json").read_text())
    out_members = []
    for mem in MEMBERS:
        files = []
        mdir = BUILD / "weights" / mem["name"]
        mdir.mkdir(parents=True)
        if mem["input"] == "fusion":
            dummy = (torch.randn(2, 1, 44, 38, 26), torch.randn(2, 1, 48, 56, 48))
        else:
            dummy = (torch.randn(*DUMMIES[mem["view"]]),)
        for run in mem["runs"]:
            for fold in range(CFG.n_folds):
                model = load_model(run, fold)
                if mem["input"] == "fusion":
                    model = FusionWrapper(model).eval()
                with torch.no_grad():
                    traced = torch.jit.trace(model, dummy, check_trace=False)
                    a = model(*dummy); b = traced(*dummy)
                    assert torch.allclose(a, b, atol=1e-4), f"trace mismatch {run} f{fold}"
                rel = f"weights/{mem['name']}/{run}_f{fold}.pt"
                traced.save(str(BUILD / rel))
                files.append(rel)
        out_members.append({"name": mem["name"], "view": mem["view"],
                            "input": mem["input"], "tta": mem["tta"], "files": files})
        print(f"traced {mem['name']}: {len(files)} models")

    spec = {"members": out_members, "temperature": ens_spec["temperature"],
            "clip": ens_spec["clip"],
            "oof_reference": ens_spec["oof_log_loss_calibrated_clipped"]}
    (BUILD / "assets" / "spec.json").write_text(json.dumps(spec, indent=2))
    shutil.copy(CFG.repo_dir / "output" / "cohort_mean_frame.npy",
                BUILD / "assets" / "mean_frame.npy")
    import numpy as np
    import pandas as pd
    folds = pd.read_csv(CFG.folds_csv)
    normals = folds.loc[folds["is_pathologic"] == 0, "uid"]
    template = np.mean([np.load(CFG.crops_dir / f"{u}.npy").astype(np.float32)
                        for u in normals], axis=0)
    np.save(BUILD / "assets" / "template_crop.npy", template)
    shutil.copy(CFG.repo_dir / "scripts" / "submission_main.py", BUILD / "main.py")

    zip_path = CFG.repo_dir / "submission.zip"
    zip_path.unlink(missing_ok=True)
    subprocess.run(["zip", "-r", "-q", str(zip_path), "."], cwd=BUILD, check=True)
    size = zip_path.stat().st_size / 1e6
    print(f"built {zip_path} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
