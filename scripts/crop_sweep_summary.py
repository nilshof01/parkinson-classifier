import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.config import TrainConfig

CFG = TrainConfig()
GROUPS = {
    -5: ["cnn3d_m-5_s0", "cnn3d_m-5_s1"],
    -3: ["cnn3d_m-3_s0", "cnn3d_m-3_s1"],
    0: ["cnn3d_crop", "cnn3d_seed1"],
    2: ["cnn3d_m2", "cnn3d_m2_s1"],
    4: ["cnn3d_m4_s0", "cnn3d_m4_s1"],
    6: ["cnn3d_m6_s0", "cnn3d_m6_s1"],
}


def main():
    feats = pd.read_csv(CFG.repo_dir / "output" / "features.csv")[
        ["uid", "sbr_putamen_min"]]
    lines = ["crop-size sweep (margin vox/side; base box 44x38x26 @2mm; "
             "seed-noise floor ~0.004):"]
    for m, runs in sorted(GROUPS.items()):
        for r in runs:
            p = CFG.runs_dir / r / "oof_summary.json"
            if not p.exists():
                lines.append(f"  m={m:+d} {r}: MISSING")
                continue
            s = json.loads(p.read_text())
            oof = pd.read_csv(CFG.runs_dir / r / "oof.csv").merge(feats, on="uid")
            band = oof[(oof.is_pathologic == 1) & oof.sbr_putamen_min.between(1.5, 2.5)]
            lines.append(
                f"  m={m:+d} {r}: auroc {s['oof_auroc']:.4f}  "
                f"cal+clip {s['oof_log_loss_calibrated_clipped']:.4f}  "
                f"band missed {(band.pred < 0.5).sum()}/210")
    text = "\n".join(lines)
    print(text)
    (CFG.repo_dir / "output" / "crop_sweep_summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
