"""Build submission.zip from the submission/ directory.

Usage (from repo root):
    python submission/build_zip.py

Steps:
  1. Checks that assets/ and models/ are populated
  2. Generates template_crop.npy from mean_frame.npy if missing
  3. Zips everything into submission.zip (main.py at root level as required)
"""

import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.preprocess import crop_slices
import config as cfg


def make_template_crop():
    mean_path = HERE / "assets" / "mean_frame.npy"
    tmpl_path = HERE / "assets" / "template_crop.npy"
    if tmpl_path.exists():
        return
    if not mean_path.exists():
        print("ERROR: assets/mean_frame.npy not found. Cannot generate template_crop.npy.")
        print("       Copy cohort_mean_frame.npy from output/ first.")
        sys.exit(1)
    mean = np.load(mean_path).astype(np.float32)
    sl = crop_slices(cfg.CROP_MARGIN)
    np.save(tmpl_path, mean[sl].astype(np.float32))
    print(f"generated assets/template_crop.npy  shape={mean[sl].shape}")


def build():
    # pre-flight checks
    mean_frame = HERE / "assets" / "mean_frame.npy"
    if not mean_frame.exists():
        print("ERROR: assets/mean_frame.npy missing — copy from output/cohort_mean_frame.npy")
        sys.exit(1)

    pts = list((HERE / "models").glob("*.pt"))
    if not pts:
        print("ERROR: no .pt files in models/ — drop your checkpoints there first")
        sys.exit(1)

    make_template_crop()

    zip_path = HERE.parent / "submission.zip"
    exclude = {".gitkeep", "build_zip.py", "README.txt", "__pycache__"}

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(HERE.rglob("*")):
            if f.is_dir() or f.name in exclude or "__pycache__" in str(f):
                continue
            arcname = f.relative_to(HERE)
            zf.write(f, arcname)
            print(f"  {arcname}")

    size_mb = zip_path.stat().st_size / 1e6
    print(f"\nwrote {zip_path}  ({size_mb:.1f} MB, {len(pts)} checkpoint(s))")
    print("verify: unzip -l submission.zip | head  # main.py should be at root")


if __name__ == "__main__":
    build()
