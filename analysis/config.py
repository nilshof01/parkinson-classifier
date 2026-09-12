import os
from dataclasses import dataclass, field
from pathlib import Path

# Set SCAN_REPO to the repository root to override the default Linux path.
# Example Windows: $env:SCAN_REPO = "C:\Users\nilsh\Projects\DaT Parkinson's Challenge"
# Example Linux:   export SCAN_REPO=/home/niho/scan-repo
_DEFAULT_REPO = Path(os.environ.get("SCAN_REPO", "/home/niho/scan-repo"))

# The raw NIfTI folder may have a site-specific suffix (e.g. niftis_utCGpHE).
# Set NIFTI_DIR to the absolute path of that folder to override auto-detection.
_NIFTI_DIR_OVERRIDE = os.environ.get("NIFTI_DIR", None)

# On Windows, multiprocessing uses spawn (not fork), so fewer workers are better.
# Default: 16 on Linux, 4 on Windows. Override with N_WORKERS env var.
import platform as _platform
_DEFAULT_WORKERS = 4 if _platform.system() == "Windows" else 16
_N_WORKERS = int(os.environ.get("N_WORKERS", _DEFAULT_WORKERS))


@dataclass
class AnalysisConfig:
    repo_dir: Path = _DEFAULT_REPO
    target_spacing_mm: float = 2.0
    # common frame: RAS, isotropic, centered on the brain-mask centroid
    frame_shape: tuple = (96, 112, 96)  # x (L->R), y (P->A), z (I->S)
    smooth_sigma_mm: float = 5.0
    mask_erode_voxels: int = 1
    # QC bounds for the accepted head-mask volume; the smoothed Otsu mask includes
    # scalp, so it runs well above pure brain volume (observed ~2-3 L)
    mask_volume_ml_bounds: tuple = (800.0, 4500.0)
    # ROI sphere radii (voxels at 2 mm)
    caudate_radius_vox: int = 4
    putamen_radius_vox: int = 5
    n_workers: int = _N_WORKERS
    n_bootstrap: int = 1000
    cv_folds: int = 5
    seed: int = 17

    data_dir: Path = field(init=False)
    nifti_dir: Path = field(init=False)
    labels_csv: Path = field(init=False)
    output_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.repo_dir / "data"
        if _NIFTI_DIR_OVERRIDE:
            self.nifti_dir = Path(_NIFTI_DIR_OVERRIDE)
        else:
            # Auto-detect: prefer "niftis", then the first "niftis_*" subfolder
            plain = self.data_dir / "niftis"
            if plain.exists():
                self.nifti_dir = plain
            else:
                candidates = sorted(self.data_dir.glob("niftis*"))
                self.nifti_dir = candidates[0] if candidates else plain
        self.labels_csv = self.data_dir / "train_labels_JNDlMjr.csv"
        self.output_dir = self.repo_dir / "output"
        self.figures_dir = self.output_dir / "figures"
        self.cache_dir = self.repo_dir / ".cache" / "frames"
        for d in (self.output_dir, self.figures_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def frame_center(self):
        return tuple(s // 2 for s in self.frame_shape)
