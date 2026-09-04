from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AnalysisConfig:
    repo_dir: Path = Path("/home/niho/scan-repo")
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
    n_workers: int = 16
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
        self.nifti_dir = self.data_dir / "niftis"
        self.labels_csv = self.data_dir / "train_labels_JNDlMjr.csv"
        self.output_dir = self.repo_dir / "output"
        self.figures_dir = self.output_dir / "figures"
        self.cache_dir = Path(
            "/tmp/claude-1007/-home-niho/3ce5ea2d-63cc-4584-86bd-7c8531b69db3/scratchpad/frame_cache"
        )
        for d in (self.output_dir, self.figures_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def frame_center(self):
        return tuple(s // 2 for s in self.frame_shape)
