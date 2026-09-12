import os
from dataclasses import dataclass, field
from pathlib import Path

# SCAN_REPO  — repository root (code + output). Defaults to the original Linux path.
# PREPARED_DIR — override just the prepared-data folder (crops, folds.csv).
#                Set this when the prepared data lives elsewhere (e.g. copied from Linux).
# Examples (Windows PowerShell):
#   $env:SCAN_REPO     = "C:\Users\nilsh\Projects\DaT Parkinson's Challenge\parkinson-classifier"
#   $env:PREPARED_DIR  = "D:\dat-prepared"        # if crops live on a different drive
_DEFAULT_REPO = Path(os.environ.get("SCAN_REPO", "/home/niho/scan-repo"))
_PREPARED_DIR_OVERRIDE = os.environ.get("PREPARED_DIR", None)


@dataclass
class TrainConfig:
    repo_dir: Path = _DEFAULT_REPO
    n_folds: int = 5
    seed: int = 17
    input_size: int = 224

    prepared_dir: Path = field(init=False)
    crops_dir: Path = field(init=False)
    folds_csv: Path = field(init=False)
    runs_dir: Path = field(init=False)

    def __post_init__(self):
        self.prepared_dir = (
            Path(_PREPARED_DIR_OVERRIDE)
            if _PREPARED_DIR_OVERRIDE
            else self.repo_dir / "prepared"
        )
        self.crops_dir = self.prepared_dir / "crops"
        self.folds_csv = self.prepared_dir / "folds.csv"
        self.runs_dir = self.repo_dir / "output" / "runs"
