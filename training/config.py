from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    repo_dir: Path = Path("/home/niho/scan-repo")
    n_folds: int = 5
    seed: int = 17
    input_size: int = 224

    prepared_dir: Path = field(init=False)
    crops_dir: Path = field(init=False)
    folds_csv: Path = field(init=False)
    runs_dir: Path = field(init=False)

    def __post_init__(self):
        self.prepared_dir = self.repo_dir / "prepared"
        self.crops_dir = self.prepared_dir / "crops"
        self.folds_csv = self.prepared_dir / "folds.csv"
        self.runs_dir = self.repo_dir / "output" / "runs"
