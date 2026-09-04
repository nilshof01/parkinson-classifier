import pandas as pd


class DatasetIndex:
    """Joins train_labels.csv with the files present on disk."""

    def __init__(self, config):
        self.config = config
        labels = pd.read_csv(config.labels_csv)
        labels["uid"] = labels["uid"].astype(str)
        on_disk = {p.name[: -len(".nii.gz")] for p in config.nifti_dir.glob("*.nii.gz")}
        labels["has_file"] = labels["uid"].isin(on_disk)
        self.labels = labels
        self.files_without_label = sorted(on_disk - set(labels["uid"]))

    @property
    def usable(self):
        return self.labels[self.labels["has_file"]].reset_index(drop=True)

    def path_for(self, uid):
        return self.config.nifti_dir / f"{uid}.nii.gz"

    def summary(self):
        df = self.labels
        usable = self.usable
        counts = usable["is_pathologic"].value_counts()
        return {
            "n_label_rows": len(df),
            "n_files_on_disk": int(df["has_file"].sum()) + len(self.files_without_label),
            "n_rows_without_file": int((~df["has_file"]).sum()),
            "n_files_without_label": len(self.files_without_label),
            "n_usable": len(usable),
            "n_normal": int(counts.get(0.0, 0)),
            "n_abnormal": int(counts.get(1.0, 0)),
            "n_label_missing_or_other": int(len(usable) - counts.get(0.0, 0) - counts.get(1.0, 0)),
            "prevalence_abnormal": float(usable["is_pathologic"].mean()),
        }
