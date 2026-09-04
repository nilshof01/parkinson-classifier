import nibabel as nib
import numpy as np
import pandas as pd


class HeaderStats:
    """Geometry statistics from NIfTI headers only (no voxel data loaded)."""

    def __init__(self, config):
        self.config = config

    def collect(self, index):
        rows = []
        for uid in index.usable["uid"]:
            img = nib.load(index.path_for(uid))
            zooms = img.header.get_zooms()[:3]
            shape = img.shape[:3]
            rows.append(
                {
                    "uid": uid,
                    "dim_x": shape[0],
                    "dim_y": shape[1],
                    "dim_z": shape[2],
                    "spacing_x": float(zooms[0]),
                    "spacing_y": float(zooms[1]),
                    "spacing_z": float(zooms[2]),
                    "n_voxels": int(np.prod(shape)),
                    "fov_x_mm": shape[0] * float(zooms[0]),
                    "fov_y_mm": shape[1] * float(zooms[1]),
                    "fov_z_mm": shape[2] * float(zooms[2]),
                    "dtype": str(img.get_data_dtype()),
                    "isotropic": bool(np.allclose(zooms, zooms[0], atol=1e-3)),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def summarize(df):
        spacing_combo = (
            df[["spacing_x", "spacing_y", "spacing_z"]].round(3).astype(str).agg("x".join, axis=1)
        )
        dim_combo = df[["dim_x", "dim_y", "dim_z"]].astype(str).agg("x".join, axis=1)
        return {
            "n_scans": len(df),
            "n_unique_spacings": spacing_combo.nunique(),
            "top_spacings": spacing_combo.value_counts().head(8).to_dict(),
            "n_unique_dims": dim_combo.nunique(),
            "top_dims": dim_combo.value_counts().head(8).to_dict(),
            "n_isotropic": int(df["isotropic"].sum()),
            "dtypes": df["dtype"].value_counts().to_dict(),
            "spacing_x_min": float(df["spacing_x"].min()),
            "spacing_x_max": float(df["spacing_x"].max()),
        }
