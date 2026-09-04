import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.config import AnalysisConfig
from training.config import TrainConfig

ACFG, TCFG = AnalysisConfig(), TrainConfig()


def export(uid, label, pred, out_dir):
    vol = np.load(TCFG.crops_dir / f"{uid}.npy").astype(np.float32)
    x, y, z = np.mgrid[: vol.shape[0], : vol.shape[1], : vol.shape[2]]
    lo, hi = np.percentile(vol[vol > 0], [55, 99.5])
    fig = go.Figure(go.Volume(
        x=x.ravel() * 2, y=y.ravel() * 2, z=z.ravel() * 2,  # mm
        value=vol.ravel(),
        isomin=float(lo), isomax=float(hi),
        opacity=0.12, surface_count=18, colorscale="Hot",
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))
    fig.update_layout(
        title=f"{uid} — expert label {'ABNORMAL' if label == 1 else 'NORMAL'}, "
              f"ensemble pred {pred:.2f} (axes in mm; x = left-right)",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    path = out_dir / f"{uid}.html"
    fig.write_html(path, include_plotlyjs=True)
    return path


def main():
    ap = argparse.ArgumentParser(description="Export scans as interactive 3D HTML")
    ap.add_argument("--uids", nargs="*", default=None)
    ap.add_argument("--errors", type=int, default=0,
                    help="export the N most costly FPs and FNs from the ensemble")
    args = ap.parse_args()
    out_dir = ACFG.output_dir / "viewer3d"
    out_dir.mkdir(exist_ok=True)

    e = pd.read_csv(ACFG.output_dir / "ensemble" / "oof_ensemble.csv").set_index("uid")
    uids = list(args.uids or [])
    if args.errors:
        y, p = e.is_pathologic, e.ensemble_pred.clip(1e-6, 1 - 1e-6)
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        for m in ((y == 0) & (p > 0.5)), ((y == 1) & (p < 0.5)):
            uids += list(ll[m].nlargest(args.errors).index)
    for uid in uids:
        r = e.loc[uid]
        print(export(uid, r.is_pathologic, r.ensemble_pred, out_dir))


if __name__ == "__main__":
    main()
