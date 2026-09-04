import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.aligner import Aligner
from analysis.config import AnalysisConfig
from analysis.dataset_index import DatasetIndex
from analysis.feature_extractor import FeatureExtractor
from analysis.frame_cache import FrameCache
from analysis.header_stats import HeaderStats
from analysis.plotter import Plotter
from analysis.preprocessor import Preprocessor
from analysis.roi_calibrator import RoiCalibrator
from analysis.roi_masks import RoiMasks
from analysis.stats_analyzer import StatsAnalyzer

CFG = AnalysisConfig()
_PRE = None
_CACHE = None


def _init_worker():
    global _PRE, _CACHE
    _PRE = Preprocessor(CFG)
    _CACHE = FrameCache(CFG)


def _process_one(args):
    uid, path = args
    try:
        if _CACHE.has(uid):
            return uid, _CACHE.load_info(uid)
        frame, info = _PRE.process(path)
        if frame is None:
            return uid, {**info, "error": "empty mask"}
        _CACHE.save(uid, frame, info)
        return uid, info
    except Exception as e:
        return uid, {"error": repr(e), "qc_pass": False}


def main():
    index = DatasetIndex(CFG)
    ds_summary = index.summary()
    print("dataset:", json.dumps(ds_summary, indent=2))

    print("collecting header stats ...")
    hs = HeaderStats(CFG)
    header_df = hs.collect(index)
    header_df.to_csv(CFG.output_dir / "header_stats.csv", index=False)
    hdr_summary = hs.summarize(header_df)
    print("headers:", json.dumps(hdr_summary, indent=2))

    usable = index.usable
    jobs = [(uid, index.path_for(uid)) for uid in usable["uid"]]
    print(f"preprocessing {len(jobs)} scans on {CFG.n_workers} workers ...")
    qc_rows = []
    with ProcessPoolExecutor(CFG.n_workers, initializer=_init_worker) as pool:
        for i, (uid, info) in enumerate(pool.map(_process_one, jobs, chunksize=8)):
            qc_rows.append({"uid": uid, **info})
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)}")
    qc_df = pd.DataFrame(qc_rows).merge(usable[["uid", "is_pathologic"]], on="uid")
    qc_df.to_csv(CFG.output_dir / "qc_per_scan.csv", index=False)
    n_err = int(qc_df.get("error").notna().sum()) if "error" in qc_df else 0
    n_fail = int((~qc_df["qc_pass"].fillna(False)).sum())
    print(f"qc: {n_err} errors, {n_fail} scans outside QC bounds")

    cache = FrameCache(CFG)
    ok_uids = qc_df.loc[qc_df["qc_pass"].fillna(False), "uid"].tolist()

    def normalized(uid):
        f = cache.load(uid)
        m = f[f > 0].mean()
        return f / m if m > 0 else None

    def build_mean(frames_by_uid):
        acc = np.zeros(CFG.frame_shape, dtype=np.float64)
        n = 0
        for f in frames_by_uid.values():
            acc += f
            n += 1
        return (acc / n).astype(np.float32)

    print(f"building cohort mean image from {len(ok_uids)} QC-passed scans ...")
    frames = {}
    for uid in ok_uids:
        f = normalized(uid)
        if f is not None:
            frames[uid] = f
    mean_frame = build_mean(frames)

    aligner = Aligner(CFG)
    shifts = {uid: (0, 0, 0) for uid in frames}
    for it in range(2):
        print(f"registration pass {it + 1}: aligning frames to the mean ...")
        for uid in list(frames):
            frames[uid], s = aligner.align(frames[uid], mean_frame)
            shifts[uid] = tuple(a + b for a, b in zip(shifts[uid], s))
        mean_frame = build_mean(frames)
    shift_mag = {u: float(np.linalg.norm(s) * CFG.target_spacing_mm) for u, s in shifts.items()}
    pd.DataFrame(
        {"uid": list(shift_mag), "shift_mm": list(shift_mag.values())}
    ).to_csv(CFG.output_dir / "registration_shifts.csv", index=False)
    print(f"registration shifts (mm): median {np.median(list(shift_mag.values())):.1f}, "
          f"max {max(shift_mag.values()):.1f}")
    np.save(CFG.output_dir / "cohort_mean_frame.npy", mean_frame)

    print("calibrating ROIs on the mean image ...")
    roi_defs = RoiCalibrator(CFG).calibrate(mean_frame)
    print("roi definitions:", json.dumps({k: [str(x) for x in v] for k, v in roi_defs.items()},
                                         indent=2))
    roi_masks = RoiMasks(CFG, roi_defs)

    print("extracting ROI features ...")
    fx = FeatureExtractor(CFG, roi_masks)
    feat_rows = []
    for uid, frame in frames.items():
        feat_rows.append({"uid": uid, **fx.extract(frame)})
    feat_df = pd.DataFrame(feat_rows).merge(usable[["uid", "is_pathologic"]], on="uid")
    feat_df.to_csv(CFG.output_dir / "features.csv", index=False)

    print("running statistics ...")
    sa = StatsAnalyzer(CFG)
    per_feat = sa.per_feature(feat_df, FeatureExtractor.FEATURES)
    per_feat.to_csv(CFG.output_dir / "per_feature_statistics.csv", index=False)
    print(per_feat.to_string(index=False))
    cv = sa.multivariable_cv(feat_df, FeatureExtractor.FEATURES)
    cv_summary = {k: v for k, v in cv.items() if k not in ("oof", "y", "uids")}
    print("multivariable CV:", json.dumps(cv_summary, indent=2))

    print("plotting ...")
    pl = Plotter(CFG)
    figs = [
        pl.geometry(header_df),
        pl.qc(qc_df),
        pl.mean_image(mean_frame, roi_masks),
        pl.feature_distributions(feat_df, FeatureExtractor.FEATURES),
        pl.roc(per_feat, cv),
    ]
    for f in figs:
        print("  figure:", f)

    summary = {
        "dataset": ds_summary,
        "headers": hdr_summary,
        "qc": {"n_errors": n_err, "n_qc_fail": n_fail, "n_analyzed": len(ok_uids)},
        "multivariable_cv": cv_summary,
    }
    (CFG.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done")


if __name__ == "__main__":
    main()
