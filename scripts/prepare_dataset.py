import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.aligner import Aligner
from analysis.config import AnalysisConfig
from analysis.crop_features import CropFeatures
from analysis.dataset_index import DatasetIndex
from analysis.preprocessor import Preprocessor
from training.config import TrainConfig

ACFG = AnalysisConfig()
TCFG = TrainConfig()
_WORKER = {}


def _init_worker(harmonize_to, crops_dir, frames_dir, affine_correct=False,
                 crop_margin=0):
    _WORKER["pre"] = Preprocessor(ACFG, affine_correct=affine_correct)
    _WORKER["crop_margin"] = crop_margin
    _WORKER["aligner"] = Aligner(ACFG)
    _WORKER["mean"] = np.load(ACFG.output_dir / "cohort_mean_frame.npy")
    _WORKER["harmonize_to"] = harmonize_to
    _WORKER["crops_dir"] = crops_dir
    _WORKER["frames_dir"] = frames_dir


def _harmonize(frame, native_spacing_mm):
    # matched blurring: bring every scan to the sampling blur of the coarsest
    # cohort. A voxel of width w adds variance w^2/12, so the missing blur is
    # sigma = sqrt(t^2 - s^2)/sqrt(12) mm (nothing to add for s >= t).
    target = _WORKER["harmonize_to"]
    var_mm2 = (target ** 2 - native_spacing_mm ** 2) / 12.0
    if var_mm2 <= 0:
        return frame
    sigma_vox = np.sqrt(var_mm2) / ACFG.target_spacing_mm
    return ndimage.gaussian_filter(frame, sigma_vox)


def _prepare_one(args):
    uid, path = args
    try:
        frame, info = _WORKER["pre"].process(path)
        if frame is None or not info["qc_pass"]:
            return uid, "qc_fail"
        m = frame[frame > 0].mean()
        if m <= 0:
            return uid, "empty"
        frame = frame / m
        for _ in range(2):
            frame, _s = _WORKER["aligner"].align(frame, _WORKER["mean"])
        if _WORKER["harmonize_to"] is not None:
            native = float(min(nib.load(path).header.get_zooms()[:3]))
            frame = _harmonize(frame, native)
        cf = CropFeatures()
        m = _WORKER.get("crop_margin", 0)
        sl = tuple(slice(s.start - m, s.stop + m) for s in (cf.X, cf.Y, cf.Z))
        crop = frame[sl].astype(np.float16)
        np.save(_WORKER["crops_dir"] / f"{uid}.npy", crop)
        if _WORKER["frames_dir"] is not None:
            # whole head frame, 2x mean-pooled to 48x56x48 (4 mm grid)
            f = frame.reshape(48, 2, 56, 2, 48, 2).mean(axis=(1, 3, 5))
            np.save(_WORKER["frames_dir"] / f"{uid}.npy", f.astype(np.float16))
        return uid, "ok"
    except Exception as e:
        return uid, f"error: {e!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harmonize-to", type=float, default=None,
                    help="target native spacing in mm for matched blurring "
                         "(e.g. 3.9 = blur finer scans to the coarsest cohort)")
    ap.add_argument("--save-frames", action="store_true",
                    help="also save the whole head frame, 2x pooled (prepared/frames)")
    ap.add_argument("--affine-correct", action="store_true",
                    help="apply header-recorded rotations during resampling; "
                         "writes crops to prepared/crops_rot")
    ap.add_argument("--crop-margin", type=int, default=0,
                    help="extra voxels per side on the striatal crop; "
                         "writes crops to prepared/crops_m<N>")
    args = ap.parse_args()
    frames_dir = None
    if args.save_frames:
        frames_dir = TCFG.prepared_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = TCFG.crops_dir
    if args.crop_margin:
        crops_dir = TCFG.prepared_dir / f"crops_m{args.crop_margin}"
        print(f"crop margin +{args.crop_margin} vox/side; crops -> {crops_dir}")
    if args.affine_correct:
        crops_dir = TCFG.prepared_dir / "crops_rot"
        print(f"affine rotation correction on; crops -> {crops_dir}")
    if args.harmonize_to is not None:
        crops_dir = TCFG.prepared_dir / f"crops_h{args.harmonize_to:g}"
        print(f"harmonizing to {args.harmonize_to} mm; crops -> {crops_dir}")
    crops_dir.mkdir(parents=True, exist_ok=True)
    index = DatasetIndex(ACFG)
    usable = index.usable.sort_values("uid").reset_index(drop=True)
    jobs = [(uid, index.path_for(uid)) for uid in usable["uid"]]
    print(f"preparing {len(jobs)} crops on {ACFG.n_workers} workers ...")
    status = {}
    with ProcessPoolExecutor(
        ACFG.n_workers, initializer=_init_worker,
        initargs=(args.harmonize_to, crops_dir, frames_dir, args.affine_correct,
                  args.crop_margin),
    ) as pool:
        for i, (uid, st) in enumerate(pool.map(_prepare_one, jobs, chunksize=8)):
            status[uid] = st
            if (i + 1) % 300 == 0:
                print(f"  {i + 1}/{len(jobs)}")

    usable["status"] = usable["uid"].map(status)
    excluded = usable[usable["status"] != "ok"]
    excluded.to_csv(TCFG.prepared_dir / "excluded.csv", index=False)
    kept = usable[usable["status"] == "ok"].reset_index(drop=True)
    print(f"kept {len(kept)}, excluded {len(excluded)} "
          f"({excluded['status'].value_counts().to_dict()})")

    if args.harmonize_to is not None or args.affine_correct or args.crop_margin:
        print("variant preparation: shared folds.csv left untouched")
        return
    skf = StratifiedKFold(TCFG.n_folds, shuffle=True, random_state=TCFG.seed)
    kept["fold"] = -1
    for k, (_, te) in enumerate(skf.split(kept, kept["is_pathologic"])):
        kept.loc[te, "fold"] = k
    kept[["uid", "is_pathologic", "fold"]].to_csv(TCFG.folds_csv, index=False)
    print(f"folds written to {TCFG.folds_csv}")
    print(kept.groupby("fold")["is_pathologic"].agg(["count", "mean"]).to_string())


if __name__ == "__main__":
    main()
