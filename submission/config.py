# ── Submission configuration ──────────────────────────────────────────────────
# Edit this file to match the training settings of your checkpoints.
# All other files (main.py, src/) should not need changes.

# ── Model architecture ────────────────────────────────────────────────────────
# Must exactly match the settings used when training the .pt checkpoints.
MODEL_POOL = "axisaware_split"  # None/"avg" / "axisaware" / "axisaware_split" / "axisaware_bins"
MODEL_WIDTH_MULT = 1.0          # width multiplier (1.0 = default)
MODEL_DROPOUT = 0.1             # used only to build the right architecture; dropout is off at eval()
MODEL_AUX_WEIGHT = 0.1          # >0 adds aux_pa_post/aux_pa_ant/aux_lr heads — must match training

# ── Input normalization ───────────────────────────────────────────────────────
NORM = "percentile"         # "zscore"  — subtract mean / std over non-zero voxels
                            # "percentile" — divide by Nth percentile (recommended)
NORM_PERCENTILE = 99.0      # only used when NORM = "percentile"

# ── Crop margin ───────────────────────────────────────────────────────────────
# Must match the --crop-margin used during prepare_dataset.py.
CROP_MARGIN = 4             # 0 = base crop (44x38x26), 4 = m4 (52x46x34)

# ── Test-time augmentation ────────────────────────────────────────────────────
TTA_FLIP = True             # average L-R flipped prediction with original
TTA_ROTATIONS = []          # additional axial rotation angles in degrees, e.g. [-5, 5]

# TTA weights — base orientation is most reliable; rotations are approximations.
# Run scripts/calibrate.py --fit-tta-weights to find optimal values from OOF.
TTA_BASE_WEIGHT = 2.0       # weight for the original orientation
TTA_FLIP_WEIGHT = 1.0       # weight for the L-R flip
TTA_ROT_WEIGHT  = 0.5       # weight per rotation (and its flip if TTA_FLIP=True)

# ── Calibration ───────────────────────────────────────────────────────────────
# "temperature" : divide mean logit by T before sigmoid. Fit T with
#                 scripts/calibrate.py --method temperature.
# "isotonic"    : apply piecewise-monotone mapping fitted on OOF predictions.
#                 Requires submission/assets/calibrator.npz (built by calibrate.py).
# "none"        : raw sigmoid of mean logit, no post-processing.
CALIBRATION = "temperature"
TEMPERATURE  = 1.0          # only used when CALIBRATION = "temperature"
                            # T > 1 softens (spreads toward 0.5), T < 1 sharpens

CLIP_LO = 0.02              # clip final probabilities to [CLIP_LO, CLIP_HI]
CLIP_HI = 0.98

# ── Fallback ──────────────────────────────────────────────────────────────────
# Prediction used when preprocessing fails.
# Set to mean(OOF predictions) from scripts/calibrate.py output, not the
# class prior — calibrated mean is a better neutral prediction.
FALLBACK_P = 0.548
