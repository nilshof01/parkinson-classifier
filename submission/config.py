# ── Submission configuration ──────────────────────────────────────────────────
# Edit this file to match the training settings of your checkpoints.
# All other files (main.py, src/) should not need changes.

# ── Model architecture ────────────────────────────────────────────────────────
# When mixing checkpoints from different training runs (different pool / aux_weight),
# define CHECKPOINTS as a list of dicts. Each entry must have "path" plus any
# architecture overrides; omitted keys fall back to the MODEL_* defaults below.
#
# Example (two runs with different pool types):
#   CHECKPOINTS = [
#       {"path": "models/final_fold0.pt",   "pool": "axisaware"},
#       {"path": "models/axaware_fold0.pt", "pool": "axisaware_split"},
#   ]
#
# Leave CHECKPOINTS = None to load all *.pt files in models/ with the same config.
CHECKPOINTS = [
    # confirmed from state dict keys:
    #   axaware_fold*.pt  → aux_pa keys        → pool="axisaware"       dim=1280
    #   diff_fold*.pt     → aux_pa_post/ant keys → pool="axisaware_split" dim=1792
    {"path": "models/axaware_fold0.pt", "pool": "axisaware",       "aux_weight": 0.1},
    {"path": "models/axaware_fold1.pt", "pool": "axisaware",       "aux_weight": 0.1},
    {"path": "models/axaware_fold2.pt", "pool": "axisaware",       "aux_weight": 0.1},
    {"path": "models/axaware_fold3.pt", "pool": "axisaware",       "aux_weight": 0.1},
    {"path": "models/axaware_fold4.pt", "pool": "axisaware",       "aux_weight": 0.1},
    {"path": "models/diff_fold0.pt",    "pool": "axisaware_split", "aux_weight": 0.1},
    {"path": "models/diff_fold1.pt",    "pool": "axisaware_split", "aux_weight": 0.1},
    {"path": "models/diff_fold2.pt",    "pool": "axisaware_split", "aux_weight": 0.1},
    {"path": "models/diff_fold3.pt",    "pool": "axisaware_split", "aux_weight": 0.1},
    {"path": "models/diff_fold4.pt",    "pool": "axisaware_split", "aux_weight": 0.1},
]

# Defaults used when CHECKPOINTS = None or a checkpoint omits a key.
MODEL_POOL       = "axisaware_split"
MODEL_WIDTH_MULT = 1.0
MODEL_DROPOUT    = 0.1
MODEL_AUX_WEIGHT = 0.1

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
CALIBRATION = "isotonic"
TEMPERATURE  = 1.0979       # only used when CALIBRATION = "temperature"
                            # T > 1 softens (spreads toward 0.5), T < 1 sharpens

CLIP_LO = 1e-4              # clip final probabilities to [CLIP_LO, CLIP_HI]
CLIP_HI = 0.9999

# ── Fallback ──────────────────────────────────────────────────────────────────
# Prediction used when preprocessing fails.
# Set to mean(OOF predictions) from scripts/calibrate.py output, not the
# class prior — calibrated mean is a better neutral prediction.
FALLBACK_P = 0.5309
