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
# m4 (+4 vox/side) is the validated best crop; base crop = slice(26,70)/slice(48,86)/slice(36,62)
CROP_MARGIN = 4             # 0 = base crop (44×38×26), 4 = m4 (52×46×34)

# ── Test-time augmentation ────────────────────────────────────────────────────
TTA_FLIP = True             # always average L-R flipped prediction with original
TTA_ROTATIONS = []          # additional axial rotation angles in degrees, e.g. [-5, 5]
                            # each is averaged in along with the base prediction

# ── Calibration ───────────────────────────────────────────────────────────────
TEMPERATURE = 1.0           # divide logit by T before sigmoid (>1 = softer, <1 = sharper)
CLIP_LO = 0.02              # clip final probabilities to [CLIP_LO, CLIP_HI]
CLIP_HI = 0.98

# ── Fallback ──────────────────────────────────────────────────────────────────
# Prediction used when preprocessing fails (dataset prior ≈ 0.548 positive rate)
FALLBACK_P = 0.548
