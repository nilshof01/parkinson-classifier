# DaT-SPECT Classification — Strategy Overview

## Cropping Pipeline (Prerequisite)

- Resample to 2.0 mm isotropic spacing, canonical orientation (RAS or LPS)
- Gaussian smooth (σ ≈ 4–6 mm) → Otsu threshold → largest connected component → fill holes → erode 1–2 voxels
- Find mid-sagittal plane by maximizing left-right symmetry of mask; rotate to axis-align
- Compute brain centroid; place fixed-size striatal bounding box at anatomical offset from centroid
- Output three crops per scan: full striatal region, left hemisphere, right hemisphere
- Fixed voxel dimensions (e.g., 64×64×40 for full crop)

---

## DL Approach A — Paradigm 1: Triaxial MIP Input

**Input construction (per scan):**
- Axial MIP → project striatal crop along Z axis → 2D image
- Coronal MIP → project along Y axis → 2D image
- Sagittal MIP → project along X axis → 2D image
- Resize each to model input size (224×224 for EfficientNet-B0)
- Stack as 3-channel image → shape (3, 224, 224)
- One forward pass per scan → one prediction

**Encoder:**
- EfficientNet-B0 (ImageNet pretrained), replace final classifier with 2-class or single-logit head
- Alternative: ConvNeXt-Tiny (ImageNet-22k pretrained) for better calibration out of the box

**Pros:** Simple, fast, uses ImageNet weights fully, mirrors clinical read pattern
**Cons:** Loses depth information (minor for compact striatum)

**Extension:** Run same model on full/left/right crops separately, average logits — captures asymmetry explicitly

---

## DL Approach B — Paradigm 3: 2.5D Adjacent-Slice Input

**Input construction (per scan):**
- Identify center slice of striatal crop (peak striatal intensity or geometric center)
- Take 3 adjacent axial slices: [center-k, center, center+k], k chosen to span striatal thickness (typically k=1 to 3 voxels at 2mm spacing)
- Resize each to 224×224
- Stack as 3-channel image → shape (3, 224, 224)
- One forward pass per scan → one prediction

**Encoder:**
- Same as Approach A — EfficientNet-B0 or ConvNeXt-Tiny, ImageNet pretrained

**Pros:** Preserves local depth relationships, single forward pass, uses ImageNet weights
**Cons:** Loses information from slices outside the 3-slice stack, dependent on reliable center-slice identification

**Extension:** Take 5 slices, modify first conv layer to accept 5 channels (loses some ImageNet transfer benefit)

---

## Comparing A vs B

- Train both on identical splits with identical augmentation
- Compare on out-of-fold predictions, both AUROC and log loss (log loss is your competition metric)
- Check calibration curves separately — one may rank slightly worse but calibrate better
- Consider ensembling both rather than choosing (they see different views of the same signal)

---

## Augmentation Recap (applies to both A and B)

**Pre-training sample augmentation (once, on disk):**
- Left-right flip (doubles dataset, label preserved)
- Optional: normal+normal hemisphere chimeras with seam blending

**Per-epoch augmentation (stochastic, during training):**
- Rotations ±5–10°, translations ±2–5 mm
- Global intensity scaling ×[0.85, 1.15]
- Smooth low-frequency multiplicative field (values [0.9, 1.1], correlation length ≥8 cm)
- Resolution jitter (Gaussian σ ∈ [0.5, 1.5 mm])
- Poisson-like noise, 5–15% of natural noise floor
- Apply each with probability 0.5–0.8, independently
- Composition order: geometric → resolution → multiplicative field → global scaling → noise

**Apply augmentation to the 3D crop *before* MIP/slice extraction**, so both A and B see augmented input consistently.

---

## Output Handling and Calibration

**Head output:**
- Single logit → sigmoid → probability of pathologic (binary classification)
- Or 2-class softmax → take softmax[1] as probability of pathologic

**Sigmoid output clipping (critical for log loss):**
- Clip final probabilities to [0.02, 0.98] before submission as a safety layer
- Rationale: log loss punishes confidently-wrong predictions catastrophically (log(0.001) on a wrong answer ≈ 6.9, while log(0.02) ≈ 3.9)
- Check competition rules — some evaluators auto-clip to [1e-15, 1-1e-15], some don't; clip explicitly regardless
- Tune clip bounds on validation set — [0.02, 0.98] is a reasonable default, but [0.05, 0.95] may be safer if your model is poorly calibrated

**Calibration (do this every time):**
- Reserve a held-out calibration set (or use cross-validation OOF predictions)
- Fit temperature scaling: single scalar T, divide logits by T before sigmoid, optimize T on validation set to minimize log loss
- Alternative: isotonic regression on OOF probabilities → non-parametric, more flexible, needs more data
- Apply calibration transform at inference before clipping

**Test-time augmentation (TTA):**
- Predict on original scan AND left-right flipped scan
- Average the two probabilities (in probability space, not logit space)
- Nearly free performance gain, especially helpful for calibration

---

## Statistical Baseline (parallel stream)

**Features to extract from cropped volumes:**
- Per-side Specific Binding Ratio (SBR) for caudate: (caudate counts − occipital counts) / occipital counts
- Per-side SBR for putamen
- Putamen/caudate ratio per side
- Left/right asymmetry index for caudate: (L−R)/(L+R)
- Left/right asymmetry index for putamen
- Anterior/posterior putamen ratio (posterior loss is early PD sign)
- Vertical/horizontal extent of thresholded striatal blob per side
- Max intensity per structure

**Classifier:**
- XGBoost or LightGBM with early stopping on validation log loss (not AUROC)
- Or logistic regression with L2 regularization — often better calibrated than tree ensembles
- Calibrate output the same way (temperature scaling or isotonic)

---

## Ensemble Strategy

**Components:**
1. DL Approach A (MIP-based EfficientNet)
2. DL Approach B (2.5D adjacent-slice EfficientNet)
3. Statistical model (XGBoost on SBRs and derived features)
4. Optional: second DL architecture (ConvNeXt-Tiny on MIPs) for diversity

**Aggregation:**
- Weighted average of probabilities (not logits) across components
- Weights fit on out-of-fold predictions by minimizing log loss (constrained: weights ≥ 0, sum to 1)
- Alternative: logistic regression stack (meta-learner) on OOF probabilities — more flexible, mild overfit risk with limited data
- Simple mean of probabilities is a strong baseline; only use weighted/stacked if it beats mean on OOF

**Order of operations at inference:**
1. Crop scan → produce full/left/right crops
2. Generate inputs for each DL model (MIPs for A, adjacent slices for B)
3. Compute statistical features for XGBoost
4. Run each model → get raw probabilities
5. Apply per-model calibration (temperature scaling)
6. Apply TTA per DL model (average with LR-flipped prediction)
7. Combine calibrated probabilities via ensemble weights
8. Clip final probability to [0.02, 0.98]
9. Write to submission.csv

**Diversity notes:**
- Statistical model and DL models make genuinely different errors — ensemble gain is real, not just noise reduction
- Two DL models trained the same way on similar inputs may be highly correlated — check OOF prediction correlation before adding to ensemble
- If Approach A and B correlate >0.95 on OOF, keep only one; ensemble diversity requires prediction independence
- Different random seeds of the same model give small but real gains — train 3 seeds of your best DL model and average

---

## Validation Protocol

- Stratified K-fold (5 or 10 folds) by label
- If site/scanner metadata available: use GroupKFold by site to estimate cross-site generalization
- Compute both log loss and AUROC per fold; log loss is the primary metric
- Final calibration and ensemble weights fit on OOF predictions across all folds
- Validate on un-augmented data always
- Use EMA of model weights or last-N-epoch average for model selection (per-epoch validation is noisy under strong augmentation)

---

## Priority Order for Implementation

1. Robust cropping pipeline (correct once, benefits everything downstream)
2. Statistical baseline with calibration → establish floor
3. DL Approach A (MIP) with basic augmentation and calibration → first DL number
4. Add TTA, tune augmentation bounds, add per-epoch training refinements
5. DL Approach B (2.5D) trained the same way
6. Ensemble the three, tune weights on OOF
7. Add second DL architecture only if ensemble hasn't plateaued
8. Final calibration and clipping on full pipeline output