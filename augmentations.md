# 1. Cropping Strategy

**Goal:** Robustly localize the striatal region (both caudates + both putamina) without relying on striatal signal itself, so it works even when uptake is nearly absent.

**Method:**
- Resample scan to fixed isotropic spacing (2.0 mm recommended) with canonical axis orientation (RAS or LPS)
- Smooth the volume with a 3D Gaussian filter, σ ≈ 4–6 mm, to suppress hotspots and noise
- Threshold the smoothed volume using Otsu or a low percentile (~20th) to separate head tissue from air
- Keep only the largest connected component to discard scatter/noise specks outside the head
- Fill interior holes in the mask (ventricles, low-signal regions inside brain)
- Erode the mask by 1–2 voxels to exclude scalp/skull edges
- Find mid-sagittal plane by maximizing left-right symmetry of the mask (or PCA on mask coordinates)
- Rotate the volume so mid-sagittal plane is axis-aligned — corrects head tilt
- Compute brain centroid from the mask
- Place a generous striatal bounding box using fixed anatomical offsets from the centroid (calibrate once on ~10 scans; approximate starting point: ~10–25 mm anterior, ~5–15 mm superior, ±10–30 mm lateral)
- Extract three crops: full striatal region, left hemisphere, right hemisphere

**Watch out for:**
- Never derive the crop location from striatal intensity — this fails on atypical PD where the striatum is nearly invisible
- Err on the side of a larger crop; missing part of the striatum is much worse than including some extra cortex
- Verify tilt correction on a sample — bad mid-sagittal detection propagates through everything downstream
- Log the brain mask volume and centroid position per scan as QC flags; outliers indicate segmentation failures
- If a scan's mask is much smaller/larger than the training distribution, flag for manual review rather than trust the crop
- Do the tilt correction even with a translation-invariant DL model — rotation invariance is harder for CNNs to learn than translation invariance

---

# 2. Voxel Alignment

**Goal:** Every scan enters the model in the same coordinate system, spacing, and orientation so that spatial features are comparable across scans.

**Method:**
- Read voxel spacing from scan metadata (DICOM PixelSpacing + SliceThickness, or NIfTI affine)
- Resample all scans to a fixed isotropic spacing (2.0 mm standard for DaT-SPECT)
- Use B-spline or linear interpolation for the intensity volume; nearest-neighbor for any mask
- Enforce canonical axis orientation (e.g., RAS): patient's right → +X, anterior → +Y, superior → +Z
- Reorient using the direction matrix from metadata — do not assume array axes correspond to anatomical axes
- After tilt correction (from cropping step), all scans share the same anatomical orientation
- Fix crop dimensions in voxels (e.g., 64×64×40 for full striatal region) so all outputs have identical shape
- Pad with zeros or edge values if a scan's brain is unusually small/positioned; never crop beyond the brain mask
- Preserve the original spacing and transform as metadata even after resampling — needed if you ever want to map predictions back

**Watch out for:**
- Voxel spacing distributions with sharp bins (as you noted) suggest multiple scanners/protocols — always resample rather than assuming spacing is consistent
- Don't confuse voxel index space with physical space; interpolation must happen in physical coordinates
- Check the sign of axes after reading — some scanners store data flipped relative to metadata
- Anisotropic input spacing (thicker slices than in-plane) is common; isotropic resampling is essential before any 3D convolution
- Verify a few scans visually after resampling and reorientation — a single sign flip can silently reverse left/right and destroy asymmetry information
- Use the same resampling pipeline at training and inference; mismatches here are a common source of deployment bugs

---

# 3. Augmentation Strategies

## Pre-training sample augmentation (applied once, expands dataset on disk)

**Purpose:** Create a fixed enlarged dataset before training starts. Useful for augmentations that are expensive to compute or where you want deterministic, inspectable training data.

**What to include:**
- **Left-right flip** — double the dataset by mirroring each scan; label unchanged (asymmetry direction is not diagnostically meaningful, only its presence)
- **Normal + normal hemisphere chimeras** (optional) — combine left hemisphere of one healthy patient with right hemisphere of another; label = normal
  - Register both to common template before combining
  - Blend across ±2–3 voxels at the midline to hide the seam
  - Match global intensity between source scans before combining
  - Only combine scans from the same scanner/site if possible
  - dont do for all and provide an argument where the user can choose in the train script the proportion of chimeras

**Watch out for:**
- Avoid pathologic+pathologic and normal+pathologic chimeras — they shift the class-conditional distribution in unpredictable ways and risk creating patterns that don't exist in real disease
- The seam can become a shortcut feature the model exploits — validate that chimeric augmentation actually improves performance on real held-out data, not just cross-validation with synthetic examples
- Do not do intensity-based synthesis of pathology (dimming a putamen artificially) — this teaches the model your synthesis artifacts

## Per-epoch augmentation (applied stochastically during training)

**Purpose:** Every training example is slightly different each epoch, forcing invariance to acquisition variance while preserving the diagnostic signal.

**Order of composition (physically motivated):** geometric → resolution → multiplicative field → global scaling → noise

**What to include:**
- **Small rotations** (±5–10°) and **translations** (±2–5 mm) — simulates patient positioning variance
- **Global intensity scaling**, factor ∈ [0.85, 1.15] — simulates dose/calibration variation; label-invariant because all discriminative features are ratios
- **Smooth low-frequency multiplicative field** — 3D Perlin noise or sum of coarse Gaussians, values bounded in [0.9, 1.1], spatial correlation length ≥ 4× striatum diameter (≥8–10 cm)
  - Captures scatter/attenuation-like regional bias without changing inter-hemispheric ratios
- **Resolution jitter** — convolve with Gaussian, σ ∈ [0.5, 1.5 mm]; simulates reconstruction filter differences
- **Additive noise** — Poisson-like (or Gaussian with σ ∝ √intensity), scaled to ~5–15% of the natural noise floor estimated from a background region

**Apply each transform independently with probability 0.5–0.8 per sample per epoch.**

**Watch out for:**
- **Never** apply regional intensity manipulation (dimming one putamen) — direct false-positive generator
- **Avoid elastic deformation** — warps the striatum, changes partial volume effects, can silently flip labels
- **Avoid aggressive Gaussian blur** — mimics disease-like uptake loss via increased partial volume effect
- **Do not use Mixup or CutMix** — blends two scans into one with no valid label
- **Do not use large rotations** (>15°) — outside the range of real acquisitions, model wastes capacity on unrealistic scenarios
- **Calibrate bounds empirically:** apply your augmentation pipeline to training scans, recompute SBRs and asymmetry indices, verify augmented feature distributions don't exceed the natural range of the dataset
- **If test-retest scans are available**, use their observed feature variance as your hard upper bound — augmentation should not exceed real intra-subject variability
- **Validate on un-augmented data** — augmented training makes per-epoch validation noisy; use EMA of weights or averaged validation over recent epochs for model selection
- **Fix random seeds during validation** so you can compare epochs meaningfully
- **Apply LR flip aggressively** — it's the single most valuable augmentation here and comes with zero label risk

---

## Harmonization blur (deterministic preprocessing, not stochastic augmentation)

**Purpose:** Remove scanner-resolution texture differences instead of only teaching invariance to them. Native voxel spacing spans 1.37–4.42 mm across centers; after resampling to a common 2 mm grid, finer-native scans still carry characteristically sharper texture that a model could use as a scanner fingerprint.

**Method (matched blurring):** applied identically to every scan at train *and* inference, per scan:
- σ_add = sqrt(t² − s²) / sqrt(12) mm, where s = the scan's native spacing and t = target spacing (default 3.9 mm, the coarsest major cohort); scans with s ≥ t are left untouched
- rationale: sampling with voxel width w adds variance w²/12, so this equalizes the sampling-blur component across cohorts
- implementation: `scripts/prepare_dataset.py --harmonize-to 3.9` writes a parallel crop set (`prepared/crops_h3.9`); train against it with `--crops-dir`

**Watch out for:**
- This stacks with (does not replace) the stochastic resolution jitter above
- It only equalizes sampling blur, not reconstruction-filter differences, which remain unknown per site
- Validate against the unharmonized baseline on the same folds — harmonization trades genuine detail in fine scans for cross-scanner consistency, and the net effect is an empirical question

---

**Cross-cutting notes:**
- Log all augmentation parameters per training example if you're debugging — knowing which combination produced a bad gradient step is useful
- Save a handful of augmented scans as images periodically during training and eyeball them — augmentations that look wrong to you probably look wrong to the model too
- The order matters: augmentation happens *after* the cropping pipeline (so the crop is stable), but rotation-based augmentation ideally happens on the pre-crop volume so the crop follows the rotated anatomy — practical compromise is to crop generously enough that in-crop rotations don't push the striatum out of frame