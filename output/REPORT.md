# Statistical Analysis of the DaT-SPECT Challenge Training Set

**Region-of-interest statistics for normal vs. abnormal dopamine transporter scans**
Analysis date: 2026-09-01 · Code: `analysis/` + `scripts/run_analysis.py` · Outputs: `output/`

---

## Introduction

The SFMN DaT scan challenge asks for models that predict the probability that a DaT-SPECT
examination is abnormal, scored by log loss on a withheld test set. Before any model
development, this analysis characterizes the training data: sample size, class balance,
acquisition heterogeneity, and — most importantly — whether the classical semi-quantitative
signal used by clinicians (dopamine transporter binding in putamen and caudate) is
recoverable from the provided volumes with a fully automatic pipeline.

**Research hypothesis (specified before feature extraction).** Semi-quantitative
region-of-interest features — specific binding ratios (SBR) of putamen and caudate,
their left/right asymmetry, and the putamen/caudate ratio — computed with automatically
placed, cohort-calibrated ROIs (a) differ significantly between normal and abnormal scans
(Mann-Whitney, Holm-corrected α = 0.05), and (b) discriminate the classes with an AUROC of
at least 0.80 for the strongest single feature (pre-specified: worst-side putamen SBR).

**Verdict up front: the hypothesis was tested; part (a) worked out, part (b) failed
narrowly.** All SBR features separate the groups at very high significance (smallest
Holm-corrected p ≈ 2×10⁻⁶⁶), with an ordering that matches clinical knowledge, but the
best single-feature AUROC reached 0.774 (95% bootstrap CI 0.749–0.798), below the 0.80
target. Details and interpretation follow.

Known limitations of this pipeline are flagged in Methodologies and revisited in the
Discussion; the headline ones are translation-only registration (no rotation or scale
correction) and fixed spherical ROIs.

## Methodologies

### Data

1,362 NIfTI volumes (`.nii.gz`, one 3D reconstruction per patient) with expert labels in
`train_labels.csv`. Every label row has a file on disk and vice versa (0 mismatches).
Per the analysis protocol, no image was inspected manually; all processing ran in scripts
and only aggregate statistics were reviewed.

### Preprocessing (per scan)

1. Reorient to RAS and resample to 2.0 mm isotropic spacing (linear interpolation).
2. Head mask: Gaussian smoothing (σ = 5 mm) → Otsu threshold → largest connected
   component → hole filling → 1-voxel erosion.
3. Anchor point: centroid of the mask restricted to the top 140 mm below the mask's
   superior extent. The restriction exists because the axial field of view varies from
   108 to 630 mm across scans, and included neck/shoulder tissue would otherwise drag
   the centroid away from the brain.
4. Extract a fixed 96×112×96-voxel frame (2 mm grid, RAS) centered on the anchor;
   voxels outside the head mask are zeroed.

QC rule: scans with a head-mask volume outside 0.8–4.5 L were excluded from feature
analysis (bounds chosen around the observed distribution; the mask includes scalp, so it
runs well above pure brain volume).

### Cohort registration and ROI calibration

- Each accepted frame was intensity-normalized (divided by its mean in-head value) and
  a cohort mean image was built. Every frame was then registered to the mean by FFT
  phase correlation (translation only, max ±28 mm), and the mean was rebuilt; two passes
  were run. Registration uses the whole head's intensity pattern, not striatal signal.
- ROI centers were calibrated **once, on the cohort mean image**, never per scan:
  the striatal blob per hemisphere (97th-percentile threshold inside a generous central
  search box) was split along its principal axis — anterior end = caudate head,
  posterior end = putamen. ROIs are spheres of radius 8 mm (caudate) and 10 mm (putamen).
  This follows the project constraint that per-scan ROI placement must never depend on
  that scan's striatal intensity, which can be nearly absent in abnormal exams.
- Occipital reference: a per-scan box in the posterior band of that scan's own head
  extent (central 60 mm in x, 4–18 mm from the posterior edge in y, striatum-level z).

### Features (per scan)

SBR per structure and side, `(ROI mean − occipital mean) / occipital mean`; worst-side
(minimum) SBR per structure; mean striatal SBR; putamen/caudate ratio per side and its
minimum; absolute left/right asymmetry index per structure `|L−R| / (|L|+|R|)`.

### Statistics

Per feature: Mann-Whitney U (two-sided) with Holm correction across the 12 features,
rank-biserial correlation as effect size, and AUROC with a 1,000-resample bootstrap 95%
CI. Multivariable check: L2 logistic regression on all 12 features, 5-fold stratified
cross-validation, evaluated on out-of-fold predictions with AUROC and log loss (the
competition metric), against a constant-prevalence baseline.

## Results

### Sample size and class balance

All 1,362 labeled scans were usable. 615 of 1,362 (45.2%) are normal and 747 of 1,362
(54.8%) abnormal — close to balanced, with a slight abnormal majority. 9 of 1,362 scans
(0.7%) failed head-mask QC (mask volumes 4.5–5.25 L, suggesting mask spill-over rather
than anatomy; 7 abnormal, 2 normal) and were excluded, leaving 1,353. One further scan
produced an empty occipital reference, so feature statistics use n = 1,352.

### Acquisition heterogeneity

The set is strongly multi-protocol, consistent with its ten-center origin
(figure: `figures/geometry.png`):

- 52 distinct voxel-spacing combinations; the largest groups are 2.46 mm isotropic
  (528 of 1,362; 38.8%), 3.895 mm (325 of 1,362; 23.9%), 2.3 mm (207; 15.2%) and
  2.398 mm (143; 10.5%). In-plane spacing spans 1.37–4.42 mm; 967 of 1,362 (71.0%)
  are exactly isotropic.
- 66 distinct matrix sizes; most common are 256³ (460 of 1,362; 33.8%), 128³
  (295; 21.7%) and 128×128×87 (143; 10.5%). Axial field of view spans 108–630 mm
  (median 294 mm), so some scans cover the head only, others reach the shoulders.
- Voxel data are unsigned 16-bit in 1,346 of 1,362 scans (98.8%), signed 16-bit in
  the remaining 16 (1.2%). Raw count scales differ; all features used here are
  count-scale-invariant ratios.

Any model pipeline must therefore resample and spatially normalize; a fixed-shape
assumption would fail immediately.

### Registration and QC

Head-mask volumes: median 2,730 mL, IQR 2,436–3,144 mL (distribution and accepted range:
`figures/qc_mask_volume.png`). Residual translation corrections after centroid anchoring
were small for most scans (median 0.0 mm, mean 2.3 mm) but material for a minority:
119 of 1,353 (8.8%) needed more than 10 mm, up to 26.1 mm. The calibrated ROIs sit
symmetric about the midline with the putamen lateral-posterior to the caudate head, as
expected anatomically (overlay: `figures/mean_image_rois.png`).

### Group differences per region of interest (n = 1,352)

Full table: `output/per_feature_statistics.csv`; distributions:
`figures/feature_distributions.png`. Key rows (medians, Holm-corrected p, AUROC with
95% CI):

| Feature | Normal median | Abnormal median | p (Holm) | AUROC (95% CI) |
|---|---|---|---|---|
| Putamen SBR, worst side | 2.22 | 1.25 | 2×10⁻⁶⁶ | 0.774 (0.749–0.798) |
| Putamen SBR, left | 2.43 | 1.44 | 3×10⁻⁶¹ | 0.763 (0.738–0.788) |
| Putamen SBR, right | 2.39 | 1.47 | 1×10⁻⁵⁸ | 0.757 (0.732–0.785) |
| Striatum mean SBR | 2.53 | 1.60 | 2×10⁻⁵⁶ | 0.752 (0.725–0.776) |
| Caudate SBR, worst side | 2.35 | 1.44 | 2×10⁻³¹ | 0.687 (0.657–0.716) |
| Putamen asymmetry | 0.069 | 0.111 | 7×10⁻¹⁵ | 0.626 (0.596–0.652) |
| Putamen/caudate ratio, min | 0.78 | 0.69 | 1×10⁻⁹ | 0.599 (0.569–0.629) |

All 12 features differ significantly after Holm correction (largest corrected
p = 1.7×10⁻⁴, for the right putamen/caudate ratio). The ordering is clinically coherent:
putamen SBR separates better than caudate SBR (putaminal binding is lost earlier in
neurodegenerative parkinsonism), abnormal scans are more asymmetric, and their
putamen/caudate ratio is lower.

### Multivariable check against the competition metric

Logistic regression on all 12 features (5-fold stratified CV, out-of-fold, n = 1,352):
AUROC 0.774, log loss 0.572 versus 0.689 for a constant-prevalence prediction
(ROC: `figures/roc.png`). The ROI feature set as a whole adds little beyond the single
best feature (0.774 vs 0.774), consistent with the strong correlation among SBR variants.

### Supplementary: gradient boosting on a determined striatal crop

To test whether the discrimination shortfall lies in the ROI featurization rather than
in the data or alignment, a deterministic 88×76×52 mm crop around both striata was taken
from the same aligned frames, mean-pooled to a 4 mm grid (5,434 voxel features,
normalized by each scan's in-head mean) and fed to XGBoost under the identical 5-fold
protocol (n = 1,353; script: `scripts/crop_model_check.py`; outputs:
`crop_xgb_result_pool2.json`, `crop_xgb_oof_pool2.csv`).

Out-of-fold AUROC 0.901, log loss 0.447 (0.435 with [0.02, 0.98] clipping) — a large
gain over the 12 SBR features (AUROC 0.774, log loss 0.572). A full-resolution 2 mm
variant (43,472 features) scored slightly worse (AUROC 0.894, log loss 0.478), so the
pooled crop is the better representation of the two; per-fold AUROC ranged 0.86–0.92 in
both variants, with fold 3 consistently weakest.

## Discussion

The hypothesis was tested and split cleanly. Part (a) worked out: every putamen- and
caudate-derived feature separates expert-labeled normal from abnormal scans at very high
significance, with effect directions and an importance ordering that match the clinical
literature. This suggests the automatic pipeline (head masking, cohort registration,
calibrated ROIs) is recovering genuine striatal binding signal, not artifact. Part (b)
failed narrowly: the pre-specified 0.80 AUROC target was not reached (0.774; the CI
upper bound of 0.798 sits just below it), so on this pipeline the ROI features alone
appear insufficient for competitive discrimination.

The gap to published SBR performance (typically AUROC > 0.9 with software-based
quantification) is largely attributable to the featurization, as the supplementary
experiment shows directly: on the same aligned frames and CV folds, XGBoost over a
determined striatal crop (pooled voxel intensities) reaches AUROC 0.901 / log loss 0.447,
versus 0.774 / 0.572 for the 12 hand-crafted SBR features. Small spherical ROIs and a
noisy occipital reference discard most of the crop's spatial information; a
gradient-boosted model over the crop recovers it. Residual misalignment likely explains
part of what remains — the pipeline corrects translation only, no head-tilt/rotation or
head-size scaling, and 8.8% of scans (119 of 1,353) needed > 10 mm translation.
This supports the planned deep-learning approaches (`training.md`), and updates the
floor to beat from ROI statistics (log loss 0.572 / AUROC 0.774) to the determined-crop
gradient-boosting baseline: **out-of-fold log loss 0.447 / AUROC 0.901**, versus
0.689 log loss for a prevalence-only prediction.

For downstream modeling, three dataset facts matter most: the classes are near-balanced
(747 of 1,362 abnormal; 54.8%), so a prevalence prior close to 0.5 is the right
calibration starting point; acquisition heterogeneity is severe (52 spacings, 66 matrix
sizes, 108–630 mm axial FOV) and must be normalized away; and QC failures are rare
(9 of 1,362; 0.7%) but real, so an inference pipeline needs a fallback prediction for
scans whose preprocessing fails.

### Rejected alternatives

- **Per-scan striatal ROI placement** (centering ROIs on each scan's own hottest striatal
  voxels): rejected by design — it fails exactly on severely abnormal scans with absent
  uptake and would bias SBRs upward for them.
- **Fixed anatomical offsets from the raw head-mask centroid without registration**
  (first iteration of this analysis): produced unstable placement (best AUROC 0.72,
  putamen/caudate ratio at chance) because the variable axial FOV shifts the centroid;
  superseded by the head-restricted anchor plus phase-correlation registration.
- **Occipital reference calibrated on the cohort mean image** (also tried first): the
  mean head is larger than many individual heads, so the box fell outside the head for
  197 of 1,353 scans (14.6%), silently discarding them; replaced by the per-scan
  posterior band.
- **Y-axis percentile split of the striatal blob for caudate/putamen**: put the two
  centers ~6 mm apart with heavily overlapping spheres; replaced by the principal-axis
  split, after which the putamen/caudate ratio behaved as clinically expected.
- **Elastic or affine registration and manually drawn ROIs**: out of scope for a
  scripted first-pass analysis; noted as follow-up below.

### Open questions

1. (Resolved 2026-09-02.) An independent run reporting AUROC 0.97 with XGBoost on a
   determined crop was traced to a faulty data split and withdrawn; the supplementary
   crop + XGBoost result here (AUROC 0.901 / log loss 0.447) stands as the project
   baseline.
2. How much of the remaining within-class variance is residual misalignment?
   Adding rotation (or full rigid/affine) registration and re-running this analysis
   would quantify it; the consistently weakest CV fold (0.86–0.87) may mark a
   subgroup of poorly aligned or otherwise atypical scans worth characterizing.
3. Do the SBR distributions differ by acquisition group (2.46 mm vs 3.895 mm cohorts,
   i.e., presumably by scanner/site)? If so, site-wise harmonization or site-stratified
   validation (GroupKFold) will matter for generalization claims.
4. The 9 QC-failed scans skew abnormal (7 of 9) — is unusual field of view or positioning
   itself weakly informative, and how should an inference pipeline predict for such scans?
5. Are the labels themselves graded by difficulty (the challenge motivation cites ~1 in 5
   equivocal exams)? A confidence-weighted error analysis on the eventual model could
   reveal whether the hard 20% dominates the log loss.
