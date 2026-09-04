# Model Roster — DaT-SPECT Ensemble

Status: 2026-09-03 · All metrics are out-of-fold (OOF) on the shared stratified 5-fold split,
n = 1,353 · "ll" = log loss after temperature calibration and [0.02, 0.98] clipping (lower = better)
· T = fitted temperature (closer to 1 = better natively calibrated)

**Current best: equal-weight average of all 8 members + one temperature → ll 0.2647
(nested-validated 0.2651), AUROC 0.955.** Spec: `output/ensemble/spec.json`.
No fitted weights — nested CV showed fitted weights overfit (0.2695 > 0.2651).

Reference points: predict-the-prevalence baseline ll 0.689 · first statistical baseline
(SBR features + logistic) ll 0.572 · best single model ll 0.2910.

---

## The 8 members

### 1. effb0_mip_ref40 — the reference
EfficientNet-B0 · triaxial MIP view · avg pooling · linear head · no chimera/zoom.
**AUROC 0.9438 · ll 0.2970 · T 1.60**
- Excels: the common presentation — bilateral, diffuse uptake reduction; best
  all-round CNN with the plain recipe; the control every ablation compares against.
- Weak: structurally blind to one-sided disease (average pooling dilutes localized
  evidence; scan 1xj7wwud → 0.13); misses the mild-abnormal band like all members.

### 2. effb0_mip_chimera25 — the robustness member
Same as ref40 + 25% of normal samples replaced by normal+normal hemisphere chimeras.
**AUROC 0.9405 · ll 0.3063 · T 1.40**
- Excels: small/underrepresented scanner cohorts (leave-one-protocol-out: 1.5 mm group
  AUROC 0.899 → 0.944); partially hemisphere-aware (1xj7wwud 0.34 vs ref's 0.13);
  better native calibration than ref.
- Weak: slightly worse overall solo; chimera brings no gain on the standard split.

### 3. effb0_mip_catavgmax_mlp — the best single model
Ref40 + concatenated avg+max pooling + MLP head (2560→256→1). The head lets the model
gate the noisy max statistics; without it (member 4) the same statistics hurt.
**AUROC 0.9464 · ll 0.2910 · T 1.13**
- Excels: best solo score; best native calibration of all members; reacts to focal
  evidence the average-pooled models suppress (1xj7wwud 0.47).
- Weak: noisier on the extreme-asymmetry subset (6/28 errors vs ref's 2/28, n small);
  656k head parameters on 1,082 training scans per fold.

### 4. effb0_mip_catavgmax — the max-statistic perspective
Same pooling as member 3 but with the plain linear head.
**AUROC 0.9276 · ll 0.3537 · T 2.92 (heavily overconfident)**
- Excels: nothing solo — kept purely as a diversity source: it reads the image through
  max statistics that no other member trusts the same way, and the ensemble mean
  measurably improves with it included.
- Weak: worst CNN solo; the linear head cannot gate max-channel noise (the lesson that
  validated member 3's design).

### 5. groupcv_chimera_blur_harmonization — the scanner-equalized member
Chimera 25% + resolution-harmonized crops (matched blur to 3.9 mm native spacing),
batch 32. Despite the name it trained on the standard stratified folds.
**AUROC 0.9388 · ll 0.3087 · T ~1.3**
- Excels: the only member that sees resolution-equalized data, so its errors are the
  least scanner-texture-dependent; robustness perspective for an unseen test-site mix.
- Weak: confounded config (three variables differ from ref40) — the clean
  harmonization-only run is still open; solo slightly behind ref.

### 6. effb0_slices25d — the true-slice perspective
Three real axial slices (center ±4 mm) as channels instead of projections.
**AUROC 0.900 · ll 0.395 · T 2.40**
- Excels: only CNN whose three channels are spatially consistent (true local in-plane
  structure, no projection artifacts); small but real contribution to the ensemble mean.
- Weak: weakest CNN by far — its 12 mm slab physically misses the top and bottom of the
  ~25–30 mm striatum. Do not invest further; MIP won this comparison decisively.

### 7. effb0_mipasym — the asymmetry specialist
Axial MIP + coronal MIP + a signed left-minus-mirrored-right difference channel
(symmetric anatomy cancels to zero; any one-sided difference stands at full contrast).
**AUROC 0.9418 · ll 0.3048 · T 1.35**
- Excels: THE solution for one-sided disease — 1xj7wwud 0.13 → **0.845**, the failure
  mode fixed at the input rather than in the architecture (an asymmetry blob survives
  average pooling because for symmetric scans the channel is ~0). Highest weight (0.27)
  whenever weights are fit.
- Weak: paid for the new channel by dropping the sagittal view (slightly worse solo);
  no effect on the mild-abnormal band — those scans are bilaterally symmetric,
  so a left/right comparison shows nothing (hypothesis tested and rejected).

### 8. xgb — the non-neural counterweight
XGBoost on 5,434 mean-pooled 4 mm cubes of the same aligned crop, shared folds.
**AUROC 0.898 · ll 0.452 (clipped) — weakest solo, most different**
- Excels: no pooling bottleneck — individual dark cubes drive decisions regardless of
  how healthy the rest looks (1xj7wwud ≈ 0.9); most decorrelated errors of any member
  (error overlap with CNNs ≈ 0.3 vs ≈ 0.5 CNN-to-CNN); CPU-only, seconds to train.
- Weak: clearly the weakest discriminator; no spatial-smoothness prior.

---

## Who covers which error type

| Error type | Covered by |
|---|---|
| Severe bilateral loss ("two dots") | everyone — trivially easy, ~0 errors |
| Diffuse mild-moderate bilateral loss | ref40, catavgmax_mlp (the bread and butter) |
| One-sided (unilateral) disease | **mipasym** (decisively), xgb, partially chimera/catavgmax_mlp |
| Unseen scanner / protocol shift | chimera, harmonized member (and the whole mean: leave-one-protocol-out AUROC 0.90–0.96) |
| Crop/scale outliers | zoom augmentation (available, not yet in any member's recipe) |
| **Mild bilateral abnormals that look normal** | **nobody — the open challenge (below)** |

## The final challenge: the mild band

58 ensemble errors sit in the mild band (worst-putamen SBR 1.5–2.5): 39 missed abnormals +
19 false alarms. They are 4.3% of scans but carry **20.3% of the total loss** (mean per-scan
loss 1.25 — confidently wrong). 55 of the 58 are in the model-independent overlap zone.

Size of the prize (on the current ensemble, uncalibrated ll 0.2638):
- Make the ensemble merely *honest* on them (predict 0.5): **ll → 0.240** (−0.024)
- Actually solve them (predict 0.75 on the right side): **ll → 0.222** (−0.041)

Tested and rejected for this band: left-right asymmetry input (mipasym — no change),
max-statistic pooling (no change), chimera, more resolution. Remaining hypotheses:
1. Within-side anterior/posterior gradient (earliest disease sign: the *rear* of the
   putamen fades first, on both sides) — an A/P difference channel, same trick as
   mipasym but front-vs-back.
2. A "hard-case detector": don't solve them, *recognize* them and output ~0.5 —
   captures more than half the prize without solving anything.
3. Some fraction is label noise / genuinely equivocal (the challenge itself says ~1 in 5
   exams are hard for experts) — irreducible; expect diminishing returns.

## Reproducing / extending

```bash
# refit ensemble after any new run completes (nested decision, equal-weight unless proven):
python3 scripts/ensemble.py effb0_mip_ref40 effb0_mip_chimera25 effb0_mip_catavgmax_mlp \
    effb0_mip_catavgmax efficientnet_b0_mip_groupcv_chimera_blur_harmonization \
    effb0_slices25d effb0_mipasym
# diagnostics: scripts/error_analysis.py <runs>, scripts/saliency.py <uid>, scripts/gallery.py
```
