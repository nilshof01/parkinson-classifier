# Status and Next Steps — DaT-SPECT Challenge

Date: 2026-09-05 · Deadline: 2026-09-16 · All metrics are out-of-fold (OOF) on the shared
stratified 5-fold split (n = 1,353) with nested validation for anything fitted.

## Current standing

**Best validated system: 14-member ensemble + shrunk logistic stack (C = 0.005, logit space)
+ temperature + clip → nested log loss 0.2253** (equal mean of the same pool: 0.2265; the
14th member is the m10-crop seed pair, adopted by nested audition).
Public leaderboard snapshot (2026-09-03): #1 = 0.2192 / AUROC 0.9709, #2 = 0.2286,
#3 = 0.2335. The OOF estimate sits between #1 and #2; the remaining gap to #1 is
predominantly discrimination (AUROC 0.971 vs ≈0.957), not calibration (ours measured
near-optimal: nested isotonic 0.2283 vs temperature 0.2295 on the earlier pool).

Pool: cnn3d seed-averages at two crop sizes (m0 ×5 runs, m4 ×4 runs), focal-γ2, fusion
(m0 + m4), r3d18 (m0 — m4 hurt it), whole-brain 3D, slicevoter (m0 + m4), four shift-TTA
2D EfficientNet variants (MIP / catavgmax-MLP / LR-asymmetry / whole-brain-MIP).

Progression: 0.447 (XGBoost baseline) → 0.297 (first 2D CNN) → 0.2592 (3D CNN) →
0.2375 → 0.2272 → **0.2255**.

## Frozen endgame recipe (for 3D native-resolution models)

- Crop m4 (+4 vox/side, 52×46×34 @ 2 mm) — sweep-validated (−5/−3 much worse, +6 flat
  within noise; m8/m10 extension running, verdict pending).
- Symmetric chimera (normal+normal 25% AND worst-side abnormal+abnormal 25%) —
  confirmed with 2 seeds (0.2431 / 0.2463 vs plain-m4 seeds 0.2462 / 0.2522).
- Zoom augmentation ±10% — OOF-neutral (0.2493 ≈ seed mean 0.2492), kept as scale-outlier
  insurance for the unseen test set.
- Standard six augmentations; lr 1e-3, 40 epochs, EMA; flip-TTA.
- Fixed-internal-resize models (r3d18) keep the original crop; 2D members keep
  shift+flip TTA.

## Next steps (ordered)

1. **First submission** (user action): rebuild zip with current pool + serialized stack,
   smoke test, then one full submission — measures the OOF↔leaderboard offset that every
   later decision needs. Risk check first: runtime repo's torch version vs our 2.10 traces.
2. **Extended crop sweep verdict — RESOLVED 2026-09-05**: m8 mean 0.2526, m10 mean 0.2499
   vs m4 mean 0.2492 → plateau from m4 through m10; m4 stays the standard. The m10 seed
   pair joined the ensemble as a diversity member (+0.0002 nested).
3. **Seed farm** (~1 GPU-day, mechanical): use `cnn3d_m4_drop01_lr3e4` as the new seed
   target (see hyperparameter sweep below); grow to 3–5 seeds once full OOF confirms.
4. **Capacity probe — RESOLVED 2026-09-12**: width ×2 hurt (fold-0 val_ll 0.1980 vs
   0.1848 baseline). More capacity overfits at this dataset size. Do not revisit.
5. **SSL pretraining** (~1 day, uncertain): masked-volume pretraining on the 1,353
   training scans only (test-set use prohibited), fine-tune the champion; the main
   untested representation lever and the most plausible source of the leader's AUROC edge.
   **STATUS: running** — see SSL pretraining section below.
6. **10-fold refit** of the surviving roster (~1–2 GPU-days): +12.5% training data per
   model; folds change, so this lands after experiments freeze.
7. **Full-data members**: one run per member on all 1,353 scans at the median best-epoch
   budget, added alongside (not replacing) fold models in the zip.
8. **Final calibration = isotonic** (nested +0.0012 over temperature) fitted on final OOF;
   final zip; smoke test; submission with buffer days.

Open user decisions: (a) timing of step 1; (b) the site-prior feature (labeling thresholds
differ by site, p ≈ 2×10⁻⁶, and scanner fingerprints identify sites — exploiting this is
rules-legal and likely worth 0.002–0.004 on this leaderboard, but it models annotator
culture rather than disease and would not transfer to new hospitals; currently unused).

## Rejected alternatives (validated negatives, do not revisit)

External data / PPMI (label-semantics mismatch, timeline risk) · DINOv2 member (failed
nested audition) · fitted ensemble weights and unregularized stacking (overfit; only
C ≤ 0.01 stacking survives) · pruning weak members (always neutral-to-worse) ·
middle-snap "clipping" to 0.5 (worsens at every width) · rotation-corrected resampling
(neutral-negative; augmentation already covers tilt) · template-correlation auto-fallback
(flags severe abnormals, 6% error rate vs 9.4% average — review tool only) · anatomical
ROI/SBR features as model inputs (superseded) · slices25d view (12 mm slab too thin) ·
smaller crops (sharply worse) · age-proxy/context hypotheses for the mild band (three
independent negative tests — the ~25-scan hard core is a site-threshold/label floor).

## Augmentation experiments (started 2026-09-04, continuing ~2026-09-11)

Three new augmentations implemented and activated via CLI args in `scripts/train.py`:

| Arg | What it does | Label effect |
|---|---|---|
| `--posterior-frac` | Bilateral half-Gaussian ramp along P→A axis, simulates early DaT loss | normal → abnormal |
| `--posterior-uni-frac` | Same but one hemisphere only | normal → abnormal |
| `--asym-jitter-frac` | Random L-R scale within healthy AI range (≤10%) | unchanged |

Implementation: `training/augment_posterior.py`, `training/augment_asymmetry.py`.
Dispatch in `training/dataset.py` is mutually exclusive (single random draw per sample).

### Running / queued experiments

**cnn3d_m4_post_bil_s0** — bilateral posterior augmentation (frac=0.20), chimera 0.25/0.25
- Fold 0 completed: val_ll **0.2023**, val_auc **0.9759** (with TTA) — best epoch 23
- STATUS: interrupted after fold 0; needs folds 1–4 to complete (rerun with `--overwrite`)
- Command:
  ```
  python scripts/train.py --model cnn3d --view volume3d \
    --chimera-frac 0.25 --chimera-pos-frac 0.25 --posterior-frac 0.20 \
    --crops-dir "C:\Users\nilsh\Projects\DaT Parkinson's Challenge\prepared\crops_m4" \
    --run-name cnn3d_m4_post_bil_s0 --workers 0 --epochs 30 --overwrite
  ```
Result: 
OOF: {
  "n": 1353,
  "oof_auroc": 0.9575724174419117,
  "oof_log_loss": 0.25683489441871643,
  "temperature": 0.9554422470512982,
  "oof_log_loss_calibrated": 0.2565964460372925,
  "oof_log_loss_calibrated_clipped": 0.26223424077033997
}

**cnn3d_m4_baseline_s0** — same recipe without posterior augmentation (control for local comparison)
- Command:
  ```
  python scripts/train.py --model cnn3d --view volume3d \
    --chimera-frac 0.25 --chimera-pos-frac 0.25 \
    --crops-dir "C:\Users\nilsh\Projects\DaT Parkinson's Challenge\prepared\crops_m4" \
    --run-name cnn3d_m4_baseline_s0 --workers 0 --epochs 30
  ```
OOF: {
  "n": 1353,
  "oof_auroc": 0.9642894934085798,
  "oof_log_loss": 0.23491588234901428,
  "temperature": 0.9933367747917264,
  "oof_log_loss_calibrated": 0.23491114377975464,
  "oof_log_loss_calibrated_clipped": 0.2416343092918396
}

**hard chimera** - look if targeted chimera to hard cases helps.

- Command:
  ```
  python scripts/train.py --model cnn3d --view volume3d \
    --chimera-frac 0.25 --chimera-pos-frac 0.25 \
    --crops-dir "C:\Users\nilsh\Projects\DaT Parkinson's Challenge\prepared\crops_m4" \
    --run-name cnn3d_m4_baseline_s0 --workers 0 --epochs 30 --hard-chimera
  ```

  OOF: {
  "n": 1353,
  "oof_auroc": 0.9624685860411799,
  "oof_log_loss": 0.24412941932678223,
  "temperature": 1.0730115648219136,
  "oof_log_loss_calibrated": 0.2435745745897293,
  "oof_log_loss_calibrated_clipped": 0.24959823489189148
}


**cnn3d_m4_post_uni_s0** — unilateral posterior augmentation (frac=0.15)
- STATUS: running on pod (2026-09-12)
- Command:
  ```
  python scripts/train.py --model cnn3d --view volume3d \
    --chimera-frac 0.25 --chimera-pos-frac 0.25 --posterior-uni-frac 0.15 \
    --crops-dir $SCAN_REPO/prepared/crops_m4 \
    --run-name cnn3d_m4_post_uni_s0 --epochs 30
  ```
OOF: {
  "n": 1353,
  "oof_auroc": 0.954955689784401,
  "oof_log_loss": 0.2668159306049347,
  "temperature": 0.9153468674809137,
  "oof_log_loss_calibrated": 0.26592326164245605,
  "oof_log_loss_calibrated_clipped": 0.27128398418426514
}
**effb0_mipasym_asymjitter_s0** — asymmetry jitter on the mipasym 2D model
- STATUS: completed
- OOF ll (cal+clip) **0.3294**, AUROC 0.9302 — worse than base effb0_mipasym (0.3048);
  asymmetry jitter did not help this model. Validated negative for this architecture.
- Command:
  ```
  python scripts/train.py --model efficientnet_b0 --view mipasym \
    --chimera-frac 0.25 --asym-jitter-frac 0.20 \
    --run-name effb0_mipasym_asymjitter_s0 --epochs 30
  ```
- OOF: auroc 0.9302 · ll 0.3435 · cal+clip ll 0.3294 · T 1.393
### Remaining research hypotheses (from MODELS.md)
1. **A/P difference channel** — same trick as mipasym but front-vs-back within each side;
   explicit anterior/posterior gradient input channel for the 3D or 2D model.
   Targets the mild bilateral band directly. See A/P channel section below.
2. **Hard-case detector / abstainer** — train a second head or separate model to recognise
   the mild-band cases and output ~0.5 rather than a confident wrong answer.
3. **Seed farm** — baseline (cnn3d_m4_baseline_s0, OOF 0.2416) is now the reference;
   grow to 3–5 seeds once sweep confirms optimal config; each seed worth ~0.001.

## cnn3d hyperparameter sweep — RESOLVED 2026-09-12

Fold-0 sweep over LR, pooling/head, weight decay, dropout, width multiplier.
All runs: cnn3d, volume3d view, chimera 0.25/0.25, crops_m4, 40 epochs.
Metric: best val_log_loss on fold 0 (no TTA).

| Run | val_ll (fold 0) | best epoch | verdict |
|---|---|---|---|
| sw_drop_01 (dropout=0.1) | **0.1783** | 26 | **new default — clear winner** |
| sw_drop_05 (dropout=0.5) | 0.1823 | 28 | better than default 0.3 |
| sw_lr_3e4 (lr=3e-4) | 0.1843 | 23 | marginal improvement over 1e-3 |
| sw_drop_03 / sw_lr_1e3 / sw_pool_avg_lin / sw_wd_1e4 / sw_width_1x | 0.1848 | 26 | current defaults |
| sw_wd_1e5 | 0.1850 | 31 | neutral |
| sw_wd_1e3 | 0.1852 | 26 | neutral |
| sw_lr_3e3 (lr=3e-3) | 0.1929 | 25 | worse |
| sw_pool_avg_mlp | 0.1946 | 30 | worse |
| sw_width_2x (width_mult=2.0) | 0.1980 | 39 | worse — overfits |
| sw_pool_cat_mlp (catavgmax+mlp) | 0.2485 | 38 | **much worse — do not use on cnn3d** |

**Key findings:**
- Dropout 0.1 is a meaningful improvement (−0.0065 on fold 0). Model was over-regularised at 0.3.
- catavgmax pooling destroys cnn3d (0.2485) — opposite effect from effb0 where it was best.
  avg pooling is correct for cnn3d.
- LR 3e-4 marginally beats 1e-3. Combined with dropout 0.1 for the next seed.
- Weight decay is insensitive across 1e-5 to 1e-3. Keep 1e-4.
- Width ×2 hurts — dataset too small for the extra capacity.

**Updated frozen recipe for cnn3d:** dropout=0.1, lr=3e-4, pool=avg, head=linear, wd=1e-4.

**Next run (all 5 folds):**
```bash
python scripts/train.py --model cnn3d --view volume3d \
    --chimera-frac 0.25 --chimera-pos-frac 0.25 \
    --crops-dir $SCAN_REPO/prepared/crops_m4 \
    --dropout3d 0.1 --lr 3e-4 \
    --run-name cnn3d_m4_drop01_lr3e4_s0 --epochs 40
```

## SSL pretraining experiment (started 2026-09-12)

**Hypothesis:** the cnn3d trains from random weights on ~1,082 scans per fold. A masked-volume
pretraining stage forces the encoder to learn 3D anatomy (striatum location, uptake shape,
left-right symmetry) before seeing any labels. This may close the AUROC gap to leaderboard
#1 (0.971 vs 0.957), which is a discrimination gap most consistent with better representations.

**Implementation:** `scripts/pretrain_ssl.py` + `training/models/decoder3d.py`.
Randomly masks 40% of 6×6×6 voxel patches (12mm cubes — ~striatum scale) and trains
encoder + decoder to reconstruct masked regions. MSE loss on masked voxels only.
Fine-tune with `--pretrained-encoder` flag in `train.py`.

**Status: running on pod (2026-09-12)**

Stage 1 — pretraining:
```bash
python scripts/pretrain_ssl.py \
    --crops-dir $SCAN_REPO/prepared/crops_m4 \
    --epochs 80 --run-name ssl_m4
```

Stage 2 — fine-tuning:
```bash
python scripts/train.py --model cnn3d --view volume3d \
    --chimera-frac 0.25 --chimera-pos-frac 0.25 \
    --crops-dir $SCAN_REPO/prepared/crops_m4 \
    --pretrained-encoder $SCAN_REPO/output/runs/ssl_m4/encoder_best.pt \
    --run-name cnn3d_ssl_m4_s0
```

**How to interpret:** compare `cnn3d_ssl_m4_s0` OOF ll against `cnn3d_m4_baseline_s0`
(0.2416). If lower → pretraining helps, grow to 3+ seeds. If neutral/worse → dataset
is too small for SSL to find useful structure; validated negative.


OOF: {
  "n": 1353,
  "oof_auroc": 0.9597802125126758,
  "oof_log_loss": 0.2488415539264679,
  "temperature": 1.0624369692973814,
  "oof_log_loss_calibrated": 0.2484322190284729,
  "oof_log_loss_calibrated_clipped": 0.254264771938324
}

## Input normalization — RESOLVED 2026-09-13

**Hypothesis:** per-scan z-score (current default) removes inter-subject absolute intensity
variation. 99th-percentile scaling (divide by P99 of in-head voxels) is more robust to
injection artifacts and preserves inter-subject relative intensity differences, giving the
model more signal to distinguish mild PD from normal.

**Result: percentile normalization is significantly better.**

| Run | norm | OOF AUROC | OOF ll | cal+clip ll |
|---|---|---|---|---|
| cnn3d_m4_baseline_s0 | zscore | 0.9643 | 0.2349 | 0.2416 |
| cnn3d_m4_norm_pct99_s0 | percentile (P99) | **0.9659** | **0.2320** | **0.2383** |

Delta: AUROC +0.0016, ll −0.0029, cal+clip ll −0.0033. Consistent improvement across all
three metrics. The absolute binding level (striatum vs background) carries genuine signal
that z-score was discarding.

**Updated frozen recipe:** `--norm percentile --norm-percentile 99` added to all future
cnn3d runs. The `--norm` flag defaults to `zscore` for backward compatibility; pass
`--norm percentile` explicitly.

## Augmentation rotation/shift sweep — RESOLVED 2026-09-13

Fold-0 sweep over rotation, shift, and gamma augmentation. Base config:
cnn3d, volume3d, chimera 0.25/0.25, crops_m4, 30 epochs.

| Run | rot (°) | shift (vox) | gamma | val_ll (fold 0) | best epoch |
|---|---|---|---|---|---|
| aug_gamma_only | 15 | 5 | on | **0.1814** | 24 |
| aug_rot15_shift5 | 15 | 5 | on | **0.1814** | 24 |
| aug_orig_rot8_shift2 | 8 | 2 | on | 0.1841 | 24 |
| aug_nogamma | 15 | 5 | off | 0.1851 | 29 |
| aug_rot25_shift10 | 25 | 10 | on | 0.1888 | 29 |
| aug_rot20_gamma | 20 | 8 | on | 0.1893 | 29 |
| aug_rot20_shift8 | 20 | 8 | on | 0.1893 | 29 |
| aug_rot20_nogamma | 20 | 8 | off | 0.1951 | 29 |

**Key findings:**
- rot15/shift5 is the sweet spot — more aggressive rotation actively hurts (too much
  anatomy distortion; the uptake pattern is small and fragile).
- Gamma augmentation helps by ~0.004 and should stay on (default `--aug all`).
- Original defaults (rot8/shift2) were suboptimal by ~0.003; new defaults confirmed.
- The train/val gap in folds 2–4 (train ~0.18 vs val ~0.27) was unaffected — it is
  caused by distribution shift (folds 2–4 val sets contain more mild/bilateral positives),
  not by augmentation strength.

**Updated frozen recipe for aug:** `--aug-rot-deg 15 --aug-shift-vox 5` (gamma on, default).

**Note:** aug_gamma_only and aug_rot15_shift5 are identical configs (both use default
`--aug all`); tied score confirms reproducibility.

## A/P difference channel experiment (not yet started)

**Hypothesis:** the earliest DaT loss sign is the posterior putamen fading before the
anterior. The mipasym view already exploits the L-R asymmetry signal by adding a
left-minus-mirrored-right channel. The same trick applied front-vs-back would make
the A/P gradient explicit — a bilateral signal invisible to L-R difference.

**Design:** new view `view_mip_apgrad.py` mirroring `view_mip_asym.py` but flipping
along the Y axis (P→A) instead of X (L→R). The three channels would be:
- Axial MIP (same as base)
- Coronal MIP (same as base)  
- Posterior-minus-anterior signed difference (new — highlights rear-fading putamen)

**Why it might work:** the 39 missed abnormals in the mild band are bilaterally
symmetric (mipasym sees nothing), but the posterior-gradient augmentation
(`--posterior-frac`) showed this signal exists — the model just can't see it
explicitly in the standard MIP. Making it an input channel removes the need for
the model to discover it from pooled projections.

**Command (once implemented):**
```bash
python scripts/train.py --model efficientnet_b0 --view mipapgrad \
    --chimera-frac 0.25 \
    --run-name effb0_mipapgrad_s0 --epochs 40
```

**Note:** requires implementing `training/view_mip_apgrad.py` (mirror of
`view_mip_asym.py` with Y-axis flip instead of X-axis flip).

### Windows setup notes (for reproducibility)
- Set `$env:SCAN_REPO = "C:\Users\nilsh\Projects\DaT Parkinson's Challenge"` each session
  (or run `[System.Environment]::SetEnvironmentVariable("SCAN_REPO", "...", "User")` once)
- Always add `--workers 0` on Windows (avoids CUDA DLL paging file errors)
- Prepared data at `SCAN_REPO\prepared\` (1186 scans; 167 excluded due to MemoryError
  during preprocessing — large-FOV outliers, consistently excluded)
- folds.csv generated from the 1186 available scans; fold assignments differ from the
  Linux 1353-scan split — OOF numbers are not directly comparable to Linux models but
  are internally consistent for comparing augmented vs. baseline runs

## Mislabel / DIP analysis — 2026-09-13

OOF predictions from `ax_split_aux03` (axisaware_split, aux_weight=0.3) were cross-referenced
with SBR features to identify the 40 highest-loss samples. Two clinically distinct groups
emerged.

### Hard negatives — labeled healthy, model strongly predicts PD

| uid | pred | SBR_put_min | interpretation |
|---|---|---|---|
| f5gzpl1q | 0.748 | **0.48** | SBR of 0.48 — pathologically low, almost certainly PD mislabeled as healthy |
| bbxux1ox | 0.882 | **0.63** | Same |
| bmae38sr | 0.806 | **0.77** | Same |
| 9tze5r5u | 0.988 | 1.07 | Low SBR + model very confident |
| gom6736z | 0.979 | 1.34 | L/R = 2.20/1.34 — 64% asymmetry, classic unilateral PD pattern |
| y08p0fod | 0.926 | 1.16 | L/R asymmetry 78% |

Most hard negatives have SBR < 1.5 despite a healthy label. In clinical DaT-SPECT, SBR < 1.5
strongly suggests dopaminergic deficit. These are the most likely mislabels in the dataset.

### Hard positives — labeled PD, model strongly predicts healthy

| uid | pred | SBR_put_min | interpretation |
|---|---|---|---|
| 7ujodck8 | 0.039 | **4.41** | SBR of 4.41 — completely normal binding. Almost certainly DIP |
| vzme9wt0 | 0.056 | 3.20 | Normal binding, DIP suspect |
| y595cpjz | 0.091 | 3.15 | Normal binding |
| dg1wjuii | 0.083 | 3.22 | Normal binding |
| 344mw04i | 0.076 | 3.30 | Normal binding |
| q10f7ia4 | 0.078 | **0.37** | GENUINE HARD PD — very low SBR, model misses it |
| rsezytkw | 0.097 | 0.86 | Genuine hard PD — low SBR |

Most hard positives with SBR > 2.0 are likely DIP (drug-induced parkinsonism): clinical
parkinsonism with structurally normal dopamine transporters. These should never have been
labeled as pathologic DaT-SPECT. The cases with low SBR (q10f7ia4, rsezytkw, dhvs41lq)
are genuine PD the model currently misses.

### What sbr_weight does

`--sbr-weight` assigns each positive (PD) training sample a weight of `1 / sbr_putamen_min`,
then normalises so the mean weight stays 1. Effect:

- DIP suspects (SBR > 3): weight ~0.1–0.3 → nearly ignored during training
- Mild PD (SBR 1.5–2.5): weight ~0.4–0.7 → moderate attention
- Clear PD (SBR 0.5–1.0): weight ~1.5–2.5 → strong focus
- All healthy controls: weight = 1.0 (unchanged)

This means DIP-like mislabeled cases contribute almost nothing to the gradient, while the
model focuses on learning the genuine hard positive cases (low SBR PD it currently misses).
Negative samples are unaffected — the hard negatives need a different intervention
(exclusion or relabeling).

### Files created

- `prepared/exclude_mislabels.csv` — all 40 suspicious UIDs with label, pred, SBR, reason
- `scripts/visualize_samples.py` — generates 3-panel MIP PNGs for inspection
- `--exclude-csv` flag added to `scripts/train.py`

### Recommended experiments

1. `--exclude-csv prepared/exclude_mislabels.csv` — train without suspicious cases, compare OOF
2. `--sbr-weight` — downweight DIP suspects without removing them
3. Both combined — cleanest dataset + weighted loss
4. Visually inspect `output/mislabel_review/` PNGs before deciding on exclusions

### Caution

Removing all 40 cases improves *reported* OOF (those hard cases no longer contribute to the
metric), but does not guarantee the model improves on the test set if some flagged cases
appear there. The hard cases with low SBR (q10f7ia4, rsezytkw, dhvs41lq, f5gzpl1q, bbxux1ox,
bmae38sr) are the ones most worth reviewing clinically before removal.

## Open questions

1. Does the OOF→public-leaderboard offset confirm our standing (step 1)?
2. ~~Is the crop-size curve a plateau at m4 or still rising?~~ Resolved: plateau
   (m4 ≈ m10, m6/m8 slightly worse; no size beats m4).
3. Can representation work (steps 4–5) close any of the ~0.014 AUROC gap to the leader,
   or is that gap public-split overfitting on their side?
4. Site-prior: exploit or abstain?
