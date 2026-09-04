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
3. **Seed farm** (~1 GPU-day, mechanical): grow cnn3d-m4(+chimera) and fusion-m4 families
   to 6–8 seeds; each seed historically worth ~0.001 to its family average.
4. **Capacity probe** (~2 h): one cnn3d ×2 width at m4 with the full frozen recipe and
   80-epoch schedule — retests capacity under the modern recipe (the old width test
   predates m4/chimera/zoom).
5. **SSL pretraining** (~1 day, uncertain): masked-volume pretraining on the 1,353
   training scans only (test-set use prohibited), fine-tune the champion; the main
   untested representation lever and the most plausible source of the leader's AUROC edge.
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

## Open questions

1. Does the OOF→public-leaderboard offset confirm our standing (step 1)?
2. ~~Is the crop-size curve a plateau at m4 or still rising?~~ Resolved: plateau
   (m4 ≈ m10, m6/m8 slightly worse; no size beats m4).
3. Can representation work (steps 4–5) close any of the ~0.014 AUROC gap to the leader,
   or is that gap public-split overfitting on their side?
4. Site-prior: exploit or abstain?
