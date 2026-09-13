#!/bin/bash
# Anisotropic stretch augmentation sweep — all 5 folds per config.
# Compares baseline (no aniso) vs aniso-only vs iso+aniso at different ranges.
# Runs 2 configs in parallel; each config runs folds 0-4 sequentially.
#
# Usage:
#   export SCAN_REPO=/workspace/parkinson-classifier
#   bash scripts/sweep_aniso.sh 2>&1 | tee output/runs/sweep_aniso.log

REPO=${SCAN_REPO:-/workspace/parkinson-classifier}
CROPS="$REPO/prepared/crops_m4"
PY="python $REPO/scripts/train.py"

BASE="--model cnn3d --view volume3d \
      --chimera-frac 0.25 --chimera-pos-frac 0.25 \
      --aug-rot-deg 15 --aug-shift-vox 5 \
      --norm percentile --norm-percentile 99 \
      --epochs 40 \
      --folds-csv $REPO/prepared/folds.csv \
      --crops-dir $CROPS --seed 0 --overwrite"

# run all 5 folds for one config, sequentially
run_all_folds() {
    local name=$1; shift
    echo "[START] $name (all folds)"
    local ok=1
    for fold in 0 1 2 3 4; do
        $PY $BASE "$@" --fold $fold --run-name "aniso_$name" \
            >> "$REPO/output/runs/aniso_${name}.log" 2>&1 \
            || { echo "[FAIL]  $name fold $fold"; ok=0; break; }
        echo "  fold $fold done"
    done
    [ $ok -eq 1 ] && echo "[DONE]  $name" || true
}

print_oof() {
    python - <<'EOF'
import csv, os
from pathlib import Path
from sklearn.metrics import log_loss, roc_auc_score
import numpy as np

runs_dir = Path(os.environ.get("SCAN_REPO", "/workspace/parkinson-classifier")) / "output" / "runs"
rows = []
for run_dir in sorted(runs_dir.glob("aniso_*")):
    all_preds, all_labels = [], []
    for k in range(5):
        vp = run_dir / f"fold{k}" / "val_preds.csv"
        if not vp.exists():
            break
        data = list(csv.DictReader(open(vp)))
        all_preds.extend(float(r["pred"]) for r in data)
        all_labels.extend(float(r["is_pathologic"]) for r in data)
    if len(all_labels) < 100:
        continue
    p = np.clip(all_preds, 1e-6, 1 - 1e-6)
    ll  = log_loss(all_labels, p)
    auc = roc_auc_score(all_labels, p)
    rows.append((run_dir.name, len(all_labels), ll, auc))

rows.sort(key=lambda x: x[2])
print(f"\n{'run':<35} {'n':>5} {'OOF ll':>8} {'AUROC':>7}")
print("-" * 58)
for name, n, ll, auc in rows:
    print(f"{name:<35} {n:>5} {ll:>8.4f} {auc:>7.4f}")
EOF
}

echo "============================================================"
echo " Anisotropic stretch sweep — all 5 folds per config"
echo "============================================================"
mkdir -p "$REPO/output/runs"

# ── Batch 1: baseline vs aniso-only ──────────────────────────────────────────
# baseline: no aniso augmentation (current default --aug all has no aniso)
run_all_folds baseline &

# aniso-only: replace isotropic zoom with anisotropic stretch, same range 0.9-1.1
run_all_folds aniso_med \
    --aug flip,geom,aniso,res,field,scale,noise \
    --aug-aniso-range 0.9 1.1 &
wait; echo "--- batch 1 done ---"

# ── Batch 2: different aniso ranges ──────────────────────────────────────────
# narrow range — subtle stretch ±7%
run_all_folds aniso_narrow \
    --aug flip,geom,aniso,res,field,scale,noise \
    --aug-aniso-range 0.93 1.07 &

# wide range — aggressive stretch ±15%
run_all_folds aniso_wide \
    --aug flip,geom,aniso,res,field,scale,noise \
    --aug-aniso-range 0.85 1.15 &
wait; echo "--- batch 2 done ---"

# ── Batch 3: iso + aniso combined ────────────────────────────────────────────
# both isotropic zoom AND per-axis stretch active
run_all_folds iso_plus_aniso \
    --aug all \
    --aug-aniso-range 0.9 1.1 &
wait; echo "--- batch 3 done ---"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " OOF results (all 5 folds, lower ll = better)"
echo "============================================================"
print_oof
