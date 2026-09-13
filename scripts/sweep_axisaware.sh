#!/bin/bash
# Axisaware architecture sweep — all 5 folds per config.
# Tests aux_weight, dropout, bottleneck_dim, norm, and chimera_frac.
# Runs 2 configs in parallel; each config runs folds 0-4 sequentially.
#
# Usage:
#   export SCAN_REPO=/workspace/parkinson-classifier
#   bash scripts/sweep_axisaware.sh 2>&1 | tee output/runs/sweep_axisaware.log

REPO=${SCAN_REPO:-/workspace/parkinson-classifier}
CROPS="$REPO/prepared/crops_m4"
PY="python $REPO/scripts/train.py"

# base: axisaware+aux, best aug, percentile norm, 40 epochs
BASE="--model cnn3d --view volume3d --pool axisaware \
      --chimera-frac 0.25 --chimera-pos-frac 0.25 \
      --aug-rot-deg 15 --aug-shift-vox 5 \
      --norm percentile --norm-percentile 99 \
      --epochs 40 \
      --folds-csv $REPO/prepared/folds.csv \
      --crops-dir $CROPS --seed 0 --overwrite"

run_all_folds() {
    local name=$1; shift
    echo "[START] $name (all folds)"
    local ok=1
    for fold in 0 1 2 3 4; do
        $PY $BASE "$@" --fold $fold --run-name "ax_$name" \
            >> "$REPO/output/runs/ax_${name}.log" 2>&1 \
            || { echo "[FAIL]  $name fold $fold"; ok=0; break; }
        echo "  $name fold $fold done"
    done
    [ $ok -eq 1 ] && echo "[DONE]  $name" || true
}

print_oof() {
    python - <<'EOF'
import csv, os
from pathlib import Path
import numpy as np
try:
    from sklearn.metrics import log_loss, roc_auc_score
except ImportError:
    print("sklearn not available"); exit()

runs_dir = Path(os.environ.get("SCAN_REPO", "/workspace/parkinson-classifier")) / "output" / "runs"
rows = []
for run_dir in sorted(runs_dir.glob("ax_*")):
    all_preds, all_labels = [], []
    for k in range(5):
        vp = run_dir / f"fold{k}" / "val_preds.csv"
        if not vp.exists():
            break
        for r in csv.DictReader(open(vp)):
            all_preds.append(float(r["pred"]))
            all_labels.append(float(r["is_pathologic"]))
    if len(all_labels) < 100:
        continue
    p = np.clip(all_preds, 1e-6, 1 - 1e-6)
    ll  = log_loss(all_labels, p)
    auc = roc_auc_score(all_labels, p)
    rows.append((run_dir.name, len(all_labels), ll, auc))

rows.sort(key=lambda x: x[2])
print(f"\n{'run':<38} {'n':>5} {'OOF ll':>8} {'AUROC':>7}")
print("-" * 62)
for name, n, ll, auc in rows:
    print(f"{name:<38} {n:>5} {ll:>8.4f} {auc:>7.4f}")
EOF
}

echo "================================================================"
echo " Axisaware architecture sweep — all 5 folds per config"
echo "================================================================"
mkdir -p "$REPO/output/runs"

# ── Batch 1: aux_weight ───────────────────────────────────────────────────────
# How hard should the P-A and L-R auxiliary heads push?
# Too low = under-enforced, too high = aux heads dominate and main MLP can't complement
echo "=== Batch 1: aux_weight ==="
run_all_folds aux01 --aux-weight 0.1 &
run_all_folds aux03 --aux-weight 0.3 &   # current reference
wait

run_all_folds aux02 --aux-weight 0.2 &
run_all_folds aux05 --aux-weight 0.5 &
wait; echo "--- batch 1 done ---"

# ── Batch 2: dropout ─────────────────────────────────────────────────────────
# hp sweep showed dropout=0.1 beats 0.3 for avg-pool. Does this hold for
# axisaware+aux where aux heads already provide regularization?
echo "=== Batch 2: dropout ==="
run_all_folds drop01 --aux-weight 0.3 --dropout3d 0.1 &
run_all_folds drop02 --aux-weight 0.3 --dropout3d 0.2 &
wait; echo "--- batch 2 done ---"

# ── Batch 3: bottleneck dim ───────────────────────────────────────────────────
# 5C → bottleneck → 1. Default 256 = 5× compression of 1280 input.
# 128 = more compressed, more regularized; 512 = richer, more capacity
echo "=== Batch 3: bottleneck dim ==="
run_all_folds bn128 --aux-weight 0.3 --bottleneck-dim 128 &
run_all_folds bn512 --aux-weight 0.3 --bottleneck-dim 512 &
wait; echo "--- batch 3 done ---"

# ── Batch 4: norm control ─────────────────────────────────────────────────────
# Confirms percentile norm improvement and whether P95 vs P99 matters
echo "=== Batch 4: norm ==="
run_all_folds zscore  --aux-weight 0.3 --norm zscore &
run_all_folds pct95   --aux-weight 0.3 --norm percentile --norm-percentile 95 &
wait; echo "--- batch 4 done ---"

# ── Batch 5: chimera fraction ─────────────────────────────────────────────────
# Is 0.25 the optimal chimera fraction? Also relevant for final-model training
# on all data — density stays constant with dataset size, but is 0.25 right?
echo "=== Batch 5: chimera_frac ==="
run_all_folds chi010 --aux-weight 0.3 --chimera-frac 0.10 --chimera-pos-frac 0.10 &
run_all_folds chi015 --aux-weight 0.3 --chimera-frac 0.15 --chimera-pos-frac 0.15 &
wait

run_all_folds chi035 --aux-weight 0.3 --chimera-frac 0.35 --chimera-pos-frac 0.35 &
run_all_folds chi050 --aux-weight 0.3 --chimera-frac 0.50 --chimera-pos-frac 0.25 &
wait; echo "--- batch 5 done ---"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " OOF results (all 5 folds, lower ll = better)"
echo " Reference: cnn3d_m4_norm_pct99_s0 OOF ll ~0.2320"
echo "================================================================"
print_oof
