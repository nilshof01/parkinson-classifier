#!/bin/bash
# Axisaware architecture sweep: aux_weight, dropout, bottleneck_dim, and
# combinations with percentile normalization.
# Runs 2 jobs in parallel on RTX 3090.
#
# Usage:
#   export SCAN_REPO=/workspace/parkinson-classifier
#   bash scripts/sweep_axisaware.sh 2>&1 | tee output/runs/sweep_axisaware.log

REPO=${SCAN_REPO:-/workspace/parkinson-classifier}
CROPS="$REPO/prepared/crops_m4"
PY="python $REPO/scripts/train.py"

# base: axisaware+aux, best aug, percentile norm, fold 0, 40 epochs
BASE="--model cnn3d --view volume3d --pool axisaware \
      --fold 0 --epochs 40 \
      --chimera-frac 0.25 --chimera-pos-frac 0.25 \
      --aug-rot-deg 15 --aug-shift-vox 5 \
      --norm percentile --norm-percentile 99 \
      --folds-csv $REPO/prepared/folds.csv \
      --crops-dir $CROPS --seed 0 --overwrite"

run() {
    local name=$1; shift
    echo "[START] $name"
    $PY $BASE "$@" --run-name "ax_$name" \
        > "$REPO/output/runs/ax_${name}.log" 2>&1 \
        && echo "[DONE]  $name" \
        || echo "[FAIL]  $name — check ax_${name}.log"
}

echo "============================================================"
echo " Axisaware architecture sweep — fold 0, 40 epochs"
echo "============================================================"
mkdir -p "$REPO/output/runs"

# ── Batch 1: aux_weight ───────────────────────────────────────────────────────
echo "=== Batch 1: aux_weight ==="
run aux01  --aux-weight 0.1 &
run aux02  --aux-weight 0.2 &
wait

run aux03  --aux-weight 0.3 &   # current default
run aux05  --aux-weight 0.5 &
wait; echo "--- batch 1 done ---"

# ── Batch 2: dropout (with best aux_weight=0.3 as placeholder) ───────────────
echo "=== Batch 2: dropout ==="
run drop01 --aux-weight 0.3 --dropout3d 0.1 &
run drop02 --aux-weight 0.3 --dropout3d 0.2 &
wait; echo "--- batch 2 done ---"

# ── Batch 3: bottleneck dim ───────────────────────────────────────────────────
echo "=== Batch 3: bottleneck dim ==="
run bn128  --aux-weight 0.3 --bottleneck-dim 128 &
run bn512  --aux-weight 0.3 --bottleneck-dim 512 &
wait; echo "--- batch 3 done ---"

# ── Batch 4: zscore vs percentile (control — base uses percentile already) ───
echo "=== Batch 4: norm control ==="
run zscore --aux-weight 0.3 --norm zscore &
run pct95  --aux-weight 0.3 --norm percentile --norm-percentile 95 &
wait; echo "--- batch 4 done ---"

# ── Results ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Results (fold 0, lower = better)"
echo "============================================================"
python - <<'EOF'
import csv, os
from pathlib import Path

runs_dir = Path(os.environ.get("SCAN_REPO", "/workspace/parkinson-classifier")) / "output" / "runs"
rows = []
for fold_dir in sorted(runs_dir.glob("ax_*/fold0")):
    mf = fold_dir / "metrics.csv"
    if not mf.exists():
        continue
    data = [r for r in csv.DictReader(open(mf))
            if r.get("val_log_loss", "").strip() and r.get("train_loss", "").strip()]
    if not data:
        continue
    best = min(data, key=lambda r: float(r["val_log_loss"]))
    vll = float(best["val_log_loss"])
    rows.append((fold_dir.parent.name, vll, int(best["epoch"])))

rows.sort(key=lambda x: x[1])
print(f"{'run':<30} {'val_ll':>8} {'epoch':>6}")
print("-" * 47)
for name, vll, ep in rows:
    print(f"{name:<30} {vll:>8.4f} {ep:>6}")
EOF
