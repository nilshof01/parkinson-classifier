#!/bin/bash
# Fold-0 sweep for cnn3d hyperparameters — runs 2 jobs in parallel.
# RTX 3090 (24 GB): ~5-6 GB per job, 2 at once = ~12 GB, leaves ~12 GB headroom.
# Wall clock: ~1.5 h instead of ~2.5 h sequential.
#
# Usage:
#   export SCAN_REPO=/workspace/parkinson-classifier
#   bash scripts/sweep_cnn3d.sh 2>&1 | tee sweep.log

REPO=${SCAN_REPO:-/workspace/parkinson-classifier}
CROPS="$REPO/prepared/crops_m4"
PY="python $REPO/scripts/train.py"
BASE="--model cnn3d --view volume3d --fold 0 --epochs 40 --crops-dir $CROPS \
      --chimera-frac 0.25 --chimera-pos-frac 0.25"

run() {
    local name=$1; shift
    echo "[START] $name"
    $PY $BASE "$@" --run-name "$name" --overwrite > "$REPO/output/runs/${name}_sweep.log" 2>&1 \
        && echo "[DONE]  $name" \
        || echo "[FAIL]  $name — check ${name}_sweep.log"
}

echo "========================================="
echo " cnn3d fold-0 sweep (2 parallel jobs)"
echo " crops: $CROPS"
echo "========================================="
mkdir -p "$REPO/output/runs"

# ── Batch 1: Learning rate ────────────────────────────────────────────────────
run sw_lr_3e4   --lr 3e-4  &
run sw_lr_1e3   --lr 1e-3  &   # current default
wait
run sw_lr_3e3   --lr 3e-3  &
run sw_lr_1e4   --lr 1e-4  &
wait; echo "--- batch 1 done ---"

# ── Batch 2: Pooling + head ───────────────────────────────────────────────────
run sw_pool_avg_lin  --lr 1e-3 --pool avg       --head linear &   # current
run sw_pool_cat_mlp  --lr 1e-3 --pool catavgmax --head mlp    &   # best on effb0
wait
run sw_pool_avg_mlp  --lr 1e-3 --pool avg       --head mlp    &
run sw_pool_cat_lin  --lr 1e-3 --pool catavgmax --head linear &
wait; echo "--- batch 2 done ---"

# ── Batch 3: Weight decay + dropout ──────────────────────────────────────────
run sw_wd_1e5    --lr 1e-3 --weight-decay 1e-5 &
run sw_wd_1e3    --lr 1e-3 --weight-decay 1e-3 &
wait
run sw_drop_01   --lr 1e-3 --dropout3d 0.1     &
run sw_drop_05   --lr 1e-3 --dropout3d 0.5     &
wait; echo "--- batch 3 done ---"

# ── Batch 4: Width multiplier ─────────────────────────────────────────────────
run sw_width_2x  --lr 1e-3 --width-mult 2.0 &
run sw_width_05  --lr 1e-3 --width-mult 0.5 &
wait; echo "--- batch 4 done ---"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================="
echo " Results (val_ll fold 0, lower = better)"
echo "========================================="
python - <<'EOF'
import json
from pathlib import Path
import os

runs_dir = Path(os.environ.get("SCAN_REPO", "/workspace/parkinson-classifier")) / "output" / "runs"
rows = []
for fold_dir in sorted(runs_dir.glob("sw_*/fold0")):
    metrics_file = fold_dir / "metrics.csv"
    if not metrics_file.exists():
        continue
    import csv
    rows_m = list(csv.DictReader(open(metrics_file)))
    best = min(rows_m, key=lambda r: float(r["val_log_loss"]))
    rows.append((fold_dir.parent.name, float(best["val_log_loss"]), int(best["epoch"])))

rows.sort(key=lambda x: x[1])
print(f"{'run':<30} {'val_ll':>8} {'best_epoch':>12}")
print("-" * 52)
for name, ll, ep in rows:
    print(f"{name:<30} {ll:>8.4f} {ep:>12}")
EOF
