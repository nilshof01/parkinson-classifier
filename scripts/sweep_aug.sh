#!/bin/bash
# Fold-0 augmentation sweep: rotation, shift, gamma, and combinations.
# Runs 2 jobs in parallel on RTX 3090 (~5-6 GB each).
#
# Usage:
#   export SCAN_REPO=/workspace/parkinson-classifier
#   bash scripts/sweep_aug.sh 2>&1 | tee sweep_aug.log

REPO=${SCAN_REPO:-/workspace/parkinson-classifier}
CROPS="$REPO/prepared/crops_m4"
PY="python $REPO/scripts/train.py"

# fixed base: matches the established baseline except we vary augmentation
BASE="--model cnn3d --view volume3d --fold 0 --epochs 30 \
      --chimera-frac 0.25 --chimera-pos-frac 0.25 \
      --folds-csv $REPO/prepared/folds.csv \
      --crops-dir $CROPS --seed 0 --overwrite"

run() {
    local name=$1; shift
    echo "[START] $name"
    $PY $BASE "$@" --run-name "aug_$name" \
        > "$REPO/output/runs/aug_${name}.log" 2>&1 \
        && echo "[DONE]  $name" \
        || echo "[FAIL]  $name — check aug_${name}.log"
}

echo "============================================"
echo " Augmentation fold-0 sweep (2 parallel)"
echo " crops: $CROPS"
echo "============================================"
mkdir -p "$REPO/output/runs"

# ── Batch 1: Rotation + shift ────────────────────────────────────────────────
# original baseline values vs new defaults vs aggressive
run orig_rot8_shift2  --aug-rot-deg 8  --aug-shift-vox 2  &   # original baseline
run rot15_shift5      --aug-rot-deg 15 --aug-shift-vox 5  &   # new default
wait

run rot20_shift8      --aug-rot-deg 20 --aug-shift-vox 8  &   # aggressive
run rot25_shift10     --aug-rot-deg 25 --aug-shift-vox 10 &   # very aggressive
wait; echo "--- batch 1 done ---"

# ── Batch 2: Gamma ───────────────────────────────────────────────────────────
# --aug all enables gamma at p=0.4; --aug all without gamma disables it
run nogamma           --aug flip,geom,zoom,res,field,scale,noise \
                      --aug-rot-deg 15 --aug-shift-vox 5  &   # gamma off
run gamma_only        --aug-rot-deg 15 --aug-shift-vox 5  &   # gamma on (--aug all default)
wait; echo "--- batch 2 done ---"

# ── Batch 3: Best rotation + gamma off/on ────────────────────────────────────
run rot20_nogamma     --aug-rot-deg 20 --aug-shift-vox 8 \
                      --aug flip,geom,zoom,res,field,scale,noise &
run rot20_gamma       --aug-rot-deg 20 --aug-shift-vox 8  &   # with gamma
wait; echo "--- batch 3 done ---"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Results (val_ll fold 0, lower = better)"
echo "============================================"
python - <<'EOF'
import csv, os
from pathlib import Path

runs_dir = Path(os.environ.get("SCAN_REPO", "/workspace/parkinson-classifier")) / "output" / "runs"
rows = []
for fold_dir in sorted(runs_dir.glob("aug_*/fold0")):
    mf = fold_dir / "metrics.csv"
    if not mf.exists():
        continue
    data = list(csv.DictReader(open(mf)))
    best = min(data, key=lambda r: float(r["val_log_loss"]))
    rows.append((fold_dir.parent.name, float(best["val_log_loss"]), int(best["epoch"])))

rows.sort(key=lambda x: x[1])
print(f"{'run':<35} {'val_ll':>8} {'epoch':>6}")
print("-" * 52)
for name, ll, ep in rows:
    print(f"{name:<35} {ll:>8.4f} {ep:>6}")
EOF
