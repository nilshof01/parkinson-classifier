#!/bin/bash
# Fold-0 core technique sweep: SBR weighting, hard chimera, focal loss,
# axisaware pooling, auxiliary heads, and combinations.
# Runs 2 jobs in parallel on RTX 3090 (~5-6 GB each).
#
# Phase 1: individual techniques (8 runs)
# Phase 2: best combinations (up to 4 runs, fill in BEST1/BEST2 from phase 1)
# Phase 3: validate winner on folds 1-4 to check if gap closes
#
# Usage:
#   export SCAN_REPO=/workspace/parkinson-classifier
#   bash scripts/sweep_core.sh 2>&1 | tee output/runs/sweep_core.log

REPO=${SCAN_REPO:-/workspace/parkinson-classifier}
CROPS="$REPO/prepared/crops_m4"
PY="python $REPO/scripts/train.py"

# best aug settings from sweep_aug results
BASE="--model cnn3d --view volume3d --fold 0 --epochs 30 \
      --chimera-frac 0.25 --chimera-pos-frac 0.25 \
      --aug-rot-deg 15 --aug-shift-vox 5 \
      --folds-csv $REPO/prepared/folds.csv \
      --crops-dir $CROPS --seed 0 --overwrite"

run() {
    local name=$1; shift
    echo "[START] $name"
    $PY $BASE "$@" --run-name "core_$name" \
        > "$REPO/output/runs/core_${name}.log" 2>&1 \
        && echo "[DONE]  $name" \
        || echo "[FAIL]  $name — check core_${name}.log"
}

print_results() {
    python - <<'EOF'
import csv, os
from pathlib import Path

runs_dir = Path(os.environ.get("SCAN_REPO", "/workspace/parkinson-classifier")) / "output" / "runs"
rows = []
for fold_dir in sorted(runs_dir.glob("core_*/fold0")):
    mf = fold_dir / "metrics.csv"
    if not mf.exists():
        continue
    data = [r for r in csv.DictReader(open(mf))
            if r.get("val_log_loss", "").strip() and r.get("train_loss", "").strip()]
    if not data:
        print(f"  (no data: {fold_dir.parent.name})")
        continue
    best = min(data, key=lambda r: float(r["val_log_loss"]))
    train_ll = float(best["train_loss"])
    val_ll = float(best["val_log_loss"])
    rows.append((fold_dir.parent.name, val_ll, train_ll, val_ll - train_ll, int(best["epoch"])))

rows.sort(key=lambda x: x[1])
print(f"{'run':<40} {'val_ll':>8} {'train_ll':>9} {'gap':>6} {'epoch':>6}")
print("-" * 72)
for name, vll, tll, gap, ep in rows:
    print(f"{name:<40} {vll:>8.4f} {tll:>9.4f} {gap:>6.4f} {ep:>6}")
EOF
}

echo "============================================================"
echo " Core technique sweep — fold 0 (2 parallel)"
echo " crops: $CROPS"
echo "============================================================"
mkdir -p "$REPO/output/runs"

# ── Phase 1: individual techniques ───────────────────────────────────────────
echo ""
echo "=== Phase 1: individual techniques ==="

# reference: plain baseline with best aug (should match aug_rot15_shift5 ~0.1814)
run baseline &

# SBR weighting: upweights mild positives (high SBR) in training loss
# addresses distribution shift — mild PD dominates folds 2-4 val gap
run sbr_weight --sbr-weight &
wait; echo "--- p1 batch 1 done ---"

# focal loss: automatically down-weights easy positives, focuses gradient on hard cases
run focal --loss focal --focal-gamma 2.0 &

# focal + sbr (the two techniques targeting the same problem from different angles)
run focal_sbr --loss focal --focal-gamma 2.0 --sbr-weight &
wait; echo "--- p1 batch 2 done ---"

# axisaware pooling: preserves P-A gradient and L-R asymmetry explicitly
run axisaware --pool axisaware &

# axisaware + auxiliary heads: forces pa_vec and lr_vec to be independently discriminative
run axisaware_aux --pool axisaware --aux-weight 0.3 &
wait; echo "--- p1 batch 3 done ---"

# hard chimera: concentrates chimera mixing on normals the model currently finds hardest
run hard_chimera --hard-chimera &

# label smoothing: reduces overconfidence, which is one symptom of the gap
run smooth --label-smoothing 0.05 &
wait; echo "--- p1 batch 4 done ---"

echo ""
echo "============================================================"
echo " Phase 1 results (fold 0, lower val_ll = better)"
echo " gap = val_ll - train_ll  (large gap = overfit/distrib shift)"
echo "============================================================"
print_results

# ── Phase 2: combine best two from phase 1 ────────────────────────────────────
# Edit BEST1 and BEST2 to the two lowest val_ll configs from phase 1.
# Example: if sbr_weight and axisaware_aux win, set:
#   BEST1="--sbr-weight"
#   BEST2="--pool axisaware --aux-weight 0.3"

echo ""
echo "=== Phase 2: combine top two (edit BEST1/BEST2 first) ==="

BEST1="${SWEEP_BEST1:-}"
BEST2="${SWEEP_BEST2:-}"

if [ -z "$BEST1" ] || [ -z "$BEST2" ]; then
    echo "  Skipping phase 2 — set SWEEP_BEST1 and SWEEP_BEST2 env vars, e.g.:"
    echo "    SWEEP_BEST1='--sbr-weight' SWEEP_BEST2='--pool axisaware --aux-weight 0.3' bash scripts/sweep_core.sh"
else
    run best_combo $BEST1 $BEST2 &
    run best_combo_focal $BEST1 $BEST2 --loss focal &
    wait; echo "--- p2 done ---"

    echo ""
    echo "============================================================"
    echo " Phase 2 results"
    echo "============================================================"
    print_results
fi

# ── Phase 3: validate winner on hard folds ────────────────────────────────────
# Run the best config from phases 1+2 on folds 2, 3, 4 to check if the
# train/val gap closes. Set SWEEP_WINNER to the flags for the best config.

echo ""
echo "=== Phase 3: gap check on hard folds (set SWEEP_WINNER) ==="

WINNER="${SWEEP_WINNER:-}"

if [ -z "$WINNER" ]; then
    echo "  Skipping phase 3 — set SWEEP_WINNER env var, e.g.:"
    echo "    SWEEP_WINNER='--sbr-weight --pool axisaware --aux-weight 0.3' bash scripts/sweep_core.sh"
else
    HARD_BASE="--model cnn3d --view volume3d --epochs 30 \
               --chimera-frac 0.25 --chimera-pos-frac 0.25 \
               --aug-rot-deg 15 --aug-shift-vox 5 \
               --folds-csv $REPO/prepared/folds.csv \
               --crops-dir $CROPS --seed 0 --overwrite"

    for fold in 2 3 4; do
        echo "[START] winner_fold${fold}"
        $PY $HARD_BASE $WINNER --fold $fold \
            --run-name "core_winner_fold${fold}" \
            > "$REPO/output/runs/core_winner_fold${fold}.log" 2>&1 \
            && echo "[DONE]  winner_fold${fold}" \
            || echo "[FAIL]  winner_fold${fold}"
    done

    echo ""
    echo "============================================================"
    echo " Phase 3: gap on folds 2-4 with winner config"
    echo "============================================================"
    python - <<'EOF2'
import csv, os
from pathlib import Path

runs_dir = Path(os.environ.get("SCAN_REPO", "/workspace/parkinson-classifier")) / "output" / "runs"
rows = []
for fold_num in [2, 3, 4]:
    fold_dir = runs_dir / f"core_winner_fold{fold_num}" / f"fold{fold_num}"
    mf = fold_dir / "metrics.csv"
    if not mf.exists():
        continue
    data = list(csv.DictReader(open(mf)))
    best = min(data, key=lambda r: float(r["val_log_loss"]))
    train_ll = float(best["train_loss"])
    val_ll = float(best["val_log_loss"])
    rows.append((f"fold{fold_num}", val_ll, train_ll, val_ll - train_ll, int(best["epoch"])))

print(f"{'fold':<10} {'val_ll':>8} {'train_ll':>9} {'gap':>6} {'epoch':>6}")
print("-" * 45)
for name, vll, tll, gap, ep in rows:
    print(f"{name:<10} {vll:>8.4f} {tll:>9.4f} {gap:>6.4f} {ep:>6}")
EOF2
fi
