#!/bin/bash
cd /home/niho/scan-repo
AUG=flip,geom,res,field,scale,noise
python3 scripts/prepare_dataset.py --crop-margin 8
python3 scripts/prepare_dataset.py --crop-margin 10
while pgrep -f "cnn3d_m4_zoom|chimboth_s1" > /dev/null; do sleep 60; done
for M in 8 10; do
  for S in 0 1; do
    D=$((S % 2))
    python3 scripts/train.py --model cnn3d --view volume3d --fold all --epochs 40 \
      --lr 1e-3 --aug $AUG --crops-dir prepared/crops_m$M --seed-offset $S \
      --device cuda:$D --run-name cnn3d_m${M}_s${S} &
  done
  wait
done
python3 - <<'PY'
import json, pandas as pd
for m in (4, 6, 8, 10):
    for r in ([f"cnn3d_m{m}_s0", f"cnn3d_m{m}_s1"] if m > 6 else
              ([f"cnn3d_m{m}_s0", f"cnn3d_m{m}_s1"] if m == 6 else ["cnn3d_m4_s0", "cnn3d_m4_s1"])):
        try:
            s = json.load(open(f"output/runs/{r}/oof_summary.json"))
            print(f"m={m} {r}: {s['oof_log_loss_calibrated_clipped']:.4f}")
        except FileNotFoundError:
            print(f"m={m} {r}: missing")
PY
