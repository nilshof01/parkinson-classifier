#!/bin/bash
# Overnight crop-size sweep: margins -5,-3,+4,+6 vox/side (x2 seeds) + m2 seed1.
# m=0 (cnn3d_crop, cnn3d_seed1) and m=+2 seed0 (cnn3d_m2) already exist.
cd /home/niho/scan-repo
set -x
for M in -5 -3 4 6; do
  python3 scripts/prepare_dataset.py --crop-margin $M
done
AUG=flip,geom,res,field,scale,noise
for M in -5 -3 4 6; do
  for S in 0 1; do
    python3 scripts/train.py --model cnn3d --view volume3d --fold all --epochs 40 \
      --lr 1e-3 --aug $AUG --crops-dir prepared/crops_m$M --seed-offset $S \
      --device cuda:0 --run-name cnn3d_m${M}_s${S}
  done
done
python3 scripts/train.py --model cnn3d --view volume3d --fold all --epochs 40 \
  --lr 1e-3 --aug $AUG --crops-dir prepared/crops_m2 --seed-offset 1 \
  --device cuda:0 --run-name cnn3d_m2_s1
python3 scripts/crop_sweep_summary.py
