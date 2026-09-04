#!/bin/bash
cd /home/niho/scan-repo
AUG=flip,geom,res,field,scale,noise
for K in 3 4; do
  python3 scripts/train.py --model cnn3d --view volume3d --fold $K --epochs 40 --lr 1e-3 \
    --aug $AUG --crops-dir prepared/frames --device cuda:0 --overwrite --run-name cnn3d_frame3d
done
for K in 1 2 3 4; do
  python3 scripts/train.py --model dinov2_small --view mip --fold $K --epochs 40 --lr 1e-4 \
    --aug $AUG --device cuda:1 --overwrite --run-name dinov2_mip &
  wait
done
