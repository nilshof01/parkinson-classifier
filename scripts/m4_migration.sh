#!/bin/bash
cd /home/niho/scan-repo
AUG=flip,geom,res,field,scale,noise
python3 scripts/train.py --model r3d18 --view volume3d --fold all --epochs 40 --lr 1e-4 \
  --batch-size 8 --aug $AUG --crops-dir prepared/crops_m4 \
  --chimera-frac 0.25 --chimera-pos-frac 0.25 --device cuda:0 --run-name r3d18_m4 &
P0=$!
(python3 scripts/train.py --model fusion3d --view fusion3d --fold all --epochs 40 --lr 1e-3 \
  --aug $AUG --crops-dir prepared/crops_m4 --device cuda:1 --run-name fusion3d_m4 && \
 python3 scripts/train.py --model slicevoter --view volume3d --fold all --epochs 40 \
  --batch-size 16 --aug $AUG --crops-dir prepared/crops_m4 \
  --chimera-frac 0.25 --chimera-pos-frac 0.25 --device cuda:1 --run-name slicevoter_m4) &
P1=$!
wait $P0 $P1
