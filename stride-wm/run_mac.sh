#!/usr/bin/env bash
# Apple-silicon Mac: PyTorch runs on the GPU through MPS (picked automatically), MuJoCo renders
# natively. Mid-scale: 8x8 patch grid, DINO-WM-sized predictor. Time one model first (README).
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1   # run any op MPS lacks on CPU instead of crashing
EPISODES=${EPISODES:-500}              # ~150 KB/frame on disk: 500 episodes ~ 15 GB
STEPS=${STEPS:-20000}
SEEDS=${SEEDS:-"0 1 2"}

[ -f data/cube_single.h5 ] || python scripts/collect.py --out data/cube_single.h5 --episodes "$EPISODES" --steps 200
[ -f data/cache/z.npy ] || python scripts/encode.py --h5 data/cube_single.h5 --out data/cache --pool 2 --n-eval 50
python -m pytest -q tests

for s in $SEEDS; do
  python scripts/train.py --arch dinowm --k 5 --seed "$s" --steps "$STEPS"
  for k in 1 2 4 5 8; do python scripts/train.py --arch chunk --k "$k" --seed "$s" --steps "$STEPS"; done
done

python scripts/eval_openloop.py --ckpts runs/*.pt --horizon 80 --every 2
# planning is the slow part (k=1 makes 8x the model calls of k=8): seed 0, reduced CEM budget
python scripts/eval_plan.py --ckpts replay runs/*_s0.pt --offsets 25,50,100 --pairs 10 \
  --samples 100 --iters 10 --topk 10
python scripts/report.py
