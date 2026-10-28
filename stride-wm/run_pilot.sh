#!/usr/bin/env bash
# CPU-scale pilot (2 cores, no GPU): exactly what produced results/ for the checkpoint.
# Full-scale settings are the script defaults; see README.
set -euo pipefail
cd "$(dirname "$0")"
ENC=${ENC:-data/dinov2-small}   # local copy of facebook/dinov2-small (or the HF id)
SMALL="--depth 3 --heads 6 --mlp-dim 1024 --steps 5000 --batch 32"

python scripts/collect.py --out data/cube_single.h5 --episodes 120 --steps 200 --fast-render
python scripts/encode.py --h5 data/cube_single.h5 --out data/cache --encoder "$ENC" --pool 4 --n-eval 20
python -m pytest -q tests

python scripts/train.py --cache data/cache --arch dinowm --k 5 $SMALL       # original DINO-WM baseline
for k in 1 2 4 5 8; do
  python scripts/train.py --cache data/cache --arch chunk --k $k $SMALL      # proposed
done

python scripts/eval_openloop.py --cache data/cache --ckpts runs/*.pt --horizon 40 --every 2
python scripts/eval_plan.py --cache data/cache --ckpts runs/*.pt --offsets 25 --pairs 6 \
  --samples 64 --iters 5 --topk 8 --horizon 40 --replan 8 --fast-render
python scripts/report.py
