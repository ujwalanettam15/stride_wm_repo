#!/usr/bin/env bash
# Preliminary pass on Apple silicon (~1 h). Same data scale, predictor and step count as the CPU
# pilot in results/; the one thing changed is the patch grid, 4x4 -> 8x8 (--pool 2). That is the
# open check the README names: whether the H3 null (k=8 no worse than k=1 in Grasp/Place) survives
# at finer resolution, and whether 4x4 pooling was hiding contact detail.
# Two seeds, so E4 (chunk vs dinowm at k=5) keeps its seed-spread error bars.
# Timings measured on this machine: 0.033 s/train-step at this model size, MPS.
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
EPISODES=${EPISODES:-120}      # ~6 min, ~3.5 GB
SEEDS=${SEEDS:-"0 1"}
SMALL="--depth 3 --heads 6 --mlp-dim 1024 --steps 5000 --batch 32"   # pilot predictor, 4.96M params

[ -f data/cube_single.h5 ] || python scripts/collect.py --out data/cube_single.h5 --episodes "$EPISODES" --steps 200
[ -f data/cache/z.npy ]    || python scripts/encode.py --h5 data/cube_single.h5 --out data/cache --pool 2 --n-eval 20
python -m pytest -q tests

# 12 models at ~3 min each: 6 conditions x 2 seeds.
for s in $SEEDS; do
  python scripts/train.py --arch dinowm --k 5 --seed "$s" $SMALL              # original DINO-WM baseline
  for k in 1 2 4 5 8; do python scripts/train.py --arch chunk --k "$k" --seed "$s" $SMALL; done
done

# E1/E4: the cheap, informative half. ~1 min per checkpoint.
python scripts/eval_openloop.py --ckpts runs/*.pt --horizon 40 --every 2
python scripts/report.py

# E2/E3 planning is the expensive half and is deliberately not run here. Seed 0 only, 5 pairs at
# the near offset, ~25-35 min (extrapolated from 6 min for one k=1 episode at the 22M model size;
# cost scales as H/k and roughly with model size). Run it once E1 looks sane:
#   python scripts/eval_plan.py --ckpts replay runs/*_s0.pt --offsets 25 --pairs 5 \
#     --samples 100 --iters 10 --topk 10 --horizon 40 --replan 8
#   python scripts/report.py
