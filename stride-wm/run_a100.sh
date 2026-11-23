#!/usr/bin/env bash
# One NVIDIA A100 (40 GB), ~40 GPU-hours: the "Next experiments" plan of the progress update.
# Stages run in priority order; each one skips work whose output already exists, so after a crash or a
# job time limit, re-running the same command resumes. Run a subset with STAGES="data base8" etc.
#
#   setup       versions, GPU, unit tests                                        minutes
#   data        collect episodes (CPU) if missing; encode 8x8 and 16x16 caches;  ~1 h
#               check the 16x16 encoding against the 8x8 cache
#   timing      300 steps of a 16x16 k=1 model -> projected hours for train16    ~2 min
#   base8       6 seed-0 models at the laptop settings (8x8, 20k steps); port    ~1.5 h fp32
#               check against the laptop's validation losses; they are also the
#               baselines for the ablations
#   cost        fit the candidate planning costs, offline rank-correlation       minutes
#               check, pick one (rule fixed before planning)
#   train16     18 models at 16x16 patches: 5 strides + DINO-WM baseline,        ~11 h fp32
#               3 seeds, STEPS each (20k, same as the laptop run)
#   openloop16  E1/E4 on the 16x16 models, report, failure analysis               ~45 min
#   plan16      E2/E3 on the seed-0 16x16 models with the chosen cost, plus      ~5.5 h fp32
#               expert replay and the random-action null
#   ablate8     A1 (8-step chain), A2 (embedding width), A3 (residual) at the     ~6 h fp32
#               laptop settings, seed 0; A4 action-shuffle and 8x8-probe controls
#   control16   E2 again with the original latent cost, same pairs (control)      ~5.5 h fp32
#
# Hours are estimates in plain fp32; TF32=1 (the default here) should roughly halve the GPU-bound parts.
# Outputs: data/cache (8x8), data/cache16 (16x16), runs8/, runs16/, results8/, results16/
set -euo pipefail
cd "$(dirname "$0")"
EPISODES=${EPISODES:-250}   # same data size as the laptop run, so 16x16 vs 8x8 is the only change
N_EVAL=${N_EVAL:-50}        # held-out episodes (laptop run: 50)
STEPS=${STEPS:-20000}       # training steps per model (laptop run: 20000)
SEEDS=${SEEDS:-"0 1 2"}
PAIRS=${PAIRS:-20}          # planning start/goal pairs per model
TF32=${TF32:-1}             # 1 = TF32 tensor cores on the A100 (~2x faster); 0 = plain fp32
DRY_RUN=${DRY_RUN:-0}       # 1 = print the commands without running them
STAGES=${STAGES:-"setup data timing base8 cost train16 openloop16 plan16 ablate8 control16"}
T=""; [ "$TF32" = 1 ] && T="--tf32"
PLAN="--horizon 40 --pairs $PAIRS --samples 100 --iters 10 --topk 10 $T"
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore

T_START=$(date +%s)
stamp() { echo "== $(date '+%F %T') [+$(( ($(date +%s) - T_START) / 60 )) min] $*  [free disk: $(df -h . | awk 'NR==2{print $4}')]"; }
run() { echo "+ $*"; [ "$DRY_RUN" = 1 ] || "$@"; }
want() { [[ " $STAGES " == *" $1 "* ]]; }
train() {  # $1 = runs dir, $2 = checkpoint name (must match train.py's naming), rest = train.py args
  local runs=$1 name=$2; shift 2
  if [ -f "$runs/$name.pt" ]; then echo "skip $runs/$name (exists)"; return; fi
  stamp "train $runs/$name"; run python scripts/train.py "$@" --steps "$STEPS" --out "$runs" $T
}
report() {  # $1 = results dir, $2 = runs dir, $3 = cache
  if [ "$DRY_RUN" != 1 ] && [ ! -f "$1/openloop.csv" ]; then echo "no $1/openloop.csv yet: report skipped"; return; fi
  run python scripts/report.py --openloop "$1/openloop.csv" --plan "$1/plan.jsonl" --runs "$2" --out "$1"
  run python scripts/analyze_failures.py --openloop "$1/openloop.csv" --plan "$1/plan.jsonl" --cache "$3" \
    --out "$1/failure_analysis.md"
}

if want setup; then
  stamp setup
  run python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
  run python -c "import stable_worldmodel, mujoco; print('stable_worldmodel + mujoco ok')"
  run python -m pytest -q tests
fi

if want data; then
  stamp data
  [ -f data/cube_single.h5 ] || run python scripts/collect.py --out data/cube_single.h5 --episodes "$EPISODES" --steps 200
  # encoder.json is written last, so it marks a finished cache (an interrupted encode leaves z.npy behind)
  [ -f data/cache/encoder.json ] || run python scripts/encode.py --h5 data/cube_single.h5 --out data/cache --pool 2 --n-eval "$N_EVAL" --batch 128
  [ -f data/cache16/encoder.json ] || run python scripts/encode.py --h5 data/cube_single.h5 --out data/cache16 --pool 1 --n-eval "$N_EVAL" --batch 128
  run python scripts/check_cache.py --cache8 data/cache --cache16 data/cache16
fi

if want timing; then
  stamp timing
  run python scripts/train.py --cache data/cache16 --k 1 --steps 300 --eval-every 150 --out runs_timing $T
  run python -c "
import torch
log = torch.load('runs_timing/chunk_k1_s0.pt', map_location='cpu', weights_only=False)['log']
s = (log[-1]['sec'] - log[0]['sec']) / 150
n = 6 * len('$SEEDS'.split())
print('16x16 training: %.3f s/step -> %.2f h per model -> train16 (%d models) ~%.1f h'
      % (s, s * $STEPS / 3600, n, s * $STEPS * n / 3600))"
  run rm -rf runs_timing
fi

if want base8; then
  stamp base8
  for k in 1 2 4 5 8; do train runs8 "chunk_k${k}_s0" --cache data/cache --arch chunk --k "$k" --seed 0; done
  train runs8 dinowm_k5_s0 --cache data/cache --arch dinowm --k 5 --seed 0
  run python scripts/check_port.py --runs runs8
fi

if want cost; then
  stamp cost
  run python scripts/fit_cost.py --cache data/cache16 --out results16/cost_check.md
  run python scripts/fit_cost.py --cache data/cache --out results8/cost_check.md   # comparable to the laptop's rho
fi

if want train16; then
  stamp train16
  for s in $SEEDS; do  # seed by seed, so seed 0 (the one planning uses) finishes first
    train runs16 "dinowm_k5_s$s" --cache data/cache16 --arch dinowm --k 5 --seed "$s"
    for k in 1 2 4 5 8; do train runs16 "chunk_k${k}_s$s" --cache data/cache16 --arch chunk --k "$k" --seed "$s"; done
  done
fi

if want openloop16; then
  stamp openloop16
  # batch 64: at 256 a k=1 rollout ran a 16 GB laptop out of memory at 8x8; 16x16 needs ~4x the memory
  [ -f results16/openloop.csv ] || run python scripts/eval_openloop.py --cache data/cache16 --ckpts runs16/*.pt \
    --horizon 80 --every 2 --batch 64 --out results16/openloop.csv $T
  report results16 runs16 data/cache16
fi

if want plan16; then
  stamp plan16
  run python scripts/eval_plan.py --cache data/cache16 --ckpts replay random runs16/*_s0.pt --cost best $PLAN \
    --out results16/plan.jsonl
  report results16 runs16 data/cache16
fi

if want ablate8; then
  stamp ablate8
  # A1 stride vs compounding: every stride trained on the same 8-step chained loss (k=8 at 8 steps is the default model)
  train runs8 chunk_k1-u8_s0 --cache data/cache --arch chunk --k 1 --unroll 8 --seed 0
  train runs8 chunk_k2-u4_s0 --cache data/cache --arch chunk --k 2 --unroll 4 --seed 0
  train runs8 chunk_k4-u2_s0 --cache data/cache --arch chunk --k 4 --unroll 2 --seed 0
  # A2 embedding width: DINO-WM at the chunk encoder's 64-d, chunk at DINO-WM's 10-d, chunk width sweep at k=8
  train runs8 dinowm_k5-e64_s0 --cache data/cache --arch dinowm --k 5 --dinowm-emb 64 --seed 0
  train runs8 chunk_k5-emb10_s0 --cache data/cache --arch chunk --k 5 --emb-dim 10 --seed 0
  train runs8 chunk_k8-emb32_s0 --cache data/cache --arch chunk --k 8 --emb-dim 32 --seed 0
  train runs8 chunk_k8-emb128_s0 --cache data/cache --arch chunk --k 8 --emb-dim 128 --seed 0
  # A3 residual prediction, both architectures
  for k in 1 5 8; do train runs8 "chunk_k${k}-res_s0" --cache data/cache --arch chunk --k "$k" --residual --seed 0; done
  train runs8 dinowm_k5-res_s0 --cache data/cache --arch dinowm --k 5 --residual --seed 0
  # open loop for every 8x8 model; A4 controls: shuffled actions, and a probe that reads the full 8x8 grid
  [ -f results8/openloop.csv ] || {
    run python scripts/eval_openloop.py --cache data/cache --ckpts runs8/*.pt --horizon 40 --every 2 --batch 128 \
      --out results8/openloop_main.csv $T
    run python scripts/eval_openloop.py --cache data/cache --ckpts runs8/chunk_k?_s0.pt runs8/dinowm_k5_s0.pt \
      --horizon 40 --every 2 --batch 128 --shuffle-actions --out results8/openloop_shuf.csv $T
    run python -c "
import pandas as pd
a, b = pd.read_csv('results8/openloop_main.csv'), pd.read_csv('results8/openloop_shuf.csv')
pd.concat([a, b[b.arch.str.endswith('-shuf')]]).to_csv('results8/openloop.csv', index=False)"
  }
  [ -f results8/openloop_probe8.csv ] || run python scripts/eval_openloop.py --cache data/cache \
    --ckpts runs8/chunk_k1_s0.pt runs8/chunk_k8_s0.pt --horizon 40 --every 2 --batch 128 --probe-grid 8 \
    --out results8/openloop_probe8.csv $T
  report results8 runs8 data/cache
fi

if want control16; then
  stamp control16
  run python scripts/eval_plan.py --cache data/cache16 --ckpts runs16/*_s0.pt --cost latent $PLAN \
    --out results16/plan.jsonl
  report results16 runs16 data/cache16
fi
stamp DONE
