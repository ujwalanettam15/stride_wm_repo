#!/usr/bin/env bash
# One NVIDIA A100 (40 GB), ~40 GPU-hours. Stages run in priority order: experiments that can confirm or
# falsify an explanation in the progress update come first; scaling up to 16x16 patches is optional and
# last. Each stage skips work whose output already exists, so after a crash or a job time limit,
# re-running the same command resumes. Run a subset with STAGES="data base8" etc.
#
# Default stages (8x8 patches, same data and settings as the laptop run), ~21 h in plain fp32:
#   setup     versions, GPU, unit tests                                              minutes
#   data      collect episodes if missing (CPU), encode the 8x8 cache, check it      0.2-1 h
#   base8     6 seed-0 models at the laptop settings; port check against the         ~0.75 h
#             laptop's validation losses; prints the measured speed and projection
#   compound  EXPLANATION TESTED: "coarse strides win because they chain fewer       ~13 h
#             calls, not because they learn better dynamics". Every stride
#             (k = 1, 2, 4, 8) trained on the same SPAN-step chained loss, 3 seeds,
#             next to the default models; plus the error carry-over decomposition.
#             Falsified if the stride ordering survives equal-span training.
#   cost      fit the candidate planning costs, offline check, pick one              minutes
#   plan8     EXPLANATION TESTED: "planning fails because the latent cost tracks     ~4.7 h
#             the arm, not the cube". Same models and pairs planned with the chosen
#             cost and with the original cost, next to replay and the random null.
#             Falsified if the chosen cost does not beat random either.
#   ablate8   EXPLANATIONS TESTED: "the E4 null hides an embedding-width effect"     ~1.5 h
#             (A2), "models drift a static cube" (A3 residual), "models use their
#             actions" (A4 shuffle), "4x4 probe pooling hides detail" (A4 probe).
#
# Optional scale-up, only with hours to spare (not in the default STAGES):
#   STAGES="data16 timing16 train16 openloop16 plan16" SEEDS="0" bash run_a100.sh
#   data16 (~0.3 h), train16 (18 models x ~37 min; 6 with SEEDS="0"), openloop16, plan16 (~5.5 h)
#
# Hours are estimates in plain fp32; TF32=1 (the default here) should roughly halve the GPU-bound parts.
# Outputs: data/cache (8x8), runs8/, results_a100/; scale-up: data/cache16, runs16/, results_a100_16x16/
set -euo pipefail
cd "$(dirname "$0")"
EPISODES=${EPISODES:-250}   # same data size as the laptop run
N_EVAL=${N_EVAL:-50}        # held-out episodes (laptop run: 50)
STEPS=${STEPS:-20000}       # training steps per model (laptop run: 20000)
SEEDS=${SEEDS:-"0 1 2"}     # seeds for the compounding ablation (and for train16)
SPAN=${SPAN:-16}            # env steps of chained prediction in the equal-span ablation (k must divide it)
PAIRS=${PAIRS:-30}          # planning start/goal pairs per model
TF32=${TF32:-1}             # 1 = TF32 tensor cores on the A100 (~2x faster); 0 = plain fp32
DRY_RUN=${DRY_RUN:-0}       # 1 = print the commands without running them
STAGES=${STAGES:-"setup data base8 compound cost plan8 ablate8"}
T=""; [ "$TF32" = 1 ] && T="--tf32"
PLAN="--horizon 40 --pairs $PAIRS --samples 100 --iters 10 --topk 10 $T"
BASE8="runs8/chunk_k1_s0.pt runs8/chunk_k2_s0.pt runs8/chunk_k4_s0.pt runs8/chunk_k5_s0.pt runs8/chunk_k8_s0.pt runs8/dinowm_k5_s0.pt"
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore

# Keep a copy of everything this run prints next to its results (logs/ is not in git; this folder is),
# so the log travels back with the results when they are committed. One file per invocation, written as
# it goes, so a job killed at its time limit still leaves its log.
if [ "$DRY_RUN" != 1 ]; then
  LOG_DIR=results_a100/run_logs
  [[ " $STAGES " == *16* ]] && LOG_DIR=results_a100_16x16/run_logs
  mkdir -p "$LOG_DIR"
  LOG="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}.log"
  exec > >(tee -a "$LOG") 2>&1
  # ${HOSTNAME}: bash sets it; the `hostname` command is missing in some containers (Singularity images)
  echo "run_a100.sh | commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown) | host ${HOSTNAME:-unknown} |" \
       "job ${SLURM_JOB_ID:-none} | log $LOG"
  echo "settings: STAGES=\"$STAGES\" EPISODES=$EPISODES STEPS=$STEPS SEEDS=\"$SEEDS\" SPAN=$SPAN PAIRS=$PAIRS TF32=$TF32"
  python -m pip freeze > "${LOG%.log}_packages.txt" 2>/dev/null || true   # exact package versions of this run
fi

# Stop if no GPU is visible: the scripts fall back to the CPU silently, which would spend the whole
# allocation on a fraction of the work. Usual causes: the session/job was requested without a GPU, or the
# container was started without --nv. ALLOW_CPU=1 overrides (e.g. to run only the `cost` stage).
if [ "$DRY_RUN" != 1 ] && [ "${ALLOW_CPU:-0}" != 1 ]; then
  python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'No CUDA GPU visible. Check that the session or job has a GPU and that the container was started with --nv. ALLOW_CPU=1 overrides.')"
fi

# MuJoCo renders through EGL on the GPU. A container started with --nv gets NVIDIA's EGL library but may lack
# the small vendor file that tells EGL where it is (NYU Torch's code-server image: "Cannot initialize a
# headless EGL display"). Write one and point EGL at it, unless the system already has one or it is set.
if [ "$DRY_RUN" != 1 ] && [ "$(uname)" = Linux ] && [ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ] \
   && ! ls /usr/share/glvnd/egl_vendor.d/*nvidia* /etc/glvnd/egl_vendor.d/*nvidia* >/dev/null 2>&1; then
  echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > "$HOME/nvidia_egl.json"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$HOME/nvidia_egl.json"
  echo "EGL: no NVIDIA vendor file in the system, using $__EGL_VENDOR_LIBRARY_FILENAMES"
fi
[ "$(uname)" = Linux ] && export MUJOCO_GL=${MUJOCO_GL:-egl}   # headless; macOS renders natively

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
openloop8() {  # open loop for every 8x8 model trained so far, plus the shuffled-action control; then the report
  run python scripts/eval_openloop.py --cache data/cache --ckpts runs8/*.pt --horizon 40 --every 2 --batch 128 \
    --out results_a100/openloop_main.csv $T
  run python scripts/eval_openloop.py --cache data/cache --ckpts $BASE8 --horizon 40 --every 2 --batch 128 \
    --shuffle-actions --out results_a100/openloop_shuf.csv $T
  run python -c "
import pandas as pd
a, b = pd.read_csv('results_a100/openloop_main.csv'), pd.read_csv('results_a100/openloop_shuf.csv')
pd.concat([a, b[b.arch.str.endswith('-shuf')]]).to_csv('results_a100/openloop.csv', index=False)"
  report results_a100 runs8 data/cache
}

if want setup; then
  stamp setup
  run python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
  run python -c "import stable_worldmodel, mujoco; print('stable_worldmodel + mujoco ok')"
  run python -m pytest -q tests
fi

if want data; then
  stamp data
  # collect into a scratch folder and move it in when done, so a cut-off collection is redone on resume
  # instead of a half-written file being taken as the dataset
  if [ ! -f data/cube_single.h5 ]; then
    run rm -rf data/collect_tmp
    run python scripts/collect.py --out data/collect_tmp/cube_single.h5 --episodes "$EPISODES" --steps 200
    run mv data/collect_tmp/cube_single.h5 data/cube_single.h5
  fi
  # encoder.json is written last, so it marks a finished cache (an interrupted encode leaves z.npy behind)
  [ -f data/cache/encoder.json ] || run python scripts/encode.py --h5 data/cube_single.h5 --out data/cache --pool 2 --n-eval "$N_EVAL" --batch 128
  run python scripts/check_cache.py --cache8 data/cache
  run python scripts/check_render.py --cache data/cache   # planning renders frames here; warns, never stops
fi

if want base8; then
  stamp base8
  for k in 1 2 4 5 8; do train runs8 "chunk_k${k}_s0" --cache data/cache --arch chunk --k "$k" --seed 0; done
  train runs8 dinowm_k5_s0 --cache data/cache --arch dinowm --k 5 --seed 0
  run python scripts/check_port.py --runs runs8
  run python -c "
import torch
s = torch.load('runs8/chunk_k1_s0.pt', map_location='cpu', weights_only=False)['log'][-1]['sec'] / $STEPS
seeds, span = len('$SEEDS'.split()), $SPAN
chained = sum(span // k for k in (1, 2, 4, 8) if span // k > 1)  # model-equivalents per seed
print('8x8 training: %.3f s/step = %.1f min per model -> compound stage ~%.1f h (%d default + %d equal-span model-equivalents)'
      % (s, s * $STEPS / 60, s * $STEPS * (4 * (seeds - 1) + chained * seeds) / 3600, 4 * (seeds - 1), chained * seeds))"
fi

if want compound; then
  stamp compound
  for s in $SEEDS; do  # seed by seed, cheapest model first, so a finished seed is a complete comparison
    for k in 8 4 2 1; do train runs8 "chunk_k${k}_s$s" --cache data/cache --arch chunk --k "$k" --seed "$s"; done
    for k in 8 4 2 1; do
      u=$((SPAN / k))
      [ $((u * k)) -eq "$SPAN" ] || { echo "SPAN=$SPAN is not a multiple of k=$k"; exit 1; }
      [ "$u" -gt 1 ] || continue   # unroll 1 is the default model
      train runs8 "chunk_k${k}-u${u}_s$s" --cache data/cache --arch chunk --k "$k" --unroll "$u" --seed "$s"
    done
  done
  openloop8
  run python scripts/eval_compounding.py --cache data/cache --ckpts runs8/chunk_k[1248]_s*.pt runs8/chunk_k[1248]-u*_s*.pt \
    --horizon 40 --every 4 --batch 128 --openloop results_a100/openloop.csv --out results_a100/compounding.csv $T
fi

if want cost; then
  stamp cost
  run python scripts/fit_cost.py --cache data/cache --out results_a100/cost_check.md
fi

if want plan8; then
  stamp plan8
  # chosen cost first (with replay and the random null), then the original cost on the same pairs as the control
  run python scripts/eval_plan.py --cache data/cache --ckpts replay random $BASE8 --cost best $PLAN --out results_a100/plan.jsonl
  run python scripts/eval_plan.py --cache data/cache --ckpts $BASE8 --cost latent $PLAN --out results_a100/plan.jsonl
  report results_a100 runs8 data/cache
fi

if want ablate8; then
  stamp ablate8
  # A2 embedding width: DINO-WM at the chunk encoder's 64-d, chunk at DINO-WM's 10-d, chunk width sweep at k=8
  train runs8 dinowm_k5-e64_s0 --cache data/cache --arch dinowm --k 5 --dinowm-emb 64 --seed 0
  train runs8 chunk_k5-emb10_s0 --cache data/cache --arch chunk --k 5 --emb-dim 10 --seed 0
  train runs8 chunk_k8-emb32_s0 --cache data/cache --arch chunk --k 8 --emb-dim 32 --seed 0
  train runs8 chunk_k8-emb128_s0 --cache data/cache --arch chunk --k 8 --emb-dim 128 --seed 0
  # A3 residual prediction, both architectures
  for k in 1 5 8; do train runs8 "chunk_k${k}-res_s0" --cache data/cache --arch chunk --k "$k" --residual --seed 0; done
  train runs8 dinowm_k5-res_s0 --cache data/cache --arch dinowm --k 5 --residual --seed 0
  openloop8   # includes the A4 shuffled-action control
  # A4: a probe that reads the full 8x8 grid instead of pooling to 4x4
  run python scripts/eval_openloop.py --cache data/cache --ckpts runs8/chunk_k1_s0.pt runs8/chunk_k8_s0.pt \
    --horizon 40 --every 2 --batch 128 --probe-grid 8 --out results_a100/openloop_probe8.csv $T
fi

# ---- optional scale-up to 16x16 patches (not in the default STAGES; see the header) -------------------
if want data16; then
  stamp data16
  [ -f data/cache16/encoder.json ] || run python scripts/encode.py --h5 data/cube_single.h5 --out data/cache16 --pool 1 --n-eval "$N_EVAL" --batch 128
  run python scripts/check_cache.py --cache8 data/cache --cache16 data/cache16
fi

if want timing16; then
  stamp timing16
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
  run python scripts/eval_openloop.py --cache data/cache16 --ckpts runs16/*.pt \
    --horizon 80 --every 2 --batch 64 --out results_a100_16x16/openloop.csv $T
  run python scripts/eval_compounding.py --cache data/cache16 --ckpts runs16/*.pt --horizon 40 --every 4 --batch 32 \
    --out results_a100_16x16/compounding.csv $T
  report results_a100_16x16 runs16 data/cache16
fi

if want plan16; then
  stamp plan16
  [ -f data/cache16/cost.pt ] || run python scripts/fit_cost.py --cache data/cache16 --out results_a100_16x16/cost_check.md
  run python scripts/eval_plan.py --cache data/cache16 --ckpts replay random runs16/*_s0.pt --cost best $PLAN \
    --out results_a100_16x16/plan.jsonl
  report results_a100_16x16 runs16 data/cache16
fi
stamp DONE
