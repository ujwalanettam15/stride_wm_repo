#!/usr/bin/env bash
# Second A100 run: why does planning fail? Plus extra seeds for the comparisons the first run did with one seed.
# Run it in the same stride-wm/ folder as the first run (run_a100.sh): it reuses that run's data/cache,
# data/cache/cost.pt and runs8/ checkpoints, and writes everything new to results_a100_2/.
#
# What the first run found (results_a100/): no learned model plans better than chance; a model with half the
# prediction error (k=8 vs k=1) plans no better; a cost that tracks the cube twice as well did not help.
# Stages, in priority order (each skips work that is already done, so re-running resumes):
#   setup     versions, GPU, unit tests, first-run inputs present                         minutes
#   oracle    the same planner with the SIMULATOR as its model and the true cube          ~0.3 h
#             distance as its cost (perfect predictions, perfect cost), goals 40 steps away.
#             EXPLANATION TESTED: "planning fails because of the world model".
#             If even this fails, the planner/task setup is the bottleneck, not the model.
#   near      k = 1, 2, 4, 8 planning toward closer goals (16 and 24 steps; both costs),   ~0.7 h
#             with replay, random and the simulator planner at each distance.
#             EXPLANATION TESTED: "long-range predictions are too blurry" (cube error at 40 steps
#             is 3.5-6.5 cm against a 4 cm success radius). Supported if success rises at closer goals.
#   budget    k=8 with stable-worldmodel's default search budget, 300 samples x 30          ~0.4 h
#             iterations (9x the first run's), both costs, goals 40 steps away.
#             EXPLANATION TESTED: "the planner does not search enough".
#   oracle_z  the simulator planner scored with our learned costs on rendered frames         ~1.5 h
#             (latent, probe_cube): perfect predictions, our cost. Separates cost from model.
#   seeds     seeds 1 and 2 for the k=5 architecture comparison (E4) and the embedding-      ~2 h
#             width ablation (A2), which the first run trained with seed 0 only (12 models).
#   eval      open loop for every model (adds the new seeds), report                          ~0.5 h
# Optional, not in the default STAGES: oracle_big = the simulator planner with the 300 x 30 budget (~2 h);
# worth it only if `oracle` fails, to tell search budget from task difficulty.
#
# Hours are estimates for one A100 with 8 CPU cores; total ~5-6 h. results_a100_2/plan.jsonl starts as a copy
# of the first run's planning rows (same 30 pairs at 40 steps), so its tables compare old and new side by side.
set -euo pipefail
cd "$(dirname "$0")"
STEPS=${STEPS:-20000}       # training steps per model (as in the first run)
PAIRS=${PAIRS:-30}          # planning start/goal pairs per condition (as in the first run)
TF32=${TF32:-1}             # 1 = TF32 tensor cores on the A100; 0 = plain fp32
DRY_RUN=${DRY_RUN:-0}       # 1 = print the commands without running them
STAGES=${STAGES:-"setup oracle near budget oracle_z seeds eval"}
NEAR=${NEAR:-"16 24"}       # closer goal distances (env steps); k = 1, 2, 4, 8 must divide them
RES=results_a100_2
T=""; [ "$TF32" = 1 ] && T="--tf32"
W=""; [ -n "${WORKERS:-}" ] && W="--workers $WORKERS"      # simulator-planner processes (default: all CPUs)
BUDGET="--pairs $PAIRS --samples 100 --iters 10 --topk 10"  # the first run's search budget
BIG="--pairs $PAIRS --samples 300 --iters 30 --topk 30"     # stable-worldmodel's default CEM budget
NEARK="runs8/chunk_k1_s0.pt runs8/chunk_k2_s0.pt runs8/chunk_k4_s0.pt runs8/chunk_k8_s0.pt"
FIRST="runs8/chunk_k1_s0.pt runs8/chunk_k2_s0.pt runs8/chunk_k4_s0.pt runs8/chunk_k5_s0.pt runs8/chunk_k8_s0.pt runs8/dinowm_k5_s0.pt"
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore

# Keep a copy of everything this run prints next to its results (logs/ is not in git; this folder is).
if [ "$DRY_RUN" != 1 ]; then
  mkdir -p "$RES/run_logs"
  LOG="$RES/run_logs/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}.log"
  exec > >(tee -a "$LOG") 2>&1
  echo "run_a100_2.sh | commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown) | host ${HOSTNAME:-unknown} |" \
       "job ${SLURM_JOB_ID:-none} | log $LOG"
  echo "settings: STAGES=\"$STAGES\" STEPS=$STEPS PAIRS=$PAIRS NEAR=\"$NEAR\" TF32=$TF32 WORKERS=${WORKERS:-auto}"
  python -m pip freeze > "${LOG%.log}_packages.txt" 2>/dev/null || true
fi

# Stop if no GPU is visible (the scripts would fall back to the CPU silently). ALLOW_CPU=1 overrides.
if [ "$DRY_RUN" != 1 ] && [ "${ALLOW_CPU:-0}" != 1 ]; then
  python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'No CUDA GPU visible. Check that the session or job has a GPU and that the container was started with --nv. ALLOW_CPU=1 overrides.')"
fi

# MuJoCo renders through EGL on the GPU; point EGL at NVIDIA's library if the container has no vendor file
# (NYU Torch's code-server image). Same fix as run_a100.sh.
if [ "$DRY_RUN" != 1 ] && [ "$(uname)" = Linux ] && [ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ] \
   && ! ls /usr/share/glvnd/egl_vendor.d/*nvidia* /etc/glvnd/egl_vendor.d/*nvidia* >/dev/null 2>&1; then
  echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > "$HOME/nvidia_egl.json"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$HOME/nvidia_egl.json"
  echo "EGL: no NVIDIA vendor file in the system, using $__EGL_VENDOR_LIBRARY_FILENAMES"
fi
[ "$(uname)" = Linux ] && export MUJOCO_GL=${MUJOCO_GL:-egl}

T_START=$(date +%s)
stamp() { echo "== $(date '+%F %T') [+$(( ($(date +%s) - T_START) / 60 )) min] $*  [free disk: $(df -h . | awk 'NR==2{print $4}')]"; }
run() { echo "+ $*"; [ "$DRY_RUN" = 1 ] || "$@"; }
want() { [[ " $STAGES " == *" $1 "* ]]; }
train() {  # $1 = checkpoint name (must match train.py's naming), rest = train.py args
  local name=$1; shift
  if [ -f "runs8/$name.pt" ]; then echo "skip runs8/$name (exists)"; return; fi
  stamp "train runs8/$name"; run python scripts/train.py "$@" --steps "$STEPS" --out runs8 $T
}
report() {
  if [ "$DRY_RUN" != 1 ] && [ ! -f "$RES/openloop.csv" ]; then echo "no $RES/openloop.csv yet: report skipped"; return; fi
  run python scripts/report.py --openloop "$RES/openloop.csv" --plan "$RES/plan.jsonl" --runs runs8 --out "$RES"
  run python scripts/analyze_failures.py --openloop "$RES/openloop.csv" --plan "$RES/plan.jsonl" --cache data/cache \
    --out "$RES/failure_analysis.md"
}

# Start from the first run's results so the tables show old and new side by side; new rows are appended.
if [ "$DRY_RUN" != 1 ]; then
  [ -f "$RES/plan.jsonl" ] || { [ -f results_a100/plan.jsonl ] && cp results_a100/plan.jsonl "$RES/plan.jsonl"; } || true
  [ -f "$RES/openloop.csv" ] || { [ -f results_a100/openloop.csv ] && cp results_a100/openloop.csv "$RES/openloop.csv"; } || true
fi

if want setup; then
  stamp setup
  run python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
  run python -m pytest -q tests
  missing=""
  for f in data/cache/encoder.json data/cache/cost.pt $FIRST; do [ -f "$f" ] || missing="$missing $f"; done
  if [ -n "$missing" ] && [ "$DRY_RUN" != 1 ]; then
    echo "missing from the first run:$missing"
    echo "run this script in the stride-wm/ folder where run_a100.sh ran (it needs that run's cache, cost.pt and runs8/)"
    exit 1
  fi
  run python -c "import os; n = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count(); print('CPU cores for the simulator planner:', n)"
fi

if want oracle; then
  stamp "oracle: simulator as model, true cube distance, goals 40 steps away"
  run python scripts/eval_oracle.py --cache data/cache --horizon 40 --cost true_cube $BUDGET $W --out "$RES/plan.jsonl"
  report
fi

if want near; then
  for H in $NEAR; do
    stamp "near: goals $H steps away"
    run python scripts/eval_plan.py --cache data/cache --ckpts replay random $NEARK --cost best --horizon "$H" $BUDGET $T \
      --out "$RES/plan.jsonl"
    run python scripts/eval_plan.py --cache data/cache --ckpts $NEARK --cost latent --horizon "$H" $BUDGET $T \
      --out "$RES/plan.jsonl"
    run python scripts/eval_oracle.py --cache data/cache --horizon "$H" --cost true_cube $BUDGET $W --out "$RES/plan.jsonl"
  done
  report
fi

if want budget; then
  stamp "budget: k=8 with 9x the search (300 samples x 30 iterations)"
  run python scripts/eval_plan.py --cache data/cache --ckpts runs8/chunk_k8_s0.pt --cost best --horizon 40 $BIG $T \
    --out "$RES/plan.jsonl"
  run python scripts/eval_plan.py --cache data/cache --ckpts runs8/chunk_k8_s0.pt --cost latent --horizon 40 $BIG $T \
    --out "$RES/plan.jsonl"
  report
fi

if want oracle_z; then
  stamp "oracle_z: simulator as model, our learned costs on rendered frames"
  for cost in latent best; do
    run python scripts/eval_oracle.py --cache data/cache --horizon 40 --cost "$cost" $BUDGET ${ZWORKERS:+--workers $ZWORKERS} \
      --out "$RES/plan.jsonl"
  done
  report
fi

if want oracle_big; then
  stamp "oracle_big: simulator as model with the 300 x 30 budget"
  run python scripts/eval_oracle.py --cache data/cache --horizon 40 --cost true_cube $BIG $W --out "$RES/plan.jsonl"
  report
fi

if want seeds; then
  stamp "seeds: seeds 1 and 2 for E4 (k=5 architectures) and A2 (embedding width)"
  for s in 1 2; do
    train "dinowm_k5_s$s" --cache data/cache --arch dinowm --k 5 --seed "$s"
    train "chunk_k5_s$s" --cache data/cache --arch chunk --k 5 --seed "$s"
    train "dinowm_k5-e64_s$s" --cache data/cache --arch dinowm --k 5 --dinowm-emb 64 --seed "$s"
    train "chunk_k5-emb10_s$s" --cache data/cache --arch chunk --k 5 --emb-dim 10 --seed "$s"
    train "chunk_k8-emb32_s$s" --cache data/cache --arch chunk --k 8 --emb-dim 32 --seed "$s"
    train "chunk_k8-emb128_s$s" --cache data/cache --arch chunk --k 8 --emb-dim 128 --seed "$s"
  done
fi

if want eval; then
  stamp "eval: open loop for every model"
  run python scripts/eval_openloop.py --cache data/cache --ckpts runs8/*.pt --horizon 40 --every 2 --batch 128 \
    --out "$RES/openloop_main.csv" $T
  # the shuffled-action control is unchanged from the first run (same seed-0 models): reuse its rows
  run python -c "
import pandas as pd
a = pd.read_csv('$RES/openloop_main.csv')
parts = [a]
try:
    b = pd.read_csv('results_a100/openloop_shuf.csv'); parts.append(b[b.arch.str.endswith('-shuf')])
except FileNotFoundError:
    print('results_a100/openloop_shuf.csv not found: no shuffled-action rows in this report')
pd.concat(parts).to_csv('$RES/openloop.csv', index=False)"
  report
fi

stamp DONE
