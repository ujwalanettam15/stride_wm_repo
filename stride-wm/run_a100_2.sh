#!/usr/bin/env bash
# Second A100 run: why does planning fail? Plus extra seeds for the comparisons the first run did with one seed.
# Run it in the same stride-wm/ folder as the first run (run_a100.sh): it reuses that run's data/cache,
# data/cache/cost.pt and runs8/ checkpoints, and writes everything new to results_a100_2/.
# Submit it as a batch job, which keeps running when the browser or VS Code session closes:
#   bash submit_a100_2.sh      (on the node, outside the container; see that file and the README)
#
# What the first run found (results_a100/): no learned model plans better than chance; a model with half the
# prediction error (k=8 vs k=1) plans no better; a cost that tracks the cube twice as well did not help.
#
# GPU use. NYU Torch cancels a job whose GPU use stays low for 2 hours: below 10% on most nodes, below 50-60% on
# gl*/gh*/ga*/gr* nodes (services.rt.nyu.edu, Slurm "Low GPU Utilization Policy"). The simulator planner uses no
# GPU at all, and three sessions were cancelled while it ran on its own. So it now runs in the BACKGROUND, next to
# the stages that use the GPU, and the job waits for it at the end. Every 30 s the trace file next to the log
# records GPU use; every 10 minutes the log prints its average over the last 2 hours.
#
# Stages (each skips work that is already done, so re-running resumes):
#   GPU stages, in order:
#   setup       versions, GPU, unit tests, first-run inputs present                          minutes
#   seeds       seeds 1 and 2 for the k=5 architecture comparison (E4) and the width       ~1.7 h
#               ablation (A2), which the first run trained with seed 0 only (12 models).
#   near        k = 1, 2, 4, 8 planning toward closer goals (16 and 24 steps; both costs),    ~0.3 h per distance
#               with replay and random at each distance.
#               EXPLANATION TESTED: "long-range predictions are too blurry" (cube error at 40 steps is
#               3.5-6.5 cm against a 4 cm success radius). Supported if success rises at closer goals.
#   budget      k=8 with stable-worldmodel's default search, 300 samples x 30 iterations      ~0.5 h
#               (9x the first run's), both costs, goals 40 steps away.
#               EXPLANATION TESTED: "the planner does not search enough".
#   eval        open loop for every model (adds the new seeds); then the final report         ~0.5 h
#   Simulator stages, one after another in the background (CPU; oracle_z also renders and runs DINOv2 on the GPU):
#   oracle      the same planner with the SIMULATOR as its model and the true cube distance as its cost
#               (perfect predictions, perfect cost), goals 40 steps away.
#               EXPLANATION TESTED: "planning fails because of the world model". If even this fails, the
#               planner/task setup is the bottleneck, not the model.
#   near_oracle the same at the closer goals (NEAR).
#   oracle_z    the simulator planner scored with our learned costs on rendered frames (latent, and the cost
#               fit_cost.py chose): perfect predictions, our cost. Separates the cost from the model. Runs on the
#               first ZPAIRS of the same pairs.
#   Optional, not in the default STAGES:
#   oracle_big  the simulator planner with the 300 x 30 budget. Uses no GPU: run it next to GPU stages, or on a
#               CPU-only node with ALLOW_CPU=1 MUJOCO_GL=disable (the true-cube planner never renders).
#
# Planning rows go to results_a100_2/plan_main.jsonl (GPU stages) and plan_sim.jsonl (simulator stages), one file
# per writer, and are merged into results_a100_2/plan.jsonl for the reports. That file starts with the first run's
# rows (same 30 pairs at 40 steps), so its tables compare old and new side by side.
set -euo pipefail
cd "$(dirname "$0")"
STEPS=${STEPS:-20000}       # training steps per model (as in the first run)
PAIRS=${PAIRS:-30}          # planning start/goal pairs per condition (as in the first run)
ZPAIRS=${ZPAIRS:-10}        # pairs for oracle_z (several minutes of simulation per plan)
TF32=${TF32:-1}             # 1 = TF32 tensor cores on the A100; 0 = plain fp32
DRY_RUN=${DRY_RUN:-0}       # 1 = print the commands without running them
STAGES=${STAGES:-"setup seeds near budget eval oracle near_oracle oracle_z"}
NEAR=${NEAR:-"16 24"}       # closer goal distances (env steps); k = 1, 2, 4, 8 must divide them
RES=results_a100_2
T=""; [ "$TF32" = 1 ] && T="--tf32"
NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
SIMW=${WORKERS:-$(( NCPU > 5 ? NCPU - 4 : 1 ))}   # simulator-planner processes; 4 cores stay with the GPU stages
ZW=${ZWORKERS:-$SIMW}                             # the same for oracle_z
SEARCH="--samples 100 --iters 10 --topk 10"       # the first run's search budget
BUDGET="--pairs $PAIRS $SEARCH"
BIG="--pairs $PAIRS --samples 300 --iters 30 --topk 30"   # stable-worldmodel's default CEM budget
MAIN=$RES/plan_main.jsonl   # written by the GPU stages
SIM=$RES/plan_sim.jsonl     # written by the simulator stages
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
  echo "settings: STAGES=\"$STAGES\" STEPS=$STEPS PAIRS=$PAIRS ZPAIRS=$ZPAIRS NEAR=\"$NEAR\" TF32=$TF32" \
       "CPUs=$NCPU simulator processes=$SIMW (oracle_z $ZW)"
  python -m pip freeze > "${LOG%.log}_packages.txt" 2>/dev/null || true
fi

# Stop if no GPU is visible (the scripts would fall back to the CPU silently). ALLOW_CPU=1 overrides.
if [ "$DRY_RUN" != 1 ] && [ "${ALLOW_CPU:-0}" != 1 ]; then
  python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'No CUDA GPU visible. Check that the session or job has a GPU and that the container was started with --nv. ALLOW_CPU=1 overrides.')"
fi
case "$(hostname -s 2>/dev/null)" in
  gl*|gh*|ga*|gr*) echo "WARNING: on $(hostname -s) NYU Torch cancels jobs below 50-60% GPU use, which planning cannot reach."
                   echo "         Request the node type of the earlier sessions (b-*). Rows already written are kept.";;
esac

# MuJoCo renders through EGL on the GPU; point EGL at NVIDIA's library if the container has no vendor file
# (NYU Torch's code-server image). Same fix as run_a100.sh.
if [ "$DRY_RUN" != 1 ] && [ "$(uname)" = Linux ] && [ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ] \
   && ! ls /usr/share/glvnd/egl_vendor.d/*nvidia* /etc/glvnd/egl_vendor.d/*nvidia* >/dev/null 2>&1; then
  echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > "$HOME/nvidia_egl.json"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$HOME/nvidia_egl.json"
  echo "EGL: no NVIDIA vendor file in the system, using $__EGL_VENDOR_LIBRARY_FILENAMES"
fi
[ "$(uname)" = Linux ] && export MUJOCO_GL=${MUJOCO_GL:-egl}

# Trace every 30 s: the job's memory use, limit and out-of-memory kills (Linux cgroup v2), and GPU use. Every
# 10 minutes, print the average GPU use over the last 2 hours, the quantity NYU Torch's cancellation rule checks.
if [ "$DRY_RUN" != 1 ] && [ -r /proc/self/cgroup ]; then
  CG=/sys/fs/cgroup$(sed -n 's/^0:://p' /proc/self/cgroup); MEMCG=$CG
  d=$CG; while [ -n "$d" ] && [ "$d" != /sys/fs/cgroup ]; do   # nearest enclosing cgroup with a memory limit
    v=$(cat "$d/memory.max" 2>/dev/null || true); if [ -n "$v" ] && [ "$v" != max ]; then MEMCG=$d; break; fi; d=${d%/*}
  done
  TRACE="${LOG%.log}_trace.tsv"
  echo -e "time\tused_GB\tlimit_GB\toom_kills\tnode_available_GB\tmy_processes\tgpu_util_pct\tgpu_mem_MiB" > "$TRACE"
  ( set +eu +o pipefail   # a failed read must not stop the trace
    i=0
    while true; do
      gb() { awk -v b="$1" 'BEGIN { if (b ~ /^[0-9]+$/) printf "%.1f", b / 2^30; else print b }'; }
      used=$(gb "$(cat "$MEMCG/memory.current" 2>/dev/null || echo NA)"); lim=$(gb "$(cat "$MEMCG/memory.max" 2>/dev/null || echo NA)")
      oom=$(awk '$1 == "oom_kill" {print $2}' "$MEMCG/memory.events" 2>/dev/null); avail=$(awk '/MemAvailable/ {printf "%.1f", $2 / 2^20}' /proc/meminfo)
      gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' | tr ',' '\t')
      echo -e "$(date +%T)\t$used\t$lim\t${oom:-NA}\t$avail\t$(ps -u "$(id -u)" --no-headers 2>/dev/null | wc -l)\t${gpu:-NA\tNA}" >> "$TRACE"
      i=$((i + 1))
      if [ $((i % 20)) = 0 ]; then
        avg=$(tail -n 240 "$TRACE" | awk -F'\t' '$7 ~ /^[0-9]+$/ {s += $7; n++} END {if (n) printf "%.0f", s / n; else print "NA"}')
        echo "[gpu] average GPU use over the last $(( i >= 240 ? 120 : i / 2 )) min: ${avg}% (NYU Torch cancels below 10% over 2 h on most nodes)"
      fi
      sleep 30
    done ) &
  TRACEPID=$!; trap 'kill $TRACEPID 2>/dev/null || true' EXIT
  echo "trace: memory limit $(awk -v b="$(cat "$MEMCG/memory.max" 2>/dev/null || echo unknown)" 'BEGIN { if (b ~ /^[0-9]+$/) printf "%.1f GB", b / 2^30; else print b }'), memory and GPU use every 30 s in $TRACE"
fi

T_START=$(date +%s)
stamp() { echo "== $(date '+%F %T') [+$(( ($(date +%s) - T_START) / 60 )) min] $*  [free disk: $(df -h . | awk 'NR==2{print $4}')]"; }
run() { echo "+ $*"; [ "$DRY_RUN" = 1 ] || "$@"; }
want() { [[ " $STAGES " == *" $1 "* ]]; }
train() {  # $1 = checkpoint name (must match train.py's naming), rest = train.py args
  local name=$1; shift
  if [ -f "runs8/$name.pt" ]; then echo "skip runs8/$name (exists)"; return; fi
  stamp "train runs8/$name"; run python scripts/train.py "$@" --steps "$STEPS" --out runs8 $T
}
report() {  # called by the GPU stages only; the merge skips a simulator row that is being written right now
  run python scripts/plan_log.py merge "$RES"
  if [ "$DRY_RUN" != 1 ] && [ ! -f "$RES/openloop.csv" ]; then echo "no $RES/openloop.csv yet: report skipped"; return; fi
  run python scripts/report.py --openloop "$RES/openloop.csv" --plan "$RES/plan.jsonl" --runs runs8 --out "$RES"
  run python scripts/analyze_failures.py --openloop "$RES/openloop.csv" --plan "$RES/plan.jsonl" --cache data/cache \
    --out "$RES/failure_analysis.md"
}

# Planning rows: one file per writer (see the top). The first time, divide the existing plan.jsonl between them
# (or start from the first run's rows). Open-loop rows start from the first run's too.
run python scripts/plan_log.py split "$RES" --first results_a100/plan.jsonl
if [ "$DRY_RUN" != 1 ]; then
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
fi

# ---------------------------------------------------------------- simulator stages, in the background
sim_stages() {
  if want oracle; then
    stamp "oracle: simulator as model, true cube distance, goals 40 steps away"
    run python scripts/eval_oracle.py --cache data/cache --horizon 40 --cost true_cube $BUDGET --workers "$SIMW" --out "$SIM"
  fi
  if want near_oracle; then
    for H in $NEAR; do
      stamp "near_oracle: simulator as model, true cube distance, goals $H steps away"
      run python scripts/eval_oracle.py --cache data/cache --horizon "$H" --cost true_cube $BUDGET --workers "$SIMW" --out "$SIM"
    done
  fi
  if want oracle_z; then
    for cost in latent best; do
      stamp "oracle_z: simulator as model, our '$cost' cost on rendered frames, $ZPAIRS pairs"
      run python scripts/eval_oracle.py --cache data/cache --horizon 40 --cost "$cost" --pairs "$ZPAIRS" $SEARCH \
        --workers "$ZW" --out "$SIM"
    done
  fi
  if want oracle_big; then
    stamp "oracle_big: simulator as model with the 300 x 30 budget"
    run python scripts/eval_oracle.py --cache data/cache --horizon 40 --cost true_cube $BIG --workers "$SIMW" --out "$SIM"
  fi
}
SIM_PID=""; SIM_LOG=""
if want oracle || want near_oracle || want oracle_z || want oracle_big; then
  if [ "$DRY_RUN" = 1 ]; then
    stamp "simulator stages (in a real run: in the background, next to the GPU stages)"; sim_stages
  else
    SIM_LOG="${LOG%.log}_sim.log"
    stamp "simulator planner: in the BACKGROUND with $SIMW processes, next to the GPU stages; log $SIM_LOG"
    sim_stages > "$SIM_LOG" 2>&1 &
    SIM_PID=$!
    if ! want seeds && ! want near && ! want budget && ! want eval; then
      echo "WARNING: no GPU stage in STAGES, so the GPU stays idle while the simulator planner runs."
      echo "         NYU Torch cancels such jobs. Add GPU stages, or for true-cube stages use a CPU-only node with"
      echo "         ALLOW_CPU=1 MUJOCO_GL=disable."
    fi
  fi
fi
wait_sim() {
  [ -n "$SIM_PID" ] || return 0
  stamp "waiting for the simulator planner in the background (log $SIM_LOG)"
  if wait "$SIM_PID"; then SIM_PID=""; stamp "simulator planner finished"
  else SIM_PID=""; echo "simulator planner failed: see $SIM_LOG"; exit 1; fi
}

# ---------------------------------------------------------------- GPU stages, in the foreground
if want seeds; then  # seeds 1 and 2 for E4 and A2
  for s in 1 2; do
    train "dinowm_k5_s$s" --cache data/cache --arch dinowm --k 5 --seed "$s"
    train "chunk_k5_s$s" --cache data/cache --arch chunk --k 5 --seed "$s"
    train "dinowm_k5-e64_s$s" --cache data/cache --arch dinowm --k 5 --dinowm-emb 64 --seed "$s"
    train "chunk_k5-emb10_s$s" --cache data/cache --arch chunk --k 5 --emb-dim 10 --seed "$s"
    train "chunk_k8-emb32_s$s" --cache data/cache --arch chunk --k 8 --emb-dim 32 --seed "$s"
    train "chunk_k8-emb128_s$s" --cache data/cache --arch chunk --k 8 --emb-dim 128 --seed "$s"
  done
fi

if want near; then
  for H in $NEAR; do
    stamp "near: learned models, goals $H steps away"
    run python scripts/eval_plan.py --cache data/cache --ckpts replay random $NEARK --cost best --horizon "$H" $BUDGET $T \
      --out "$MAIN"
    run python scripts/eval_plan.py --cache data/cache --ckpts $NEARK --cost latent --horizon "$H" $BUDGET $T --out "$MAIN"
  done
  report
fi

if want budget; then
  stamp "budget: k=8 with 9x the search (300 samples x 30 iterations)"
  run python scripts/eval_plan.py --cache data/cache --ckpts runs8/chunk_k8_s0.pt --cost best --horizon 40 $BIG $T --out "$MAIN"
  run python scripts/eval_plan.py --cache data/cache --ckpts runs8/chunk_k8_s0.pt --cost latent --horizon 40 $BIG $T --out "$MAIN"
  report
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
fi

wait_sim
report
stamp DONE
