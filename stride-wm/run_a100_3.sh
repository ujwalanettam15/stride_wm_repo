#!/usr/bin/env bash
# Third A100 run: fix the planning cost, then ask again whether prediction stride matters for planning.
# Run it in the same stride-wm/ folder as the first two runs: it reuses data/cache, data/cache/cost.pt, runs8/ and
# results_a100_2/, and writes everything new to results_a100_3/ (plus data/cache_rand/ and data/cache/cost_rand.pt).
# Submit it as a batch job:   bash submit_a100_3.sh      (on the node, outside the container; see the README)
#
# What the second run found (results_a100_2/): with perfect predictions (the simulator as the model) the planner
# solves 7/10 pairs with the true cube distance but 2/10 with either learned cost, so the cost is the bottleneck.
# Likely reason: the probe_cube cost reads the cube position with a linear probe fit on expert frames only, where the
# cube rides in the gripper, so the probe can place the cube by looking at the arm. The planner's candidates move the
# arm without the cube.
#
# Stages (each skips work that is already done, so re-running resumes):
#   setup      versions, GPU, unit tests, inputs from the first two runs present                     minutes
#   old        BACKGROUND from the start (GPU): k = 1, 2, 4, 8 seeds 1 and 2 with the ORIGINAL probe cost, so the
#              old-vs-new comparison has 3 seeds for both costs (seed 0 is in results_a100_2)          ~45 min
#   rand_data  frames from random, held-random and noisy-expert actions, with their true cube and arm state
#              (scripts/collect_rand.py -> data/cache_rand)                                            ~10 min
#   fit        refit the cube probe on expert + off-expert frames; linear (4x4 and 8x8 grids) and a small MLP; keep
#              the lowest held-out off-expert cube error (scripts/fit_probe_rand.py -> data/cache/cost_rand.pt,
#              results_a100_3/cost_rand_check.md)                                                      ~5 min
#   oracle_r   BACKGROUND after fit: perfect predictions + the refit probe, first ZPAIRS of the same pairs
#              QUESTION: does the new cost let the planner solve the task when predictions are perfect? Compare with
#              7/10 (true cube distance) and 2/10 (old probe) on the first 10 pairs.                     ~1 h
#   new        k = 1, 2, 4, 8 x seeds 0, 1, 2 with the refit probe, one process per seed, side by side
#              QUESTION: once the cost works, does stride change planning success?                     ~45 min
# Then: report (results_a100_3/summary.md, failure_analysis.md), DONE. Expected total: ~1.5 h.
#
# GPU use: NYU Torch cancels jobs below 10% GPU use over 2 hours. Planning processes run side by side from the first
# minute (old), and the trace next to the log prints the 2-hour average every 10 minutes.
set -euo pipefail
cd "$(dirname "$0")"
PAIRS=${PAIRS:-30}          # planning pairs per model (as before)
ZPAIRS=${ZPAIRS:-20}        # pairs for oracle_r; the first 10 are the pairs of results_a100_2's oracle_z rows
RSTARTS=${RSTARTS:-3000}    # off-expert start frames from training episodes (x 5 depths = frames)
REVAL=${REVAL:-600}         # off-expert start frames from held-out episodes (the check)
TF32=${TF32:-1}
DRY_RUN=${DRY_RUN:-0}
STAGES=${STAGES:-"setup old rand_data fit oracle_r new"}
SEEDS_NEW=${SEEDS_NEW:-"0 1 2"}
SEEDS_OLD=${SEEDS_OLD:-"1 2"}
RES=results_a100_3
PREV=results_a100_2
T=""; [ "$TF32" = 1 ] && T="--tf32"
NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
NOLD=$(echo $SEEDS_OLD | wc -w); NNEW=$(echo $SEEDS_NEW | wc -w)
CW=${CWORKERS:-$(( NCPU - NOLD - 2 > 1 ? NCPU - NOLD - 2 : 1 ))}        # collection processes (old runs alongside)
ZW=${ZWORKERS:-$(( NCPU - NOLD - NNEW > 1 ? NCPU - NOLD - NNEW : 1 ))}  # oracle_r processes (old and new alongside)
SEARCH="--samples 100 --iters 10 --topk 10"   # the search budget of all earlier planning rows
CK() { local s=$1; echo "runs8/chunk_k1_s$s.pt runs8/chunk_k2_s$s.pt runs8/chunk_k4_s$s.pt runs8/chunk_k8_s$s.pt"; }
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore

if [ "$DRY_RUN" != 1 ]; then
  mkdir -p "$RES/run_logs"
  LOG="$RES/run_logs/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}.log"
  exec > >(tee -a "$LOG") 2>&1
  echo "run_a100_3.sh | commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown) | host ${HOSTNAME:-unknown} |" \
       "job ${SLURM_JOB_ID:-none} | log $LOG"
  echo "settings: STAGES=\"$STAGES\" PAIRS=$PAIRS ZPAIRS=$ZPAIRS RSTARTS=$RSTARTS REVAL=$REVAL SEEDS_NEW=\"$SEEDS_NEW\"" \
       "SEEDS_OLD=\"$SEEDS_OLD\" TF32=$TF32 CPUs=$NCPU collection processes=$CW oracle processes=$ZW"
  python -m pip freeze > "${LOG%.log}_packages.txt" 2>/dev/null || true
else
  LOG=$RES/run_logs/dry_run.log
fi

if [ "$DRY_RUN" != 1 ] && [ "${ALLOW_CPU:-0}" != 1 ]; then
  python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'No CUDA GPU visible. Check that the session or job has a GPU and that the container was started with --nv. ALLOW_CPU=1 overrides.')"
fi
case "$(hostname -s 2>/dev/null)" in
  gl*|gh*|ga*|gr*) echo "WARNING: on $(hostname -s) NYU Torch cancels jobs below 50-60% GPU use, which planning cannot reach."
                   echo "         Request the node type of the earlier sessions (b-*). Rows already written are kept.";;
esac

# MuJoCo renders through EGL on the GPU; point EGL at NVIDIA's library if the container has no vendor file.
if [ "$DRY_RUN" != 1 ] && [ "$(uname)" = Linux ] && [ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ] \
   && ! ls /usr/share/glvnd/egl_vendor.d/*nvidia* /etc/glvnd/egl_vendor.d/*nvidia* >/dev/null 2>&1; then
  echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > "$HOME/nvidia_egl.json"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$HOME/nvidia_egl.json"
  echo "EGL: no NVIDIA vendor file in the system, using $__EGL_VENDOR_LIBRARY_FILENAMES"
fi
[ "$(uname)" = Linux ] && export MUJOCO_GL=${MUJOCO_GL:-egl}

# Trace every 30 s (memory and GPU use); every 10 minutes the log prints the 2-hour GPU average.
if [ "$DRY_RUN" != 1 ] && [ -r /proc/self/cgroup ]; then
  CG=/sys/fs/cgroup$(sed -n 's/^0:://p' /proc/self/cgroup); MEMCG=$CG
  d=$CG; while [ -n "$d" ] && [ "$d" != /sys/fs/cgroup ]; do
    v=$(cat "$d/memory.max" 2>/dev/null || true); if [ -n "$v" ] && [ "$v" != max ]; then MEMCG=$d; break; fi; d=${d%/*}
  done
  TRACE="${LOG%.log}_trace.tsv"
  echo -e "time\tused_GB\tlimit_GB\toom_kills\tnode_available_GB\tmy_processes\tgpu_util_pct\tgpu_mem_MiB" > "$TRACE"
  ( set +eu +o pipefail
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
fi

T_START=$(date +%s)
stamp() { echo "== $(date '+%F %T') [+$(( ($(date +%s) - T_START) / 60 )) min] $*  [free disk: $(df -h . | awk 'NR==2{print $4}')]"; }
run() { echo "+ $*"; [ "$DRY_RUN" = 1 ] || "$@"; }
want() { [[ " $STAGES " == *" $1 "* ]]; }
PIDS=""; FAILED=""
bg() {  # bg <name> <command...>: run in the background, output to <log>_<name>.log; collected by wait_all
  local name=$1; shift
  echo "+ [background: $name, log ${LOG%.log}_$name.log] $*"
  if [ "$DRY_RUN" = 1 ]; then return; fi
  ( "$@" ) > "${LOG%.log}_$name.log" 2>&1 &
  PIDS="$PIDS $!:$name"
}
wait_all() {
  for p in $PIDS; do
    if wait "${p%%:*}"; then stamp "finished: ${p#*:}"; else FAILED="$FAILED ${p#*:}"; echo "FAILED: ${p#*:} (see ${LOG%.log}_${p#*:}.log)"; fi
  done
  PIDS=""
}
report() {
  run python scripts/plan_log.py merge "$RES"
  run python scripts/report.py --openloop "$RES/openloop.csv" --plan "$RES/plan.jsonl" --runs runs8 --out "$RES"
  run python scripts/analyze_failures.py --openloop "$RES/openloop.csv" --plan "$RES/plan.jsonl" --cache data/cache \
    --out "$RES/failure_analysis.md"
}
plan_seeds() {  # plan_seeds <cost> <tag> <seed>: one seed's k = 1, 2, 4, 8 models, its own results file
  local cost=$1 tag=$2 s=$3
  OMP_NUM_THREADS=${PLAN_THREADS:-2} MKL_NUM_THREADS=${PLAN_THREADS:-2} python scripts/eval_plan.py --cache data/cache --ckpts $(CK "$s") --cost "$cost" --horizon 40 --pairs "$PAIRS" $SEARCH \
    $T --out "$RES/plan_main_${tag}_s$s.jsonl"
}

# Planning rows start from the second run's (same 30 pairs at 40 steps), so the tables show old and new side by side.
run python scripts/plan_log.py split "$RES" --first "$PREV/plan.jsonl"
if [ "$DRY_RUN" != 1 ] && [ ! -f "$RES/openloop.csv" ]; then cp "$PREV/openloop.csv" "$RES/openloop.csv"; fi

if want setup; then
  stamp setup
  run python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
  run python -m pytest -q tests
  missing=""
  for f in data/cache/encoder.json data/cache/cost.pt "$PREV/plan.jsonl" "$PREV/openloop.csv" $(CK 0) $(CK 1) $(CK 2); do
    [ -f "$f" ] || missing="$missing $f"
  done
  if [ -n "$missing" ] && [ "$DRY_RUN" != 1 ]; then
    echo "missing:$missing"
    echo "run this script in the stride-wm/ folder of the first two runs (git pull first: results_a100_2/ is in git)"
    exit 1
  fi
fi

if want old; then
  stamp "old: original probe cost, seeds $SEEDS_OLD, in the background"
  for s in $SEEDS_OLD; do bg "old_s$s" plan_seeds best old "$s"; done
fi

if want rand_data; then
  stamp "rand_data: off-expert frames ($RSTARTS + $REVAL starts x 5 depths)"
  run python scripts/collect_rand.py --cache data/cache --out data/cache_rand --starts "$RSTARTS" --eval-starts "$REVAL" \
    --workers "$CW"
fi

if want fit; then
  if [ -f data/cache/cost_rand.pt ] && [ -f "$RES/cost_rand_check.md" ]; then
    echo "skip fit: data/cache/cost_rand.pt exists (delete it and $RES/cost_rand_check.md to refit)"
  else
    stamp "fit: refit the cube probe"
    run python scripts/fit_probe_rand.py --cache data/cache --rand data/cache_rand --out "$RES/cost_rand_check.md" ${FIT_ARGS:-}
  fi
fi

if want oracle_r; then
  stamp "oracle_r: perfect predictions + the refit probe, $ZPAIRS pairs, $ZW processes, in the background"
  bg sim python scripts/eval_oracle.py --cache data/cache --horizon 40 --cost probe_cube_rand --pairs "$ZPAIRS" $SEARCH \
    --workers "$ZW" --out "$RES/plan_sim.jsonl"
fi

if want new; then
  stamp "new: refit probe cost, k = 1, 2, 4, 8, seeds $SEEDS_NEW side by side"
  NEWPIDS=""
  for s in $SEEDS_NEW; do
    echo "+ [seed $s, log ${LOG%.log}_new_s$s.log] plan_seeds probe_cube_rand new $s"
    if [ "$DRY_RUN" != 1 ]; then plan_seeds probe_cube_rand new "$s" > "${LOG%.log}_new_s$s.log" 2>&1 & NEWPIDS="$NEWPIDS $!:new_s$s"; fi
  done
  for p in $NEWPIDS; do
    if wait "${p%%:*}"; then stamp "finished: ${p#*:}"; else FAILED="$FAILED ${p#*:}"; echo "FAILED: ${p#*:} (see ${LOG%.log}_${p#*:}.log)"; fi
  done
  report
fi

stamp "waiting for the background stages"
wait_all
report
if [ -n "$FAILED" ]; then stamp "DONE with failures:$FAILED"; exit 1; fi
stamp DONE
