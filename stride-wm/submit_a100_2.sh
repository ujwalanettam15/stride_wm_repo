#!/usr/bin/env bash
# Submit run_a100_2.sh as a Slurm batch job. Unlike a run in the VS Code terminal, it keeps going when the
# browser, the VS Code server or the interactive session ends.
# Run it ON THE NODE, OUTSIDE the container (code-server's container has no sbatch), from a GPU session like the
# earlier ones:
#   ssh ${SLURMD_NODENAME:-$HOSTNAME}
#   cd ~/stride_wm_repo/stride-wm && bash submit_a100_2.sh
# It copies the account, partition, GPU request, node features, CPU count and memory of the session you are in, so
# the job gets the same kind of node (NYU Torch's GPU-use rule is much stricter on some node types). Override any:
#   ACCOUNT=... PARTITION=... GRES=gpu:1 CONSTRAINT=... CPUS=12 MEM=64G TIME=6:00:00 bash submit_a100_2.sh
# (PARTITION= or CONSTRAINT= with nothing after it leaves that one out.) Settings for run_a100_2.sh (STAGES,
# PAIRS, WORKERS, ...) are passed to the job. Extra arguments go to sbatch.
set -euo pipefail
cd "$(dirname "$0")"
command -v sbatch >/dev/null || { echo "sbatch not found here: run this on the node outside the container (ssh to the node first)"; exit 1; }

JOB=${FROM_JOB:-$(squeue -h -u "$USER" -t R -w "$(hostname -s)" -o %i 2>/dev/null | head -1)}
INFO=""; [ -n "$JOB" ] && INFO=$(scontrol show job "$JOB" 2>/dev/null || true)
field() { echo "$INFO" | tr ' ' '\n' | sed -n "s/^$1=//p" | head -1 | sed 's/^(null)$//'; }
reqtres() { echo "$INFO" | tr ' ' '\n' | sed -n 's/^ReqTRES=//p' | head -1 | tr ',' '\n' | sed -n "s/^$1=//p" | head -1; }
ACCOUNT=${ACCOUNT:-$(field Account)}
PARTITION=${PARTITION-$(field Partition)}
gpu_req() {  # the session's GPU request as a --gres value, e.g. gpu:1 or gpu:a100:1
  local g; g=$(field TresPerNode); [ -n "$g" ] || g=$(field TresPerJob)
  [ -n "$g" ] || g=$(echo "$INFO" | tr ' ' '\n' | sed -n 's/^ReqTRES=//p' | tr ',' '\n' | grep -m1 '^gres/gpu' | tr '=' ':')
  echo "$g" | tr ',' '\n' | grep -m1 gpu | sed 's#^gres/##; s#^gres:##'
}
GRES=${GRES:-$(gpu_req)}
CONSTRAINT=${CONSTRAINT-$(field Features)}
CPUS=${CPUS:-$(field NumCPUs)}; CPUS=${CPUS:-12}
MEM=${MEM:-$(reqtres mem)}; MEM=${MEM:-64G}
TIME=${TIME:-6:00:00}
export SCRIPT=run_a100_2.sh SIF=${SIF:-/share/apps/code-server/code-server-4.93.1-39.sif}

[ -n "$ACCOUNT" ] || { echo "no account found (session job: ${JOB:-none}). Set ACCOUNT=...; list yours with: my_slurm_accounts"; exit 1; }
[ -n "$GRES" ] || { echo "the session (job ${JOB:-none}) has no GPU to copy. Submit from a GPU session like the earlier ones, or set GRES=gpu:1 (and PARTITION/CONSTRAINT)"; exit 1; }
[ -f "$SIF" ] || { echo "container image $SIF not found: set SIF=<the .sif your .venv was built in>"; exit 1; }

OPTS=(--job-name=stride-a100-2 --account="$ACCOUNT" --gres="$GRES" --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME"
      --output=logs/%x-%j.out)
[ -n "$PARTITION" ] && OPTS+=(--partition="$PARTITION")
[ -n "$CONSTRAINT" ] && OPTS+=(--constraint="$CONSTRAINT")
mkdir -p logs
echo "copied from your session (job ${JOB:-none}): account $ACCOUNT | partition ${PARTITION:-(Slurm picks)} | gres $GRES |" \
     "constraint ${CONSTRAINT:-none} | $CPUS CPUs | memory $MEM; time limit $TIME"
echo "+ sbatch ${OPTS[*]} $* slurm_a100.sbatch"
sbatch "${OPTS[@]}" "$@" slurm_a100.sbatch
echo "Check it with:  squeue -u \$USER     Follow it with:  tail -f logs/stride-a100-2-<job id>.out"
echo "You can now close this session (Open OnDemand > My Interactive Sessions > Delete): the batch job does not need it."
