#!/bin/bash
# SLURM wrapper for run_a100.sh. Submit from the stride-wm/ folder:
#   mkdir -p logs && sbatch slurm_a100.sbatch
# Resubmitting the same file resumes: every stage skips work whose output already exists.
# Adjust the lines marked CHANGE to your cluster (check its docs for the A100 request syntax).
#SBATCH --job-name=stride-a100
#SBATCH --gres=gpu:a100:1          # CHANGE if your cluster uses e.g. --gres=gpu:1 --constraint=a100
#SBATCH --cpus-per-task=8          # data collection and MuJoCo rendering run on the CPU
#SBATCH --mem=64G
#SBATCH --time=40:00:00            # CHANGE to your allocation
#SBATCH --output=logs/%x-%j.out
##SBATCH --account=YOUR_ACCOUNT    # CHANGE: remove one '#' and fill in, if your cluster needs it
##SBATCH --partition=YOUR_PARTITION

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # compute nodes often have no internet: use the cached DINOv2 weights
nvidia-smi || true

# The .venv must be used inside the same environment it was created in.
# CHANGE (Singularity/Apptainer): if you built .venv inside a container, set SIF to that image. Find it by
# running `echo $SINGULARITY_CONTAINER` (or `echo $APPTAINER_CONTAINER`) inside the container shell. Add any
# --overlay/--bind options your cluster's container setup uses to SIF_OPTS.
SIF=${SIF:-}
SIF_OPTS=${SIF_OPTS:-}
RUN="source .venv/bin/activate && bash ${SCRIPT:-run_a100.sh}"   # SCRIPT=run_a100_2.sh for the second run
if [ -n "$SIF" ]; then
  CTR=$(command -v singularity || command -v apptainer || true)
  [ -n "$CTR" ] || { echo "SIF is set but neither singularity nor apptainer is on PATH (try: module load singularity)"; exit 1; }
  # --nv exposes the GPU inside the container; the environment (STAGES, SEEDS, ...) is passed through
  $CTR exec --nv $SIF_OPTS "$SIF" bash -c "$RUN"
else
  bash -c "$RUN"
fi
