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
# CHANGE: activate the environment you installed requirements.txt into (venv, conda, or a container)
source .venv/bin/activate
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # compute nodes often have no internet: use the cached DINOv2 weights
nvidia-smi || true
bash run_a100.sh
