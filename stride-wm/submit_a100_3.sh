#!/usr/bin/env bash
# Submit run_a100_3.sh (third A100 run: refit the cube probe, plan again) as a Slurm batch job. Same as
# submit_a100_2.sh, which copies your session's account, partition, GPU, CPUs and memory; run it ON THE NODE,
# OUTSIDE the container:
#   ssh ${SLURMD_NODENAME:-$HOSTNAME}
#   cd ~/stride_wm_repo/stride-wm && bash submit_a100_3.sh
# Overrides as in submit_a100_2.sh (ACCOUNT=, PARTITION=, TIME=, ...); settings for run_a100_3.sh (STAGES, PAIRS,
# ZPAIRS, ...) are passed to the job.
cd "$(dirname "$0")" && SCRIPT=run_a100_3.sh JOBNAME=stride-a100-3 TIME=${TIME:-4:00:00} exec bash submit_a100_2.sh "$@"
