#!/bin/bash
# C1 solver-dynamics sweep (Chen Fig 3a-b) as a per-seed array job.
#
# ILLUSTRATIVE cluster script — NOT required to reproduce the paper. The portable,
# account-free command is `make c1` (see the Makefile), which runs the same sweep
# serially on one GPU.
#
# Each array task owns ONE seed and writes its own shard
# (results/c1/c1_solver_dynamics_s<seed>.csv), so tasks never race on a file. Merge
# the shards with `python -m scripts.c1_report` (it globs the shards).
#
#   sbatch --account=YOUR_ACCT --partition=YOUR_PART scripts/run_c1_solver_dynamics.slurm
#
#SBATCH --job-name=c1_solver
#SBATCH --partition=CHANGE_ME_PARTITION
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=180
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs results/c1

cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-neural_odes}"

SEED="${SLURM_ARRAY_TASK_ID:-0}"
echo "C1 seed ${SEED} on $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -m scripts.run_c1_solver_dynamics \
  --seeds "${SEED}" \
  --tag "_s${SEED}" \
  --epochs "${C1_EPOCHS:-5}" \
  --filters "${C1_FILTERS:-64}" \
  --eval_batch "${C1_EVAL_BATCH:-128}"
