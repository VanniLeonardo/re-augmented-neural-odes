#!/bin/bash
# ILLUSTRATIVE cluster script — NOT required to reproduce the paper.
# This launches an OPTIONAL Weights & Biases hyperparameter sweep and therefore
# needs your OWN W&B account + sweep id. It is not part of `make reproduce-all`.
# The original personal W&B entity/sweep id has been removed.
# Edit the ACCOUNT/PARTITION placeholders, or override on the CLI:
#   sbatch --account=YOUR_ACCT --partition=YOUR_PART scripts/run_sweep_discrete.slurm
#SBATCH --job-name=sweep_disc
#SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --partition=CHANGE_ME_PARTITION
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --time=120
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-neural_odes}"

# Create a sweep first with:  wandb sweep sweep_discrete.yaml
# then export WANDB_SWEEP_ID=<entity>/<project>/<id> before submitting this job.
: "${WANDB_SWEEP_ID:?Set WANDB_SWEEP_ID=<entity>/<project>/<sweep-id> (your own W&B account)}"
wandb agent "${WANDB_SWEEP_ID}"
