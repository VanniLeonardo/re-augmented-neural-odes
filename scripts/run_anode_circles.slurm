#!/bin/bash
# ILLUSTRATIVE cluster script — NOT required to reproduce the paper.
# Portable, account-free commands live in the repo Makefile / README.
# Edit the ACCOUNT/PARTITION placeholders, or override on the CLI:
#   sbatch --account=YOUR_ACCT --partition=YOUR_PART scripts/run_anode_circles.slurm
#SBATCH --job-name=anode_circles
#SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --partition=CHANGE_ME_PARTITION
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --time=240
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-neural_odes}"

python -m scripts.train_anode_circles \
  --epochs 500 \
  --n_samples 1000 \
  --batch_size 64 \
  --lr 3e-3 \
  --augment_dims 0,1,2,5 \
  --seeds 0,1,2,3,4 \
  --hidden_dim 2 \
  --ode_hidden_dim 64 \
  --log_every 50 \
  --write_results
