#!/bin/bash
# ILLUSTRATIVE cluster script — NOT required to reproduce the paper.
# Portable, account-free commands live in the repo Makefile / README.
# Edit the ACCOUNT/PARTITION placeholders, or override on the CLI:
#   sbatch --account=YOUR_ACCT --partition=YOUR_PART scripts/run_continuous_mnist.slurm
#SBATCH --job-name=ode_mnist
#SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --partition=CHANGE_ME_PARTITION
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --time=60
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# Portable: run from the submission directory (falls back to the repo root).
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

# Activate the pinned environment (edit CONDA_ENV / conda base for your site).
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-neural_odes}"

python -m scripts.train_continuous_mnist --network_type cnn --seed 0
