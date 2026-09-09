#!/bin/bash
# ILLUSTRATIVE cluster script — NOT required to reproduce the paper.
# Portable, account-free commands live in the repo Makefile / README.
# Edit the ACCOUNT/PARTITION placeholders, or override on the CLI:
#   sbatch --account=YOUR_ACCT --partition=YOUR_PART scripts/run_discrete_mnist.slurm
#SBATCH --job-name=disc_mnist
#SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --partition=CHANGE_ME_PARTITION
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --time=60
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# NOT `set -u`: conda's activate.d hooks reference unbound vars (e.g.
# MKL_INTERFACE_LAYER), which aborts the job at `conda activate` under -u.
set -eo pipefail
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-neural_odes}"

python -m scripts.train_discrete_mnist --num_layers 5 --seed 0
