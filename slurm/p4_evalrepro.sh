#!/usr/bin/env bash
#SBATCH --job-name=p4-evalrepro
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 1
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
python scripts/probe4_eval_reproduction.py
