#!/usr/bin/env bash
# Offline only: entropy-derivative check and the figure package.
# Reads exported checkpoints and frozen artifacts. Starts no training.
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/entfig_%j.out
set -uo pipefail
cd "$HOME/repos/reppo"
echo "host $(hostname)  commit $(git rev-parse HEAD)  $(date -Is)"
./.venv/bin/python scripts/analysis/entropy_and_figures.py
echo "rc=$?  $(date -Is)"
