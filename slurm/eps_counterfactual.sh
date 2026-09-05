#!/usr/bin/env bash
# Offline analysis only: state-bank collection and per-checkpoint sigma/saturation.
# No training, no parameter update.
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/epscf_%j.out
set -uo pipefail
cd "$HOME/repos/reppo"
echo "host $(hostname)  commit $(git rev-parse HEAD)  $(date -Is)"
./.venv/bin/python scripts/analysis/walker_eps_counterfactual.py
echo "done $(date -Is)"
