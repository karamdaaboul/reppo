#!/usr/bin/env bash
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/sst_%j.out
set -uo pipefail
cd "$HOME/repos/reppo"
echo "host $(hostname)  commit $(git rev-parse HEAD)  $(date -Is)"
./.venv/bin/python scripts/analysis/same_state_test.py
echo "rc=$?  $(date -Is)"
