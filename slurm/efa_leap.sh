#!/usr/bin/env bash
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/efa_leap_%j.out
set -uo pipefail
cd "$HOME/repos/reppo"
echo "host $(hostname)  commit $(git rev-parse HEAD)  $(date -Is)"
echo "########## regression: walker and g1 must reproduce committed values"
EF_TASK=walker ./.venv/bin/python scripts/analysis/entropy_factorial_analysis.py 2>&1 | grep -E "Delta_ent |Delta_noent |full     log_ratio|score_window3 +\|"
EF_TASK=g1 ./.venv/bin/python scripts/analysis/entropy_factorial_analysis.py 2>&1 | grep -E "Delta_ent |Delta_noent |full     log_ratio|score_window3 +\|"
echo
echo "########## LEAP d=16"
EF_TASK=leap ./.venv/bin/python scripts/analysis/entropy_factorial_analysis.py
echo "rc=$?  $(date -Is)"
