#!/usr/bin/env bash
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/efa_g1_%j.out
set -uo pipefail
cd "$HOME/repos/reppo"
echo "host $(hostname)  commit $(git rev-parse HEAD)  $(date -Is)"
echo "########## Walker regression (must reproduce committed values)"
EF_TASK=walker ./.venv/bin/python scripts/analysis/entropy_factorial_analysis.py 2>&1 | grep -E "score_window3 +\||bank median sigma|Delta_ent |Delta_noent |full     log_ratio"
echo
echo "########## G1 d=29"
EF_TASK=g1 ./.venv/bin/python scripts/analysis/entropy_factorial_analysis.py
echo "rc=$?  $(date -Is)"
