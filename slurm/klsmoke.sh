#!/usr/bin/env bash
# Smoke test for the KL expansion audit: part 1 plus one seed per arm.
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/klsmoke_%j.out
set -uo pipefail
cd "$HOME/repos/reppo"
echo "host $(hostname)  commit $(git rev-parse HEAD)  $(date -Is)"
KLAUDIT_SEEDS=301 ./.venv/bin/python scripts/analysis/kl_expansion_audit.py
echo "rc=$?  $(date -Is)"
