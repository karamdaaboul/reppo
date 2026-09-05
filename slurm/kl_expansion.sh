#!/usr/bin/env bash
# Offline only: KL expansion audit over all eight corrected Walker seeds, both arms,
# three checkpoint fractions. Reads exported checkpoints; starts no training.
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/klexp_%j.out
set -uo pipefail
cd "$HOME/repos/reppo"
echo "host $(hostname)  commit $(git rev-parse HEAD)  $(date -Is)"
./.venv/bin/python scripts/analysis/kl_expansion_audit.py
echo "rc=$?  $(date -Is)"
