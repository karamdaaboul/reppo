#!/usr/bin/env bash
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/t2t5_%j.out
set -uo pipefail
cd "$HOME/repos/reppo"
echo "host $(hostname)  commit $(git rev-parse HEAD)  $(date -Is)"
./.venv/bin/python tests/test_entropy_factorial.py
echo "rc=$?  $(date -Is)"
