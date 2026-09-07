#!/usr/bin/env bash
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/hpcwork/qzi10910/estep_wt/slurm/logs/lst_%j.out
set -uo pipefail
cd /hpcwork/qzi10910/estep_wt
echo "host $(hostname)  commit $(git rev-parse HEAD)  $(date -Is)"
echo "script sha256: $(sha256sum scripts/analysis/logsigma_theory.py | cut -d' ' -f1)"
./.venv/bin/python scripts/analysis/logsigma_theory.py
echo "rc=$?  $(date -Is)"
