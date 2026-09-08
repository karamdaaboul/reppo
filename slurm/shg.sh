#!/usr/bin/env bash
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=shgate
#SBATCH --output=/hpcwork/qzi10910/estep_wt/slurm/logs/shgate_%j.out
set -uo pipefail
cd /hpcwork/qzi10910/estep_wt
echo "host $(hostname)  commit $(git rev-parse HEAD)  $(date -Is)"
echo "script sha256: $(sha256sum scripts/analysis/split_half_gate.py | cut -d' ' -f1)"
./.venv/bin/python scripts/analysis/split_half_gate.py
echo "rc=$?  $(date -Is)"
