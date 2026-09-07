#!/usr/bin/env bash
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=opbanks
#SBATCH --output=/hpcwork/qzi10910/estep_wt/slurm/logs/opbanks_%j.out
set -uo pipefail
cd /hpcwork/qzi10910/estep_wt
echo "host $(hostname)  commit $(git rev-parse HEAD)  $(date -Is)"
./.venv/bin/python scripts/analysis/build_onpolicy_banks.py
echo "rc=$?  $(date -Is)"
