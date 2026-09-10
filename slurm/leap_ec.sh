#!/usr/bin/env bash
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --job-name=leapec
#SBATCH --output=/hpcwork/qzi10910/estep_wt/slurm/logs/leapec_%j.out
set -uo pipefail
cd /hpcwork/qzi10910/estep_wt
echo "host $(hostname)  commit $(git rev-parse HEAD)  $(date -Is)"
./.venv/bin/python /hpcwork/qzi10910/gates/leap_ec.py
echo "rc=$?  $(date -Is)"
