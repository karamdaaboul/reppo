#!/usr/bin/env bash
#SBATCH --job-name=sat-kl
#SBATCH --partition=c25g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=00:25:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mapfile -t CK < <(ls -d "${EXPORTS:?}"/*_final | grep -vE "_s2[0-9][0-9]_final$" | sort)
./.venv/bin/python scripts/analysis/audit_sat_kl.py "${CK[$SLURM_ARRAY_TASK_ID]}" --outdir "${OUTDIR:?}"
