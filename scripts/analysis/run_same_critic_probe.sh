#!/usr/bin/env bash
#SBATCH --job-name=sc-probe
#SBATCH --partition=c25g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=00:30:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mapfile -t CK < <(ls -d "${EXPORTS:?}"/{G1JoystickFlatTerrain,LeapCubeRotateZAxis}_*_final \
                  | grep -E "_s10[1-8]_final$" | sort)
./.venv/bin/python scripts/analysis/same_critic_probe.py "${CK[$SLURM_ARRAY_TASK_ID]}" --outdir "${OUTDIR:?}"
