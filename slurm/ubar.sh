#!/usr/bin/env bash
# Step 2 of docs/prereg_ubar_ratio.md over the 74 registered checkpoints.
# Frozen-checkpoint, read-only: no training, nothing written into the run tree.
#SBATCH --job-name=ubar
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 1
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p reports/artifacts/ubar
IDX=${SLURM_ARRAY_TASK_ID:?}
mapfile -t CKPTS < reports/artifacts/ubar_ckpt_list.txt
CK="${CKPTS[$IDX]}"
echo "commit $(git rev-parse HEAD) | $CK | $(hostname)"
python scripts/analysis/ubar_ratio.py "$CK" "reports/artifacts/ubar/$(basename "$CK").npz"
