#!/usr/bin/env bash
#SBATCH --job-name=ubar-batch
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 1
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p reports/artifacts/ubar_batch
IDX=${SLURM_ARRAY_TASK_ID:?}
mapfile -t CKPTS < reports/artifacts/ubar_ckpt_list.txt
CK="${CKPTS[$IDX]}"
# Shuffles needed to reach >=512 reconstructed minibatches per task-arm condition
# (docs/prereg_ubar_ratio.md Sec. 8): g1 yields 16 minibatches/epoch over 8 seeds -> 4;
# the padded Walker conditions have 5 seeds at 64/epoch -> 2; the rest 8 seeds at 64 -> 1.
case "$CK" in
  *G1Joystick*)  NSHUF=4 ;;
  *pad16*)       NSHUF=2 ;;
  *)             NSHUF=1 ;;
esac
NSHUF=${NSHUF_OVERRIDE:-$NSHUF}
echo "commit $(git rev-parse HEAD) | $CK | shuffles=$NSHUF | $(hostname)"
python scripts/analysis/ubar_batch_gradient.py "$CK"   "reports/artifacts/ubar_batch/$(basename "$CK").npz" "$NSHUF" 8
