#!/usr/bin/env bash
# Probe 4 (crossed same-critic operator table) over the regenerated k=16 checkpoints.
# array 0-4 -> seeds 0-4 under the CHECKPOINT law (primary)
# array 5-9 -> seeds 0-4 under the COMMON STANDARDIZED law N(0,I_k) (mandatory sensitivity)
#SBATCH --job-name=probe4
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -uo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_ROOT" || exit 1
IDX=${SLURM_ARRAY_TASK_ID:?}
if [ "$IDX" -lt 5 ]; then SEED=$IDX;        LAW=ckpt
else                      SEED=$((IDX-5));  LAW=std; fi
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p reports/artifacts
echo "commit $(git rev-parse HEAD) | seed $SEED | law $LAW | $(hostname)"
python scripts/probe4_crossed.py "$SEED" "$LAW" "reports/artifacts/probe4_s${SEED}_${LAW}.npz"
