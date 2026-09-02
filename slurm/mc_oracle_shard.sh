#!/usr/bin/env bash
# One replicate shard of a pilot-2 run. Sharding is exact, not approximate: rollout
# keys fold on the ABSOLUTE replicate index, so shards tile the run bit-identically
# (verified against an unsharded run before launch).
#SBATCH --job-name=mco2
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
export CUDA_DEVICE_ORDER=PCI_BUS_ID JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=true WANDB_MODE=disabled
CK=${MCO_CKPT:?}; OUT=${MCO_OUT:?}; BANK=${MCO_BANK:?}
NREP=${MCO_NREP:?}; NSHARD=${MCO_NSHARD:?}; I=${SLURM_ARRAY_TASK_ID:?}
if (( NREP % NSHARD )); then echo "NREP not divisible by NSHARD" >&2; exit 2; fi
BLK=$(( NREP / NSHARD ))
export MCO_REPS="$(( I*BLK )):$(( (I+1)*BLK ))"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "shard $I of $NSHARD -> MCO_REPS=$MCO_REPS"
python scripts/analysis/mc_oracle_walker.py pilot "$CK" "$BANK" "${OUT}.shard${I}.npz" p2
