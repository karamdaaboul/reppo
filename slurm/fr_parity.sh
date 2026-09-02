#!/usr/bin/env bash
# Phase 4A legacy parity: with every faithful flag OFF, the modified code must
# reproduce 07319d4 exactly. Two worktrees, identical config and seed, short horizon.
# index 0 = 07319d4 (pristine) ; 1 = HEAD (flags off)
#SBATCH --job-name=fr-parity
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=00:40:00
#SBATCH --output=/hpcwork/qzi10910/parity/logs/%x_%A_%a.out
#SBATCH --error=/hpcwork/qzi10910/parity/logs/%x_%A_%a.err
set -uo pipefail
IDX=${SLURM_ARRAY_TASK_ID:?}
if [ "$IDX" = 0 ]; then ROOT=$HOME/repos/reppo_07319d4; TAG=pristine07319d4
else                    ROOT=$HOME/repos/reppo_headparity; TAG=headflagsoff; fi
cd "$ROOT" || exit 1
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=true WANDB_MODE=disabled
echo "commit $(git rev-parse HEAD) | root $ROOT | $(hostname)"
python scripts/train_and_export.py \
  env=mjx_dmc env.name=WalkerRun experiment_overrides=mjx_dmc_large_data \
  seed=401 num_trials=1 num_seeds=1 wandb.mode=disabled \
  hyperparameters.actor_update_mode=weighted_mle \
  hyperparameters.update_entropy_lagrangian=false \
  hyperparameters.ent_start=0.014509912580251694 \
  hyperparameters.total_time_steps=4000000 \
  hydra.run.dir=outputs/frparity_${TAG}
echo "PARITY-DONE $TAG rc=$?"
