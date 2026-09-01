#!/usr/bin/env bash
# Instrumentation-parity test for the Probe 4 regeneration.
#
# Amendment A records that the ORIGINAL k=16 padded runs were produced at 3b96deb.
# The regeneration runs at 07319d4, whose only src/ change relative to 3b96deb is
# additive and default-off (log_estimator_diag, log_eval_iqm) plus extra meta.json
# curves.  This job runs the SAME config and SAME seed under both commits for a
# short budget and exports both, so the parameters can be compared array-by-array.
#
# array index 0 -> 3b96deb (original) ; 1 -> 07319d4 (regeneration commit)
#SBATCH --job-name=pad16-parity
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
if [ "$IDX" = 0 ]; then ROOT=$HOME/repos/reppo_3b96deb; TAGSUF=orig3b96deb
else                    ROOT=$HOME/repos/reppo;         TAGSUF=regen07319d4; fi
cd "$ROOT" || exit 1
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=true WANDB_MODE=disabled
echo "commit: $(git rev-parse HEAD)  root: $ROOT  host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader

python scripts/train_and_export.py \
  env=mjx_dmc env.name=WalkerRun experiment_overrides=mjx_dmc_large_data \
  env.action_pad=16 seed=0 num_trials=1 num_seeds=1 wandb.mode=disabled \
  hyperparameters.actor_update_mode=pathwise \
  hyperparameters.update_entropy_lagrangian=false \
  hyperparameters.ent_start=0.01528 \
  hyperparameters.total_time_steps=4000000 \
  hydra.run.dir=outputs/parity_${TAGSUF}
echo "PARITY-DONE $TAGSUF rc=$?"
