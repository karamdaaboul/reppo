#!/usr/bin/env bash
# Corrected-path smoke, seeds 402-403 (disjoint from confirmatory 301-308).
# index 0 = PW-1 corrected ; 1 = WML-32 corrected
#SBATCH --job-name=fr-smoke
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 1
IDX=${SLURM_ARRAY_TASK_ID:?}
if [ "$IDX" = 0 ]; then MODE=pathwise; SEED=402; else MODE=weighted_mle; SEED=403; fi
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=true WANDB_MODE=disabled
echo "smoke mode=$MODE seed=$SEED commit=$(git rev-parse HEAD)"
python scripts/train_and_export.py \
  env=mjx_dmc env.name=WalkerRun experiment_overrides=mjx_dmc_large_data \
  seed=$SEED num_trials=1 num_seeds=1 wandb.mode=disabled \
  hyperparameters.actor_update_mode=$MODE \
  hyperparameters.update_entropy_lagrangian=false \
  hyperparameters.ent_start=0.014509912580251694 \
  hyperparameters.faithful_same_point=true \
  hyperparameters.fresh_minibatch_key=true \
  hyperparameters.log_faithful_diag=true \
  hyperparameters.total_time_steps=4000000 \
  hydra.run.dir=outputs/fr_smoke/${MODE}_s${SEED}
echo "SMOKE-DONE $MODE rc=$?"
