#!/usr/bin/env bash
# Phase 1.8 corrected-path parity: with freeze_sigma=null the new code must reproduce
# the corrected replication EXACTLY, for both operators.
#
# Both arms run in the SAME worktree with the SAME venv and the SAME command line;
# only the checked-out source differs. reppo_headparity/exports is a symlink into
# /hpcwork/qzi10910/frparity_head, a different directory from the main export tree,
# so nothing here can touch the corrected artifacts.
#SBATCH --job-name=cf-parity
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
WT=${CF_WT:-$HOME/repos/reppo_headparity}
DEST=${CF_DEST:-/hpcwork/qzi10910/cfparity}
RESTORE=${CF_RESTORE:-}
BASE=${CF_BASE:?}          # pre-freeze commit
NEW=${CF_NEW:?}            # freeze commit, flag off
mkdir -p "$DEST"
cd "$WT" || exit 1
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=true WANDB_MODE=disabled
nvidia-smi --query-gpu=name --format=csv,noheader

# park anything left over from an earlier parity rather than deleting it
mkdir -p "$DEST/preexisting"
find exports/ -mindepth 1 -maxdepth 1 -exec mv -t "$DEST/preexisting" {} + 2>/dev/null

run () {  # $1 = arm tag, $2 = actor_update_mode
  local tag=$1 mode=$2
  echo "=== RUN $tag ($mode) at $(git rev-parse --short HEAD) ==="
  python scripts/train_and_export.py \
    env=mjx_dmc env.name=WalkerRun experiment_overrides=mjx_dmc_large_data \
    seed=499 num_trials=1 num_seeds=1 wandb.mode=disabled \
    hyperparameters.actor_update_mode="$mode" \
    hyperparameters.update_entropy_lagrangian=false \
    hyperparameters.ent_start=0.014509912580251694 \
    hyperparameters.faithful_same_point=true \
    hyperparameters.fresh_minibatch_key=true \
    hyperparameters.log_faithful_diag=true \
    hyperparameters.total_time_steps=4000000 \
    hydra.run.dir="$DEST/run_${tag}" || { echo "RUN FAILED $tag"; exit 1; }
  mkdir -p "$DEST/$tag"
  find exports/ -mindepth 1 -maxdepth 1 -exec mv -t "$DEST/$tag" {} +
  echo "=== captured $tag: $(ls "$DEST/$tag" | tr '\n' ' ')"
}

git checkout -q "$BASE" || exit 1
run A_PW  pathwise
run A_WML weighted_mle

git checkout -q "$NEW" || exit 1
run B_PW  pathwise
run B_WML weighted_mle

# leave the scratch worktree where it was found
git checkout -q "${RESTORE:-$BASE}"
echo "PARITY-RUNS-DONE wt=$WT dest=$DEST"
