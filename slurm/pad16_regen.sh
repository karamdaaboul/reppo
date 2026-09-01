#!/usr/bin/env bash
# Regenerate the 10 k=16 padded WalkerRun checkpoints required by Probe 4 of
# docs/prospective_padding_error_field_analysis.md.
#
# Runs the code at 3b96deb (a detached worktree), NOT the estep-study tree.
# pyproject.toml is byte-identical between 3b96deb and 07319d4 and uv.lock did
# not exist at 3b96deb, so the venv built for the ladder is a valid environment
# for that commit; it is symlinked into the worktree rather than rebuilt.
#
# Ledger entries are written BEFORE launch: ledger/runs_pad16_regen.jsonl.
#
#SBATCH --job-name=pad16-regen
#SBATCH --partition=c25g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"

IDX=${SLURM_ARRAY_TASK_ID:?}
if [ "$IDX" -lt 5 ]; then ARM=pathwise; SEED=$IDX; else ARM=weighted_mle; SEED=$((IDX-5)); fi

ALPHA=0.01528          # reports/probe_k6_report.md:32 -- the padding-experiment alpha
PAD=16

export PATH="$HOME/.local/bin:$PATH"
echo "== pad16 regen: arm=$ARM seed=$SEED k=$PAD alpha=$ALPHA sha=$(git rev-parse HEAD)"

./.venv/bin/python scripts/train_and_export.py \
  env=mjx_dmc \
  env.name=WalkerRun \
  experiment_overrides=mjx_dmc_large_data \
  env.action_pad=$PAD \
  seed=$SEED \
  num_trials=1 \
  num_seeds=1 \
  wandb.mode=disabled \
  hyperparameters.actor_update_mode=$ARM \
  hyperparameters.update_entropy_lagrangian=false \
  hyperparameters.ent_start=$ALPHA \
  hydra.run.dir=outputs/pad16/${ARM}_s${SEED}
