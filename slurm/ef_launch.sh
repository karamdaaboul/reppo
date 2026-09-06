#!/usr/bin/env bash
# Walker entropy factorial. One arm per array; seeds 301-308 as array tasks 1-8.
# Every scientific value pinned explicitly so Hydra inheritance cannot alter it.
set -uo pipefail
cd "$HOME/repos/reppo"
SEED=$((300 + SLURM_ARRAY_TASK_ID))
ARM="$EF_ARM"
case "$ARM" in
  PW-H)  MODE=pathwise;     FLAG="hyperparameters.pw_drop_actor_entropy=true";;
  WML+H) MODE=weighted_mle; FLAG="hyperparameters.wml_add_actor_entropy=true";;
  *) echo "unknown arm $ARM"; exit 1;;
esac
echo "arm $ARM  seed $SEED  sha $(git rev-parse HEAD)  job ${SLURM_JOB_ID}  $(date -Is)"
./.venv/bin/python scripts/train_and_export.py \
  env=mjx_dmc env.name=WalkerRun experiment_overrides=mjx_dmc_large_data \
  seed=$SEED num_trials=1 num_seeds=1 wandb.mode=disabled \
  hyperparameters.actor_update_mode=$MODE $FLAG \
  hyperparameters.num_mini_batches=128 hyperparameters.num_epochs=4 \
  hyperparameters.num_envs=1024 hyperparameters.num_steps=128 \
  hyperparameters.total_time_steps=50000000 hyperparameters.num_eval=20 \
  hyperparameters.lr=0.0003 hyperparameters.max_grad_norm=0.5 \
  hyperparameters.anneal_lr=false hyperparameters.gamma=0.99 hyperparameters.lmbda=0.95 \
  hyperparameters.kl_bound=0.1 hyperparameters.reduce_kl=true \
  hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped \
  hyperparameters.estep_num_samples=32 hyperparameters.eps_e=0.5 \
  hyperparameters.mstep_decoupled=false hyperparameters.sqrt_rho=1.0 \
  hyperparameters.ent_loss_per_dim=false \
  hyperparameters.ent_start=0.014509912580251694 \
  hyperparameters.update_entropy_lagrangian=false \
  hyperparameters.faithful_same_point=true hyperparameters.fresh_minibatch_key=true \
  hyperparameters.log_faithful_diag=true \
  hydra.run.dir=/hpcwork/qzi10910/reppo_runs/outputs/entropy_factorial/walker_${ARM}_s${SEED}
echo "rc=$?  $(date -Is)"
