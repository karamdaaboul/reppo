#!/usr/bin/env bash
# LEAP entropy factorial. Every scientific value pinned explicitly, recovered from LEAP's
# own executed config, not copied from Walker or G1.
set -uo pipefail
cd "$HOME/repos/reppo"
SEED=$((300 + SLURM_ARRAY_TASK_ID))
case "$EF_ARM" in
  PW_noent) MODE=pathwise;     FLAG="hyperparameters.pw_drop_actor_entropy=true";;
  WML_ent)  MODE=weighted_mle; FLAG="hyperparameters.wml_add_actor_entropy=true";;
  *) echo "unknown arm $EF_ARM"; exit 1;;
esac
echo "arm $EF_ARM seed $SEED partition $SLURM_JOB_PARTITION sha $(git rev-parse HEAD) job ${SLURM_JOB_ID} $(date -Is)"
./.venv/bin/python scripts/train_and_export.py \
  env=mjx_dmc env.name=LeapCubeRotateZAxis env.asymmetric_obs=false \
  experiment_overrides=mjx_dmc_large_data \
  env.vmin=-10 env.vmax=60 env.max_episode_steps=500 \
  hyperparameters.max_episode_steps=500 \
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
  hyperparameters.ent_start=0.000782382907345891 \
  hyperparameters.update_entropy_lagrangian=false \
  hyperparameters.freeze_sigma=null hyperparameters.log_cov_diag=false \
  hyperparameters.faithful_same_point=true hyperparameters.fresh_minibatch_key=true \
  hyperparameters.log_faithful_diag=true \
  hydra.run.dir=/hpcwork/qzi10910/reppo_runs/outputs/leap_entropy_factorial/leap_${EF_ARM}_s${SEED}
echo "rc=$?  $(date -Is)"
