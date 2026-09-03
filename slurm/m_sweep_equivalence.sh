#!/usr/bin/env bash
# Code equivalence for the confirmatory M-sweep: the WML-32 arm at seeds 301-308
# launched at 7edb8e8c (ledger runs_faithful_repair.jsonl). HEAD is 4eb4713 and
# src/ differs (399411d5 -> 2e604b3c). Bitwise test at the ARM'S OWN config, not
# at defaults: that arm sets faithful_same_point, fresh_minibatch_key,
# log_faithful_diag and a frozen alpha, so a defaults-only test would miss it.
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=slurm/logs/eqv_%j.out
set -uo pipefail
P=/hpcwork/qzi10910/eqv
OUT=$P/results; mkdir -p "$OUT"
SEED=301
ALPHA=0.014509912580251694     # ledger ent_start for walker_WML32_s301
STEPS=2621440                  # 20 iterations; reppo.py:1464 needs iterations >= num_eval
echo "host $(hostname) $(date -Is)"
nvidia-smi --query-gpu=name,uuid --format=csv,noheader

run () {
  local label=$1 tree=$2
  echo "=================== $label ($(cd $tree && git rev-parse --short HEAD))"
  rm -rf "$tree/exports"
  ( cd "$tree" && CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_PREALLOCATE=false \
    ./.venv/bin/python scripts/train_and_export.py \
      env=mjx_dmc env.name=WalkerRun experiment_overrides=mjx_dmc_large_data \
      seed=$SEED num_trials=1 num_seeds=1 wandb.mode=disabled \
      hyperparameters.actor_update_mode=weighted_mle \
      hyperparameters.update_entropy_lagrangian=false \
      hyperparameters.ent_start=$ALPHA \
      hyperparameters.faithful_same_point=true \
      hyperparameters.fresh_minibatch_key=true \
      hyperparameters.log_faithful_diag=true \
      hyperparameters.total_time_steps=$STEPS ) 2>&1 | tail -12
  local d; d=$(ls -d "$tree"/exports/*_final 2>/dev/null | head -1)
  [ -z "$d" ] && { echo "!! $label produced no export"; return 1; }
  echo "  export $(basename "$d")"
  rm -rf "$OUT/$label"; mkdir -p "$OUT/$label"; cp "$d"/* "$OUT/$label/"
}
run launch  $P/launch
run head1   $P/head
run head2   $P/head
echo
echo "=================== BITWISE"
for f in actor.npz critic.npz; do
  for pair in "head1:head2" "launch:head1"; do
    a=${pair%%:*}; b=${pair##*:}
    if [ -f "$OUT/$a/$f" ] && [ -f "$OUT/$b/$f" ]; then
      cmp -s "$OUT/$a/$f" "$OUT/$b/$f" && r=IDENTICAL || r=DIFFERS
    else r=MISSING; fi
    printf "  %-11s %-7s vs %-7s : %s\n" "$f" "$a" "$b" "$r"
  done
done
echo "done $(date -Is)"
