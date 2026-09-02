#!/usr/bin/env bash
# Merge parity: merged-at-defaults must equal BOTH parents bitwise on actor and
# critic parameters. merged is run TWICE first as a self-control: if two runs of
# the SAME code disagree, GPU nondeterminism makes the whole comparison void.
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=slurm/logs/parity_%j.out
set -uo pipefail
P=/hpcwork/qzi10910/parity
OUT=$P/results
mkdir -p "$OUT" "$HOME/repos/reppo/slurm/logs"
SEED=777
STEPS=2621440   # 20 iterations at 1024x128; reppo.py:1464 needs iterations >= num_eval(20)
ARM=weighted_mle
echo "host $(hostname)  $(date -Is)"
echo "seed $SEED  total_time_steps $STEPS  arm $ARM"
nvidia-smi --query-gpu=name,uuid --format=csv,noheader

run () {                       # run <label> <worktree>
  local label=$1 tree=$2
  echo "=================== $label  ($(cd $tree && git rev-parse --short HEAD))"
  rm -rf "$tree/exports"
  ( cd "$tree" && \
    CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_PREALLOCATE=false \
    ./.venv/bin/python scripts/train_and_export.py \
      env=mjx_dmc env.name=WalkerRun experiment_overrides=mjx_dmc_large_data \
      hyperparameters.actor_update_mode=$ARM \
      hyperparameters.total_time_steps=$STEPS \
      seed=$SEED num_trials=1 num_seeds=1 wandb.mode=disabled ) 2>&1 | tail -6
  local d
  d=$(ls -d "$tree"/exports/*_final 2>/dev/null | head -1)
  if [ -z "$d" ]; then echo "!! $label produced no export"; return 1; fi
  echo "  export: $(basename "$d")"
  rm -rf "$OUT/$label"; mkdir -p "$OUT/$label"; cp "$d"/* "$OUT/$label/"
  sha256sum "$OUT/$label"/actor.npz "$OUT/$label"/critic.npz | sed 's|.*/results/|  |'
}

run merged1 $P/merged
run merged2 $P/merged
run parentA $P/parentA
run parentB $P/parentB
echo
echo "=================== BITWISE COMPARISON"
for f in actor.npz critic.npz; do
  for pair in "merged1:merged2" "merged1:parentA" "merged1:parentB" "parentA:parentB"; do
    a=${pair%%:*}; b=${pair##*:}
    if [ -f "$OUT/$a/$f" ] && [ -f "$OUT/$b/$f" ]; then
      if cmp -s "$OUT/$a/$f" "$OUT/$b/$f"; then r=IDENTICAL; else r=DIFFERS; fi
    else r="MISSING"; fi
    printf "  %-10s %-8s vs %-8s : %s\n" "$f" "$a" "$b" "$r"
  done
done
echo "done $(date -Is)"
