#!/usr/bin/env bash
# M=512 memory probe. One iteration at the FULL registered batch geometry
# (num_envs 1024, num_steps 128, num_mini_batches 128, num_epochs 4), arm config
# as in the prereg. Outputs are discarded: this measures feasibility only.
# num_eval=1 solely so reppo.py:1464 (eval_interval = iterations // num_eval)
# stays >= 1 at a single iteration; it does not affect memory.
#SBATCH --account=rwth2182
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=slurm/logs/memprobe_%j.out
set -uo pipefail
cd "$HOME/repos/reppo"
M=${1:?M}
echo "host $(hostname)  partition ${SLURM_JOB_PARTITION}  $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

POLL=/tmp/memprobe_$$.log
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; sleep 1; done ) > "$POLL" 2>/dev/null &
PID=$!

D=/hpcwork/$USER/memprobe_$SLURM_JOB_ID
rm -rf "$D"; mkdir -p "$D"
set +e
CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_PREALLOCATE=false \
HYDRA_FULL_ERROR=1 \
./.venv/bin/python scripts/train_and_export.py \
  env=mjx_dmc env.name=WalkerRun experiment_overrides=mjx_dmc_large_data \
  seed=301 num_trials=1 num_seeds=1 wandb.mode=disabled \
  hyperparameters.actor_update_mode=weighted_mle \
  hyperparameters.update_entropy_lagrangian=false \
  hyperparameters.ent_start=0.014509912580251694 \
  hyperparameters.faithful_same_point=true \
  hyperparameters.fresh_minibatch_key=true \
  hyperparameters.log_faithful_diag=true \
  hyperparameters.estep_num_samples=$M \
  hyperparameters.total_time_steps=131072 \
  hyperparameters.num_eval=1 \
  hydra.run.dir="$D" > "$D/run.log" 2>&1
RC=$?
set -e
kill $PID 2>/dev/null
PEAK=$(sort -n "$POLL" | tail -1); rm -f "$POLL"

echo "--------- RESULT M=$M partition=${SLURM_JOB_PARTITION}"
echo "  exit_code       $RC"
echo "  memory.total    ${TOTAL} MiB"
echo "  peak memory.used ${PEAK} MiB"
if [ "$RC" -eq 0 ]; then echo "  VERDICT         FITS"; else
  echo "  VERDICT         DID NOT COMPLETE"
  echo "  --- last 15 lines ---"; tail -15 "$D/run.log"
  grep -i -m3 "out of memory\|RESOURCE_EXHAUSTED\|OOM" "$D/run.log" && echo "  (OOM signature present)" || echo "  (no OOM signature; different failure)"
fi
rm -rf "$D"                    # outputs discarded
echo "done $(date -Is)"
