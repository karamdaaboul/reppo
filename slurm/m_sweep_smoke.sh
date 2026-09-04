#!/usr/bin/env bash
# Smoke test for the confirmatory M-sweep, prereg Sec. 11.
# Runs on ONE partition and covers BOTH arms on it. Sec. 11 asks the budget
# assertion to pass at M=512 on both partitions, so M=512 is smoked on c25g too
# even though Sec. 2 places no real M=512 run there. Outputs are discarded.
#SBATCH --account=rwth2182
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=slurm/logs/mssmoke_%j.out
set -uo pipefail
cd "$HOME/repos/reppo"
PREREG_COMMIT=$(git log -1 --format=%H -- docs/prereg_m_sweep_confirmatory.md)
echo "=========== M-sweep smoke ==========="
echo "host        $(hostname)"
echo "partition   ${SLURM_JOB_PARTITION}"
echo "git commit  $(git rev-parse HEAD)"
echo "prereg      $PREREG_COMMIT"
echo "dirty       $(git status --porcelain | wc -l) files"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

echo "----------- Sec. 4.2 / Sec. 11 budget + tag assertions (all 16 cells)"
./.venv/bin/python scripts/analysis/m_sweep_assert.py || { echo "ASSERT FAILED"; exit 1; }
echo

RC=0
for M in 128 512; do
  echo "----------- tiny run, M=$M on ${SLURM_JOB_PARTITION} (outputs discarded)"
  D=/hpcwork/$USER/mssmoke_${SLURM_JOB_ID}_M${M}
  rm -rf "$D"; mkdir -p "$D"
  CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_PREALLOCATE=false HYDRA_FULL_ERROR=1 \
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
  r=$?
  if [ $r -eq 0 ]; then
    echo "  M=$M OK at full batch geometry on ${SLURM_JOB_PARTITION}"
    ls "$D"/../ >/dev/null 2>&1
  else
    echo "  M=$M FAILED (rc=$r)"; tail -12 "$D/run.log"; RC=1
  fi
  rm -rf "$D"
done
# the smoke must not leave an export behind that a real run would later collide with
rm -rf exports/WalkerRun_weighted_mle_m128_s301 exports/WalkerRun_weighted_mle_m512_s301 2>/dev/null
for t in WalkerRun_weighted_mle_m128_s301 WalkerRun_weighted_mle_m512_s301; do
  for suf in _final _p25 _p50; do rm -rf "exports/${t}${suf}"; done
done
echo
[ $RC -eq 0 ] && echo "SMOKE PASSED on ${SLURM_JOB_PARTITION}" || echo "SMOKE FAILED on ${SLURM_JOB_PARTITION}"
echo "done $(date -Is)"
exit $RC
