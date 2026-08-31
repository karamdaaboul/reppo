#!/usr/bin/env bash
# Registered confirmatory ladder: 4 tasks x 8 seeds x 2 arms = 64 runs.
#
# Registered historical implementation, NO KL decoupling:
#   arm A = pathwise operator,      KL Monte-Carlo samples = 16 (implied by the mode)
#   arm B = weighted_mle, M = 32,   KL Monte-Carlo samples = 32 (implied by the mode)
# log_estimator_diag=false and log_eval_iqm=false so every confirmatory run stays
# bit-identical to the pristine reference (parity check (a), PASS).
#
# GPU counterbalancing, FROZEN BEFORE LAUNCH and never changed on outcomes:
#   odd  seeds (101,103,105,107): A -> GPU 0,  B -> GPU 1
#   even seeds (102,104,106,108): A -> GPU 1,  B -> GPU 0
# so within every task each arm gets 4 seeds on Blackwell and 4 on Ada.
#
# Fixed task priority, set before outcomes: G1 -> LEAP -> Hopper -> Walker.
set -u
cd /home/human/workspaces/reppo_original || exit 1
J=/home/human/.claude/jobs/2ef546ab/tmp
LEDGER=ledger/runs.jsonl
export CUDA_DEVICE_ORDER=PCI_BUS_ID JAX_PLATFORMS=cuda,cpu
SHA=$(git rev-parse HEAD)
GPUNAME_0="NVIDIA RTX PRO 4500 Blackwell"
GPUNAME_1="NVIDIA RTX 4000 Ada Generation"

# Frozen task/alpha/env definitions are shared with the SLURM launcher so the two
# can never drift. See slurm/ladder_matrix.sh.
source slurm/ladder_matrix.sh

TASKS="$TASKS_DEFAULT"   # Walker runs separately, after its own seed-901 alpha is frozen

one () {  # one <gpu> <task> <arm> <seed>
  local gpu="$1" task="$2" arm="$3" seed="$4"
  local a; a=$(alpha_of "$task")
  local mode; mode=$(mode_of "$arm")
  local gname; [ "$gpu" = 0 ] && gname="$GPUNAME_0" || gname="$GPUNAME_1"
  local tag="${task}_${arm}_s${seed}"
  local cmd="python scripts/train_and_export.py hydra.run.dir=outputs/conf/${tag} $(env_args "$task") \
seed=${seed} num_trials=1 num_seeds=1 wandb.mode=disabled \
hyperparameters.actor_update_mode=${mode} \
hyperparameters.update_entropy_lagrangian=false hyperparameters.ent_start=${a} \
hyperparameters.log_estimator_diag=false hyperparameters.log_eval_iqm=false"
  local t0 t1 st
  t0=$(date +%s); st=$(date -Is)
  echo "START $tag gpu=$gpu $st" >> "$J/conf_timing.txt"
  if CUDA_VISIBLE_DEVICES="$gpu" $cmd > "$J/conf_${tag}.log" 2>&1; then st_out=completed; else st_out=failed; fi
  t1=$(date +%s)
  echo "$st_out $tag gpu=$gpu wall_s=$((t1-t0)) $(date -Is)" >> "$J/conf_timing.txt"
  python - "$tag" "$(taskname "$task")" "$arm" "$seed" "$SHA" "$gname" "$st_out" "$((t1-t0))" "$st" "$cmd" <<'PY' >> "$LEDGER"
import json,sys
tag,task,arm,seed,sha,gpu,status,wall,start,cmd = sys.argv[1:11]
print(json.dumps(dict(run_id=f"conf-{tag}", namespace="confirmatory", label=None,
    task=task, arm=arm, seed=int(seed), git_sha=sha, gpu=gpu, command=cmd,
    algorithm_version=("A-pathwise" if arm=="A" else "B-weighted-mle"),
    changed_params=[], reason="Registered confirmatory ladder (prereg L.1)",
    status=status, start=start, wall_clock_s=int(wall),
    gpu_hours=round(int(wall)/3600,4), return_metrics=None, estimator_diag=None)))
PY
}

worker () {  # worker <gpu>
  local gpu="$1"
  for task in $TASKS; do
    for seed in 101 102 103 104 105 106 107 108; do
      local odd=$(( seed % 2 ))
      # odd -> A on gpu0 / B on gpu1 ; even -> A on gpu1 / B on gpu0
      if [ "$odd" = 1 ]; then aG=0; bG=1; else aG=1; bG=0; fi
      [ "$aG" = "$gpu" ] && one "$gpu" "$task" A "$seed"
      [ "$bG" = "$gpu" ] && one "$gpu" "$task" B "$seed"
    done
    echo "TASK_GPU_DONE $task gpu=$gpu $(date -Is)" >> "$J/conf_timing.txt"
  done
  echo "WORKER_DONE gpu=$gpu $(date -Is)" >> "$J/conf_timing.txt"
}

worker 0 & W0=$!
worker 1 & W1=$!
wait $W0 $W1
echo CONFIRMATORY_ALL_DONE >> "$J/conf_timing.txt"
