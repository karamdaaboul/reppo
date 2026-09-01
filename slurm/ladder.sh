#!/usr/bin/env bash
# Registered confirmatory dimension ladder as a SLURM job array, one GPU per run.
#
#   tasks x 8 seeds (101-108) x 2 arms   ->  48 runs (g1/leap/hopper), 64 with walker
#
# Submit via slurm/submit_ladder.sh, which fills in --account and --array. Always
# submit from the repo root: Slurm's working directory is wherever sbatch is called
# from and every path below is relative to it.
#
# Billing (RWTH, c23g): one GPU is capped at 24 cores and 122 GB and is charged as
# 24 core-hours per GPU-hour. The header requests exactly that share of a node
# (4x H100 94GB, 96 cores).

#SBATCH --job-name=reppo-ladder
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -uo pipefail

# Under sbatch the script is staged into /var/spool/slurm, so ${BASH_SOURCE[0]}
# points outside the repo and "$(dirname ...)/.." resolves to /var/spool/slurm.
# $SLURM_SUBMIT_DIR is the repo root because submit_ladder.sh cds there before
# submitting; fall back to the script path for direct local invocation.
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# shellcheck source=slurm/ladder_matrix.sh
source "$REPO_ROOT/slurm/ladder_matrix.sh"

TASKS=${TASKS:-$TASKS_DEFAULT}
IDX=${SLURM_ARRAY_TASK_ID:-0}
DRY_RUN=${DRY_RUN:-0}
SMOKE=${SMOKE:-0}
ALLOW_DIRTY=${ALLOW_DIRTY:-0}        # SMOKE=1 implies it: those runs are exploratory
TOTAL_STEPS=${TOTAL_STEPS:-}          # empty => the 50M default in config/reppo.yaml

ladder_decode "$IDX" "$TASKS" || exit 1
task=$LADDER_TASK; arm=$LADDER_ARM; seed=$LADDER_SEED
mode=$(mode_of "$arm")
alpha=$(alpha_of "$task")

# Seeds 101-108 are the RESERVED confirmatory namespace (ledger/README.md) and must
# never be spent on a smoke test. SMOKE=1 moves the run into the 201+ exploratory
# namespace and into a separate output/ledger tree.
if [ "$SMOKE" = 1 ]; then
  seed=$(( 201 + IDX ))
  NAMESPACE=exploratory
  RUN_ROOT=outputs/smoke
  LEDGER_DIR=ledger/runs.d.smoke
  REASON="Cluster smoke test for the confirmatory ladder (not confirmatory evidence)"
else
  NAMESPACE=confirmatory
  RUN_ROOT=outputs/conf
  LEDGER_DIR=ledger/runs.d
  REASON="Registered confirmatory ladder (prereg L.1)"
fi

tag="${task}_${arm}_s${seed}"
run_id="conf-${tag}"

# ---------------------------------------------------------------------------
# Command. Byte-identical to the registered launch command (prereg L.1.7) except
# for hydra.run.dir, which MUST be unique per run: 48 array tasks starting in the
# same second would otherwise all resolve to the same outputs/<date>/<time>/ and
# overwrite each other's metrics.npz (bug L.1.21, at 48x scale).
# ---------------------------------------------------------------------------
cmd="python scripts/train_and_export.py hydra.run.dir=${RUN_ROOT}/${tag} $(env_args "$task") \
seed=${seed} num_trials=1 num_seeds=1 wandb.mode=disabled \
hyperparameters.actor_update_mode=${mode} \
hyperparameters.update_entropy_lagrangian=false hyperparameters.ent_start=${alpha} \
hyperparameters.log_estimator_diag=false hyperparameters.log_eval_iqm=false"
[ -n "$TOTAL_STEPS" ] && cmd="$cmd hyperparameters.total_time_steps=${TOTAL_STEPS}"

if [ "$DRY_RUN" = 1 ]; then
  echo "[$IDX] $tag :: $cmd"
  exit 0
fi

cd "$REPO_ROOT" || exit 1

# ---------------------------------------------------------------------------
# Provenance. A confirmatory git_sha that does not describe the code is worthless,
# so refuse to run from a dirty tree unless explicitly overridden.
# ---------------------------------------------------------------------------
SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)
DIRTY=$(git status --porcelain 2>/dev/null | wc -l)
if [ "$DIRTY" -ne 0 ] && [ "$ALLOW_DIRTY" != 1 ] && [ "$SMOKE" != 1 ]; then
  echo "ERROR: working tree has $DIRTY modified/untracked files. Commit them so the" >&2
  echo "       ledger's git_sha describes the code that produced the run, or set" >&2
  echo "       ALLOW_DIRTY=1 for a throwaway run." >&2
  git status --short >&2
  exit 1
fi

echo "================= REPPO ladder array task ================="
echo "host      : $(hostname)"
echo "date      : $(date -Is)"
echo "job       : ${SLURM_JOB_ID:-local}  array ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}"
echo "commit    : $SHA  (dirty: $DIRTY)"
echo "namespace : $NAMESPACE"
echo "index     : $IDX of $(ladder_size "$TASKS")   tasks: $TASKS"
echo "task      : $task ($(taskname "$task"))"
echo "arm       : $arm  ->  actor_update_mode=$mode"
echo "seed      : $seed"
echo "alpha     : $alpha (frozen, entropy dual off)"
echo "steps     : ${TOTAL_STEPS:-50000000 (config default)}"
echo "-----------------------------------------------------------"
nvidia-smi
echo "==========================================================="

# ---------------------------------------------------------------------------
# Environment. pyproject pins jax[cuda12]==0.5.2, which ships the CUDA runtime,
# cuBLAS and cuDNN as pip wheels inside .venv; JAX loads those in preference to any
# system install, so there is deliberately NO `module load CUDA`.
#
# Compute nodes have no internet. slurm/bootstrap.sh builds the venv and the
# $HPCWORK symlinks once on a login node; this script only checks them.
# ---------------------------------------------------------------------------
VENV=${VENV:-$REPO_ROOT/.venv}
if [ ! -x "$VENV/bin/python" ]; then
  echo "ERROR: no venv at $VENV. Run slurm/bootstrap.sh on a LOGIN node first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

for d in exports outputs logs; do
  if [ ! -e "$d" ]; then
    echo "ERROR: $d/ does not exist. Run slurm/bootstrap.sh on a login node first." >&2
    exit 1
  fi
done

# Match the launch environment registered in prereg L.1.7.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=true   # one job per GPU: take the whole card
export WANDB_MODE=${WANDB_MODE:-disabled}

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_NAME=${GPU_NAME:-unknown}

# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------
t0=$(date +%s); start=$(date -Is)
echo "RUN $tag :: $cmd"
if $cmd; then status=completed; else status=failed; fi
t1=$(date +%s)
echo "DONE $tag status=$status wall_s=$((t1-t0)) $(date -Is)"

# ---------------------------------------------------------------------------
# Ledger. One file per array task -- 48 concurrent appends to a single
# ledger/runs.jsonl would interleave and corrupt lines. slurm/collect_ledger.py
# merges these into the tracked jsonl afterwards.
# ---------------------------------------------------------------------------
mkdir -p "$LEDGER_DIR"
python - "$LEDGER_DIR/$run_id.json" "$run_id" "$NAMESPACE" "$(taskname "$task")" "$arm" \
        "$seed" "$SHA" "$GPU_NAME" "$status" "$((t1-t0))" "$start" "$REASON" \
        "$(algorithm_version_of "$arm")" "${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}" \
        "$cmd" <<'PY'
import json, sys
(out, run_id, ns, task, arm, seed, sha, gpu, status, wall, start,
 reason, algo, slurm_id, cmd) = sys.argv[1:16]
rec = dict(run_id=run_id, namespace=ns, label=f"slurm:{slurm_id}",
           task=task, arm=arm, seed=int(seed), git_sha=sha, gpu=gpu, command=cmd,
           algorithm_version=algo, changed_params=[], reason=reason,
           status=status, start=start, wall_clock_s=int(wall),
           gpu_hours=round(int(wall) / 3600, 4), return_metrics=None,
           estimator_diag=None)
with open(out, "w") as f:
    f.write(json.dumps(rec) + "\n")
PY

[ "$status" = completed ] || exit 1
