#!/usr/bin/env bash
# REPPO training on RWTH CLAIX-2023, one array task per seed.
#
# Submit from the repo root -- Slurm's working directory is wherever sbatch is
# called from, and every path below is relative to it.
#
#   sbatch slurm/train.sh                                  # 3 seeds, default arm
#   ENVNAME=HumanoidRun ACTOR_UPDATE_MODE=weighted_mle sbatch -A rwth1234 slurm/train.sh
#
# For the registered confirmatory ladder use slurm/ladder.sh instead -- this
# script maps one array index to one SEED and cannot express task x arm x seed.
#
# Billing (RWTH, from 01.11.2025): one GPU on c23g is limited to 24 cores and
# 122 GB, and 1 GPU-hour is charged as 24 core-hours. The header below requests
# exactly that share of a node (4x H100, 96 cores).

#SBATCH --job-name=reppo
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=02:00:00                # ~2x the slowest measured 50M-step run
# --account is NOT set here: pass it at submit time, e.g. `sbatch -A rwth1234 ...`,
# or export SBATCH_ACCOUNT=rwth1234 once per session.
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs   # slurm --output target; see README for the $HPCWORK symlink

# ---------------------------------------------------------------------------
# Config via environment variables. Every default matches config/reppo.yaml +
# experiment_overrides/mjx_dmc_large_data.yaml, so an unset variable changes
# nothing. Override on the sbatch command line to switch arms.
# ---------------------------------------------------------------------------
ENVNAME=${ENVNAME:-WalkerRun}                        # mujoco_playground id, CamelCase
TASK=${TASK:-mjx_dmc}                                # hydra env group: mjx_dmc | mjx_humanoid | brax
ACTOR_UPDATE_MODE=${ACTOR_UPDATE_MODE:-pathwise}     # pathwise | weighted_mle
ALPHA=${ALPHA:-}                                     # set to FREEZE the entropy dual at this value
EPS_E=${EPS_E:-0.5}                                  # E-step KL budget (weighted_mle)
EPS_MU=${EPS_MU:-0.1}                                # decoupled M-step mean bound
EPS_SIGMA=${EPS_SIGMA:-5.0e-5}                       # decoupled M-step scale bound
BETA_SIGMA_FIXED=${BETA_SIGMA_FIXED:-}               # set to hold beta_sigma constant
MSTEP_DECOUPLED=${MSTEP_DECOUPLED:-false}            # true enables the decoupled M-step
ESTEP_NUM_SAMPLES=${ESTEP_NUM_SAMPLES:-32}           # M samples per state
TOTAL_STEPS=${TOTAL_STEPS:-50000000}
OVERRIDES=${OVERRIDES:-mjx_dmc_large_data}           # hydra experiment_overrides group

SEED=${SLURM_ARRAY_TASK_ID:-0}

# Optional overrides are only appended when set, so the config default applies
# otherwise -- exactly as it does for a local run.
EXTRA=()
if [[ -n "$ALPHA" ]]; then
  EXTRA+=(hyperparameters.update_entropy_lagrangian=false "hyperparameters.ent_start=$ALPHA")
fi
if [[ -n "$BETA_SIGMA_FIXED" ]]; then
  EXTRA+=("hyperparameters.beta_sigma_fixed=$BETA_SIGMA_FIXED")
fi

# ---------------------------------------------------------------------------
# Provenance. Every run must be traceable to a commit.
# ---------------------------------------------------------------------------
echo "================= REPPO array task ================="
echo "host      : $(hostname)"
echo "date      : $(date -Is)"
echo "job       : ${SLURM_JOB_ID:-local}  array ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}"
echo "commit    : $(git rev-parse HEAD 2>/dev/null || echo 'not a git checkout')"
echo "dirty     : $(git status --porcelain 2>/dev/null | wc -l) modified/untracked files"
echo "seed      : $SEED"
echo "env       : $TASK / $ENVNAME"
echo "arm       : actor_update_mode=$ACTOR_UPDATE_MODE decoupled=$MSTEP_DECOUPLED"
echo "alpha     : ${ALPHA:-learned}"
echo "eps_e     : $EPS_E   eps_mu: $EPS_MU   eps_sigma: $EPS_SIGMA"
echo "beta_sig  : ${BETA_SIGMA_FIXED:-learned}"
echo "M         : $ESTEP_NUM_SAMPLES   steps: $TOTAL_STEPS   overrides: $OVERRIDES"
echo "-----------------------------------------------------"
nvidia-smi
echo "====================================================="

# ---------------------------------------------------------------------------
# Environment. pyproject pins `jax[cuda12]==0.5.2`, which installs the CUDA
# runtime, cuBLAS, cuDNN, etc. as pip wheels under .venv/lib/.../nvidia/. JAX
# loads those in preference to any system CUDA, so NO `module load CUDA` is
# needed -- and loading one risks a version mismatch with the bundled libs.
#
# Compute nodes have no internet, so the venv must ALREADY exist on shared
# storage. Build it ONCE on a login node from the repo root:
#     uv sync --frozen
# This script only checks for it and refuses to run otherwise.
# ---------------------------------------------------------------------------
VENV=${VENV:-$PWD/.venv}
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "ERROR: no venv at $VENV. Run 'uv sync --frozen' on a LOGIN node first." >&2
  exit 1
fi
source "$VENV/bin/activate"

# ---------------------------------------------------------------------------
# Outputs. $HOME has a small quota; checkpoints and slurm logs go to $HPCWORK.
# exports/ and logs/ in the repo are symlinks into it, so the entrypoint and
# scripts/load_ckpt.py keep their relative paths unchanged.
# ---------------------------------------------------------------------------
OUT_ROOT=${OUT_ROOT:-${HPCWORK:-$PWD}/reppo_runs}
mkdir -p "$OUT_ROOT/exports" "$OUT_ROOT/logs"
[[ -L exports || ! -e exports ]] && ln -sfn "$OUT_ROOT/exports" exports
echo "outputs   : $OUT_ROOT  (exports/ -> $(readlink -f exports))"

export XLA_PYTHON_CLIENT_PREALLOCATE=true   # one job per GPU: take the whole card
export WANDB_MODE=${WANDB_MODE:-disabled}

# ---------------------------------------------------------------------------
# Same entrypoint and override syntax as a local run (scripts/train_and_export.py).
# Checkpoints land in exports/<env>_<arm>_s<seed>_{p25,p50,final}/.
# ---------------------------------------------------------------------------
python scripts/train_and_export.py \
    env="$TASK" \
    env.name="$ENVNAME" \
    experiment_overrides="$OVERRIDES" \
    seed="$SEED" \
    num_trials=1 \
    num_seeds=1 \
    wandb.mode="$WANDB_MODE" \
    hyperparameters.total_time_steps="$TOTAL_STEPS" \
    hyperparameters.actor_update_mode="$ACTOR_UPDATE_MODE" \
    hyperparameters.estep_num_samples="$ESTEP_NUM_SAMPLES" \
    hyperparameters.eps_e="$EPS_E" \
    hyperparameters.mstep_decoupled="$MSTEP_DECOUPLED" \
    hyperparameters.eps_mu="$EPS_MU" \
    hyperparameters.eps_sigma="$EPS_SIGMA" \
    "${EXTRA[@]}"
