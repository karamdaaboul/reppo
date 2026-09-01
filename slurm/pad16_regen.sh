#!/usr/bin/env bash
# Regenerate the 10 k=16 padded WalkerRun checkpoints required by Probe 4 of
# docs/prospective_padding_error_field_analysis.md.
#
# The ORIGINAL checkpoints were produced at 3b96deb on a different machine and are
# absent from CLAIX.  This runs at 07319d4-lineage HEAD, whose only src/ difference
# from 3b96deb is additive and default-off; slurm/pad16_parity.sh verified the two
# commits produce BYTE-IDENTICAL parameters on this exact config (35/35 arrays, max
# abs diff 0.0).  The three known defects are therefore PRESERVED, which is the point:
# Probe 4 analyses critics produced by the original training process, not corrected
# training.
#
# Supersedes the variant in c3b5e05, which was registered but never executed.  That
# one ran a detached 3b96deb worktree on c25g to avoid touching the estep-study tree;
# the byte-identical parity result above makes the worktree unnecessary, so this runs
# in-tree at the analysis commit and additionally writes a post-run ledger entry with
# the checkpoint checksum, which the c3b5e05 version did not.
#
# index 0-4 -> arm A (pathwise), seeds 0-4 ; 5-9 -> arm B (weighted_mle), seeds 0-4
#SBATCH --job-name=pad16-regen
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -uo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_ROOT" || exit 1

IDX=${SLURM_ARRAY_TASK_ID:?}
if [ "$IDX" -lt 5 ]; then MODE=pathwise;     SEED=$IDX;        ARM=A
else                      MODE=weighted_mle; SEED=$((IDX-5));  ARM=B; fi
ALPHA=0.01528     # reports/probe_k6_report.md:32 -- the padding-experiment alpha,
                  # NOT the ladder's recalibrated walker alpha (that one postdates these runs)
PAD=16
TAG="WalkerRun_${MODE}_s${SEED}_pad${PAD}"

SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)
DIRTY=$(git status --porcelain -- ':!*.out' ':!*.err' 2>/dev/null | wc -l)

source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=true WANDB_MODE=disabled
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)

echo "=========== Probe 4 checkpoint regeneration ==========="
echo "host $(hostname) | job ${SLURM_ARRAY_JOB_ID:-}_${IDX} | gpu ${GPU_NAME:-unknown}"
echo "commit $SHA (dirty $DIRTY) | arm $ARM | mode $MODE | seed $SEED | pad $PAD | alpha $ALPHA"
echo "======================================================="

cmd="python scripts/train_and_export.py \
env=mjx_dmc env.name=WalkerRun experiment_overrides=mjx_dmc_large_data \
env.action_pad=${PAD} seed=${SEED} num_trials=1 num_seeds=1 wandb.mode=disabled \
hyperparameters.actor_update_mode=${MODE} \
hyperparameters.update_entropy_lagrangian=false hyperparameters.ent_start=${ALPHA} \
hyperparameters.log_estimator_diag=false hyperparameters.log_eval_iqm=false \
hydra.run.dir=outputs/pad16/${MODE}_s${SEED}"

t0=$(date +%s); start=$(date -Is)
echo "RUN $TAG :: $cmd"
if $cmd; then status=completed; else status=failed; fi
t1=$(date +%s)
echo "DONE $TAG status=$status wall_s=$((t1-t0))"

mkdir -p ledger/runs.d.pad16
python - "ledger/runs.d.pad16/pad16-regen-${ARM}-s${SEED}.json" "$ARM" "$SEED" "$MODE" \
         "$SHA" "${GPU_NAME:-unknown}" "$status" "$((t1-t0))" "$start" \
         "${SLURM_ARRAY_JOB_ID:-}_${IDX}" "$cmd" <<'PY'
import json, sys, os, hashlib, glob
out, arm, seed, mode, sha, gpu, status, wall, start, slurm_id, cmd = sys.argv[1:12]
variant = "_fa_pad16" if mode == "pathwise" else "_pad16"
exp = "exports/WalkerRun_%s%s_s%s_final" % (mode, variant, seed)
csum = None
if os.path.isdir(exp):
    h = hashlib.sha256()
    for f in sorted(glob.glob(exp + "/*")):
        with open(f, "rb") as fh:
            h.update(os.path.basename(f).encode()); h.update(fh.read())
    csum = h.hexdigest()
rec = dict(run_id="pad16-regen-%s-s%s" % (arm, seed), namespace="probe4_regeneration",
           task="WalkerRun", arm=arm, actor_update_mode=mode, seed=int(seed),
           action_pad=16, action_dim=22, git_sha=sha, gpu=gpu, command=cmd,
           label="slurm:%s" % slurm_id, status=status, start=start,
           wall_clock_s=int(wall), gpu_hours=round(int(wall) / 3600, 4),
           export_path=exp, export_present=os.path.isdir(exp), checkpoint_sha256=csum)
with open(out, "w") as f:
    f.write(json.dumps(rec) + "\n")
PY
[ "$status" = completed ] || exit 1
