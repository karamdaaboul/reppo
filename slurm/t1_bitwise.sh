#!/usr/bin/env bash
# T1: with both flags OFF the new code must reproduce the pre-change baseline
# BITWISE on actor and critic parameters after N updates, same RNG.
# Builds a pristine worktree at the pre-change commit and compares exported params.
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/t1_bitwise_%j.out
set -uo pipefail
cd "$HOME/repos/reppo"
PRE="${PRE_SHA:-b130ac7c406b2597188bb8f050614c6eb5af8780}"
NEW="$(git rev-parse HEAD)"
echo "pre-change $PRE   new $NEW   $(date -Is)"
PY="$HOME/repos/reppo/.venv/bin/python"
WT=/tmp/t1_pristine_$$
rm -rf "$WT"; git worktree add --detach "$WT" "$PRE" >/dev/null 2>&1 || exit 1
ln -sfn "$HOME/repos/reppo/.venv" "$WT/.venv" 2>/dev/null

# a tiny but real config: same scientific shape, tiny budget
COMMON="env=mjx_dmc env.name=WalkerRun experiment_overrides=mjx_dmc_large_data \
seed=301 num_trials=1 num_seeds=1 wandb.mode=disabled \
hyperparameters.num_mini_batches=128 hyperparameters.num_epochs=4 \
hyperparameters.num_envs=1024 hyperparameters.num_steps=128 \
hyperparameters.total_time_steps=2621440 hyperparameters.num_eval=4 \
hyperparameters.ent_start=0.014509912580251694 \
hyperparameters.update_entropy_lagrangian=false"

for MODE in pathwise weighted_mle; do
  echo "=== $MODE : pristine ($PRE) ==="
  (cd "$WT" && $PY scripts/train_and_export.py $COMMON \
      hyperparameters.actor_update_mode=$MODE \
      hydra.run.dir=/tmp/t1_pre_$MODE 2>&1 | tail -2)
  echo "=== $MODE : new code, flags OFF ($NEW) ==="
  $PY scripts/train_and_export.py $COMMON \
      hyperparameters.actor_update_mode=$MODE \
      hyperparameters.pw_drop_actor_entropy=false \
      hyperparameters.wml_add_actor_entropy=false \
      hydra.run.dir=/tmp/t1_new_$MODE 2>&1 | tail -2
done

echo; echo "=================== T1 BITWISE COMPARISON"
$PY - <<'PY'
import glob, os, numpy as np
worst = 0.0; nfile = 0
for mode in ("pathwise", "weighted_mle"):
    for base in ("actor.npz", "critic.npz"):
        a = sorted(glob.glob("/tmp/t1_pre_%s/**/%s" % (mode, base), recursive=True))
        b = sorted(glob.glob("/tmp/t1_new_%s/**/%s" % (mode, base), recursive=True))
        if not a or not b:
            print("  %-14s %-11s MISSING (pre=%d new=%d)" % (mode, base, len(a), len(b)))
            continue
        za, zb = np.load(a[-1]), np.load(b[-1])
        m = 0.0; ident = True
        for k in sorted(za.files):
            x, y = za[k], zb[k]
            if x.shape != y.shape or x.dtype != y.dtype:
                ident = False; continue
            if not np.array_equal(x, y):
                ident = False
            m = max(m, float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))))
        worst = max(worst, m); nfile += 1
        print("  %-14s %-11s %s   max|diff| = %.3e" % (
            mode, base, "IDENTICAL" if ident else "DIFFERS", m))
print("\n  T1_BITWISE_NOOP max abs diff over %d files = %.3e" % (nfile, worst))
print("  T1 =", "PASS" if worst == 0.0 and nfile == 4 else "FAIL")
PY
git worktree remove --force "$WT" 2>/dev/null
echo "done $(date -Is)"
