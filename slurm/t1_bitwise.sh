#!/usr/bin/env bash
# T1: with both flags OFF the new code must reproduce the pre-change commit BITWISE
# on exported actor and critic parameters after a short run, same seed and RNG.
#
# BOTH sides run inside their own detached worktree. train_and_export.py sets
# REPO_ROOT from __file__ and inserts it at sys.path[0], so each side loads its own
# source, and its exports land inside that worktree. Neither side can write to the
# canonical exports/ tree -- an earlier version of this script ran the new side from
# the main repo and destroyed two baseline checkpoints.
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:50:00
#SBATCH --output=slurm/logs/t1_bitwise_%j.out
set -uo pipefail
cd "$HOME/repos/reppo"
PRE="${PRE_SHA:-b130ac7c406b2597188bb8f050614c6eb5af8780}"
NEW="$(git rev-parse HEAD)"
PY="$HOME/repos/reppo/.venv/bin/python"
# Task is parameterised; the defaults are the Walker values, so running with no
# environment set reproduces the previously validated Walker T1 exactly.
ENVNAME="${T1_ENV_NAME:-WalkerRun}"
ENVCFG="${T1_ENV_CFG:-mjx_dmc}"
STEPS="${T1_STEPS:-2621440}"
EXPOVR="${T1_EXP_OVERRIDES:-mjx_dmc_large_data}"
EXTRA="${T1_EXTRA:-}"
echo "pre-change $PRE   new $NEW   $(date -Is)"
WPRE=/hpcwork/qzi10910/t1_pre_${ENVNAME}
WNEW=/hpcwork/qzi10910/t1_new_${ENVNAME}
rm -rf "$WPRE" "$WNEW"
git worktree add --detach "$WPRE" "$PRE" >/dev/null 2>&1 || exit 1
git worktree add --detach "$WNEW" "$NEW" >/dev/null 2>&1 || exit 1

echo "=== source identity check: each side must load its own tree ==="
for W in "$WPRE" "$WNEW"; do
  $PY -c "
import sys
sys.path.insert(0, '$W')
import src.jaxrl.reppo as r
has = 'pw_drop_actor_entropy' in open('$W/src/jaxrl/reppo.py').read()
print('  %-26s module=%s flags_present=%s' % ('$W', r.__file__, has))
"
done

COMMON="env=$ENVCFG env.name=$ENVNAME experiment_overrides=$EXPOVR $EXTRA seed=301 num_trials=1 num_seeds=1 wandb.mode=disabled hyperparameters.num_mini_batches=128 hyperparameters.num_epochs=4 hyperparameters.num_envs=1024 hyperparameters.num_steps=128 hyperparameters.total_time_steps=$STEPS hyperparameters.num_eval=4 hyperparameters.ent_start=${T1_ENT_START:-0.014509912580251694} hyperparameters.update_entropy_lagrangian=false"

for MODE in pathwise weighted_mle; do
  echo "=== $MODE pristine ==="
  (cd "$WPRE" && $PY scripts/train_and_export.py $COMMON hyperparameters.actor_update_mode=$MODE hydra.run.dir=$WPRE/outputs/t1_$MODE 2>&1 | tail -1)
  echo "=== $MODE new, flags OFF ==="
  (cd "$WNEW" && $PY scripts/train_and_export.py $COMMON hyperparameters.actor_update_mode=$MODE hyperparameters.pw_drop_actor_entropy=false hyperparameters.wml_add_actor_entropy=false hydra.run.dir=$WNEW/outputs/t1_$MODE 2>&1 | tail -1)
done

echo
echo "=================== T1 BITWISE COMPARISON"
WPRE="$WPRE" WNEW="$WNEW" T1_ENV_NAME="$ENVNAME" $PY - <<'PYEOF'
import os, numpy as np
wpre, wnew = os.environ["WPRE"], os.environ["WNEW"]
worst, nfile, allident = 0.0, 0, True
for mode, tag in (("pathwise", "pathwise_fa"), ("weighted_mle", "weighted_mle")):
    for base in ("actor.npz", "critic.npz"):
        env = os.environ.get("T1_ENV_NAME", "WalkerRun")
        a = os.path.join(wpre, "exports", "%s_%s_s301_final" % (env, tag), base)
        b = os.path.join(wnew, "exports", "%s_%s_s301_final" % (env, tag), base)
        if not (os.path.exists(a) and os.path.exists(b)):
            print("  %-14s %-11s MISSING" % (mode, base)); allident = False; continue
        za, zb = np.load(a), np.load(b)
        m, ident = 0.0, True
        for k in sorted(za.files):
            x, y = za[k], zb[k]
            if x.shape != y.shape or x.dtype != y.dtype or not np.array_equal(x, y):
                ident = False
            m = max(m, float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))))
        worst = max(worst, m); nfile += 1; allident = allident and ident
        print("  %-14s %-11s %s   max|diff| = %.3e"
              % (mode, base, "IDENTICAL" if ident else "DIFFERS", m))
print()
print("  T1_BITWISE_NOOP max abs diff over %d files = %.3e" % (nfile, worst))
print("  T1 =", "PASS" if (allident and worst == 0.0 and nfile == 4) else "FAIL")
PYEOF

echo "=== canonical exports untouched? ==="
ls -la --time-style=+%Y-%m-%dT%H:%M "$HOME/repos/reppo/exports/${ENVNAME}_pathwise_fa_s302_final/actor.npz"
git worktree remove --force "$WPRE" 2>/dev/null
git worktree remove --force "$WNEW" 2>/dev/null
echo "done $(date -Is)"
