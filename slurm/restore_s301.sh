#!/usr/bin/env bash
# RESTORATION ONLY. Rebuild the two seed-301 Walker baseline checkpoints that were
# destroyed by a faulty T1 harness on 2026-09-06. Runs at BASE_SHA in an isolated
# worktree so exports land inside the worktree, never in the canonical exports/ tree.
# The restored files are verified against exports_manifest.csv before any copy.
set -uo pipefail
cd "$HOME/repos/reppo"
BASE=1c6259ef959400eba617e1d1392bfaf58248688c
ARM="$1"          # pathwise | weighted_mle
WT=/hpcwork/qzi10910/restore_wt_${ARM}
PY="$HOME/repos/reppo/.venv/bin/python"
echo "host $(hostname)  restoring $ARM at $BASE  $(date -Is)"
rm -rf "$WT"; git worktree add --detach "$WT" "$BASE" >/dev/null 2>&1 || { echo "worktree failed"; exit 1; }
cd "$WT"
echo "REPO_ROOT will be $WT ; verifying pristine source"
$PY -c "import sys; sys.path.insert(0,'$WT'); import src.jaxrl.reppo as r; print('  reppo module:', r.__file__)"
$PY -c "
import subprocess
print('  worktree HEAD:', subprocess.run(['git','rev-parse','HEAD'],cwd='$WT',capture_output=True,text=True).stdout.strip())
print('  flags present in this source:', 'pw_drop_actor_entropy' in open('$WT/src/jaxrl/reppo.py').read())
"
$PY scripts/train_and_export.py env=mjx_dmc env.name=WalkerRun \
  experiment_overrides=mjx_dmc_large_data seed=301 num_trials=1 num_seeds=1 \
  wandb.mode=disabled hyperparameters.actor_update_mode=$ARM \
  hyperparameters.update_entropy_lagrangian=false \
  hyperparameters.ent_start=0.014509912580251694 \
  hyperparameters.faithful_same_point=true \
  hyperparameters.fresh_minibatch_key=true \
  hyperparameters.log_faithful_diag=true \
  hydra.run.dir=$WT/outputs/restore_${ARM}_s301 2>&1 | tail -3
echo "=== restored exports ==="
ls -d $WT/exports/WalkerRun_*_s301_final 2>/dev/null
echo "done $(date -Is)"
