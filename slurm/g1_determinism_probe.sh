#!/usr/bin/env bash
# CONTROL for the G1 T1 failure: is G1 reproducible at all?
# Runs the SAME commit with the SAME config twice, flags untouched, and compares
# exported parameters bitwise. If these two differ, G1 is not run-to-run
# deterministic and a bitwise no-op test cannot discriminate on this task.
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/g1_determinism_%j.out
set -uo pipefail
cd "$HOME/repos/reppo"
PY="$HOME/repos/reppo/.venv/bin/python"
NEW="$(git rev-parse HEAD)"
echo "commit $NEW  node $(hostname)  $(date -Is)"
WA=/hpcwork/qzi10910/g1_det_a; WB=/hpcwork/qzi10910/g1_det_b
rm -rf "$WA" "$WB"
git worktree add --detach "$WA" "$NEW" >/dev/null 2>&1 || exit 1
git worktree add --detach "$WB" "$NEW" >/dev/null 2>&1 || exit 1

COMMON="env=mjx_humanoid env.name=G1JoystickFlatTerrain env.asymmetric_obs=false \
experiment_overrides=mjx_humanoid_large_data seed=301 num_trials=1 num_seeds=1 \
wandb.mode=disabled hyperparameters.num_mini_batches=128 hyperparameters.num_epochs=4 \
hyperparameters.num_envs=1024 hyperparameters.num_steps=128 \
hyperparameters.total_time_steps=2621440 hyperparameters.num_eval=4 \
hyperparameters.ent_start=0.00020752247655764222 \
hyperparameters.update_entropy_lagrangian=false \
hyperparameters.actor_update_mode=pathwise"

for W in "$WA" "$WB"; do
  echo "=== run in $W (identical code, identical config) ==="
  (cd "$W" && $PY scripts/train_and_export.py $COMMON hydra.run.dir=$W/outputs/det 2>&1 | tail -1)
done

echo
echo "=================== SAME-CODE REPRODUCIBILITY"
WA="$WA" WB="$WB" $PY - <<'PYEOF'
import os, numpy as np
wa, wb = os.environ["WA"], os.environ["WB"]
worst, ident_all = 0.0, True
for base in ("actor.npz", "critic.npz"):
    a = os.path.join(wa, "exports", "G1JoystickFlatTerrain_pathwise_fa_s301_final", base)
    b = os.path.join(wb, "exports", "G1JoystickFlatTerrain_pathwise_fa_s301_final", base)
    if not (os.path.exists(a) and os.path.exists(b)):
        print("  %-11s MISSING" % base); ident_all = False; continue
    za, zb = np.load(a), np.load(b)
    m, ident = 0.0, True
    for k in sorted(za.files):
        x, y = za[k], zb[k]
        if not np.array_equal(x, y): ident = False
        m = max(m, float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))))
    worst = max(worst, m); ident_all &= ident
    print("  %-11s %s   max|diff| = %.3e" % (base, "IDENTICAL" if ident else "DIFFERS", m))
print()
print("  G1_SAME_CODE_REPRODUCIBLE =", "YES" if ident_all and worst == 0.0 else "NO")
print("  -> if NO, G1 is not run-to-run deterministic and the T1 bitwise premise")
print("     does not hold for this task; the earlier T1 FAIL is uninformative.")
PYEOF
git worktree remove --force "$WA" 2>/dev/null; git worktree remove --force "$WB" 2>/dev/null
echo "done $(date -Is)"
