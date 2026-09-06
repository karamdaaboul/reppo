#!/usr/bin/env bash
# T1b: fixed-batch no-op test at d=29. Both sides consume the SAME dumped batch and the
# SAME PRNG keys, so environment nondeterminism cannot enter the comparison.
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/t1b_%j.out
set -uo pipefail
cd "$HOME/repos/reppo"
PY="$HOME/repos/reppo/.venv/bin/python"
PRE=b130ac7c406b2597188bb8f050614c6eb5af8780
NEW="$(git rev-parse HEAD)"
BATCH=/hpcwork/qzi10910/t1b_batch
echo "pre $PRE  new $NEW  node $(hostname)  $(date -Is)"
WP=/hpcwork/qzi10910/t1b_pre; WN=/hpcwork/qzi10910/t1b_new
rm -rf "$WP" "$WN"
git worktree add --detach "$WP" "$PRE" >/dev/null 2>&1 || exit 1
git worktree add --detach "$WN" "$NEW"  >/dev/null 2>&1 || exit 1
cp scripts/analysis/t1b_fixed_batch.py "$WP/scripts/analysis/"
cp scripts/analysis/t1b_fixed_batch.py "$WN/scripts/analysis/"

for W in "$WP" "$WN"; do
  echo "=== step in $W ==="
  ( cd "$W" && $PY -c "
import sys; sys.path.insert(0,'$W')
print('  flags in this tree:', 'pw_drop_actor_entropy' in open('$W/src/jaxrl/reppo.py').read())
" && $PY scripts/analysis/t1b_fixed_batch.py step "$BATCH" "$W/t1b_out.npz" 2>&1 \
      | grep -vE "^Failed to import|^INFO|RuntimeWarning|return np.asarray" | tail -4 )
done

echo
echo "=================== T1b FIXED-BATCH COMPARISON"
WP="$WP" WN="$WN" $PY - <<'PYEOF'
import os, numpy as np
a = np.load(os.path.join(os.environ["WP"], "t1b_out.npz"))
b = np.load(os.path.join(os.environ["WN"], "t1b_out.npz"))
def cmp(prefix, label):
    ks = sorted(k for k in a.files if k.startswith(prefix))
    worst, ident = 0.0, True
    for k in ks:
        x, y = a[k], b[k]
        if x.shape != y.shape or not np.array_equal(x, y): ident = False
        worst = max(worst, float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))))
    print("  %-28s %-9s over %3d leaves   max|diff| = %.3e"
          % (label, "IDENTICAL" if ident else "DIFFERS", len(ks), worst))
    return ident, worst
i1, w1 = cmp("i_a", "initial actor params")
i2, w2 = cmp("i_c", "initial critic params")
i3, w3 = cmp("a", "post-step actor params")
i4, w4 = cmp("c", "post-step critic params")
init_ok = i1 and i2
post_ok = i3 and i4
print()
print("  INITIAL_STATE_IDENTICAL =", "PASS" if init_ok else "FAIL",
      " (precondition: the two trees must start from the same weights)")
print("  T1b_FIXED_BATCH_NOOP    =", "PASS" if (init_ok and post_ok) else "FAIL",
      " max abs diff = %.3e" % max(w3, w4))
PYEOF
git worktree remove --force "$WP" 2>/dev/null; git worktree remove --force "$WN" 2>/dev/null
echo "done $(date -Is)"
