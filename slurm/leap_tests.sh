#!/usr/bin/env bash
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:50:00
#SBATCH --output=slurm/logs/leaptests_%j.out
set -uo pipefail
cd "$HOME/repos/reppo"
PY="$HOME/repos/reppo/.venv/bin/python"
echo "host $(hostname)  commit $(git rev-parse HEAD)  $(date -Is)"
echo "########## T2-T5 at d=16 (LEAP)"
EF_TEST_CKPT=exports/LeapCubeRotateZAxis_weighted_mle_s301_final \
EF_TEST_CKPT_PW=exports/LeapCubeRotateZAxis_pathwise_fa_s301_final \
EF_TEST_BANK=reports/artifacts/leap_fixed_state_bank.npz \
EF_TEST_LABEL="LEAP d=16" $PY tests/test_entropy_factorial.py
echo "rc_t2t5=$?"
echo
echo "########## T1b fixed-batch no-op at d=16"
export T1B_TASK=leap
BATCH=/hpcwork/qzi10910/t1b_leap_batch
$PY scripts/analysis/t1b_fixed_batch.py dump "$BATCH" 2>&1 | tail -3
PRE=b130ac7c406b2597188bb8f050614c6eb5af8780
NEW="$(git rev-parse HEAD)"
WP=/hpcwork/qzi10910/t1b_leap_pre; WN=/hpcwork/qzi10910/t1b_leap_new
rm -rf "$WP" "$WN"
git worktree add --detach "$WP" "$PRE" >/dev/null 2>&1 || exit 1
git worktree add --detach "$WN" "$NEW"  >/dev/null 2>&1 || exit 1
cp scripts/analysis/t1b_fixed_batch.py "$WP/scripts/analysis/"
cp scripts/analysis/t1b_fixed_batch.py "$WN/scripts/analysis/"
for W in "$WP" "$WN"; do
  ( cd "$W" && $PY -c "print('  flags in tree:', 'pw_drop_actor_entropy' in open('$W/src/jaxrl/reppo.py').read())" \
    && T1B_TASK=leap $PY scripts/analysis/t1b_fixed_batch.py step "$BATCH" "$W/t1b_out.npz" 2>&1 \
       | grep -vE "^Failed to import|^INFO|RuntimeWarning|return np.asarray" | tail -3 )
done
echo
echo "=================== T1b LEAP COMPARISON"
WP="$WP" WN="$WN" $PY - <<'PYEOF'
import os, numpy as np
a = np.load(os.path.join(os.environ["WP"], "t1b_out.npz"))
b = np.load(os.path.join(os.environ["WN"], "t1b_out.npz"))
def cmp(pre, label):
    ks = sorted(k for k in a.files if k.startswith(pre))
    worst, ident = 0.0, True
    for k in ks:
        x, y = a[k], b[k]
        if x.shape != y.shape or not np.array_equal(x, y): ident = False
        worst = max(worst, float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))))
    print("  %-26s %-9s over %3d leaves   max|diff| = %.3e" % (label, "IDENTICAL" if ident else "DIFFERS", len(ks), worst))
    return ident, worst
i1,_ = cmp("i_a", "initial actor params"); i2,_ = cmp("i_c", "initial critic params")
i3,w3 = cmp("a", "post-step actor params"); i4,w4 = cmp("c", "post-step critic params")
print()
print("  INITIAL_STATE_IDENTICAL =", "PASS" if (i1 and i2) else "FAIL")
print("  LEAP_T1B =", "PASS" if (i1 and i2 and i3 and i4) else "FAIL", " max abs diff = %.3e" % max(w3, w4))
PYEOF
git worktree remove --force "$WP" 2>/dev/null; git worktree remove --force "$WN" 2>/dev/null
echo "done $(date -Is)"
