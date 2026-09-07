#!/usr/bin/env bash
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --job-name=opbsmoke
#SBATCH --output=/hpcwork/qzi10910/estep_wt/slurm/logs/opbsmoke_%j.out
set -uo pipefail
cd /hpcwork/qzi10910/estep_wt
echo "host $(hostname)  commit $(git rev-parse HEAD)  $(date -Is)"
./.venv/bin/python - <<'PY'
"""Smoke test for build_onpolicy_banks: tiny depths, one block per task+arm, writes nothing."""
import os, sys, numpy as np
sys.path.insert(0, os.getcwd())
import importlib.util
spec = importlib.util.spec_from_file_location("bob", "scripts/analysis/build_onpolicy_banks.py")
bob = importlib.util.module_from_spec(spec); spec.loader.exec_module(bob)

# determinism of the key derivation, twice in this process and against a literal
k1, o1 = bob.block_key("walker", "PW_noent", 301)
k2, o2 = bob.block_key("walker", "PW_noent", 301)
assert o1 == o2, "block_key not deterministic within a process"
print("  block_key walker/PW_noent/301 offset =", o1)
assert o1 != bob.block_key("walker", "PW_noent", 302)[1], "offsets collide across seeds"
assert o1 != bob.block_key("walker", "WML_ent", 301)[1], "offsets collide across arms"
assert o1 != bob.block_key("g1", "PW_noent", 301)[1], "offsets collide across tasks"
print("  offsets distinct across seed / arm / task: OK")

SMOKE = (3, 6)
ok = True
for task, cfg in bob.TASKS.items():
    for arm, tag in bob.ARMS.items():
        ck = "exports/%s_%s_s301_final" % (cfg["env"], tag)
        if not os.path.isdir(ck):
            print("  MISSING", ck); ok = False; continue
        key, _ = bob.block_key(task, arm, 301)
        got = bob.rollout(ck, SMOKE, key)
        obs = np.concatenate(got, 0)
        finite = bool(np.isfinite(obs).all())
        print("  %-7s %-9s  blocks=%d  obs=%s  finite=%s"
              % (task, arm, len(got), obs.shape, finite), flush=True)
        ok &= (len(got) == len(SMOKE)) and obs.shape[0] == bob.NENV * len(SMOKE) and finite

# every checkpoint the real run needs must exist
missing = [ "exports/%s_%s_s%d_final" % (c["env"], t, s)
            for task, c in bob.TASKS.items() for a, t in bob.ARMS.items() for s in bob.SEEDS
            if not os.path.isdir("exports/%s_%s_s%d_final" % (c["env"], t, s)) ]
print("  checkpoints required=%d  missing=%d" % (3*2*8, len(missing)))
for m in missing: print("    MISSING", m)
ok &= not missing
print("\nSMOKE = %s" % ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PY
echo "rc=$?  $(date -Is)"
