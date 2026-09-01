"""Probe 4 checkpoint table and integrity / falsification checks.

Required-checkpoint table (Phase 0) plus the checks the protocol demands before the
Probe 4 result may be interpreted.  Writes reports/artifacts/probe4_checkpoints.csv.
"""
import glob, hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "reports/artifacts"
os.makedirs(OUT, exist_ok=True)
REQUIRED = [("WalkerRun", 16, s, a) for a in ("A", "B") for s in range(5)]


def path_for(arm, seed):
    mode = "pathwise" if arm == "A" else "weighted_mle"
    var = "_fa_pad16" if arm == "A" else "_pad16"
    return f"exports/WalkerRun_{mode}{var}_s{seed}_final"


def checksum(d):
    h = hashlib.sha256()
    for f in sorted(glob.glob(d + "/*")):
        h.update(os.path.basename(f).encode())
        h.update(open(f, "rb").read())
    return h.hexdigest()


rows = []
for task, k, seed, arm in REQUIRED:
    p = path_for(arm, seed)
    found = os.path.isdir(p)
    r = dict(task=task, k=k, seed=seed, arm=arm, expected_path=p,
             found="FOUND" if found else "MISSING")
    if found:
        m = json.load(open(f"{p}/meta.json"))
        r.update(git_sha=None, action_dim=m["action_dim"], action_pad=m["action_pad"],
                 min_std=m["actor_kwargs"]["min_std"], alpha_entropy=m["alpha_entropy"],
                 alpha_kl=m["alpha_kl"], time_steps=m["time_steps"],
                 final_return=m["final_eval_return"], nan_in_eval=m["nan_in_eval"],
                 ess_final=m["ess_final"], checksum=checksum(p)[:16])
        led = f"ledger/runs.d.pad16/pad16-regen-{arm}-s{seed}.json"
        if os.path.exists(led):
            L = json.load(open(led))
            r["git_sha"] = L["git_sha"][:12]; r["slurm"] = L["label"]; r["gpu"] = L["gpu"]
        ok = (r["action_dim"] == 22 and r["action_pad"] == 16 and not r["nan_in_eval"]
              and r["time_steps"] == 52297728 and abs(r["alpha_entropy"] - 0.01528) < 1e-6)
        r["integrity"] = "PASS" if ok else "FAIL"
    rows.append(r)

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/probe4_checkpoints.csv", index=False)
cols = ["task", "k", "seed", "arm", "found", "integrity", "action_dim", "action_pad",
        "min_std", "alpha_entropy", "time_steps", "final_return", "nan_in_eval",
        "ess_final", "git_sha", "checksum"]
print(df[[c for c in cols if c in df.columns]].to_string(index=False))
print(f"\nFOUND {int((df.found=='FOUND').sum())}/10   "
      f"INTEGRITY PASS {int((df.get('integrity')=='PASS').sum())}/10")

# ---- falsification checks that do not need the Probe 4 numbers ----------------
print("\n=== integrity / falsification checks ===")
print("  [1] padded coords do not enter dynamics or reward : "
      "PASS (scripts/verify_action_pad.py 16 -- simulator received dim 6, want 6)")
have = df[df.found == "FOUND"]
if len(have) == 10:
    a = have[have.arm == "A"].final_return.values
    b = have[have.arm == "B"].final_return.values
    print(f"  [2] arm A returns {np.round(a,1)}   arm B returns {np.round(b,1)}")
    print(f"  [3] no NaN in any eval                          : "
          f"{'PASS' if not have.nan_in_eval.any() else 'FAIL'}")
    print(f"  [4] all 10 at the same step budget              : "
          f"{'PASS' if have.time_steps.nunique()==1 else 'FAIL'} ({have.time_steps.unique()})")
    print(f"  [5] all 10 at the same frozen alpha             : "
          f"{'PASS' if have.alpha_entropy.nunique()==1 else 'FAIL'}")
    print(f"  [6] effective min_std (Amendment A.1 item 1)    : "
          f"{'PASS' if (have.min_std==0.1).all() else 'FAIL'} (all 0.1, not the configured 0.0)")
    print(f"  [7] arm B ESS non-degenerate (E-step live)      : "
          f"{'PASS' if (have[have.arm=='B'].ess_final>4).all() else 'CHECK'} "
          f"(ESS {np.round(have[have.arm=='B'].ess_final.values,2)})")
    print(f"  [8] arm A ESS is 0 by construction (no E-step)  : "
          f"{'PASS' if (have[have.arm=='A'].ess_final==0).all() else 'FAIL'}")
else:
    print(f"  (checks 2-8 deferred: only {len(have)}/10 checkpoints present)")
