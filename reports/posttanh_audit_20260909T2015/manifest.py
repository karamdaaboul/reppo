#!/usr/bin/env python3
"""Phase 2 run manifest: 3 tasks x 3 arms x 8 seeds = 72 expected cells. READ-ONLY."""
import csv, hashlib, json, os, sys

REPO = "/hpcwork/qzi10910/estep_wt"
os.chdir(REPO)
OUT = sys.argv[1]
SEEDS = list(range(301, 309))
TASKS = {"walker": "WalkerRun", "g1": "G1JoystickFlatTerrain", "leap": "LeapCubeRotateZAxis"}
ARMS = {"PW": "pathwise_fa_noent", "WML05": "weighted_mle", "WML01": "weighted_mle_eps01"}
BANK_SHA = {
 "walker": "8adfeb0bf70bddcdbd64a84b972b4dbd62617c64e1c948589a7adc30cf64aa21",
 "g1": "cf6f7880b3a5b59433727f7a245b5ce97749096981f03233fd434ab3371935e9",
 "leap": "0053b0f361e45b9227d15b982ff667a5fc665469dd54e673c70c0682bcb158b2",
}
# code defaults for fields that may be absent from stored configs (src/jaxrl/reppo.py)
DEFAULTS = {"eps_e": 0.5, "mstep_decoupled": False, "update_entropy_lagrangian": True,
            "sqrt_rho": 1.0, "wml_add_actor_entropy": False, "pw_drop_actor_entropy": False,
            "actor_min_std": 0.0, "normalize_env": True}
REQ = ("actor.npz", "critic.npz", "normalizer.npz", "meta.json")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


rows, missing = [], []
for tk, env in TASKS.items():
    for arm, tag in ARMS.items():
        for s in SEEDS:
            d = "exports/%s_%s_s%d_final" % (env, tag, s)
            present = [f for f in REQ if os.path.exists(os.path.join(d, f))]
            complete = len(present) == len(REQ)
            r = dict(task=tk, env=env, arm=arm, tag=tag, seed=s, dir=d,
                     complete=complete, files_present="|".join(present))
            if not complete:
                missing.append((tk, arm, s)); rows.append(r); continue
            m = json.load(open(os.path.join(d, "meta.json")))
            ak = m.get("actor_kwargs") or {}
            eff = lambda k: m.get(k) if m.get(k) is not None else DEFAULTS.get(k)
            r.update(
                algo=m.get("actor_update_mode"),
                eps_e_stored=m.get("eps_e"), eps_e_effective=eff("eps_e"),
                seed_meta=m.get("seed"), seed_index=m.get("seed_index"),
                time_steps=m.get("time_steps"), iteration=m.get("iteration"),
                checkpoint_frac=m.get("checkpoint_frac"), num_eval=m.get("num_eval"),
                estep_num_samples=m.get("estep_num_samples"),
                obs_dim=m.get("obs_dim"), action_dim=m.get("action_dim"),
                action_pad=m.get("action_pad"),
                actor_hidden=ak.get("hidden_dim"), actor_min_std=ak.get("min_std"),
                ent_start=ak.get("ent_start"),
                normalize_env=eff("normalize_env"), normalizer_eps=m.get("normalizer_eps"),
                sha_actor=sha(os.path.join(d, "actor.npz"))[:16],
                sha_critic=sha(os.path.join(d, "critic.npz"))[:16],
                sha_normalizer=sha(os.path.join(d, "normalizer.npz"))[:16],
                git_commit=m.get("git_commit"),
                is_final=d.endswith("_final"),
            )
            rows.append(r)

print("=== 2.1 CELL COMPLETENESS (72 expected) ===")
print("  %-8s %-7s %s" % ("task", "arm", "seeds complete"))
inter = {}
for tk in TASKS:
    ok_by_arm = {}
    for arm in ARMS:
        ok = sorted(r["seed"] for r in rows if r["task"] == tk and r["arm"] == arm and r["complete"])
        ok_by_arm[arm] = set(ok)
        print("  %-8s %-7s %d/8  %s" % (tk, arm, len(ok), ok))
    inter[tk] = sorted(set.intersection(*ok_by_arm.values()))
    print("  %-8s MATCHED SEED INTERSECTION (all 3 arms): n=%d  %s" % (tk, len(inter[tk]), inter[tk]))
print("\n  total complete cells: %d/72   missing: %s" % (sum(r["complete"] for r in rows), missing or "none"))

print("\n=== 2.2 EFFECTIVE CONFIG CONSISTENCY (per task x arm, over seeds) ===")
CHK = ("algo", "eps_e_effective", "time_steps", "iteration", "checkpoint_frac", "num_eval",
       "estep_num_samples", "obs_dim", "action_dim", "action_pad", "actor_hidden",
       "actor_min_std", "normalize_env", "normalizer_eps", "ent_start")
for tk in TASKS:
    for arm in ARMS:
        sub = [r for r in rows if r["task"] == tk and r["arm"] == arm and r["complete"]]
        if not sub:
            continue
        bad = {k: sorted({json.dumps(r.get(k)) for r in sub}) for k in CHK
               if len({json.dumps(r.get(k)) for r in sub}) > 1}
        v = sub[0]
        print("  %-8s %-7s algo=%-12s eps_e=%-4s steps=%-9s iter=%-4s frac=%-4s evals=%-3s "
              "M=%-4s obs=%-4s act=%-3s pad=%-4s hid=%-4s min_std=%-5s norm=%-5s eps=%-5s"
              % (tk, arm, v["algo"], v["eps_e_effective"], v["time_steps"], v["iteration"],
                 v["checkpoint_frac"], v["num_eval"], v["estep_num_samples"], v["obs_dim"],
                 v["action_dim"], v["action_pad"], v["actor_hidden"], v["actor_min_std"],
                 v["normalize_env"], v["normalizer_eps"]))
        if bad:
            print("     !! VARIES ACROSS SEEDS: %s" % bad)

print("\n=== 2.3 NORMALIZER IDENTITY (each policy has its own; check they differ) ===")
for tk in TASKS:
    for arm in ARMS:
        sub = [r for r in rows if r["task"] == tk and r["arm"] == arm and r["complete"]]
        n = len({r["sha_normalizer"] for r in sub})
        print("  %-8s %-7s distinct normalizer hashes across %d seeds: %d" % (tk, arm, len(sub), n))

print("\n=== 2.4 STATE BANK IDENTITY ===")
for tk in TASKS:
    p = "reports/artifacts/%s_fixed_state_bank.npz" % tk
    g = sha(p)
    print("  %-8s neutral bank sha256 %s  MATCHES REGISTERED: %s" % (tk, g[:16], g == BANK_SHA[tk]))
for tk in TASKS:
    for b in ("onpolicy_bank_newarms", "onpolicy_bank_eps01"):
        p = "reports/artifacts/%s_%s.npz" % (tk, b)
        print("  %-8s %-22s sha256 %s  exists=%s" % (tk, b, sha(p)[:16] if os.path.exists(p) else "-", os.path.exists(p)))

print("\n=== 2.5 PW BASELINES AND BANKS UNMODIFIED BY THE eps_E EXPERIMENT ===")
B = "/hpcwork/qzi10910/logsplit/baselines_before.txt"
if os.path.exists(B):
    ok = bad = 0
    for ln in open(B):
        p = ln.split()
        if len(p) != 3:
            continue
        d, ah, ch = p
        a = sha("exports/%s/actor.npz" % d)[:16]; c = sha("exports/%s/critic.npz" % d)[:16]
        if a == ah and c == ch:
            ok += 1
        else:
            bad += 1; print("     CHANGED: %s" % d)
    print("  pre-eps01 baseline hashes: %d match, %d changed  -> BASELINES_UNCHANGED = %s"
          % (ok, bad, "YES" if bad == 0 else "NO"))
else:
    print("  reference hash file absent: %s" % B)

with open(os.path.join(OUT, "run_manifest.csv"), "w", newline="") as fh:
    keys = sorted(set().union(*[set(r) for r in rows]))
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
json.dump({k: v for k, v in inter.items()}, open(os.path.join(OUT, "matched_seeds.json"), "w"), indent=1)
print("\n  wrote %s/run_manifest.csv (%d rows) and matched_seeds.json" % (OUT, len(rows)))
