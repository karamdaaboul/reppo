#!/usr/bin/env python3
"""P2: the decisive same-state test. Baseline pair only, all three tasks.

Offline. No rollouts, no new banks, no launches. Both policies are evaluated on the
EXACT SAME states, separately on each half of the existing bank. Halves are never
averaged together. Definitions per docs/protocol_width_populations.md (commit 2cb7ab7).
"""
import csv, hashlib, json, os, sys
import numpy as np
import jax, jax.numpy as jnp
from scipy.stats import norm

sys.path.insert(0, os.getcwd())
from scripts.load_ckpt import load

ART = "reports/artifacts"
SEEDS = list(range(301, 309))
T95, T99 = float(np.arctanh(0.95)), float(np.arctanh(0.99))
TASKS = {
 "walker": dict(env="WalkerRun", bank="walker_fixed_state_bank.npz",
   sha="8adfeb0bf70bddcdbd64a84b972b4dbd62617c64e1c948589a7adc30cf64aa21"),
 "g1": dict(env="G1JoystickFlatTerrain", bank="g1_fixed_state_bank.npz",
   sha="cf6f7880b3a5b59433727f7a245b5ce97749096981f03233fd434ab3371935e9"),
 "leap": dict(env="LeapCubeRotateZAxis", bank="leap_fixed_state_bank.npz",
   sha="0053b0f361e45b9227d15b982ff667a5fc665469dd54e673c70c0682bcb158b2"),
}
CELLS = {"PW_ent": "pathwise_fa", "WML_noent": "weighted_mle"}

def sat(mu, sg, c):
    return norm.cdf((-c - mu) / sg) + 1.0 - norm.cdf((c - mu) / sg)

def summarise(ck, states):
    mu, sg = ck.policy_dist(states)
    mu, sg = np.asarray(mu), np.asarray(sg)
    return dict(median=float(np.median(sg)), mean=float(sg.mean()),
                median_log=float(np.exp(np.median(np.log(sg)))),
                p75=float(np.percentile(sg, 75)), p90=float(np.percentile(sg, 90)),
                p95=float(np.percentile(sg, 95)), p99=float(np.percentile(sg, 99)),
                max=float(sg.max()),
                sat95=float(sat(mu, sg, T95).mean()), sat99=float(sat(mu, sg, T99).mean()),
                median_abs_mu=float(np.median(np.abs(mu))))

def main():
    rows, split_rows = [], []
    for task, T in TASKS.items():
        bp = os.path.join(ART, T["bank"])
        assert hashlib.sha256(open(bp, "rb").read()).hexdigest() == T["sha"], task
        z = np.load(bp); obs = jnp.asarray(z["obs"]); src = z["source"]
        arm = np.array([s.split("-")[0] for s in src])
        halves = {"PW_half": np.where(arm == "PW")[0], "WML_half": np.where(arm == "WML")[0]}
        for cell, tag in CELLS.items():
            for sd in SEEDS:
                ck = load("exports/%s_%s_s%d_final" % (T["env"], tag, sd))
                for hname, hidx in halves.items():
                    r = summarise(ck, obs[jnp.asarray(hidx)])
                    rows.append(dict(task=task, cell=cell, seed=sd, half=hname, **r))
                    harm = hname.split("_")[0]
                    same = np.where(src == "%s-s%d" % (harm, sd))[0]
                    other = np.array([i for i in hidx if src[i] != "%s-s%d" % (harm, sd)])
                    split_rows.append(dict(task=task, cell=cell, seed=sd, half=hname,
                        same_seed_median=summarise(ck, obs[jnp.asarray(same)])["median"],
                        other_seed_median=summarise(ck, obs[jnp.asarray(other)])["median"],
                        n_same=len(same), n_other=len(other)))

    def med(task, cell, half, k):
        v = [r[k] for r in rows if r["task"] == task and r["cell"] == cell and r["half"] == half]
        return float(np.median(v))

    print("=" * 122)
    print("P2. SAME-STATE TEST — both policies on identical states, halves never averaged")
    print("=" * 122)
    for task in TASKS:
        print("\n########## %s" % task)
        print("  %-10s %-9s | %8s %8s %8s | %7s %7s %7s %7s %9s | %7s %7s | %8s"
              % ("cell","half","median","mean","med-log","p75","p90","p95","p99","max",
                 "sat95","sat99","med|mu|"))
        for half in ("PW_half", "WML_half"):
            for cell in CELLS:
                print("  %-10s %-9s | %8.4f %8.3f %8.4f | %7.3f %7.3f %7.3f %7.3f %9.1f | %7.4f %7.4f | %8.4f"
                      % (cell, half, med(task,cell,half,"median"), med(task,cell,half,"mean"),
                         med(task,cell,half,"median_log"), med(task,cell,half,"p75"),
                         med(task,cell,half,"p90"), med(task,cell,half,"p95"),
                         med(task,cell,half,"p99"), med(task,cell,half,"max"),
                         med(task,cell,half,"sat95"), med(task,cell,half,"sat99"),
                         med(task,cell,half,"median_abs_mu")))
            print()

    print("=" * 122)
    print("PAIRED MATCHED-STATE RATIOS  sigma_WML / sigma_PW on identical states")
    print("  convention: median over seeds of per-seed log ratios, exponentiated")
    print("=" * 122)
    ratios = []
    for task in TASKS:
        for half, name in (("PW_half", "r_PWstates"), ("WML_half", "r_WMLstates")):
            per = []
            for sd in SEEDS:
                w = [r for r in rows if r["task"]==task and r["cell"]=="WML_noent"
                     and r["half"]==half and r["seed"]==sd][0]["median"]
                p = [r for r in rows if r["task"]==task and r["cell"]=="PW_ent"
                     and r["half"]==half and r["seed"]==sd][0]["median"]
                per.append(w / p)
            rat = float(np.exp(np.median(np.log(per))))
            ratios.append(dict(task=task, half=half, name=name, ratio=rat,
                               per_seed=[float(x) for x in per]))
            print("  %-7s %-12s %8.3fx | per seed %s"
                  % (task, name, rat, " ".join("%6.3f" % v for v in per)))
        a = [r for r in ratios if r["task"]==task and r["half"]=="PW_half"][0]["ratio"]
        b = [r for r in ratios if r["task"]==task and r["half"]=="WML_half"][0]["ratio"]
        print("  %-7s  both halves agree in sign (%s 1): %s\n"
              % (task, ">" if a > 1 else "<", "YES" if (a > 1) == (b > 1) else "NO"))

    print("=" * 122)
    print("SEED SPLIT — same-seed states versus the other seven seeds, within each half")
    print("=" * 122)
    print("  %-7s %-10s %-9s | %10s %10s %7s | n_same n_other"
          % ("task","cell","half","same-seed","other-seed","ratio"))
    for task in TASKS:
        for cell in CELLS:
            for half in ("PW_half", "WML_half"):
                s = [r for r in split_rows if r["task"]==task and r["cell"]==cell and r["half"]==half]
                a = float(np.median([x["same_seed_median"] for x in s]))
                b = float(np.median([x["other_seed_median"] for x in s]))
                print("  %-7s %-10s %-9s | %10.4f %10.4f %7.2f | %6d %6d"
                      % (task, cell, half, a, b, b/a, s[0]["n_same"], s[0]["n_other"]))
        print()

    with open(os.path.join(ART, "same_state_test.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(ART, "same_state_seedsplit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(split_rows[0])); w.writeheader(); w.writerows(split_rows)
    with open(os.path.join(ART, "same_state_test.json"), "w") as f:
        json.dump(dict(cells=rows, splits=split_rows, ratios=ratios), f, indent=1, default=float)
    print("  wrote same_state_test.csv, same_state_seedsplit.csv, same_state_test.json")

if __name__ == "__main__":
    main()
