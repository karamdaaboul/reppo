#!/usr/bin/env python3
"""On-policy versus neutral-bank policy width, all four arms, no PROXY rows.

Closes the gap left by scripts/analysis/onarm_vs_neutral.py, which had to substitute
same-arm BASELINE states for PW_noent and WML_ent because no rollouts of their own
policies existed. docs/protocol_onpolicy_banks.md generated those rollouts; this script
reports every cell against a real on-policy population.

  sigma_onpolicy  that (arm, seed) checkpoint's own 192 rollout states
  sigma_neutral   the full 3072-state frozen neutral bank for that task

Both pre-tanh, sigma = exp(log_std) + min_std, over states and action dims jointly.
Mean and median are both reported: they are known to disagree in this project.

Offline. No training, no launches, no new rollouts.
"""
from __future__ import annotations
import csv, hashlib, json, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
import jax.numpy as jnp                                    # noqa: E402
from scripts.load_ckpt import load                         # noqa: E402

ART = "reports/artifacts"
SEEDS = list(range(301, 309))

TASKS = {
 "walker": dict(env="WalkerRun", neutral="walker_fixed_state_bank.npz",
   sha="8adfeb0bf70bddcdbd64a84b972b4dbd62617c64e1c948589a7adc30cf64aa21"),
 "g1": dict(env="G1JoystickFlatTerrain", neutral="g1_fixed_state_bank.npz",
   sha="cf6f7880b3a5b59433727f7a245b5ce97749096981f03233fd434ab3371935e9"),
 "leap": dict(env="LeapCubeRotateZAxis", neutral="leap_fixed_state_bank.npz",
   sha="0053b0f361e45b9227d15b982ff667a5fc665469dd54e673c70c0682bcb158b2"),
}
NEW = "%s_onpolicy_bank_newarms.npz"

# cell -> (export tag, which bank holds its own rollouts, the source label prefix)
CELLS = {
 "PW_ent":    ("pathwise_fa",        "neutral",  "PW"),
 "PW_noent":  ("pathwise_fa_noent",  "onpolicy", "PW_noent"),
 "WML_ent":   ("weighted_mle_ent",   "onpolicy", "WML_ent"),
 "WML_noent": ("weighted_mle",       "neutral",  "WML"),
}


def sigma(ck, states):
    _, sg = ck.policy_dist(states)
    sg = np.asarray(sg, np.float64)
    return float(sg.mean()), float(np.median(sg))


def main():
    man = json.load(open(os.path.join(ART, "onpolicy_banks_manifest.json")))
    print("on-policy bank generator sha256: %s" % man["script_sha256"])
    rows = []
    for task, T in TASKS.items():
        np_ = os.path.join(ART, T["neutral"])
        got = hashlib.sha256(open(np_, "rb").read()).hexdigest()
        assert got == T["sha"], "neutral bank hash mismatch for %s" % task
        zn = np.load(np_); obs_n = jnp.asarray(zn["obs"]); src_n = zn["source"]

        op = os.path.join(ART, NEW % task)
        assert hashlib.sha256(open(op, "rb").read()).hexdigest() == man["banks"][task]["sha256"]
        zo = np.load(op); obs_o = jnp.asarray(zo["obs"]); src_o = zo["source"]
        print("  %-7s neutral %d states (sha %s)  on-policy %d states (sha %s)"
              % (task, obs_n.shape[0], got[:12], obs_o.shape[0],
                 man["banks"][task]["sha256"][:12]))

        for cell, (tag, which, pref) in CELLS.items():
            for sd in SEEDS:
                ck = load("exports/%s_%s_s%d_final" % (T["env"], tag, sd))
                obs, src = (obs_n, src_n) if which == "neutral" else (obs_o, src_o)
                idx = np.where(src == "%s-s%d" % (pref, sd))[0]
                assert idx.size == 192, "%s %s s%d: %d own states" % (task, cell, sd, idx.size)
                on_mean, on_med = sigma(ck, obs[jnp.asarray(idx)])
                ne_mean, ne_med = sigma(ck, obs_n)
                rows.append(dict(task=task, cell=cell, seed=sd, n_onpolicy=int(idx.size),
                                 sigma_onpolicy_mean=on_mean, sigma_onpolicy_median=on_med,
                                 sigma_neutral_mean=ne_mean, sigma_neutral_median=ne_med,
                                 ratio_neutral_over_onpolicy_median=ne_med / on_med))

    print("\n" + "=" * 104)
    print("PER (TASK, ARM, SEED)  --  pre-tanh sigma.  Every row is a REAL on-policy population.")
    print("=" * 104)
    print("  %-7s %-10s %-5s | %11s %11s | %11s %11s | %8s"
          % ("task", "arm", "seed", "on-pol mean", "on-pol med",
             "neutral mean", "neutral med", "neu/on"))
    print("  " + "-" * 100)
    last = None
    for r in rows:
        if last and (r["task"], r["cell"]) != last:
            print("  " + "-" * 100)
        last = (r["task"], r["cell"])
        print("  %-7s %-10s %-5d | %11.4f %11.4f | %11.4f %11.4f | %8.3f"
              % (r["task"], r["cell"], r["seed"], r["sigma_onpolicy_mean"],
                 r["sigma_onpolicy_median"], r["sigma_neutral_mean"],
                 r["sigma_neutral_median"], r["ratio_neutral_over_onpolicy_median"]))

    print("\n" + "=" * 104)
    print("PER-CELL SUMMARY  --  median over the eight seeds")
    print("=" * 104)
    print("  %-7s %-10s | %11s %11s | %11s %11s | %8s"
          % ("task", "arm", "on-pol mean", "on-pol med", "neutral mean", "neutral med", "neu/on"))
    print("  " + "-" * 100)
    summ = []
    for task in TASKS:
        for cell in CELLS:
            sub = [r for r in rows if r["task"] == task and r["cell"] == cell]
            g = lambda k: float(np.median([r[k] for r in sub]))
            s = dict(task=task, cell=cell,
                     on_mean=g("sigma_onpolicy_mean"), on_med=g("sigma_onpolicy_median"),
                     ne_mean=g("sigma_neutral_mean"), ne_med=g("sigma_neutral_median"))
            s["ratio"] = s["ne_med"] / s["on_med"]
            summ.append(s)
            print("  %-7s %-10s | %11.4f %11.4f | %11.4f %11.4f | %8.3f"
                  % (task, cell, s["on_mean"], s["on_med"], s["ne_mean"], s["ne_med"], s["ratio"]))
        print("  " + "-" * 100)

    print("\n" + "=" * 104)
    print("PAIRED WML/PW WIDTH RATIO ON BOTH POPULATIONS, ENTROPY-MATCHED PAIRS")
    print("  convention: median over seeds of per-seed log ratios, exponentiated")
    print("=" * 104)
    pairs = []
    for task in TASKS:
        for lab, wc, pc in (("entropy on ", "WML_ent", "PW_ent"),
                            ("entropy off", "WML_noent", "PW_noent"),
                            ("baseline    ", "WML_noent", "PW_ent")):
            out = {}
            for sub, k in (("on-policy", "sigma_onpolicy_median"),
                           ("neutral  ", "sigma_neutral_median")):
                per = []
                for sd in SEEDS:
                    w = [r for r in rows if r["task"] == task and r["cell"] == wc and r["seed"] == sd][0][k]
                    p = [r for r in rows if r["task"] == task and r["cell"] == pc and r["seed"] == sd][0][k]
                    per.append(np.log(w / p))
                out[sub] = float(np.exp(np.median(per)))
                pairs.append(dict(task=task, pair=lab.strip(), population=sub.strip(),
                                  ratio=out[sub]))
            agree = (out["on-policy"] > 1.0) == (out["neutral  "] > 1.0)
            print("  %-7s %s  on-policy %8.3fx   neutral %8.3fx   SIGN %s"
                  % (task, lab, out["on-policy"], out["neutral  "],
                     "AGREE" if agree else "REVERSES"))

    with open(os.path.join(ART, "onarm_vs_neutral_v2.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(ART, "onarm_vs_neutral_v2.json"), "w") as f:
        json.dump(dict(rows=rows, per_cell=summ, pairs=pairs), f, indent=1, default=float)
    print("\n  wrote onarm_vs_neutral_v2.csv and onarm_vs_neutral_v2.json")
    print("  PROXY rows: 0 of %d" % len(rows))


if __name__ == "__main__":
    main()
