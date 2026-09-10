#!/usr/bin/env python3
"""E-step concentration arm on its OWN occupancy vs the neutral bank.

Closes docs/prereg_estep_concentration.md section 4.2, the secondary geometry endpoint.
The primary endpoint (matched-state, neutral bank) is already reported: the eps_e = 0.1 arm
collapses matched-state width 0.148x on walker, 0.476x on g1, 1.386x on leap. This script
asks whether that collapse also appears when each arm is measured on the states it actually
visits, which is the population contrast the project's headline result rests on.

Conventions, all frozen elsewhere and reused verbatim:
  sigma           pre-tanh, via ck.policy_dist, over states and action dims jointly
  ratio           median over seeds of per-seed log ratios, exponentiated
  bootstrap       10000 resamples over seeds, percentile interval
Populations per docs/protocol_width_populations.md.

Offline. No training, no launches, no new rollouts.
"""
from __future__ import annotations
import hashlib, json, os, sys
import numpy as np
from scipy.special import erf

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
import jax.numpy as jnp                                    # noqa: E402
from scripts.load_ckpt import load                         # noqa: E402

ART, SEEDS, NB, BOOT = "reports/artifacts", list(range(301, 309)), 10000, 20260911
T95, T99 = float(np.arctanh(0.95)), float(np.arctanh(0.99))   # prereg 4.3

TASKS = {
 "walker": dict(env="WalkerRun", neutral="walker_fixed_state_bank.npz",
   sha="8adfeb0bf70bddcdbd64a84b972b4dbd62617c64e1c948589a7adc30cf64aa21"),
 "g1": dict(env="G1JoystickFlatTerrain", neutral="g1_fixed_state_bank.npz",
   sha="cf6f7880b3a5b59433727f7a245b5ce97749096981f03233fd434ab3371935e9"),
 "leap": dict(env="LeapCubeRotateZAxis", neutral="leap_fixed_state_bank.npz",
   sha="0053b0f361e45b9227d15b982ff667a5fc665469dd54e673c70c0682bcb158b2"),
}

# arm -> (export tag, which bank carries its own rollouts, source-label prefix)
ARMS = {
 "WML_eps01": ("weighted_mle_eps01",  "eps01",   "WML_eps01"),
 "WML_eps05": ("weighted_mle",        "neutral", "WML"),
 "PW_noent":  ("pathwise_fa_noent",   "newarms", "PW_noent"),
}


def _Phi(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def sigma(ck, states):
    """Pre-tanh sigma summaries plus prereg 4.3 saturation, exact under the policy."""
    mu, sg = ck.policy_dist(states)
    mu = np.asarray(mu, np.float64); sg = np.asarray(sg, np.float64)
    sat = lambda c: float((_Phi((-c - mu) / sg) + 1.0 - _Phi((c - mu) / sg)).mean())
    return float(sg.mean()), float(np.median(sg)), sat(T95), sat(T99)


def boot_ratio(per_seed_log, rng):
    a = np.asarray(per_seed_log, float)
    i = rng.integers(0, a.size, size=(NB, a.size))
    s = np.exp(np.median(a[i], axis=1))
    return float(np.exp(np.median(a))), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    man_new = json.load(open(os.path.join(ART, "onpolicy_banks_manifest.json")))
    man_e01 = json.load(open(os.path.join(ART, "onpolicy_banks_eps01_manifest.json")))
    print("neutral-bank hashes verified inline;")
    print("newarms generator sha256 %s" % man_new["script_sha256"][:16])
    print("eps01   generator sha256 %s" % man_e01["script_sha256"][:16])
    print("bootstrap rng %d, %d resamples\n" % (BOOT, NB))

    rows = []
    for task, T in TASKS.items():
        p = os.path.join(ART, T["neutral"])
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        assert got == T["sha"], "neutral bank hash mismatch for %s" % task
        zn = np.load(p); obs_n = jnp.asarray(zn["obs"]); src_n = zn["source"]

        banks = {"neutral": (obs_n, src_n)}
        for key, fn, man in (("newarms", "%s_onpolicy_bank_newarms.npz", man_new),
                             ("eps01",   "%s_onpolicy_bank_eps01.npz",   man_e01)):
            fp = os.path.join(ART, fn % task)
            h = hashlib.sha256(open(fp, "rb").read()).hexdigest()
            assert h == man["banks"][task]["sha256"], "%s bank hash mismatch for %s" % (key, task)
            z = np.load(fp); banks[key] = (jnp.asarray(z["obs"]), z["source"])

        print("  %-7s neutral %d  newarms %d  eps01 %d states, all hashes OK"
              % (task, obs_n.shape[0], banks["newarms"][0].shape[0], banks["eps01"][0].shape[0]))

        for arm, (tag, which, pref) in ARMS.items():
            for sd in SEEDS:
                ck = load("exports/%s_%s_s%d_final" % (T["env"], tag, sd))
                obs, src = banks[which]
                idx = np.where(src == "%s-s%d" % (pref, sd))[0]
                assert idx.size == 192, "%s %s s%d: %d own states" % (task, arm, sd, idx.size)
                on_mean, on_med, on_s95, on_s99 = sigma(ck, obs[jnp.asarray(idx)])
                ne_mean, ne_med, ne_s95, ne_s99 = sigma(ck, obs_n)
                rows.append(dict(task=task, arm=arm, seed=sd,
                                 own_mean=on_mean, own_med=on_med,
                                 own_sat95=on_s95, own_sat99=on_s99,
                                 neu_mean=ne_mean, neu_med=ne_med,
                                 neu_sat95=ne_s95, neu_sat99=ne_s99))

    def get(task, arm, sd, k):
        return [r for r in rows if r["task"] == task and r["arm"] == arm and r["seed"] == sd][0][k]

    print("\n" + "=" * 92)
    print("PER-ARM SUMMARY  --  median over the eight seeds, pre-tanh sigma")
    print("=" * 92)
    print("  %-7s %-11s | %3s | %11s %11s | %11s %11s | %-26s"
          % ("task", "arm", "n", "own mean", "own med", "neutral mean", "neutral med",
             "common-to-own amplification"))
    print("  " + "-" * 104)
    rng = np.random.default_rng(BOOT)
    for task in TASKS:
        for arm in ARMS:
            sub = [r for r in rows if r["task"] == task and r["arm"] == arm]
            g = lambda k: float(np.median([r[k] for r in sub]))
            # FROZEN CONVENTION: median over seeds of per-seed log ratios, exponentiated.
            # (Was a ratio of two across-seed medians, which diverges from this under the
            #  heavy skew these distributions have. Corrected; CI added.)
            per = [np.log(r["neu_med"] / r["own_med"]) for r in sub
                   if r["own_med"] and r["neu_med"] and r["own_med"] > 0 and r["neu_med"] > 0]
            amp = boot_ratio(per, rng) if per else (float("nan"),) * 3
            print("  %-7s %-11s | %3d | %11.4f %11.4f | %11.4f %11.4f | %8.3fx [%.3f, %.3f]"
                  % (task, arm, len(per), g("own_mean"), g("own_med"), g("neu_mean"),
                     g("neu_med"), amp[0], amp[1], amp[2]))
        print("  " + "-" * 104)
    print("\n" + "=" * 92)
    print("PRIMARY  --  does the eps01 width collapse survive on OWN occupancy?")
    print("  eps01 / eps05 width ratio, paired within seed, both populations")
    print("=" * 92)
    print("  %-7s | %-34s | %-34s" % ("task", "OWN states (each arm's own)", "NEUTRAL bank (matched-state)"))
    print("  " + "-" * 88)
    out = []
    for task in TASKS:
        cells = []
        for pop, ka, kb in (("own", "own_med", "own_med"), ("neutral", "neu_med", "neu_med")):
            per = [np.log(get(task, "WML_eps01", sd, ka) / get(task, "WML_eps05", sd, kb)) for sd in SEEDS]
            m, lo, hi = boot_ratio(per, rng)
            cells.append((m, lo, hi))
            out.append(dict(task=task, stat="eps01_over_eps05", population=pop,
                            ratio=m, lo=lo, hi=hi))
        f = lambda c: "%6.3fx [%.3f, %.3f]%s" % (c[0], c[1], c[2], "  *" if (c[1] - 1) * (c[2] - 1) > 0 else "   ")
        print("  %-7s | %-34s | %-34s" % (task, f(cells[0]), f(cells[1])))
    print("  " + "-" * 88)
    print("  * interval excludes 1")

    print("\n" + "=" * 92)
    print("OPERATOR GAP vs PW_noent, both populations")
    print("=" * 92)
    print("  %-7s %-11s | %-34s | %-34s" % ("task", "arm", "OWN states", "NEUTRAL bank"))
    print("  " + "-" * 88)
    for task in TASKS:
        for arm in ("WML_eps05", "WML_eps01"):
            cells = []
            for pop, k in (("own", "own_med"), ("neutral", "neu_med")):
                per = [np.log(get(task, arm, sd, k) / get(task, "PW_noent", sd, k)) for sd in SEEDS]
                m, lo, hi = boot_ratio(per, rng)
                cells.append((m, lo, hi))
                out.append(dict(task=task, stat="%s_over_PW" % arm, population=pop,
                                ratio=m, lo=lo, hi=hi))
            f = lambda c: "%6.3fx [%.3f, %.3f]%s" % (c[0], c[1], c[2], "  *" if (c[1] - 1) * (c[2] - 1) > 0 else "   ")
            print("  %-7s %-11s | %-34s | %-34s" % (task, arm, f(cells[0]), f(cells[1])))
        print("  " + "-" * 88)

    print("\n" + "=" * 92)
    print("SATURATION  --  prereg 4.3, exact under the policy, median over seeds")
    print("=" * 92)
    print("  %-7s %-11s | %9s %9s | %9s %9s"
          % ("task", "arm", "own s95", "own s99", "neu s95", "neu s99"))
    print("  " + "-" * 88)
    for task in TASKS:
        for arm in ARMS:
            sub = [r for r in rows if r["task"] == task and r["arm"] == arm]
            g = lambda k: float(np.median([r[k] for r in sub]))
            print("  %-7s %-11s | %9.4f %9.4f | %9.4f %9.4f"
                  % (task, arm, g("own_sat95"), g("own_sat99"), g("neu_sat95"), g("neu_sat99")))
        print("  " + "-" * 88)

    import csv
    with open(os.path.join(ART, "onarm_eps01_rows.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(ART, "onarm_eps01_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print("\n  wrote reports/artifacts/onarm_eps01_rows.csv and onarm_eps01_summary.csv")


if __name__ == "__main__":
    main()
