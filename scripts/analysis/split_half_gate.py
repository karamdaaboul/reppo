#!/usr/bin/env python3
"""STEP 1 GATE: split-half robustness of the matched-state width divergence.

The union bank for each task holds 1536 PW-sourced and 1536 WML-sourced states. Both
halves are MATCHED-STATE populations: within a half, every arm is evaluated on exactly
the same states. This recomputes the paired WML/PW width ratio separately on each half,
per task, with a paired bootstrap CI over the eight seeds.

Three pairs are reported. `same_state_width.md` covers only the baseline pair and
carries no intervals, so the entropy-matched pairs and every CI here are new.

Offline. No rollouts, no new banks, no launches, no training.
"""
from __future__ import annotations
import csv, hashlib, json, os, sys
import numpy as np
from scipy.special import erf

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
import jax.numpy as jnp                                    # noqa: E402
from scripts.load_ckpt import load                         # noqa: E402

ART = "reports/artifacts"
SEEDS = list(range(301, 309))
BOOT_SEED = 20260908          # fresh, distinct from the frozen 20260902
NBOOT = 10000
T95, T99 = float(np.arctanh(0.95)), float(np.arctanh(0.99))

TASKS = {
 "walker": dict(env="WalkerRun", bank="walker_fixed_state_bank.npz",
   sha="8adfeb0bf70bddcdbd64a84b972b4dbd62617c64e1c948589a7adc30cf64aa21"),
 "g1": dict(env="G1JoystickFlatTerrain", bank="g1_fixed_state_bank.npz",
   sha="cf6f7880b3a5b59433727f7a245b5ce97749096981f03233fd434ab3371935e9"),
 "leap": dict(env="LeapCubeRotateZAxis", bank="leap_fixed_state_bank.npz",
   sha="0053b0f361e45b9227d15b982ff667a5fc665469dd54e673c70c0682bcb158b2"),
}
TAGS = {"PW_ent": "pathwise_fa", "PW_noent": "pathwise_fa_noent",
        "WML_ent": "weighted_mle_ent", "WML_noent": "weighted_mle"}
# (label, numerator cell, denominator cell)
PAIRS = [("entropy_off", "WML_noent", "PW_noent"),
         ("entropy_on",  "WML_ent",   "PW_ent"),
         ("baseline",    "WML_noent", "PW_ent")]


def Phi(z): return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def sat(mu, sg, c):
    """Exact P(|tanh Y| > tanh(c)) for Y ~ N(mu, sg^2), elementwise."""
    return Phi((-c - mu) / sg) + 1.0 - Phi((c - mu) / sg)


def boot_ci(per_seed_log, rng, nboot=NBOOT):
    """Paired bootstrap over SEEDS of exp(median(log ratio)). Percentile 95%."""
    a = np.asarray(per_seed_log, float)
    n = a.size
    idx = rng.integers(0, n, size=(nboot, n))
    stat = np.exp(np.median(a[idx], axis=1))
    return float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))


def main():
    rng = np.random.default_rng(BOOT_SEED)
    print("bootstrap: paired over seeds, %d resamples, np.random.default_rng(%d), 95%% percentile"
          % (NBOOT, BOOT_SEED))
    print("ratio convention: median over seeds of per-seed log ratios, exponentiated")
    print("per-seed statistic: MEDIAN of pre-tanh sigma over states x action dims\n")

    med, mean_, sats, sib = {}, {}, {}, []
    for task, T in TASKS.items():
        bp = os.path.join(ART, T["bank"])
        assert hashlib.sha256(open(bp, "rb").read()).hexdigest() == T["sha"], task
        z = np.load(bp); obs = jnp.asarray(z["obs"]); src = np.array(z["source"])
        arm = np.array([s.split("-")[0] for s in src])
        halves = {"PW_states": np.where(arm == "PW")[0],
                  "WML_states": np.where(arm == "WML")[0]}
        print("  %-7s bank %s  halves: PW %d  WML %d"
              % (task, T["sha"][:12], halves["PW_states"].size, halves["WML_states"].size))
        for cell, tag in TAGS.items():
            for sd in SEEDS:
                ck = load("exports/%s_%s_s%d_final" % (T["env"], tag, sd))
                mu_f, sg_f = ck.policy_dist(obs)
                mu_f, sg_f = np.asarray(mu_f, np.float64), np.asarray(sg_f, np.float64)
                for hn, hi in halves.items():
                    med[(task, cell, sd, hn)] = float(np.median(sg_f[hi]))
                    mean_[(task, cell, sd, hn)] = float(sg_f[hi].mean())
                if cell in ("PW_noent", "WML_noent"):
                    sats[(task, cell, sd)] = dict(
                        full95=float(sat(mu_f, sg_f, T95).mean()),
                        full99=float(sat(mu_f, sg_f, T99).mean()),
                        pw95=float(sat(mu_f[halves["PW_states"]], sg_f[halves["PW_states"]], T95).mean()),
                        wml95=float(sat(mu_f[halves["WML_states"]], sg_f[halves["WML_states"]], T95).mean()))
                if task == "walker" and cell == "WML_noent":
                    own = np.where(src == "WML-s%d" % sd)[0]
                    oth = np.array([i for i in halves["WML_states"] if src[i] != "WML-s%d" % sd])
                    sib.append(dict(seed=sd, own=float(np.median(sg_f[own])),
                                    sibling=float(np.median(sg_f[oth])),
                                    n_own=int(own.size), n_sib=int(oth.size)))

    rows = []
    print("\n" + "=" * 112)
    print("SPLIT-HALF MATCHED-STATE WIDTH RATIO  (numerator / denominator, both on the SAME states)")
    print("=" * 112)
    print("  %-7s %-12s %-11s %9s %-22s %7s  %s"
          % ("task", "pair", "half", "ratio", "95% CI", "signs", "per-seed ratios"))
    print("  " + "-" * 108)
    for task in TASKS:
        for lab, wc, pc in PAIRS:
            for hn in ("PW_states", "WML_states"):
                per = [np.log(med[(task, wc, s, hn)] / med[(task, pc, s, hn)]) for s in SEEDS]
                r = float(np.exp(np.median(per)))
                lo, hi = boot_ci(per, rng)
                npos = int(sum(1 for x in per if x > 0))
                rows.append(dict(task=task, pair=lab, half=hn, ratio=r, ci_lo=lo, ci_hi=hi,
                                 n_seeds_gt1=npos, excludes_1=bool(lo > 1.0 or hi < 1.0),
                                 per_seed=[float(np.exp(x)) for x in per]))
                print("  %-7s %-12s %-11s %9.3f [%9.3f, %9.3f] %4d/8  %s"
                      % (task, lab, hn, r, lo, hi, npos,
                         " ".join("%.2f" % np.exp(x) for x in per)))
        print("  " + "-" * 108)

    print("\n" + "=" * 112)
    print("GATE  --  same direction on BOTH halves, all three tasks, CI excluding 1")
    print("=" * 112)
    verdict = {}
    for lab, _, _ in PAIRS:
        sub = [r for r in rows if r["pair"] == lab]
        same_dir = all(r["ratio"] > 1 for r in sub) or all(r["ratio"] < 1 for r in sub)
        all_ci = all(r["excludes_1"] for r in sub)
        ok = same_dir and all_ci
        verdict[lab] = ok
        bad = [("%s/%s" % (r["task"], r["half"])) for r in sub if not r["excludes_1"]]
        print("  %-12s  same direction on all 6 half-task cells: %-3s   all CIs exclude 1: %-3s   -> %s"
              % (lab, "YES" if same_dir else "NO", "YES" if all_ci else "NO",
                 "HOLDS" if ok else "FAILS"))
        if bad:
            print("               CI includes 1 at: %s" % ", ".join(bad))
    print("\n  GATE_ENTROPY_OFF (WML_noent / PW_noent) = %s"
          % ("HOLDS" if verdict["entropy_off"] else "FAILS"))

    print("\n" + "=" * 112)
    print("NEUTRAL-BANK SATURATION, matched-state, PW_noent vs WML_noent")
    print("  definition: exact expectation over the STOCHASTIC policy, not Monte Carlo --")
    print("  P(|tanh Y| > t) = Phi((-c-mu)/sigma) + 1 - Phi((c-mu)/sigma), c = atanh(t),")
    print("  Y ~ N(mu, sigma^2) the PRE-TANH Gaussian; averaged over all bank states x action dims.")
    print("  States: the full 3072-state frozen neutral bank for that task. Medians over 8 seeds.")
    print("=" * 112)
    print("  %-7s %-10s %11s %11s | %11s %11s"
          % ("task", "arm", "P|a|>0.95", "P|a|>0.99", "on PW half", "on WML half"))
    satrows = []
    for task in TASKS:
        for cell in ("PW_noent", "WML_noent"):
            g = lambda k: float(np.median([sats[(task, cell, s)][k] for s in SEEDS]))
            satrows.append(dict(task=task, cell=cell, sat95=g("full95"), sat99=g("full99"),
                                sat95_pw_half=g("pw95"), sat95_wml_half=g("wml95")))
            print("  %-7s %-10s %11.5f %11.5f | %11.5f %11.5f"
                  % (task, cell, g("full95"), g("full99"), g("pw95"), g("wml95")))
        print("  " + "-" * 70)

    print("\n" + "=" * 112)
    print("SIBLING-SEED DIAGNOSTIC  --  walker WML_noent, existing bank only, no new rollouts")
    print("  each seed's policy on ITS OWN 192 states vs the other seven seeds' 1344 states")
    print("=" * 112)
    print("  %-6s %11s %11s %9s" % ("seed", "own med", "sibling med", "sib/own"))
    for r in sib:
        r["ratio"] = r["sibling"] / r["own"]
        print("  %-6d %11.4f %11.4f %9.2f" % (r["seed"], r["own"], r["sibling"], r["ratio"]))
    print("  %-6s %11.4f %11.4f %9.2f"
          % ("median", float(np.median([r["own"] for r in sib])),
             float(np.median([r["sibling"] for r in sib])),
             float(np.exp(np.median([np.log(r["ratio"]) for r in sib])))))

    with open(os.path.join(ART, "split_half_gate.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "per_seed"], extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(ART, "split_half_gate.json"), "w") as f:
        json.dump(dict(bootstrap_seed=BOOT_SEED, n_boot=NBOOT, rows=rows,
                       gate=verdict, saturation=satrows, sibling_walker_WML_noent=sib),
                  f, indent=1, default=float)
    print("\n  wrote split_half_gate.csv and split_half_gate.json")


if __name__ == "__main__":
    main()
