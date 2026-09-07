#!/usr/bin/env python3
"""On-arm versus neutral-bank width and saturation, all tasks, all available cells.

Offline. No training, no launches. Reports both populations side by side and does not
choose between them.

CAVEAT, stated in the output: each bank was built from that task's two BASELINE runs, so
its `source` labels are PW-sNNN (from PW_ent) and WML-sNNN (from WML_noent). For the two
baseline cells "on-arm" therefore means the states that cell's own policy visited. For the
two new cells (PW_noent, WML_ent) no states from their own rollouts exist, so "on-arm" is
the same-arm BASELINE states, a proxy. Marked PROXY in the table.
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
FR = "/rwthfs/rz/cluster/hpcwork/qzi10910/reppo_runs/outputs/faithful_repair"
EF = "/rwthfs/rz/cluster/hpcwork/qzi10910/reppo_runs/outputs/entropy_factorial"
G1EF = "/rwthfs/rz/cluster/hpcwork/qzi10910/reppo_runs/outputs/g1_entropy_factorial"
LC = "/rwthfs/rz/cluster/hpcwork/qzi10910/reppo_runs/outputs/leap_corrected"

TASKS = {
 "walker": dict(env="WalkerRun", bank="walker_fixed_state_bank.npz",
   sha="8adfeb0bf70bddcdbd64a84b972b4dbd62617c64e1c948589a7adc30cf64aa21",
   cells={"PW_ent":("pathwise_fa","PW",FR+"/walker_PW1_s%d",False),
          "PW_noent":("pathwise_fa_noent","PW",EF+"/walker_PW-H_s%d",True),
          "WML_ent":("weighted_mle_ent","WML",EF+"/walker_WML+H_s%d",True),
          "WML_noent":("weighted_mle","WML",FR+"/walker_WML32_s%d",False)}),
 "g1": dict(env="G1JoystickFlatTerrain", bank="g1_fixed_state_bank.npz",
   sha="cf6f7880b3a5b59433727f7a245b5ce97749096981f03233fd434ab3371935e9",
   cells={"PW_ent":("pathwise_fa","PW",FR+"/g1_PW1_s%d",False),
          "PW_noent":("pathwise_fa_noent","PW",G1EF+"/g1_PW_noent_s%d",True),
          "WML_ent":("weighted_mle_ent","WML",G1EF+"/g1_WML_ent_s%d",True),
          "WML_noent":("weighted_mle","WML",FR+"/g1_WML32_s%d",False)}),
 "leap": dict(env="LeapCubeRotateZAxis", bank="leap_fixed_state_bank.npz",
   sha="0053b0f361e45b9227d15b982ff667a5fc665469dd54e673c70c0682bcb158b2",
   cells={"PW_ent":("pathwise_fa","PW",LC+"/leap_PW_s%d",False),
          "WML_noent":("weighted_mle","WML",LC+"/leap_WML_s%d",False)}),
}

def sat(mu, sg, c):
    return norm.cdf((-c - mu) / sg) + 1.0 - norm.cdf((c - mu) / sg)

def logged_sigma(pat, sd):
    p = (pat % sd) + "/metrics.npz"
    if not os.path.exists(p): return float("nan")
    z = np.load(p)
    k = "train/pi_sigma_mean"
    return float(np.asarray(z[k]).reshape(-1)[-1]) if k in z.files else float("nan")

def stats(ck, states):
    mu, sg = ck.policy_dist(states)
    mu, sg = np.asarray(mu), np.asarray(sg)
    return (float(np.median(sg)), float(sat(mu, sg, T95).mean()), float(sat(mu, sg, T99).mean()))

def main():
    rows, halves, per_seed = [], [], {}
    for task, T in TASKS.items():
        bp = os.path.join(ART, T["bank"])
        assert hashlib.sha256(open(bp, "rb").read()).hexdigest() == T["sha"], task
        z = np.load(bp); obs = jnp.asarray(z["obs"]); src = z["source"]
        armlab = np.array([s.split("-")[0] for s in src])
        for cell, (tag, armkey, runpat, proxy) in T["cells"].items():
            acc = {"on_arm": [], "neutral": []}
            for sd in SEEDS:
                ck = load("exports/%s_%s_s%d_final" % (T["env"], tag, sd))
                idx = np.where(src == "%s-s%d" % (armkey, sd))[0]
                on = stats(ck, obs[jnp.asarray(idx)])
                ne = stats(ck, obs)
                acc["on_arm"].append(on); acc["neutral"].append(ne)
                per_seed[(task, cell, sd)] = dict(on_med=on[0], neu_med=ne[0])
                if task == "walker":
                    hp = np.where(armlab == "PW")[0]; hw = np.where(armlab == "WML")[0]
                    halves.append(dict(task=task, cell=cell, seed=sd,
                        full=ne[0], pw_half=stats(ck, obs[jnp.asarray(hp)])[0],
                        wml_half=stats(ck, obs[jnp.asarray(hw)])[0]))
            m = lambda sub, i: float(np.median([a[i] for a in acc[sub]]))
            rows.append(dict(task=task, cell=cell, proxy_on_arm=proxy,
                on_arm_median_sigma=m("on_arm", 0), neutral_median_sigma=m("neutral", 0),
                ratio_neutral_over_on_arm=m("neutral", 0) / m("on_arm", 0),
                on_arm_sat95=m("on_arm", 1), on_arm_sat99=m("on_arm", 2),
                neutral_sat95=m("neutral", 1), neutral_sat99=m("neutral", 2),
                logged_pi_sigma_mean=float(np.median([logged_sigma(runpat, s) for s in SEEDS]))))

    print("=" * 118)
    print("1. ON-ARM VERSUS NEUTRAL, ALL TASKS, ALL AVAILABLE CELLS  (medians over seeds 301-308)")
    print("=" * 118)
    print("  PROXY = the bank has no states from that policy's own rollouts (it was built from")
    print("  the two baselines), so on-arm uses the same-arm BASELINE states.\n")
    print("  %-7s %-10s %-6s | %9s %9s %7s | %7s %7s | %7s %7s | %9s"
          % ("task","cell","proxy","on-arm","neutral","ratio","on s95","on s99","neu s95","neu s99","logged sg"))
    print("  " + "-" * 114)
    for r in rows:
        print("  %-7s %-10s %-6s | %9.4f %9.4f %7.2f | %7.4f %7.4f | %7.4f %7.4f | %9.4f"
              % (r["task"], r["cell"], "PROXY" if r["proxy_on_arm"] else "own",
                 r["on_arm_median_sigma"], r["neutral_median_sigma"], r["ratio_neutral_over_on_arm"],
                 r["on_arm_sat95"], r["on_arm_sat99"], r["neutral_sat95"], r["neutral_sat99"],
                 r["logged_pi_sigma_mean"]))

    print("\n" + "=" * 118)
    print("2. PAIRED WML/PW WIDTH RATIO, BOTH POPULATIONS")
    print("   convention: median over seeds of per-seed log ratios, exponentiated")
    print("=" * 118)
    pairs = []
    for task in TASKS:
        for lab, wc, pc in (("ent", "WML_ent", "PW_ent"), ("noent", "WML_noent", "PW_noent"),
                            ("baseline pair", "WML_noent", "PW_ent")):
            if (task, wc, 301) not in per_seed or (task, pc, 301) not in per_seed: continue
            for sub, kk in (("neutral", "neu_med"), ("on_arm", "on_med")):
                per = [np.log(per_seed[(task, wc, s)][kk] / per_seed[(task, pc, s)][kk]) for s in SEEDS]
                pairs.append(dict(task=task, pair=lab, subset=sub,
                    ratio=float(np.exp(np.median(per))),
                    per_seed=[float(np.exp(x)) for x in per]))
    for task in TASKS:
        sub = [p for p in pairs if p["task"] == task]
        if not sub: continue
        print("\n  --- %s ---" % task)
        for lab in ("ent", "noent", "baseline pair"):
            rs = [p for p in sub if p["pair"] == lab]
            if not rs: continue
            for p in rs:
                print("   %-14s %-8s ratio %7.3fx | per seed %s"
                      % (p["pair"], p["subset"], p["ratio"],
                         " ".join("%6.3f" % v for v in p["per_seed"])))
            a = [p for p in rs if p["subset"] == "neutral"][0]["ratio"]
            b = [p for p in rs if p["subset"] == "on_arm"][0]["ratio"]
            agree = (a > 1.0) == (b > 1.0)
            print("   %-14s SIGN AGREE (both %s 1): %s   neutral %.3fx vs on-arm %.3fx"
                  % (lab, ">" if a > 1 else "<", "YES" if agree else "NO -- REVERSES", a, b))

    print("\n" + "=" * 118)
    print("3. BANK_HALVES_AGREE, WALKER, ALL FOUR CELLS")
    print("=" * 118)
    print("  %-10s | %9s %9s %9s | %s" % ("cell", "full", "pw_half", "wml_half", "max half-to-half rel. diff"))
    hv = []
    for cell in ("PW_ent", "PW_noent", "WML_ent", "WML_noent"):
        sub = [h for h in halves if h["cell"] == cell]
        f, p, w = (float(np.median([x[k] for x in sub])) for k in ("full", "pw_half", "wml_half"))
        rel = abs(p - w) / max(p, w)
        hv.append(dict(cell=cell, full=f, pw_half=p, wml_half=w, rel_diff=rel,
                       agree_within_10pct=bool(rel <= 0.10)))
        print("  %-10s | %9.4f %9.4f %9.4f | %.1f%%  %s"
              % (cell, f, p, w, 100 * rel, "AGREE <=10%" if rel <= 0.10 else "DISAGREE >10%"))
    print("\n  BANK_HALVES_AGREE (walker, all four within 10%%) = %s"
          % ("YES" if all(h["agree_within_10pct"] for h in hv) else "NO"))

    with open(os.path.join(ART, "onarm_vs_neutral_cells.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(ART, "onarm_vs_neutral.json"), "w") as f:
        json.dump(dict(cells=rows, pairs=pairs, walker_halves=hv), f, indent=1, default=float)
    print("\n  wrote onarm_vs_neutral_cells.csv and onarm_vs_neutral.json")

if __name__ == "__main__":
    main()
