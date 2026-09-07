#!/usr/bin/env python3
"""Stage A: measure the actor-entropy force F_H on existing baseline checkpoints.

Offline. No training, no launches. Six baseline cells, seeds 301-308 each.

Convention:
    F := -dL/dlog_sigma      F > 0 widens, F < 0 contracts
    F_H = alpha * [ 1 - 2 * sigma^2 * E(sech^2(y)) ],  y = mu + sigma*u, u ~ N(0,I)

F_H is the closed form reached from the pathwise derivative through Stein's lemma,
E[u*tanh(mu+s*u)] = s*E[sech^2(y)], so it equals the autodiff derivative only in
expectation. Part 0 verifies that against JAX autodiff on the repo's own squashed
log-prob and reports the max abs error and its 1/sqrt(N) convergence.

Each task uses its OWN frozen alpha, read from the exports.
"""
import csv, hashlib, json, os, sys
import numpy as np
import jax, jax.numpy as jnp
import distrax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from scripts.load_ckpt import load

ART = "reports/artifacts"
FIGD = os.path.join(ART, "figs_entropy_force")
SEEDS = list(range(301, 309))
K = 256                      # u-samples per (state, coordinate)
RNG = 20260907
os.makedirs(FIGD, exist_ok=True)

TASKS = {
    "walker": dict(env="WalkerRun", bank="walker_fixed_state_bank.npz",
                   sha="8adfeb0bf70bddcdbd64a84b972b4dbd62617c64e1c948589a7adc30cf64aa21", d=6),
    "g1":     dict(env="G1JoystickFlatTerrain", bank="g1_fixed_state_bank.npz",
                   sha="cf6f7880b3a5b59433727f7a245b5ce97749096981f03233fd434ab3371935e9", d=29),
    "leap":   dict(env="LeapCubeRotateZAxis", bank="leap_fixed_state_bank.npz",
                   sha="0053b0f361e45b9227d15b982ff667a5fc665469dd54e673c70c0682bcb158b2", d=16),
}
ARMS = {"PW_ent": ("pathwise_fa", "PW"), "WML_noent": ("weighted_mle", "WML")}
out = {}

def sech2(y):
    return 1.0 - jnp.tanh(y) ** 2

# ----------------------------------------------------------------- part 0
def verify():
    print("=" * 78 + "\n0. VERIFY F_H AGAINST AUTODIFF ON THE REPO'S SQUASHED LOG-PROB\n" + "=" * 78)
    print("  L_H = alpha * E[log pi(a|s)] with a = tanh(y); F := -dL/dlog_sigma")
    print("  closed form  F_H = alpha * [1 - 2 sigma^2 E(sech^2 y)]\n")
    print("    mu   sigma |   -autodiff        F_H        |err|")
    worst = 0.0
    for mu0 in (-1.0, 0.0, 0.7):
        for sg0 in (0.2, 0.5, 1.5):
            n = 2_000_000
            u = jax.random.normal(jax.random.PRNGKey(0), (n,))
            mu = jnp.full((n,), mu0); sg = jnp.full((n,), sg0)
            def L(t):
                s = sg * jnp.exp(t)
                y = mu + s * u
                d = distrax.Transformed(distrax.Normal(mu, s), distrax.Tanh())
                return jnp.mean(d.log_prob(jnp.clip(jnp.tanh(y), -1 + 1e-4, 1 - 1e-4)))
            F_ad = -float(jax.grad(L)(0.0))
            y = mu + sg * u
            F_cf = float(1.0 - 2.0 * jnp.mean(sg**2 * sech2(y)))
            worst = max(worst, abs(F_ad - F_cf))
            print("  %5.2f %6.2f | %+11.6f %+11.6f  %.2e" % (mu0, sg0, F_ad, F_cf, abs(F_ad - F_cf)))
    print("\n  AUTODIFF_MATCH max abs error = %.3e  (alpha factored out; Stein form,\n"
          "  so this is MC convergence, not a discrepancy)" % worst)
    print("  convergence at mu=0, sigma=1.5:")
    for n in (10**4, 10**5, 10**6, 10**7):
        u = jax.random.normal(jax.random.PRNGKey(7), (n,))
        mu = jnp.zeros((n,)); sg = jnp.full((n,), 1.5)
        def L(t):
            s = sg * jnp.exp(t); y = mu + s * u
            d = distrax.Transformed(distrax.Normal(mu, s), distrax.Tanh())
            return jnp.mean(d.log_prob(jnp.clip(jnp.tanh(y), -1 + 1e-4, 1 - 1e-4)))
        F_ad = -float(jax.grad(L)(0.0))
        F_cf = float(1.0 - 2.0 * jnp.mean(sg**2 * sech2(mu + sg * u)))
        print("    N=%-9d |err| = %.2e" % (n, abs(F_ad - F_cf)))
    out["autodiff_match_max_abs_err"] = worst
    return worst

# ----------------------------------------------------------------- measurement
def measure(ck, states, alpha, key):
    mu, sg = ck.policy_dist(states)
    u = jax.random.normal(key, (K, *mu.shape))
    es = jnp.mean(sech2(mu[None] + sg[None] * u), axis=0)     # (N, d)
    two_s2_es = 2.0 * sg**2 * es
    FH = alpha * (1.0 - two_s2_es)
    return (np.asarray(FH), np.asarray(sg), np.asarray(es), np.asarray(two_s2_es))

def main():
    verify()
    rows_cell, rows_coord, per_seed = [], [], []
    print("\n" + "=" * 78 + "\n1. F_H ON SIX BASELINE CELLS\n" + "=" * 78)
    print("  On-arm = the states that cell's own policy visited (bank source label).")
    print("  Neutral = the full 3072-state bank, reported alongside for comparison.\n")
    for task, T in TASKS.items():
        bp = os.path.join(ART, T["bank"])
        h = hashlib.sha256(open(bp, "rb").read()).hexdigest()
        assert h == T["sha"], "%s bank hash mismatch" % task
        z = np.load(bp)
        obs, src = jnp.asarray(z["obs"]), z["source"]
        for cell, (tag, armkey) in ARMS.items():
            alpha = None
            acc = {"on_arm": [], "neutral": []}
            seedmed = {"on_arm": [], "neutral": []}
            for sd in SEEDS:
                d = "exports/%s_%s_s%d_final" % (T["env"], tag, sd)
                ck = load(d)
                if alpha is None:
                    alpha = float(ck.meta["alpha_entropy"])
                m = np.array([s == "%s-s%d" % (armkey, sd) for s in src])
                for sub, st in (("on_arm", obs[jnp.asarray(np.where(m)[0])]), ("neutral", obs)):
                    FH, sg, es, t2 = measure(ck, st, alpha, jax.random.PRNGKey(RNG + sd))
                    acc[sub].append((FH, sg, es, t2))
                    seedmed[sub].append(float(np.median(FH)))
                    if sub == "on_arm":
                        for j in range(FH.shape[1]):
                            rows_coord.append(dict(task=task, cell=cell, seed=sd, coord=j,
                                mean_FH=float(FH[:, j].mean()), median_FH=float(np.median(FH[:, j])),
                                frac_pos=float((FH[:, j] > 0).mean()),
                                median_sigma=float(np.median(sg[:, j])),
                                E_sech2=float(es[:, j].mean()),
                                two_s2_E_sech2=float(t2[:, j].mean())))
            for sub in ("on_arm", "neutral"):
                FH = np.concatenate([a[0].ravel() for a in acc[sub]])
                sg = np.concatenate([a[1].ravel() for a in acc[sub]])
                es = np.concatenate([a[2].ravel() for a in acc[sub]])
                t2 = np.concatenate([a[3].ravel() for a in acc[sub]])
                sm = seedmed[sub]
                seeds_pos, seeds_neg = sum(1 for x in sm if x > 0), sum(1 for x in sm if x < 0)
                stable = (seeds_pos == 8) or (seeds_neg == 8)
                cls = "WIDENS" if seeds_pos == 8 else ("CONTRACTS" if seeds_neg == 8 else "MIXED")
                r = dict(task=task, cell=cell, subset=sub, alpha=alpha, d=T["d"],
                         mean_FH=float(FH.mean()), median_FH=float(np.median(FH)),
                         frac_FH_pos=float((FH > 0).mean()),
                         seed_medians_positive=seeds_pos, seed_medians_negative=seeds_neg,
                         sign_stable_across_seeds=bool(stable),
                         median_sigma=float(np.median(sg)),
                         E_sech2=float(es.mean()), two_s2_E_sech2=float(t2.mean()),
                         classification=cls,
                         seed_medians=[float(x) for x in sm])
                rows_cell.append(r)
                if sub == "on_arm":
                    per_seed.append(r)
    # print
    hdr = ("  %-7s %-10s %-8s | %-10s %-11s %-8s | %-8s %-9s | %s"
           % ("task", "cell", "subset", "median F_H", "mean F_H", "frac>0", "med sig", "2s2Esech2", "class"))
    print(hdr); print("  " + "-" * 104)
    for r in rows_cell:
        print("  %-7s %-10s %-8s | %+10.3e %+11.3e %8.4f | %8.4f %9.4f | %s  (seeds +%d/-%d)"
              % (r["task"], r["cell"], r["subset"], r["median_FH"], r["mean_FH"],
                 r["frac_FH_pos"], r["median_sigma"], r["two_s2_E_sech2"], r["classification"],
                 r["seed_medians_positive"], r["seed_medians_negative"]))
        if r["subset"] == "neutral":
            print()
    print("  per-seed medians of F_H (on-arm), sign stability visible:")
    for r in rows_cell:
        if r["subset"] != "on_arm": continue
        print("    %-7s %-10s %s" % (r["task"], r["cell"],
              " ".join("%+.2e" % x for x in r["seed_medians"])))
    out["cells"] = rows_cell

    # ---------------- validation
    print("\n" + "=" * 78 + "\n2. VALIDATION ON FOUR KNOWN OUTCOMES\n" + "=" * 78)
    print("  F_H measured at the baseline checkpoint predicts the direction the entropy")
    print("  term pushes width. The observed transition must agree.\n")
    KNOWN = [
        ("walker", "PW_ent",    "removing entropy", 0.455, 0.340, "narrower", "WIDENS"),
        ("walker", "WML_noent", "adding entropy",   8.687, 2.058, "narrower", "CONTRACTS"),
        ("g1",     "PW_ent",    "removing entropy", 0.300, 0.276, "narrower", "WIDENS"),
        ("g1",     "WML_noent", "adding entropy",   0.497, 0.574, "wider",    "WIDENS"),
    ]
    hits = {"on_arm": 0, "neutral": 0}
    for subset in ("on_arm", "neutral"):
        print("  --- measured on %s states ---" % subset)
        print("  %-7s %-10s %-17s %-22s %-11s %-11s %s"
              % ("task", "cell", "transition", "observed", "implies", "measured", "match"))
        for task, cell, trans, a, b, direction, expect in KNOWN:
            got = [r for r in rows_cell if r["task"] == task and r["cell"] == cell
                   and r["subset"] == subset][0]["classification"]
            ok = (got == expect)
            hits[subset] += ok
            print("  %-7s %-10s %-17s %6.3f -> %-13.3f %-11s %-11s %s"
                  % (task, cell, trans, a, b, direction, got, "MATCH" if ok else "MISS"))
        print("    %s: %d/4\n" % (subset, hits[subset]))
    print("  NOTE the four observed targets are NEUTRAL-BANK medians, taken from the")
    print("  factorial reports, so the commensurate comparison is the neutral row.")
    print("  The on-arm row predicts the force on a different state population.")
    print("\n  VALIDATION_4_OF_4 (on_arm, the preregistered primary) = %s  (%d/4)"
          % ("YES" if hits["on_arm"] == 4 else "NO", hits["on_arm"]))
    print("  VALIDATION_4_OF_4 (neutral, recorded for comparison)   = %s  (%d/4)"
          % ("YES" if hits["neutral"] == 4 else "NO", hits["neutral"]))
    out["validation_hits"] = hits

    # ---------------- LEAP prediction
    print("\n" + "=" * 78 + "\n3. LEAP PREDICTION\n" + "=" * 78)
    for cell, target in (("PW_ent", "LEAP PW_noent"), ("WML_noent", "LEAP WML_ent")):
        r = [x for x in rows_cell if x["task"] == "leap" and x["cell"] == cell
             and x["subset"] == "on_arm"][0]
        print("  measured on LEAP %-10s : %s  median F_H %+.3e  frac>0 %.4f  seeds +%d/-%d"
              % (cell, r["classification"], r["median_FH"], r["frac_FH_pos"],
                 r["seed_medians_positive"], r["seed_medians_negative"]))
        print("    median sigma %.4f ; 2 sigma^2 E[sech^2] = %.4f (crosses 1 at the sign change)"
              % (r["median_sigma"], r["two_s2_E_sech2"]))
    out["leap"] = {c: [x for x in rows_cell if x["task"] == "leap" and x["cell"] == c
                       and x["subset"] == "on_arm"][0] for c in ARMS}

    # ---------------- artifacts + figure
    with open(os.path.join(ART, "entropy_force_cells.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows_cell[0] if k != "seed_medians"])
        w.writeheader()
        for r in rows_cell:
            w.writerow({k: v for k, v in r.items() if k != "seed_medians"})
    with open(os.path.join(ART, "entropy_force_percoord.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_coord[0])); w.writeheader(); w.writerows(rows_coord)
    with open(os.path.join(ART, "entropy_force.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharey=False)
    for i, task in enumerate(TASKS):
        for j, cell in enumerate(ARMS):
            ax = axes[j, i]
            sub = [r for r in rows_coord if r["task"] == task and r["cell"] == cell]
            x = [r["median_sigma"] for r in sub]; y = [r["median_FH"] for r in sub]
            ax.scatter(x, y, s=14, alpha=.6,
                       color="#1f77b4" if cell == "PW_ent" else "#d62728")
            ax.axhline(0, color="k", lw=1)
            ax.axvline(1 / np.sqrt(2), color="0.5", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_title("%s  %s  (alpha %.2e)" % (task, cell,
                         [r for r in rows_cell if r["task"] == task and r["cell"] == cell][0]["alpha"]),
                         fontsize=8)
            ax.set_xlabel("median pre-tanh sigma (per coordinate)", fontsize=8)
            ax.set_ylabel("median F_H", fontsize=8)
            ax.tick_params(labelsize=7); ax.grid(alpha=.3)
    fig.suptitle("Actor-entropy force F_H vs sigma, per action coordinate, on-arm states\n"
                 "F > 0 widens; dashed line marks sigma = 1/sqrt(2), sufficient-but-not-necessary",
                 fontsize=10)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(FIGD, "fig_entropy_force." + e), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("\n  wrote artifacts and figure under %s" % FIGD)

if __name__ == "__main__":
    main()
