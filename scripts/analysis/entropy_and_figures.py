#!/usr/bin/env python3
"""Section 4 (entropy derivative) + Section 5 (figures) of the read-only package.

No training, no actor-loss modification. Reads exported checkpoints and frozen
artifacts only.
"""
import json, os, re, sys
import numpy as np
import jax, jax.numpy as jnp
import distrax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from scripts.load_ckpt import load

ART   = "reports/artifacts"
FIGD  = os.path.join(ART, "figs_package")
BANK  = os.path.join(ART, "walker_fixed_state_bank.npz")
SEEDS = list(range(301, 309))
BOOT_N, BOOT_RNG = 10000, 20260905  # figure-side reproduction seed set below
PREREG_RNG = 20260902               # the frozen preregistered bootstrap seed
os.makedirs(FIGD, exist_ok=True)
out = {}

# ----------------------------------------------------------------- section 4
def sech2(y):
    return 1.0 - jnp.tanh(y) ** 2

def LH_repo(t, mu, sg, u, alpha):
    """alpha * E[log pi(a|s)] using the repo's own Tanh(Normal) log-prob."""
    s = sg * jnp.exp(t)
    y = mu + s * u
    a = jnp.tanh(y)
    d = distrax.Transformed(distrax.Normal(mu, s), distrax.Tanh())
    return alpha * jnp.mean(d.log_prob(a))

def hand_pathwise(mu, sg, u, alpha, t=0.0):
    """Exact per-sample derivative: d/dt [-log s - u^2/2 - log sech^2(y)] with y = mu+s*u.
    This is what autodiff computes, so it must match to machine precision."""
    s = sg * jnp.exp(t)
    y = mu + s * u
    return alpha * jnp.mean(-1.0 + 2.0 * s * u * jnp.tanh(y))

def hand_stein(mu, sg, u, alpha, t=0.0):
    """The reported closed form. Reaches hand_pathwise only in expectation, via Stein:
    E[u*tanh(mu+s*u)] = s*E[sech^2(y)]. On a finite sample the two differ by MC noise
    that grows with sigma, so this is checked for CONVERGENCE, not exact equality."""
    s = sg * jnp.exp(t)
    y = mu + s * u
    return alpha * (-1.0 + 2.0 * jnp.mean(s**2 * sech2(y)))

def section4():
    print("=" * 74 + "\n4. ENTROPY DERIVATIVE CHECK\n" + "=" * 74)
    print("  L_H = alpha * E[log pi(a|s)],  a = tanh(y),  y = mu + sigma*u")
    print("  hand:  dL_H/dlog_sigma = alpha * [ -1 + 2*sigma^2*E(sech^2(y)) ]")
    print("  F := -dL/dlog_sigma  so  F > 0 widens, F < 0 contracts\n")

    # (a) autodiff match on synthetic cells, against the repo log-prob
    key = jax.random.PRNGKey(0)
    worst = 0.0
    print("    mu     sigma |   pathwise      autodiff      |err| |   stein-form  |gap|")
    for mu0 in (-1.0, 0.0, 0.7):
        for sg0 in (0.2, 0.5, 1.5):
            u = jax.random.normal(key, (400000,))
            mu = jnp.full((400000,), mu0); sg = jnp.full((400000,), sg0)
            g = float(jax.grad(lambda t: LH_repo(t, mu, sg, u, 1.0))(0.0))
            h = float(hand_pathwise(mu, sg, u, 1.0))
            st = float(hand_stein(mu, sg, u, 1.0))
            worst = max(worst, abs(g - h))
            print("  %6.2f %7.2f | %+11.6f %+13.6f  %.2e | %+11.6f  %.2e"
                  % (mu0, sg0, h, g, abs(g - h), st, abs(st - g)))
    verdict = "PASS" if worst < 1e-4 else "FAIL"
    print("\n  ENTROPY_DERIVATIVE_AUTODIFF_MATCH = %s  (max abs err %.2e, tol 1e-4)"
          % (verdict, worst))
    # the closed form differs from the pathwise derivative only by Stein MC noise:
    # show it falls as 1/sqrt(N) rather than to a constant
    print("\n  Stein-form convergence at mu=0, sigma=1.5 (worst cell above):")
    for N in (10**4, 10**5, 10**6, 10**7):
        uu = jax.random.normal(jax.random.PRNGKey(7), (N,))
        m0 = jnp.zeros((N,)); s0 = jnp.full((N,), 1.5)
        gp = float(hand_pathwise(m0, s0, uu, 1.0)); gs = float(hand_stein(m0, s0, uu, 1.0))
        print("    N=%-9d pathwise %+0.6f  stein %+0.6f  |diff| %.2e" % (N, gp, gs, abs(gs-gp)))

    # (b) on real Walker states, on-arm, at each arm's own operating sigma
    z = np.load(BANK); bank = jnp.asarray(z["states" if "states" in z.files else z.files[0]])
    h = bank.shape[0] // 2
    halves = {"pathwise": bank[:h], "weighted_mle": bank[h:]}
    tags = {"pathwise": "pathwise_fa", "weighted_mle": "weighted_mle"}
    print("\n  On-arm Walker states, final checkpoints, medians over 8 seeds.")
    print("  arm            | alpha      | sigma_op | F_H at sigma_op | sign-change sigma")
    res = {}
    for arm, bk in halves.items():
        FHs, sigops, crosses, alphas = [], [], [], []
        for sd in SEEDS:
            d = "exports/WalkerRun_%s_s%d_final" % (tags[arm], sd)
            if not os.path.isdir(d): continue
            ck = load(d)
            alpha = float(jnp.squeeze(ck.actor.temperature()))
            mu, sg = ck.policy_dist(bk)
            u = jax.random.normal(jax.random.PRNGKey(1000 + sd), (64, *mu.shape))
            # per state-coordinate F_H, averaged over the u draws
            y = mu[None] + sg[None] * u
            FH = alpha * (1.0 - 2.0 * sg**2 * jnp.mean(sech2(y), axis=0))
            FHs.append(float(jnp.median(FH)))
            sigops.append(float(jnp.median(sg)))
            alphas.append(alpha)
            # sign change: scale sigma, find where median F_H crosses zero
            facs = np.geomspace(0.02, 20.0, 220)
            med = []
            for f in facs:
                s2 = sg * f
                y2 = mu[None] + s2[None] * u
                med.append(float(jnp.median(alpha * (1.0 - 2.0 * s2**2
                                                     * jnp.mean(sech2(y2), axis=0)))))
            med = np.asarray(med); sig_grid = facs * float(jnp.median(sg))
            idx = np.where(np.sign(med[:-1]) != np.sign(med[1:]))[0]
            crosses.append(float(sig_grid[idx[0]]) if len(idx) else float("nan"))
        res[arm] = dict(alpha=float(np.median(alphas)),
                        sigma_op=float(np.median(sigops)),
                        F_H_at_sigma_op=float(np.median(FHs)),
                        sign_change_sigma=float(np.nanmedian(crosses)),
                        per_seed_F_H=FHs, per_seed_sigma_op=sigops,
                        per_seed_sign_change=crosses)
        print("  %-14s | %.8f | %8.4f | %+15.3e | %.4f"
              % (arm, res[arm]["alpha"], res[arm]["sigma_op"],
                 res[arm]["F_H_at_sigma_op"], res[arm]["sign_change_sigma"]))
    print("\n  The sign of F_H depends on mu and the state distribution, so the crossing")
    print("  is reported per arm on that arm's own states. No universal crossover.")
    out["section4"] = dict(autodiff_match=verdict, max_abs_err=worst, arms=res)
    return verdict

# ----------------------------------------------------------------- figures
def boot_ci(diffs, stat, rng_seed, n=BOOT_N):
    rng = np.random.default_rng(rng_seed)
    d = np.asarray(diffs, float); k = len(d)
    bs = [stat(d[rng.integers(0, k, k)]) for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def load_returns():
    rows = {}
    import csv
    with open(os.path.join(ART, "corrected_paired_results.csv")) as f:
        for r in csv.DictReader(f):
            rows[r["task"]] = dict(
                d=int(r["d"]),
                pw=[float(x) for x in r["pw"].split(";")],
                wml=[float(x) for x in r["wml"].split(";")],
                diffs=[float(x) for x in r["diffs"].split(";")],
                ci=(float(r["ci_lo"]), float(r["ci_hi"])),
                median=float(r["median_diff"]), mean=float(r["mean_diff"]),
                n_pos=int(r["n_pos"]), p=float(r["p_exact"]))
    # LEAP from its frozen report table
    txt = open("reports/leap_corrected.md").read()
    pw, wml, df = [], [], []
    for m in re.finditer(r"^\|\s*(30[1-8])\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
                         r"\s*\*{0,2}([+−-][\d.]+)\*{0,2}\s*\|", txt, re.M):
        pw.append(float(m.group(2))); wml.append(float(m.group(3)))
        df.append(float(m.group(4).replace("−", "-").replace("+", "")))
    rows["leap"] = dict(d=16, pw=pw, wml=wml, diffs=df, ci=(-1.3388, 10.5281),
                        median=4.2957, mean=float(np.mean(df)),
                        n_pos=int(sum(1 for x in df if x > 0)), p=float("nan"))
    return rows

def fig1(R):
    print("\n=== FIG 1: corrected return comparison ===")
    tasks = [("walker", "WalkerRun\nd=6"), ("g1", "G1JoystickFlat\nd=29"),
             ("leap", "LeapCubeRotateZ\nd=16")]
    # which statistic reproduces the frozen CI: mean or median?
    which = {}
    for t, _ in tasks:
        dfs = R[t]["diffs"]
        for name, fn in (("median", np.median), ("mean", np.mean)):
            lo, hi = boot_ci(dfs, fn, PREREG_RNG)
            ok = (abs(lo - R[t]["ci"][0]) < 0.02 and abs(hi - R[t]["ci"][1]) < 0.02)
            print("   %-7s %-6s boot CI [%+9.4f,%+9.4f]  frozen [%+9.4f,%+9.4f] %s"
                  % (t, name, lo, hi, R[t]["ci"][0], R[t]["ci"][1], "MATCH" if ok else ""))
            if ok: which[t] = name
    out["fig1_statistic_reproducing_frozen_CI"] = which

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (t, lab) in zip(axes, tasks):
        r = R[t]
        x = np.arange(len(SEEDS))
        # markers only: seeds have no ordering, so a joining line would imply
        # a trend that does not exist. The grey verticals are the pairing.
        ax.plot(x, r["pw"], "o", color="#1f77b4", label="PW-1", ms=6)
        ax.plot(x, r["wml"], "s", color="#d62728", label="WML-32", ms=6)
        for i in x:
            ax.plot([i, i], [r["wml"][i], r["pw"][i]], color="0.7", lw=1, zorder=0)
        ax.set_xticks(x); ax.set_xticklabels([str(s) for s in SEEDS], fontsize=7, rotation=45)
        ax.set_title("%s\npaired diff %+.2f  95%% CI [%+.2f, %+.2f]  %d/8 pos"
                     % (lab, r["median"], r["ci"][0], r["ci"][1], r["n_pos"]), fontsize=8)
        ax.set_xlabel("seed", fontsize=8); ax.tick_params(labelsize=7)
        ax.grid(alpha=.3)
    axes[0].set_ylabel("score_window3\n(mean of final 3 of 21 evals)", fontsize=8)
    axes[0].legend(fontsize=7)
    fig.suptitle("Corrected operator replication: PW-1 vs WML-32, paired seeds 301-308",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGD, "fig1_corrected_returns." + ext), dpi=180,
                    bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(FIGD, "fig1_corrected_returns_data.csv"), "w") as f:
        f.write("task,d,seed,pw_score_window3,wml_score_window3,diff_pw_minus_wml\n")
        for t, _ in tasks:
            r = R[t]
            for i, s in enumerate(SEEDS):
                f.write("%s,%d,%d,%.4f,%.4f,%.4f\n" % (t, r["d"], s, r["pw"][i],
                                                       r["wml"][i], r["diffs"][i]))
    return which

def fig2_3(W):
    print("\n=== FIG 2/3: fixed-bank width and saturation ===")
    tasks = [("walker", "WalkerRun d=6"), ("g1", "G1 d=29"), ("leap", "LEAP d=16")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (t, lab) in zip(axes, tasks):
        p = W[t]["paired"]
        pw = [p["s%d" % s]["pw"] for s in SEEDS]
        wm = [p["s%d" % s]["wml"] for s in SEEDS]
        x = np.arange(len(SEEDS))
        for i in x:
            ax.plot([i, i], [pw[i], wm[i]], color="0.7", lw=1, zorder=0)
        ax.plot(x, pw, "o", color="#1f77b4", label="PW-1", ms=6)
        ax.plot(x, wm, "s", color="#d62728", label="WML-32", ms=6)
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels([str(s) for s in SEEDS], fontsize=7, rotation=45)
        ax.set_title("%s\nmedian paired ratio %.2fx, %d/8 pairs"
                     % (lab, W[t]["summary"]["median_paired_ratio"],
                        W[t]["summary"]["wml_gt_pw_seed_pairs"]), fontsize=8)
        ax.grid(alpha=.3, which="both"); ax.tick_params(labelsize=7)
        ax.set_xlabel("seed", fontsize=8)
    axes[0].set_ylabel("median pre-tanh sigma\n(frozen state bank)", fontsize=8)
    axes[0].legend(fontsize=7)
    fig.suptitle("Final policy width on a frozen state bank, paired seeds", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGD, "fig2_fixed_bank_width." + ext), dpi=180,
                    bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (t, lab) in zip(axes, tasks):
        S = W[t]["sat"]
        x = np.arange(len(SEEDS)); w = 0.2
        for off, key, c, nm in ((-1.5*w, "p95", "#1f77b4", "PW |a|>.95"),
                                (-0.5*w, "p99", "#7fb3d5", "PW |a|>.99"),
                                (0.5*w, "p95", "#d62728", "WML |a|>.95"),
                                (1.5*w, "p99", "#f1948a", "WML |a|>.99")):
            arm = "PW" if "PW" in nm else "WML"
            v = [S["%s_s%d" % (arm, s)][key] for s in SEEDS]
            ax.bar(x + off, v, w, color=c, label=nm)
        ax.set_xticks(x); ax.set_xticklabels([str(s) for s in SEEDS], fontsize=7, rotation=45)
        ax.set_title("%s\nWML>PW in %d/8 pairs at |a|>0.95"
                     % (lab, W[t]["summary"]["sat95_wml_gt_pw_pairs"]), fontsize=8)
        ax.set_ylim(0, 1); ax.grid(alpha=.3, axis="y"); ax.tick_params(labelsize=7)
        ax.set_xlabel("seed", fontsize=8)
    axes[0].set_ylabel("action saturation probability", fontsize=8)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Action saturation on the frozen state bank, paired seeds", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGD, "fig3_saturation." + ext), dpi=180,
                    bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(FIGD, "fig2_fig3_width_saturation_data.csv"), "w") as f:
        f.write("task,d,seed,pw_median_sigma,wml_median_sigma,ratio,"
                "pw_sat95,wml_sat95,pw_sat99,wml_sat99\n")
        for t, _ in tasks:
            for s in SEEDS:
                p = W[t]["paired"]["s%d" % s]; S = W[t]["sat"]
                f.write("%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n"
                        % (t, W[t]["d"], s, p["pw"], p["wml"], p["ratio"],
                           S["PW_s%d" % s]["p95"], S["WML_s%d" % s]["p95"],
                           S["PW_s%d" % s]["p99"], S["WML_s%d" % s]["p99"]))

def fig4():
    print("\n=== FIG 4: controlled critic-error phase diagram ===")
    import csv
    d, sg, om, re_ = [], [], [], []
    with open(os.path.join(ART, "planted_sweep.csv")) as f:
        for r in csv.DictReader(f):
            d.append(float(r["d"])); sg.append(float(r["sigma"]))
            om.append(float(r["omega"])); re_.append(float(r["ratio_e"]))
    d, sg, om, re_ = map(np.asarray, (d, sg, om, re_))
    x = sg * om / np.sqrt(d)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for dd in sorted(set(d)):
        m = d == dd
        ax.scatter(x[m], re_[m], s=18, alpha=.75, label="d=%d" % dd)
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma\,\omega/\sqrt{d}$  (predicted boundary at 1)", fontsize=9)
    ax.set_ylabel(r"error-induced variance ratio  Var$[g_{ZO}]_e$ / Var$[g_{PW}]_e$",
                  fontsize=9)
    ax.set_title("Controlled planted critic error: estimator-level sensitivity", fontsize=10)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3, which="both")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGD, "fig4_controlled_theory." + ext), dpi=180,
                    bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(FIGD, "fig4_controlled_theory_data.csv"), "w") as f:
        f.write("d,sigma,omega,sigma_omega_over_sqrt_d,var_ratio_zo_over_pw\n")
        for i in range(len(d)):
            f.write("%d,%.6g,%.6g,%.6g,%.6g\n" % (d[i], sg[i], om[i], x[i], re_[i]))
    print("   %d cells, d in %s" % (len(d), sorted(set(int(v) for v in d))))

def main():
    v = section4()
    R = load_returns()
    which = fig1(R)
    W = json.load(open(os.path.join(ART, "fixed_bank_width_saturation.json")))
    fig2_3(W)
    fig4()
    with open(os.path.join(ART, "entropy_derivative_check.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\n  wrote figures to %s" % FIGD)
    print("  ENTROPY_DERIVATIVE_AUTODIFF_MATCH = %s" % v)
    print("  fig1 statistic reproducing frozen CI: %s" % which)

if __name__ == "__main__":
    main()
