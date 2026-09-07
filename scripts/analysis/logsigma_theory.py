#!/usr/bin/env python3
"""Steps 1-3: log-sigma decomposition, the linear-Q tilting prediction, and the M-sweep.

Offline. Implements docs/prereg_logsigma_decomp.md (5edaca3) for the decomposition, and
adds the quantitative tilting test and the M-sweep scaling test.

Theory under test (exact for locally linear Q and exponential tilting of a Gaussian):
    w_i ∝ exp(g·y/eta) on the PRE-TANH variable y tilts N(mu,sigma^2) to
    N(mu + sigma^2 g/eta, sigma^2), so
        Dmu    = sigma^2 g / eta
        A_pred = ||Dmu||^2/sigma^2 = sigma^2 ||g||^2 / eta^2
        B      -> 0 as M -> inf ; B ~ -d/ESS at finite M
    giving a stationary width  sigma* = (eta/||g||) * sqrt(d/ESS).
The pre-tanh gradient is g = grad_a Q(tanh(y)) * sech^2(y), evaluated at y = mu_old.
"""
import csv, hashlib, json, os, sys
import numpy as np
import jax, jax.numpy as jnp
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
from scripts.load_ckpt import load

ART = "reports/artifacts"
SEEDS = list(range(301, 309))
MS = [16, 32]
EPS_E, KL_BOUND, NSTATE, RNG = 0.5, 0.10, 512, 20260907
CLIP_A = 1.0 - 1e-4
TASKS = {
 "walker": dict(env="WalkerRun", bank="walker_fixed_state_bank.npz", d=6,
   sha="8adfeb0bf70bddcdbd64a84b972b4dbd62617c64e1c948589a7adc30cf64aa21"),
 "g1": dict(env="G1JoystickFlatTerrain", bank="g1_fixed_state_bank.npz", d=29,
   sha="cf6f7880b3a5b59433727f7a245b5ce97749096981f03233fd434ab3371935e9"),
 "leap": dict(env="LeapCubeRotateZAxis", bank="leap_fixed_state_bank.npz", d=16,
   sha="0053b0f361e45b9227d15b982ff667a5fc665469dd54e673c70c0682bcb158b2"),
}
ARMS = {"PW_ent": "pathwise_fa", "WML_noent": "weighted_mle"}

def fsha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

def dual_g(Q, eta, eps):
    qm = Q.max(axis=0, keepdims=True)
    return eta * eps + eta * float(np.mean(np.log(np.mean(np.exp((Q - qm) / eta), 0)) + qm[0] / eta))

def solve_eta(Q, eps, iters=200):
    lo, hi = np.log(1e-4), np.log(10.0); gr = (np.sqrt(5) - 1) / 2
    a, b = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fa, fb = dual_g(Q, np.exp(a), eps), dual_g(Q, np.exp(b), eps)
    for _ in range(iters):
        if fa < fb:
            hi, b, fb = b, a, fa; a = hi - gr * (hi - lo); fa = dual_g(Q, np.exp(a), eps)
        else:
            lo, a, fa = a, b, fb; b = lo + gr * (hi - lo); fb = dual_g(Q, np.exp(b), eps)
    return float(np.clip(np.exp(0.5 * (lo + hi)), 1e-4, 10.0))

def kl_diag(m0, s0, m1, s1):
    return np.sum(np.log(s1 / s0) + (s0**2 + (m1 - m0)**2) / (2 * s1**2) - 0.5, -1)

def eval_ckpt(ck, S, M, d, key):
    """Everything the decomposition needs on one (checkpoint, state set, M)."""
    mu, sg = ck.policy_dist(S)
    mu, sg = np.asarray(mu), np.asarray(sg)
    eps = np.asarray(jax.random.normal(key, (M, *mu.shape)))
    y = mu[None] + sg[None] * eps                       # pre-tanh samples
    a = jnp.clip(jnp.tanh(jnp.asarray(y)), -CLIP_A, CLIP_A)
    Q = np.asarray(jax.vmap(lambda aa: ck.q_scalar(S, aa))(a))
    eta = solve_eta(Q, EPS_E)
    qm = Q.max(0, keepdims=True); ew = np.exp((Q - qm) / eta)
    w = ew / ew.sum(0, keepdims=True)
    ess = 1.0 / np.sum(w**2, 0)
    mu_w = np.sum(w[..., None] * y, 0)
    dmu = mu_w - mu
    s2 = np.mean(sg**2, -1)
    A = np.sum(dmu**2, -1) / s2
    spread = np.sum(w[..., None] * (y - mu_w[None])**2, 0)
    B = np.sum(spread, -1) / s2 - d
    kl = kl_diag(mu, sg, mu_w, np.sqrt(np.maximum(spread, 1e-12)))
    # pre-tanh gradient at the policy mean: grad_a Q(tanh(mu)) * sech^2(mu)
    a_mu = jnp.clip(jnp.tanh(jnp.asarray(mu)), -CLIP_A, CLIP_A)
    gq = np.asarray(ck.q_grad_a(S, a_mu))
    g_eff = gq * (1.0 - np.tanh(mu)**2)
    gnorm = np.linalg.norm(g_eff, axis=-1)
    A_pred = s2 * gnorm**2 / eta**2
    sig_star = (eta / np.maximum(gnorm, 1e-12)) * np.sqrt(d / np.maximum(ess, 1e-12))
    curv = -np.mean(np.sum(sg[None] * np.asarray(
        jax.vmap(lambda aa: ck.q_grad_a(S, aa))(a)) * eps, -1), 0)
    return dict(A=A, B=B, net=A + B, kl=kl, ess=ess, eta=eta, sigma=np.median(sg),
                sigma_state=np.sqrt(s2), gnorm=gnorm, A_pred=A_pred, sig_star=sig_star,
                curv=curv, mu=mu, sg=sg)

def main():
    prov, rows, keep = {}, [], []
    for task, T in TASKS.items():
        bp = os.path.join(ART, T["bank"]); assert fsha(bp) == T["sha"], task
        z = np.load(bp); src = z["source"]; obs = jnp.asarray(z["obs"])
        arm_lab = np.array([s.split("-")[0] for s in src])
        prov[task] = dict(bank_sha256=fsha(bp), n=int(len(src)), d=T["d"],
                          balanced=bool((arm_lab == "PW").sum() == (arm_lab == "WML").sum()),
                          checkpoints={})
        for arm, tag in ARMS.items():
            for sd in SEEDS:
                dck = "exports/%s_%s_s%d_final" % (T["env"], tag, sd)
                ck = load(dck)
                prov[task]["checkpoints"]["%s_s%d" % (arm, sd)] = dict(
                    actor=fsha(dck + "/actor.npz"), critic=fsha(dck + "/critic.npz"))
                k = jax.random.PRNGKey(RNG + sd); ks, ke = jax.random.split(k)
                idx = np.asarray(jax.random.choice(ks, obs.shape[0], (NSTATE,), replace=False))
                S = obs[jnp.asarray(idx)]
                for M in MS:
                    r = eval_ckpt(ck, S, M, T["d"], ke)
                    op = r["kl"] < KL_BOUND
                    md = lambda v, m=None: float(np.median(v if m is None else v[m])) \
                        if (m is None or m.any()) else float("nan")
                    rows.append(dict(task=task, arm=arm, seed=sd, M=M, eta=r["eta"],
                        gate_open_frac=float(op.mean()),
                        A_open=md(r["A"], op), B_open=md(r["B"], op), net_open=md(r["net"], op),
                        A_pooled=md(r["A"]), B_pooled=md(r["B"]), net_pooled=md(r["net"]),
                        A_pred_pooled=md(r["A_pred"]), gnorm=md(r["gnorm"]),
                        sigma=float(r["sigma"]), sig_star=md(r["sig_star"]),
                        curv_pooled=md(r["curv"]), abs_curv=md(np.abs(r["curv"])),
                        frac_curv_pos=float((r["curv"] > 0).mean()),
                        ess=md(r["ess"]), kl_med=md(r["kl"]),
                        spearman_A_dmu=float(spearmanr(r["A"], np.sqrt(r["A"])).statistic),
                        spearman_A_Apred=float(spearmanr(r["A"], r["A_pred"]).statistic),
                        slope_A_on_Apred=float(np.sum(r["A"] * r["A_pred"]) /
                                               max(np.sum(r["A_pred"]**2), 1e-30))))
                    if M == 32:
                        keep.append(dict(task=task, arm=arm, seed=sd, idx=idx,
                                         A=r["A"], curv=r["curv"], open_=op,
                                         gnorm=r["gnorm"], A_pred=r["A_pred"],
                                         sig=r["sigma_state"], src=src[idx]))
    return prov, rows, keep

if __name__ == "__main__":
    prov, rows, keep = main()
    with open(os.path.join(ART, "logsigma_theory_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(ART, "logsigma_theory_provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)
    sel = lambda t, a, M: [r for r in rows if r["task"] == t and r["arm"] == a and r["M"] == M]
    md = lambda rr, k: float(np.nanmedian([r[k] for r in rr]))

    print("=" * 118 + "\nSTEP 1  DECOMPOSITION   F = -dL/dlog_sigma, F>0 widens\n" + "=" * 118)
    for M in MS:
        print("\n--- M = %d ---" % M)
        print("  %-7s %-10s | gate |    A_open     B_open   net_open |   A_pool   net_pool |"
              "    curv   |curv|  frac+ |  eta   ESS" % ("task", "arm"))
        for t in TASKS:
            for a in ARMS:
                rr = sel(t, a, M)
                print("  %-7s %-10s |%.3f | %10.4f %10.4f %10.4f | %8.4f %10.4f |"
                      " %8.4f %7.4f %.3f | %.3f %5.2f"
                      % (t, a, md(rr,"gate_open_frac"), md(rr,"A_open"), md(rr,"B_open"),
                         md(rr,"net_open"), md(rr,"A_pooled"), md(rr,"net_pooled"),
                         md(rr,"curv_pooled"), md(rr,"abs_curv"), md(rr,"frac_curv_pos"),
                         md(rr,"eta"), md(rr,"ess")))

    print("\n" + "=" * 118 + "\nSTEP 2  LINEAR-Q TILTING PREDICTION   A_pred = sigma^2 ||g||^2 / eta^2\n" + "=" * 118)
    print("  %-7s %-10s |   A_meas   A_pred   ratio | slope(A~A_pred) spearman | ||g||   sigma  sigma*"
          % ("task", "arm"))
    for t in TASKS:
        for a in ARMS:
            rr = sel(t, a, 32)
            am, ap = md(rr,"A_pooled"), md(rr,"A_pred_pooled")
            print("  %-7s %-10s | %8.4f %8.4f %7.3f | %14.3f %9.3f | %6.4f %7.4f %7.4f"
                  % (t, a, am, ap, am/ap if ap else float("nan"),
                     md(rr,"slope_A_on_Apred"), md(rr,"spearman_A_Apred"),
                     md(rr,"gnorm"), md(rr,"sigma"), md(rr,"sig_star")))

    print("\n" + "=" * 118 + "\nSTEP 2b  SEED-POPULATION TEST   is ||g|| larger on a policy's OWN states?\n" + "=" * 118)
    print("  prediction: sigma ~ 1/||g||, so own states (narrow) should have LARGER ||g||")
    print("  %-7s %-10s | ||g|| own-seed  ||g|| sibling   ratio | sigma own  sigma sibling"
          % ("task", "arm"))
    seedpop = []
    for t in TASKS:
        for a in ARMS:
            arm_pref = "PW" if a.startswith("PW") else "WML"
            go, gs, so, ss = [], [], [], []
            for p in [x for x in keep if x["task"] == t and x["arm"] == a]:
                own = np.array([s == "%s-s%d" % (arm_pref, p["seed"]) for s in p["src"]])
                sib = np.array([s.startswith(arm_pref + "-") and
                                s != "%s-s%d" % (arm_pref, p["seed"]) for s in p["src"]])
                if own.sum() and sib.sum():
                    go.append(np.median(p["gnorm"][own])); gs.append(np.median(p["gnorm"][sib]))
                    so.append(np.median(p["sig"][own]));   ss.append(np.median(p["sig"][sib]))
            if go:
                seedpop.append(dict(task=t, arm=a, g_own=float(np.median(go)),
                                    g_sib=float(np.median(gs)), sigma_own=float(np.median(so)),
                                    sigma_sib=float(np.median(ss))))
                print("  %-7s %-10s | %13.5f %14.5f %7.2f | %9.4f %13.4f"
                      % (t, a, np.median(go), np.median(gs), np.median(go)/np.median(gs),
                         np.median(so), np.median(ss)))

    print("\n" + "=" * 118 + "\nSTEP 3  BOOTSTRAP over states, M = 32, 10,000 resamples\n" + "=" * 118)
    boot = {}
    for t in TASKS:
        Aw = np.concatenate([p["A"][p["open_"]] for p in keep
                             if p["task"] == t and p["arm"] == "WML_noent"])
        Cp = np.concatenate([np.abs(p["curv"]) for p in keep
                             if p["task"] == t and p["arm"] == "PW_ent"])
        rng = np.random.default_rng(20260902)
        b1 = [np.median(Aw[rng.integers(0, len(Aw), len(Aw))]) for _ in range(10000)]
        rng = np.random.default_rng(20260902)
        b2 = [np.median(Aw[rng.integers(0, len(Aw), len(Aw))]) -
              np.median(Cp[rng.integers(0, len(Cp), len(Cp))]) for _ in range(10000)]
        c1 = (float(np.percentile(b1,2.5)), float(np.percentile(b1,97.5)))
        c2 = (float(np.percentile(b2,2.5)), float(np.percentile(b2,97.5)))
        boot[t] = dict(A_open=float(np.median(Aw)), ci=c1,
                       diff=float(np.median(Aw)-np.median(Cp)), diff_ci=c2,
                       pw_abs_curv=float(np.median(Cp)))
        print("  %-7s A_open median %9.4f  CI [%9.4f, %9.4f]  excl 0: %s"
              % (t, np.median(Aw), c1[0], c1[1], "YES" if c1[0] > 0 else "NO"))
        print("          A_WML - |curv|_PW = %9.4f  CI [%9.4f, %9.4f]  excl 0: %s"
              % (np.median(Aw)-np.median(Cp), c2[0], c2[1], "YES" if c2[0] > 0 else "NO"))
    with open(os.path.join(ART, "logsigma_theory_bootstrap.json"), "w") as f:
        json.dump(dict(boot=boot, seedpop=seedpop), f, indent=1, default=float)
    print("\n  wrote logsigma_theory_rows.csv, _provenance.json, _bootstrap.json")
