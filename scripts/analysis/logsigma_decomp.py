#!/usr/bin/env python3
"""Per-term log-sigma gradient decomposition on frozen checkpoints.

Offline. Implements docs/prereg_logsigma_decomp.md (commit 5edaca3) exactly.
Convention F = -dL/dlog_sigma, so F > 0 widens. Pre-tanh throughout; tanh enters only
as the critic's argument when forming the weights.
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
EPS_E = 0.5
KL_BOUND = 0.10
NSTATE = 512          # states sampled from the neutral bank per (task, arm, seed)
RNG = 20260907
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

def file_sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

def dual_g(Q, eta, eps):
    """MPO eta dual, log-sum-exp stabilised. Q is (M, S)."""
    qm = Q.max(axis=0, keepdims=True)
    lse = np.log(np.mean(np.exp((Q - qm) / eta), axis=0)) + qm[0] / eta
    return eta * eps + eta * float(np.mean(lse))

def solve_eta(Q, eps, iters=200):
    """Golden section on log eta, clipped to [1e-4, 10]."""
    lo, hi = np.log(1e-4), np.log(10.0)
    gr = (np.sqrt(5) - 1) / 2
    a, b = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fa, fb = dual_g(Q, np.exp(a), eps), dual_g(Q, np.exp(b), eps)
    for _ in range(iters):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - gr * (hi - lo); fa = dual_g(Q, np.exp(a), eps)
        else:
            lo, a, fa = a, b, fb
            b = lo + gr * (hi - lo); fb = dual_g(Q, np.exp(b), eps)
    return float(np.clip(np.exp(0.5 * (lo + hi)), 1e-4, 10.0))

def kl_diag(mu0, s0, mu1, s1):
    """Forward KL(N(mu0,s0) || N(mu1,s1)), summed over coordinates."""
    return np.sum(np.log(s1 / s0) + (s0**2 + (mu1 - mu0)**2) / (2 * s1**2) - 0.5, axis=-1)

def main():
    prov, rows, per_state = {}, [], []
    for task, T in TASKS.items():
        bp = os.path.join(ART, T["bank"])
        bh = file_sha(bp)
        assert bh == T["sha"], "%s bank hash mismatch" % task
        z = np.load(bp)
        src = z["source"]
        arm_lab = np.array([s.split("-")[0] for s in src])
        # confirm the balanced neutral bank, not training states
        npw, nwml = int((arm_lab == "PW").sum()), int((arm_lab == "WML").sum())
        nseed = len(set(s.split("-")[1] for s in src))
        prov[task] = dict(bank=bp, bank_sha256=bh, n_states=int(len(src)),
                          pw_states=npw, wml_states=nwml, seeds_in_bank=nseed,
                          balanced=bool(npw == nwml), d=T["d"], checkpoints={})
        obs_all = jnp.asarray(z["obs"])
        print("  %-7s bank %s  n=%d  PW=%d WML=%d  seeds=%d  balanced=%s"
              % (task, bh[:16], len(src), npw, nwml, nseed, npw == nwml))

        for arm, tag in ARMS.items():
            for sd in SEEDS:
                d_ck = "exports/%s_%s_s%d_final" % (T["env"], tag, sd)
                ck = load(d_ck)
                prov[task]["checkpoints"]["%s_s%d" % (arm, sd)] = dict(
                    actor=file_sha(os.path.join(d_ck, "actor.npz")),
                    critic=file_sha(os.path.join(d_ck, "critic.npz")))
                key = jax.random.PRNGKey(RNG + sd)
                k_sub, k_eps = jax.random.split(key)
                idx = jax.random.choice(k_sub, obs_all.shape[0], (NSTATE,), replace=False)
                S = obs_all[idx]
                mu_old, sg_old = ck.policy_dist(S)
                mu_old, sg_old = np.asarray(mu_old), np.asarray(sg_old)

                for M in MS:
                    eps = np.asarray(jax.random.normal(k_eps, (M, *mu_old.shape)))
                    a_i = mu_old[None] + sg_old[None] * eps          # (M,S,d) pre-tanh
                    a_sq = jnp.clip(jnp.tanh(jnp.asarray(a_i)), -CLIP_A, CLIP_A)
                    Q = np.asarray(jax.vmap(lambda aa: ck.q_scalar(S, aa))(a_sq))  # (M,S)
                    eta = solve_eta(Q, EPS_E)
                    qm = Q.max(axis=0, keepdims=True)
                    ew = np.exp((Q - qm) / eta)
                    w = ew / ew.sum(axis=0, keepdims=True)           # (M,S)
                    ess = 1.0 / np.sum(w**2, axis=0)                 # (S,)

                    mu_w = np.sum(w[..., None] * a_i, axis=0)        # (S,d)
                    dmu = mu_w - mu_old
                    A = np.sum(dmu**2, axis=-1) / np.mean(sg_old**2, axis=-1)
                    spread = np.sum(w[..., None] * (a_i - mu_w[None])**2, axis=0)  # (S,d)
                    B = np.sum(spread, axis=-1) / np.mean(sg_old**2, axis=-1) - T["d"]
                    net = A + B

                    sg_w = np.sqrt(np.maximum(spread, 1e-12))
                    kl = kl_diag(mu_old, sg_old, mu_w, sg_w)
                    open_ = kl < KL_BOUND

                    # pathwise comparators on the same states and the same eps draws
                    alpha = float(jnp.squeeze(ck.actor.temperature()))
                    y = a_i                                          # pre-tanh samples
                    sech2 = 1.0 - np.tanh(y)**2
                    F_H = alpha * np.sum(1.0 - 2.0 * sg_old**2 * sech2.mean(axis=0), axis=-1)
                    gq = np.asarray(jax.vmap(lambda aa: ck.q_grad_a(S, aa))(a_sq))  # (M,S,d)
                    curv = -np.mean(np.sum(sg_old[None] * gq * eps, axis=-1), axis=0)

                    rows.append(dict(task=task, arm=arm, seed=sd, M=M, eta=eta,
                        gate_open_frac=float(open_.mean()),
                        A_open=float(np.median(A[open_])) if open_.any() else np.nan,
                        B_open=float(np.median(B[open_])) if open_.any() else np.nan,
                        net_open=float(np.median(net[open_])) if open_.any() else np.nan,
                        A_closed=float(np.median(A[~open_])) if (~open_).any() else np.nan,
                        B_closed=float(np.median(B[~open_])) if (~open_).any() else np.nan,
                        net_closed=float(np.median(net[~open_])) if (~open_).any() else np.nan,
                        A_pooled=float(np.median(A)), B_pooled=float(np.median(B)),
                        net_pooled=float(np.median(net)),
                        F_H_pooled=float(np.median(F_H)),
                        curv_pooled=float(np.median(curv)),
                        abs_curv_pooled=float(np.median(np.abs(curv))),
                        curv_open=float(np.median(curv[open_])) if open_.any() else np.nan,
                        frac_curv_pos=float((curv > 0).mean()),
                        ess=float(np.median(ess)), kl_med=float(np.median(kl)),
                        spearman_A_dmu=float(spearmanr(A, np.linalg.norm(dmu, axis=-1)).statistic)))
                    if M == 32:
                        per_state.append(dict(task=task, arm=arm, seed=sd,
                            A=A, curv=curv, open_=open_))
    return prov, rows, per_state

if __name__ == "__main__":
    prov, rows, per_state = main()
    with open(os.path.join(ART, "logsigma_decomp_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(ART, "logsigma_decomp_provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)

    def sel(task, arm, M):
        return [r for r in rows if r["task"] == task and r["arm"] == arm and r["M"] == M]
    print("\n" + "=" * 112)
    print("PER-TERM LOG-SIGMA GRADIENT DECOMPOSITION   F = -dL/dlog_sigma, F>0 widens")
    print("=" * 112)
    for M in MS:
        print("\n########## M = %d" % M)
        print("  %-7s %-10s | gate_open |    A_open    B_open  net_open |  A_pooled  net_pool |"
              "   F_H      curv   |curv|  frac+ |  eta   ESS  Spear(A,|Dmu|)"
              % ("task", "arm"))
        for task in TASKS:
            for arm in ARMS:
                rr = sel(task, arm, M)
                m = lambda k: float(np.nanmedian([r[k] for r in rr]))
                print("  %-7s %-10s |   %.3f   | %9.4f %9.4f %9.4f | %9.4f %9.4f |"
                      " %8.4f %8.4f %7.4f %.3f | %.3f %5.2f  %+.3f"
                      % (task, arm, m("gate_open_frac"), m("A_open"), m("B_open"), m("net_open"),
                         m("A_pooled"), m("net_pooled"), m("F_H_pooled"), m("curv_pooled"),
                         m("abs_curv_pooled"), m("frac_curv_pos"), m("eta"), m("ess"),
                         m("spearman_A_dmu")))
    # bootstrap over states, M=32
    print("\n" + "=" * 112)
    print("BOOTSTRAP over states, 10,000 resamples, M = 32")
    print("=" * 112)
    boot = {}
    for task in TASKS:
        wml = [p for p in per_state if p["task"] == task and p["arm"] == "WML_noent"]
        pw = [p for p in per_state if p["task"] == task and p["arm"] == "PW_ent"]
        Aw = np.concatenate([p["A"][p["open_"]] for p in wml])
        Cp = np.concatenate([np.abs(p["curv"]) for p in pw])
        rng = np.random.default_rng(20260902)
        b1 = [np.median(Aw[rng.integers(0, len(Aw), len(Aw))]) for _ in range(10000)]
        rng = np.random.default_rng(20260902)
        b2 = [np.median(Aw[rng.integers(0, len(Aw), len(Aw))])
              - np.median(Cp[rng.integers(0, len(Cp), len(Cp))]) for _ in range(10000)]
        ci1 = (float(np.percentile(b1, 2.5)), float(np.percentile(b1, 97.5)))
        ci2 = (float(np.percentile(b2, 2.5)), float(np.percentile(b2, 97.5)))
        boot[task] = dict(A_open_median=float(np.median(Aw)), A_open_ci=ci1,
                          diff_median=float(np.median(Aw) - np.median(Cp)), diff_ci=ci2,
                          pw_abs_curv_median=float(np.median(Cp)))
        print("  %-7s gate-open median A = %9.4f  95%% CI [%9.4f, %9.4f]  excludes 0: %s"
              % (task, np.median(Aw), ci1[0], ci1[1], "YES" if ci1[0] > 0 else "NO"))
        print("          median A_WML - median |curv|_PW = %9.4f  95%% CI [%9.4f, %9.4f]"
              "  excludes 0: %s"
              % (np.median(Aw) - np.median(Cp), ci2[0], ci2[1], "YES" if ci2[0] > 0 else "NO"))
    with open(os.path.join(ART, "logsigma_decomp_bootstrap.json"), "w") as f:
        json.dump(boot, f, indent=1, default=float)
    print("\n  wrote logsigma_decomp_rows.csv, _provenance.json, _bootstrap.json")
