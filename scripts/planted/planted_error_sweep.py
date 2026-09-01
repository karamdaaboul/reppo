#!/usr/bin/env python
"""Controlled planted-error phase diagram for Claim 4 (crossover).

Ground truth is known exactly: Q_pi is quadratic, e is a planted sinusoid, so
e = Q_phi - Q_pi holds by construction and omega = ||grad e||_inf / ||e||_inf is
analytic.  Nothing here touches the learned-critic experiments; its conclusions
must not be used to reinterpret Probe 4.

Protocol is fixed by planted_sweep_config.json, committed before this ran.
"""
import json, os, sys, itertools
import numpy as np

# Importable for the validation suite: the config is only required when the sweep
# is actually run, so the estimator functions can be imported and unit-tested alone.
if len(sys.argv) > 2:
    CFG = json.load(open(sys.argv[1]))
    OUT = sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    G = CFG["grid"]
else:
    CFG, OUT, G = None, None, {"M": 32, "eps_E": 0.5}
M, EPS_E = G["M"], G["eps_E"]
EPS_TR = 0.1          # trust-region radius; cancels in the normalised comparison
G_SIG = 1.0           # true signal magnitude, constant across d


# ---------------------------------------------------------------- estimators
def q_pi(a, astar):
    return -0.5 * np.sum((a - astar) ** 2, axis=-1)


def grad_q_pi(a, astar):
    return -(a - astar)


def e_field(a, ve, om, ph, eps):
    return eps * np.sin(om * (a @ ve) + ph)


def grad_e_field(a, ve, om, ph, eps):
    return (eps * om * np.cos(om * (a @ ve) + ph))[..., None] * ve


def solve_eta(Q, eps_e, lo=1e-4, hi=1e4):
    """MPO E-step dual: minimise eta*eps_e + eta*log mean exp(Q/eta). Batched bisection
    on the derivative, which is monotone in eta."""
    def dual_grad(eta):
        z = Q / eta[:, None]
        zm = z.max(axis=1, keepdims=True)
        lse = zm[:, 0] + np.log(np.mean(np.exp(z - zm), axis=1))
        w = np.exp(z - zm); w /= w.sum(axis=1, keepdims=True)
        return eps_e + lse - (w * Q).sum(axis=1) / eta
    n = Q.shape[0]
    a, b = np.full(n, lo), np.full(n, hi)
    for _ in range(60):
        m = np.sqrt(a * b)
        neg = dual_grad(m) < 0
        a = np.where(neg, m, a); b = np.where(neg, b, m)
    return np.sqrt(a * b)


def estimators(u, sig, astar, ve, om, ph, eps, with_error):
    """u: (R, M, d) whitened draws. Returns g_PW, g_ZO, dmu_WML for Q_pi (+e if asked)."""
    a = sig * u                                   # mu = 0, Sigma^{1/2} = sig*I
    Q = q_pi(a, astar)
    gr = grad_q_pi(a, astar)
    if with_error:
        Q = Q + e_field(a, ve, om, ph, eps)
        gr = gr + grad_e_field(a, ve, om, ph, eps)
    g_pw = gr.mean(axis=1)
    Qc = Q - Q.mean(axis=1, keepdims=True)
    g_zo = (Qc[..., None] * u).mean(axis=1) / sig          # Sigma^{-1/2} = 1/sig
    eta = solve_eta(Q, EPS_E)
    z = Q / eta[:, None]; z -= z.max(axis=1, keepdims=True)
    w = np.exp(z); w /= w.sum(axis=1, keepdims=True)
    dmu_wml = (w[..., None] * a).sum(axis=1)               # sum_i w_i (a_i - mu)
    return g_pw, g_zo, dmu_wml


def tr_step(g, sig):
    """Delta_mu = sqrt(2 eps) Sigma g / ||g||_Sigma, with Sigma = sig^2 I."""
    nrm = sig * np.linalg.norm(g, axis=-1, keepdims=True)   # ||g||_Sigma = sig*||g||
    return np.sqrt(2 * EPS_TR) * (sig ** 2) * g / np.maximum(nrm, 1e-300)


def tr_err(dmu, dmu_star, sig):
    """||v||^2_{Sigma^-1} of the difference."""
    return np.sum((dmu - dmu_star) ** 2, axis=-1) / sig ** 2


# ---------------------------------------------------------------- sweep
def run_sweep():
    """Execute the committed grid. Importing the module must not run it, so the
    estimator functions can be unit-tested on their own."""
    rng = np.random.default_rng(G["seed"])
    rows = []
    combos = list(itertools.product(G["d"], G["sigma"], G["omega"]))
    for ci, (d, sig, om) in enumerate(combos):
        r = sig * om / np.sqrt(d)
        acc = {k: [] for k in ("var_pw_e", "var_zo_e", "mse_pw", "mse_zo", "mse_wml",
                               "cos_pw", "cos_zo", "cos_wml", "win_wml_vs_pw",
                               "win_zo_vs_pw", "om_meas")}
        for _ in range(G["n_directions"]):
            v_sig = rng.normal(size=d); v_sig /= np.linalg.norm(v_sig)
            ve = rng.normal(size=d); ve /= np.linalg.norm(ve)
            ph = rng.uniform(0, 2 * np.pi)
            astar = G_SIG * v_sig
            u = rng.normal(size=(G["n_batches"], M, d))

            # common random numbers: identical u for the clean and contaminated critic
            pw0, zo0, wml0 = estimators(u, sig, astar, ve, om, ph, 0.0, False)
            pw1, zo1, wml1 = estimators(u, sig, astar, ve, om, ph, CFG["planted_error"]["eps"], True)

            # EXACT error-induced component (Claim 4's Var[.]_e), isolated by CRN
            dpw, dzo = pw1 - pw0, zo1 - zo0
            acc["var_pw_e"].append(np.trace(np.cov(dpw.T)) if d > 1 else np.var(dpw))
            acc["var_zo_e"].append(np.trace(np.cov(dzo.T)) if d > 1 else np.var(dzo))

            # operational: trust-region update error against the EXACT oracle
            g_star = grad_q_pi(np.zeros(d), astar)          # exact; blur-invariant for a quadratic
            dmu_star = tr_step(g_star[None], sig)
            for nm, gg in (("pw", pw1), ("zo", zo1)):
                dm = tr_step(gg, sig)
                acc[f"mse_{nm}"].append(tr_err(dm, dmu_star, sig).mean())
                acc[f"cos_{nm}"].append(np.mean(
                    (gg @ g_star) / (np.linalg.norm(gg, axis=1) * np.linalg.norm(g_star) + 1e-300)))
            dm_w = np.sqrt(2 * EPS_TR) * wml1 / np.maximum(
                np.linalg.norm(wml1, axis=-1, keepdims=True) / sig, 1e-300)
            acc["mse_wml"].append(tr_err(dm_w, dmu_star, sig).mean())
            acc["cos_wml"].append(np.mean(
                (wml1 @ g_star) / (np.linalg.norm(wml1, axis=1) * np.linalg.norm(g_star) + 1e-300)))
            acc["win_wml_vs_pw"].append(np.mean(
                tr_err(dm_w, dmu_star, sig) < tr_err(tr_step(pw1, sig), dmu_star, sig)))
            acc["win_zo_vs_pw"].append(np.mean(
                tr_err(tr_step(zo1, sig), dmu_star, sig) < tr_err(tr_step(pw1, sig), dmu_star, sig)))
            # numerical calibration of omega against the analytic value
            ag = rng.normal(size=(20000, d)) * sig
            acc["om_meas"].append(
                np.abs(grad_e_field(ag, ve, om, ph, 1.0)).max() and
                (np.linalg.norm(grad_e_field(ag, ve, om, ph, 1.0), axis=-1).max()
                 / max(np.abs(e_field(ag, ve, om, ph, 1.0)).max(), 1e-12)))

        row = dict(d=d, sigma=sig, omega=om, r=r,
                   **{k: float(np.mean(v)) for k, v in acc.items()})
        row["ratio_e"] = row["var_zo_e"] / row["var_pw_e"]
        row["mse_ratio_zo_pw"] = row["mse_zo"] / row["mse_pw"]
        row["mse_ratio_wml_pw"] = row["mse_wml"] / row["mse_pw"]
        rows.append(row)
        if ci % 40 == 0:
            print(f"  [{ci+1}/{len(combos)}] d={d} sig={sig} om={om} r={r:.3g} "
                  f"ratio_e={row['ratio_e']:.4g}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/planted_sweep.csv", index=False)
    print(f"\nwrote {OUT}/planted_sweep.csv  ({len(df)} cells)")
    print(f"omega calibration: max |measured/nominal - 1| = "
          f"{np.abs(df.om_meas / df.omega - 1).max():.3e}")
    print("\n=== primary: ratio_e = Var[g_ZO]_e / Var[g_PW]_e, binned by r ===")
    df["rbin"] = pd.cut(np.log10(df.r), bins=np.arange(-2.5, 2.6, 0.5))
    print(df.groupby("rbin", observed=True).agg(
        n=("r", "size"), r_med=("r", "median"),
        ratio_e_med=("ratio_e", "median"),
        frac_zo_better=("ratio_e", lambda s: float((s < 1).mean()))).to_string())
    lo = df[df.r < 1].ratio_e
    hi = df[df.r > 1].ratio_e
    print(f"\nr<1 : median ratio_e = {lo.median():.4g}   ZO better in {(lo<1).mean():.1%} of cells")
    print(f"r>1 : median ratio_e = {hi.median():.4g}   ZO better in {(hi<1).mean():.1%} of cells")



if __name__ == "__main__":
    run_sweep()