"""Offline epsilon_E counterfactual on Walker WML. NO TRAINING.

For each corrected Walker WML checkpoint and the canonical neutral Walker bank:
draw ONE fixed candidate set (M=32) with the actual training generator, evaluate Q
with the actual critic path, then solve the implementation's MPO eta dual twice on
the IDENTICAL candidates and Q values -- once at eps_E=0.5, once at eps_E=0.1.

  dual:  g(eta) = eta*eps_E + eta * E_s[ log (1/M) sum_i exp(Q_i/eta) ]
  eta parameterisation/constraint: clip to [1e-4, 10] as in jax_models.py:451-455

Reported per condition: eta, ESS, ESS/M, w_max, the weighted pre-tanh radial
statistic sum_i w_i (y_i-mu)^T Sigma^-1 (y_i-mu), and the weighted-MLE target
moments, with rho_j = sigma_target_j^2 / sigma_old_j^2.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
import jax, jax.numpy as jnp                                          # noqa: E402
from scripts.load_ckpt import load                                    # noqa: E402

BANK = "reports/artifacts/walker_fixed_state_bank.npz"
CK   = "exports/WalkerRun_weighted_mle_s%d_final"
SEEDS = list(range(301, 309))
M, NSTATE, NDRAW = 32, 512, 8
ETA_LO, ETA_HI = 1e-4, 10.0
EPS = (0.5, 0.1)

def dual_g(Q, eta, eps):
    """g(eta) = eta*eps + eta * mean_s log mean_i exp(Q/eta), max-shifted."""
    qm = Q.max(axis=0, keepdims=True)
    lse = np.log(np.mean(np.exp((Q - qm) / eta), axis=0)) + qm[0] / eta
    return eta * eps + eta * float(np.mean(lse))

def solve_eta(Q, eps, iters=200):
    """Golden-section on log(eta) over the implementation's clip range."""
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = np.log(ETA_LO), np.log(ETA_HI)
    for _ in range(iters):
        c, d_ = b - phi * (b - a), a + phi * (b - a)
        if dual_g(Q, np.exp(c), eps) < dual_g(Q, np.exp(d_), eps): b = d_
        else: a = c
    return float(np.clip(np.exp(0.5 * (a + b)), ETA_LO, ETA_HI))

def main():
    z = np.load(BANK, allow_pickle=True)
    obs = np.asarray(z["obs"], np.float32)
    idx = np.random.default_rng(20260905).choice(obs.shape[0], NSTATE, replace=False)
    obs = obs[np.sort(idx)]
    ob = jnp.asarray(obs)
    out = {"bank": BANK, "n_states": NSTATE, "M": M, "n_draws": NDRAW,
           "eta_clip": [ETA_LO, ETA_HI], "seeds": SEEDS, "per_seed": {}}
    print("Walker WML, %d states, M=%d, %d independent candidate draws per seed" % (NSTATE, M, NDRAW))
    print("  %-5s %-6s %8s %9s %8s %9s %11s %10s %10s %9s %9s"
          % ("seed","eps_E","eta","ESS","ESS/M","w_max","radial","rho_med","rho_mean","rho>1","rho_p95"))
    agg = {e: {k: [] for k in ("eta","ess","wmax","radial","rho_med","rho_mean","rho_gt1","rho_p95","ref")} for e in EPS}
    for sd in SEEDS:
        c = load(CK % sd)
        mu, sg = c.policy_dist(ob)
        mu, sg = np.asarray(mu, np.float64), np.asarray(sg, np.float64)
        d = mu.shape[-1]
        rows = {e: [] for e in EPS}
        for dr in range(NDRAW):
            u = np.asarray(jax.random.normal(jax.random.PRNGKey(1000*sd + dr), (M, NSTATE, d)), np.float64)
            y = mu[None] + sg[None] * u
            a = np.tanh(y)
            obr = np.broadcast_to(obs[None], (M, NSTATE, obs.shape[-1]))
            Q = np.asarray(c.q_scalar(jnp.asarray(obr.reshape(-1, obs.shape[-1])),
                                      jnp.asarray(a.reshape(-1, d))), np.float64).reshape(M, NSTATE)
            for eps in EPS:
                eta = solve_eta(Q, eps)
                zz = Q / eta; zz -= zz.max(0, keepdims=True)
                w = np.exp(zz); w /= w.sum(0, keepdims=True)              # (M,S)
                ess = 1.0 / np.sum(w**2, axis=0)                          # (S,)
                wmax = w.max(0)
                radial = np.sum(w * np.sum(u**2, axis=-1), axis=0)        # (S,)
                mw = np.sum(w[..., None] * y, axis=0)                     # (S,d)
                var = np.sum(w[..., None] * (y - mw[None])**2, axis=0)    # (S,d)
                rho = var / (sg**2)
                rows[eps].append(dict(eta=eta, ess=float(ess.mean()), wmax=float(wmax.mean()),
                                      radial=float(radial.mean()), rho_med=float(np.median(rho)),
                                      rho_mean=float(rho.mean()), rho_gt1=float((rho > 1).mean()),
                                      rho_p95=float(np.percentile(rho, 95)),
                                      ref=float(np.mean(1.0 - 1.0/ess))))
        out["per_seed"]["s%d" % sd] = {str(e): rows[e] for e in EPS}
        for eps in EPS:
            m = {k: float(np.mean([r[k] for r in rows[eps]])) for k in rows[eps][0]}
            for k in agg[eps]: agg[eps][k].append(m[k])
            print("  %-5d %-6.1f %8.5f %9.3f %8.4f %9.4f %11.4f %10.4f %10.4f %9.4f %9.4f"
                  % (sd, eps, m["eta"], m["ess"], m["ess"]/M, m["wmax"], m["radial"],
                     m["rho_med"], m["rho_mean"], m["rho_gt1"], m["rho_p95"]))
    print("\n=== across-seed means (d = %d, so unweighted E||u||^2 = %d) ===" % (d, d))
    print("  %-6s %8s %9s %8s %9s %11s %10s %10s %9s %9s %12s"
          % ("eps_E","eta","ESS","ESS/M","w_max","radial","rho_med","rho_mean","rho>1","rho_p95","1-1/ESS ref"))
    summ = {}
    for eps in EPS:
        m = {k: float(np.mean(v)) for k, v in agg[eps].items()}
        summ[str(eps)] = m
        print("  %-6.1f %8.5f %9.3f %8.4f %9.4f %11.4f %10.4f %10.4f %9.4f %9.4f %12.4f"
              % (eps, m["eta"], m["ess"], m["ess"]/M, m["wmax"], m["radial"],
                 m["rho_med"], m["rho_mean"], m["rho_gt1"], m["rho_p95"], m["ref"]))
    dr_med = summ["0.1"]["rho_med"] - summ["0.5"]["rho_med"]
    print("\n  Delta rho (median)  = rho(0.1) - rho(0.5) = %+.5f" % dr_med)
    print("  Delta rho (mean)    = %+.5f" % (summ["0.1"]["rho_mean"] - summ["0.5"]["rho_mean"]))
    print("  Delta eta           = %+.5f  (x%.2f)" % (summ["0.1"]["eta"]-summ["0.5"]["eta"],
                                                      summ["0.1"]["eta"]/summ["0.5"]["eta"]))
    print("  Delta ESS/M         = %+.4f" % (summ["0.1"]["ess"]/M - summ["0.5"]["ess"]/M))
    print("  Delta radial        = %+.4f  (unweighted reference %d)" % (summ["0.1"]["radial"]-summ["0.5"]["radial"], d))
    print("  radial excess over unweighted: eps=0.5 %+.4f   eps=0.1 %+.4f"
          % (summ["0.5"]["radial"]-d, summ["0.1"]["radial"]-d))
    print("  rho vs finite-sample reference (1-1/ESS):  eps=0.5 %+.5f   eps=0.1 %+.5f"
          % (summ["0.5"]["rho_med"]-summ["0.5"]["ref"], summ["0.1"]["rho_med"]-summ["0.1"]["ref"]))
    out["summary"] = summ
    out["deltas"] = dict(rho_median=dr_med, rho_mean=summ["0.1"]["rho_mean"]-summ["0.5"]["rho_mean"],
                         eta=summ["0.1"]["eta"]-summ["0.5"]["eta"],
                         ess_over_M=summ["0.1"]["ess"]/M-summ["0.5"]["ess"]/M,
                         radial=summ["0.1"]["radial"]-summ["0.5"]["radial"], d=d)
    json.dump(out, open("reports/artifacts/walker_eps_counterfactual.json","w"), indent=1)
    print("\nwrote reports/artifacts/walker_eps_counterfactual.json")

if __name__ == "__main__":
    main()
