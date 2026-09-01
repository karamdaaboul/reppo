"""Does the ESTEP-minus-ZO cosine gap shrink as M grows?

If it does, the shipped E-step's deficit is FINITE-SAMPLE WEIGHTING BIAS -- the softmax
weights are self-normalised, so sum_i w_i u_i carries an O(1/M) bias that the centred ZO
(unbiased after M/(M-1)) does not -- and not a property of the operator. That distinction
decides whether Section 7.6's d=21 result is a crossover or an artefact of M=32.

d=21 is included because that is HumanoidRun's action dimension.
"""
import sys, os, json, time
sys.path.insert(0, "/home/human/workspaces/reppo_original")
import scripts.lqr_crossover as L
from scripts.lqr_crossover import lqr, error_field as EF, OUT, SEED_ROOT
from scripts.lqr_crossover.sweep import SIGMAS, OMEGAS
from src.jaxrl.estimators import centred_zo, pathwise_mean, softmax_displacement, whitened_pathwise
import numpy as np, jax, jax.numpy as jnp

ETA = jnp.asarray(np.exp(np.linspace(np.log(1e-4), np.log(1e4), 140)))
EPS_E = 0.5
DS = (4, 16, 21, 64)
MS = (8, 32, 128, 512)
SIGMA = float(SIGMAS[12])                 # 0.367, comfortably above production min_std 0.1
OMEGA = float(OMEGAS[18])                 # c = sigma*omega ~ 8
NST, NREP, CH = 32, 200, 25

def solve_eta(q):
    def dual(e):
        qm = q.max(-1, keepdims=True)
        lse = jnp.log(jnp.mean(jnp.exp((q - qm) / e), axis=-1)) + qm[..., 0] / e
        return e * EPS_E + e * lse.mean()
    v = jax.vmap(dual)(ETA)
    return ETA[jnp.argmin(v)]

def cosv(a, b):
    return jnp.sum(a * b, -1) / jnp.maximum(
        jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1), 1e-300)

rows = []
print(f"sigma={SIGMA:.4f}  omega={OMEGA:.3f}  c={SIGMA*OMEGA:.3f}  "
      f"states={NST} reps={NREP}", flush=True)
print(f"{'d':>4} {'M':>5} {'cos PW':>8} {'cos ZO':>8} {'cos ES':>8} {'gap ZO-ES':>10} "
      f"{'gap*M':>8} {'ESS/M':>7} {'eta':>8}", flush=True)
for d in DS:
    s = lqr.build_system(d, seed=SEED_ROOT + d)
    rng = np.random.default_rng(SEED_ROOT + 6000 + d)
    st = lqr.sample_states(s, rng, NST)
    H, g, mu = lqr.q_coeffs(s, st)
    pe = EF.draw_error(rng, NST, d, kind="rank1", omega=OMEGA)
    eps = 0.05 * lqr.q_spread_closed_form(s, st, SIGMA)
    gstar = jnp.asarray(np.asarray(g) + EF.blurred_e_grad(pe, eps, mu, SIGMA))
    mj = jnp.asarray(mu)[:, None, None, :]; ej = jnp.asarray(eps)[:, None, None]
    qf = lqr.q_of_u_factory(s, st, SIGMA)
    q_of_u = lambda u: qf(u) + EF.e_value(pe, ej, mj, SIGMA, u)
    for M in MS:
        acc = {k: 0.0 for k in ("pw", "zo", "es", "ess")}
        eta_last = 0.0
        key = jax.random.PRNGKey(SEED_ROOT + 7000 + d * 13 + M)
        done = 0
        while done < NREP:
            b = min(CH, NREP - done)
            key, sub = jax.random.split(key)
            u = jax.random.normal(sub, (NST, b, M, d))
            q, Hw = whitened_pathwise(q_of_u, u)
            g_pw = pathwise_mean(Hw, axis=-2) / SIGMA
            g_zo = centred_zo(q, u, axis=2, deattenuate=True, reduce="broadcast") / SIGMA
            eta = solve_eta(q); eta_last = float(eta)
            w = jax.nn.softmax(q / eta, axis=-1)
            d_es = softmax_displacement(w, u, axis=2) / SIGMA
            gs = gstar[:, None, :]
            acc["pw"] += float(cosv(g_pw, gs).sum()); acc["zo"] += float(cosv(g_zo, gs).sum())
            acc["es"] += float(cosv(d_es, gs).sum())
            acc["ess"] += float((1.0 / (w ** 2).sum(-1)).sum())
            done += b
        n = NST * NREP
        cp, cz, ce = acc["pw"] / n, acc["zo"] / n, acc["es"] / n
        gap = cz - ce
        rows.append(dict(d=d, M=M, cos_pw=cp, cos_zo=cz, cos_es=ce, gap=gap,
                         ess_frac=acc["ess"] / n / M, eta=eta_last))
        print(f"{d:4d} {M:5d} {cp:8.4f} {cz:8.4f} {ce:8.4f} {gap:10.4f} "
              f"{gap*M:8.2f} {acc['ess']/n/M:7.3f} {eta_last:8.3f}", flush=True)
json.dump(rows, open(os.path.join(OUT, "m_sweep_estep.json"), "w"), indent=1)
print("\nwrote m_sweep_estep.json")
