"""Arm 3: the operator that actually ships, measured against the same estimand.

    python scripts/lqr_crossover/estep_arm.py [--smoke]

WHY THIS ARM EXISTS. Claim 4 and Proposition 1 are stated about the centred zeroth-order
estimator `g_ZO = (1/(sigma M)) sum_i (Q_i - Qbar) u_i`. That is NOT what
`actor_update_mode="weighted_mle"` optimises. The shipped arm maximises a
softmax-weighted MLE, whose mean displacement is

    d_ESTEP = sum_i w_i u_i,     w = softmax(Q(s, a_i) / eta)

with eta solved against the E-step KL budget eps_e. The two coincide only to first order
in 1/eta. Comparing them on identical draws is the difference between a paper whose
theory describes its code and one where the correspondence is assumed.

WHY COSINE AND NOT MSE. d_ESTEP is a displacement, not a gradient: its magnitude is set
by eta and the trust region, not by ||grad Q||. An MSE against g* would mostly measure
eta. The scale-free question -- does the shipped operator point where the theory says --
is a direction comparison, so this module reports cosine to the exact blurred estimand
and leaves magnitude to the dual diagnostics.

eta is grid-minimised on the dual, matching `eta_dual_loss` in src/jaxrl/reppo.py and the
practice of scripts/verify_estep.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# three levels: this package sits at scripts/lqr_crossover/, not scripts/
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import scripts.lqr_crossover  # noqa: F401,E402

import jax  # noqa: E402
import numpy as np  # noqa: E402
from jax import numpy as jnp  # noqa: E402

from scripts.lqr_crossover import OUT, SEED_ROOT, error_field as EF, lqr  # noqa: E402
from scripts.lqr_crossover.sweep import DS, OMEGAS, SIGMAS, git_sha  # noqa: E402
from src.jaxrl.estimators import (  # noqa: E402
    centred_zo, pathwise_mean, softmax_displacement, whitened_pathwise,
)

ETA_GRID = jnp.asarray(np.exp(np.linspace(np.log(1e-4), np.log(10.0), 80)))
EPS_E = 0.5                              # config/reppo.yaml
SUB_SIGMA = (2, 7, 12, 17)               # subgrid indices into SIGMAS
SUB_OMEGA = (4, 11, 18, 25, 32)          # subgrid indices into OMEGAS


def solve_eta(q):
    """Grid-minimise the MPO dual  g(eta) = eta*eps_e + eta*mean_j logmeanexp(q_ji/eta).

    Same form as `eta_dual_loss` (src/jaxrl/reppo.py), with the per-state max pulled out
    so the exponent is <= 0 everywhere. q: (..., M).
    """
    def dual(eta):
        qm = q.max(-1, keepdims=True)
        lse = jnp.log(jnp.mean(jnp.exp((q - qm) / eta), axis=-1)) + qm[..., 0] / eta
        return eta * EPS_E + eta * lse.mean()
    vals = jax.vmap(dual)(ETA_GRID)
    return ETA_GRID[jnp.argmin(vals)]


def cos(a, b):
    na = jnp.linalg.norm(a, axis=-1)
    nb = jnp.linalg.norm(b, axis=-1)
    return jnp.sum(a * b, -1) / jnp.maximum(na * nb, 1e-300)


def run(d, *, M=32, n_states=32, n_rep=400, kind="rank1", eps_frac=0.05, smoke=False):
    if smoke:
        n_states, n_rep = 4, 50
    t0 = time.time()
    s = lqr.build_system(d, seed=SEED_ROOT + d)
    rng = np.random.default_rng(SEED_ROOT + 4000 + d)
    states = lqr.sample_states(s, rng, n_states)
    H, g, mu = lqr.q_coeffs(s, states)
    pe = EF.draw_error(rng, n_states, d, kind=kind, omega=1.0)

    ns, no = len(SUB_SIGMA), len(SUB_OMEGA)
    out = {k: np.zeros((n_states, ns, no)) for k in
           ("cos_pw", "cos_zo", "cos_estep", "eta", "ess", "norm_ratio")}

    for si, i in enumerate(SUB_SIGMA):
        sigma = float(SIGMAS[i])
        eps = eps_frac * lqr.q_spread_closed_form(s, states, sigma)
        for oj, j in enumerate(SUB_OMEGA):
            omega = float(OMEGAS[j])
            pe_o = EF.PlantedError(pe.kind, pe.rank, omega, pe.V, pe.phi)
            # exact estimand: smooth part (grad_a Q^pi at mu) + blurred error gradient
            gstar = np.asarray(g) + EF.blurred_e_grad(pe_o, eps, mu, sigma)
            key = jax.random.fold_in(jax.random.PRNGKey(SEED_ROOT + 5000 + d), i * 97 + j)
            u = jax.random.normal(key, (n_states, n_rep, M, d))
            mu_j = jnp.asarray(mu)[:, None, None, :]
            eps_j = jnp.asarray(eps)[:, None, None]
            qf = lqr.q_of_u_factory(s, states, sigma)

            def q_of_u(uu):
                return qf(uu) + EF.e_value(pe_o, eps_j, mu_j, sigma, uu)

            q, Hw = whitened_pathwise(q_of_u, u)
            g_pw = pathwise_mean(Hw, axis=-2) / sigma
            g_zo = centred_zo(q, u, axis=2, deattenuate=True, reduce="broadcast") / sigma
            eta = solve_eta(q)
            w = jax.nn.softmax(q / eta, axis=-1)
            d_es = softmax_displacement(w, u, axis=2) / sigma

            gs = jnp.asarray(gstar)[:, None, :]
            out["cos_pw"][:, si, oj] = np.asarray(cos(g_pw, gs).mean(-1))
            out["cos_zo"][:, si, oj] = np.asarray(cos(g_zo, gs).mean(-1))
            out["cos_estep"][:, si, oj] = np.asarray(cos(d_es, gs).mean(-1))
            out["eta"][:, si, oj] = float(eta)
            out["ess"][:, si, oj] = np.asarray((1.0 / (w ** 2).sum(-1)).mean(-1))
            out["norm_ratio"][:, si, oj] = np.asarray(
                (jnp.linalg.norm(d_es, axis=-1)
                 / jnp.maximum(jnp.linalg.norm(g_zo, axis=-1), 1e-300)).mean(-1))

    path = os.path.join(OUT, f"estep_d{d}_{kind}_M{M}{'_smoke' if smoke else ''}.npz")
    np.savez_compressed(
        path, d=d, M=M, kind=kind, n_states=n_states, n_rep=n_rep, eps_frac=eps_frac,
        sigmas=SIGMAS[list(SUB_SIGMA)], omegas=OMEGAS[list(SUB_OMEGA)],
        eps_e=EPS_E, git_sha=git_sha(), seconds=time.time() - t0, **out)
    print(f"  d={d:3d} -> {os.path.basename(path)} ({time.time() - t0:.1f}s)  "
          f"cos: PW {out['cos_pw'].mean():.4f}  ZO {out['cos_zo'].mean():.4f}  "
          f"ESTEP {out['cos_estep'].mean():.4f}  ESS {out['ess'].mean():.1f}/{M}")
    with open(os.path.join(OUT, "index.jsonl"), "a") as f:
        f.write(json.dumps(dict(name=os.path.basename(path), arm="estep", d=d, M=M,
                                kind=kind, git_sha=git_sha(),
                                seconds=round(time.time() - t0, 2))) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--kind", default="rank1")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print(f"E-step arm (the shipped weighted_mle operator), eps_e={EPS_E}")
    for d in ([a.d] if a.d else list(DS)):
        run(d, kind=a.kind, smoke=a.smoke)


if __name__ == "__main__":
    main()
