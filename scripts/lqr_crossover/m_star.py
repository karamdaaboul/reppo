"""Track A: the M* surface. What sets the E-step's sample requirement?

    python scripts/lqr_crossover/m_star.py [--d ...] [--regime low|high] [--smoke]

M*(tau; d, eps_E, arm) is the smallest M at which the mean cosine to the EXACT estimand
g* reaches tau, interpolated on log M. Cosine rather than MSE because it is scale-free
and so comparable across d; MSE is recorded too.

Chatterjee & Diaconis (2018) give M* ~ exp(D(q*||pi_old)) = exp(eps_E), independent of d,
for SELF-NORMALISED IS of a SCALAR. g_ZO and d_ESTEP are d-VECTORS. This sweep separates
the two axes. See docs/prereg_m_star.md -- in particular Sec. 2, which records that
b ~ 1 for the centred ZO arm is analytically determined by E0a and is NOT a finding.

PAIRING. One draw of u per cell serves all three arms and all six eps_E values: q and H
are computed once and only w = softmax(q/eta) changes. The arms are therefore paired on
identical samples, not merely compared.
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
from scripts.lqr_crossover.sweep import OMEGAS, SIGMAS, git_sha  # noqa: E402
from src.jaxrl.estimators import (  # noqa: E402
    centred_zo, pathwise_mean, softmax_displacement, whitened_pathwise,
)

DS = (2, 4, 8, 16, 21, 32, 64)
MS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
EPS_ES = (0.05, 0.1, 0.5, 1.0, 2.0, 4.0)
TAUS = (0.90, 0.95, 0.99)
SIGMA = float(SIGMAS[12])                      # 0.3669
REGIMES = {"low": float(OMEGAS[18]),           # c = sigma*omega ~ 8.15
           "high": float(OMEGAS[31])}          # c ~ 404
ETA_GRID = jnp.asarray(np.exp(np.linspace(np.log(1e-4), np.log(1e4), 160)))
NSTATES, NREPS = 32, 48
EPS_FRAC = 0.05


def prereg_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", "docs/prereg_m_star.md"],
            cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _solve_eta(q, eps_e):
    """Grid-minimise the MPO dual, same form as eta_dual_loss in src/jaxrl/reppo.py.

    One eta per (M, eps_E) batch, matching production where eta is a single learned
    scalar shared across the minibatch rather than per state.
    """
    def dual(e):
        qm = q.max(-1, keepdims=True)
        lse = jnp.log(jnp.mean(jnp.exp((q - qm) / e), axis=-1)) + qm[..., 0] / e
        return e * eps_e + e * lse.mean()
    return ETA_GRID[jnp.argmin(jax.vmap(dual)(ETA_GRID))]


def _cos(a, b):
    return jnp.sum(a * b, -1) / jnp.maximum(
        jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1), 1e-300)


def run(d, regime, *, n_states=NSTATES, n_reps=NREPS, eps_frac=EPS_FRAC, tag=""):
    t0 = time.time()
    omega = REGIMES[regime]
    s = lqr.build_system(d, seed=SEED_ROOT + d)
    rng = np.random.default_rng(SEED_ROOT + 8000 + d)
    st = lqr.sample_states(s, rng, n_states)
    H, g, mu = lqr.q_coeffs(s, st)
    pe = EF.draw_error(rng, n_states, d, kind="rank1", omega=omega)
    eps = eps_frac * lqr.q_spread_closed_form(s, st, SIGMA)
    gstar = jnp.asarray(np.asarray(g) + EF.blurred_e_grad(pe, eps, mu, SIGMA))
    mj = jnp.asarray(mu)[:, None, None, :]
    ej = jnp.asarray(eps)[:, None, None]
    qf = lqr.q_of_u_factory(s, st, SIGMA)

    def q_of_u(u):
        return qf(u) + EF.e_value(pe, ej, mj, SIGMA, u)

    nM, nE = len(MS), len(EPS_ES)
    Z = lambda *sh: np.zeros(sh)
    out = dict(
        cos_pw=Z(nM, n_states), cos_zo=Z(nM, n_states), cos_es=Z(nM, nE, n_states),
        mse_pw=Z(nM, n_states), mse_zo=Z(nM, n_states), mse_es=Z(nM, nE, n_states),
        ess=Z(nM, nE, n_states), kl_unif=Z(nM, nE, n_states),
        logit_sd=Z(nM, nE, n_states), eta=Z(nM, nE),
    )
    gs = gstar[:, None, :]
    for im, M in enumerate(MS):
        # keep the live array near 2e7 elements regardless of (M, d)
        ch = max(1, int(2e7 // max(n_states * M * d, 1)))
        key = jax.random.fold_in(jax.random.PRNGKey(SEED_ROOT + 9000 + d), im)
        done = 0
        etas = None
        while done < n_reps:
            b = min(ch, n_reps - done)
            key, sub = jax.random.split(key)
            u = jax.random.normal(sub, (n_states, b, M, d))
            q, Hw = whitened_pathwise(q_of_u, u)
            g_pw = pathwise_mean(Hw, axis=-2) / SIGMA
            g_zo = centred_zo(q, u, axis=2, deattenuate=True,
                              reduce="broadcast") / SIGMA
            out["cos_pw"][im] += np.asarray(_cos(g_pw, gs).sum(1))
            out["cos_zo"][im] += np.asarray(_cos(g_zo, gs).sum(1))
            out["mse_pw"][im] += np.asarray(((g_pw - gs) ** 2).sum(-1).sum(1))
            out["mse_zo"][im] += np.asarray(((g_zo - gs) ** 2).sum(-1).sum(1))
            if etas is None:
                etas = [float(_solve_eta(q, e)) for e in EPS_ES]
                for ie, e in enumerate(etas):
                    out["eta"][im, ie] = e
            for ie, eta in enumerate(etas):
                lg = q / eta
                w = jax.nn.softmax(lg, axis=-1)
                d_es = softmax_displacement(w, u, axis=2) / SIGMA
                out["cos_es"][im, ie] += np.asarray(_cos(d_es, gs).sum(1))
                out["mse_es"][im, ie] += np.asarray(((d_es - gs) ** 2).sum(-1).sum(1))
                out["ess"][im, ie] += np.asarray((1.0 / (w ** 2).sum(-1)).sum(1))
                # KL(w || uniform) = log M + sum_i w_i log w_i
                kl = np.log(M) + np.asarray(jnp.sum(w * jnp.log(jnp.maximum(w, 1e-300)),
                                                    axis=-1))
                out["kl_unif"][im, ie] += kl.sum(1)
                out["logit_sd"][im, ie] += np.asarray(lg.std(axis=-1).sum(1))
            done += b
        for k in out:
            if k != "eta":
                out[k][im] /= n_reps

    name = f"mstar_d{d}_{regime}{tag}"
    path = os.path.join(OUT, name + ".npz")
    np.savez_compressed(
        path, d=d, regime=regime, sigma=SIGMA, omega=omega, c=SIGMA * omega,
        Ms=np.array(MS), eps_Es=np.array(EPS_ES), taus=np.array(TAUS),
        n_states=n_states, n_reps=n_reps, eps_frac=eps_frac,
        rho_closed=s.rho_closed, cond_H=s.cond_H, tr_H2=s.tr_H2,
        git_sha=git_sha(), prereg_sha=prereg_sha(),
        seconds=time.time() - t0, **out)
    print(f"  d={d:3d} {regime:>4} -> {name}.npz ({time.time()-t0:6.1f}s) "
          f"cos@M=32: PW {out['cos_pw'][3].mean():.4f} ZO {out['cos_zo'][3].mean():.4f} "
          f"ES(eps=0.5) {out['cos_es'][3, 2].mean():.4f}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--regime", default=None, choices=list(REGIMES))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    kw = dict(tag=a.tag)
    if a.smoke:
        kw.update(n_states=4, n_reps=8, tag=a.tag + "_smoke")
    os.makedirs(OUT, exist_ok=True)
    ds = [a.d] if a.d else list(DS)
    regs = [a.regime] if a.regime else list(REGIMES)
    print(f"Track A: M* surface. sigma={SIGMA:.4f} sha={git_sha()[:10]}")
    print(f"  prereg committed at {prereg_sha()} (docs/prereg_m_star.md)")
    t0 = time.time()
    for reg in regs:
        for d in ds:
            run(d, reg, **kw)
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
