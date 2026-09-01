"""Crossover location, exponent fit, and the hierarchical bootstrap.

Conventions are the house ones (`scripts/probe1_report.py`): ratios are formed PER STATE
before any aggregation, the bootstrap is hierarchical over states then batches, seeds are
date-derived, and 10 000 resamples with percentile CIs.

Two things this module must not do, both registered in docs/prereg_lqr_crossover.md:
  * never subtract two variances -- everything is a log-ratio, because the crossover is
    exactly where the two are equal;
  * never accept a root at a grid edge -- that means the range was inadequate, not that
    a crossover was found.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

BOOT_RNG = 20260902
NBOOT = 10000


def load(path):
    return np.load(path, allow_pickle=True)


def eps_over_sigma(z):
    """(n_states, n_sigma). eps is set per (state, sigma) from the Q spread."""
    return z["eps"] / z["sigmas"][None, :]


def err_only(z, batch_mean=True):
    """Error-only squared error per arm. (n_states[, n_batch], n_sigma, n_omega).

    In units of (eps/sigma)^2 -- and since BOTH arms carry that same prefactor, it
    cancels from the E1a ratio entirely. That is why the E1a contour is exactly
    eps-independent (prereg Sec. 3), and it is why no eps appears in this function.
    """
    p, q = z["s3_pw"], z["s3_zo"]
    return (p.mean(1), q.mean(1)) if batch_mean else (p, q)


def full_mse(z, batch_mean=True):
    """E1b full-estimator MSE, reconstructed from the three statistics.

        ||g - g*||^2 = s1(sigma) + 2 (eps/sigma) s2(sigma) + (eps/sigma)^2 s3(sigma,omega)

    Only the DIAGONAL of s2 is physical: a cell (sigma_i, omega_j) pairs the error part
    with the smooth part at the same sigma_i.
    """
    r = eps_over_sigma(z)[:, :, None]                       # (states, sigma, 1)
    out = []
    for pw in (True, False):
        s1 = z["s1_pw" if pw else "s1_zo"]                   # (st, b, sigma)
        s2 = z["s2_pw" if pw else "s2_zo"]                   # (st, b, S, sigma, omega)
        s3 = z["s3_pw" if pw else "s3_zo"]                   # (st, b, sigma, omega)
        s2d = np.einsum("bsso->bso", s2) if s2.ndim == 4 else \
            np.stack([s2[:, :, i, i, :] for i in range(s2.shape[2])], axis=2)
        m = s1[..., None] + 2.0 * r[:, None] * s2d + (r[:, None] ** 2) * s3
        out.append(m.mean(1) if batch_mean else m)
    return out[0], out[1]


def cross_share(z):
    """|2 (eps/sigma) s2| / ((eps/sigma)^2 s3): the spec Sec. 5.2 reconciliation.

    Claim 4's decomposition drops this term. If it is comparable to the error-only part
    anywhere in the swept region, that is a limitation of the claim and belongs in the
    paper (registered threshold: 0.25).
    """
    r = eps_over_sigma(z)[:, :, None]
    out = {}
    for nm, pw in (("pw", True), ("zo", False)):
        s2 = z["s2_pw" if pw else "s2_zo"]
        s3 = z["s3_pw" if pw else "s3_zo"]
        s2d = np.stack([s2[:, :, i, i, :] for i in range(s2.shape[2])], axis=2)
        num = np.abs(2.0 * r[:, None] * s2d).mean(1)
        den = ((r[:, None] ** 2) * s3).mean(1)
        out[nm] = num / np.maximum(den, 1e-300)
    return out


def _c_index_groups(n_sigma, n_omega):
    """Cells (i, j) grouped by k = i + j, i.e. by the value of c = sigma*omega.

    The sigma and omega grids share a log ratio, so sigma_i * omega_j depends only on
    i + j and the 2-D grid collapses to n_sigma + n_omega - 1 distinct c values. That
    redundancy is not waste: it is the falsifiable collapse check of the prereg.
    """
    groups = {}
    for i in range(n_sigma):
        for j in range(n_omega):
            groups.setdefault(i + j, []).append((i, j))
    return groups


def c_grid(z):
    n_s, n_o = len(z["sigmas"]), len(z["omegas"])
    return z["sigmas"][0] * z["omegas"][0] * \
        (z["omegas"][1] / z["omegas"][0]) ** np.arange(n_s + n_o - 1)


def log_ratio_by_c(pw, zo, n_sigma, n_omega):
    """Per-state log(Var_ZO_e / Var_PW_e), averaged within each c level-set.

    Per-state-then-aggregate: the ratio is formed inside a state before anything is
    pooled across states (house convention).
    """
    g = _c_index_groups(n_sigma, n_omega)
    ks = sorted(g)
    r = np.log(np.maximum(zo, 1e-300)) - np.log(np.maximum(pw, 1e-300))
    return np.stack([np.mean([r[..., i, j] for i, j in g[k]], axis=0) for k in ks], -1)


def solve_crossover(logc, rbar, *, require_interior=True):
    """Root of a monotone decreasing log-ratio in log c. Returns (c_star, ok).

    Uniqueness is structural: Var_PW ~ c^4 and Var_ZO ~ c^2 as c -> 0, while Var_PW ~ c^2
    and Var_ZO saturates as c -> inf, so the log-ratio runs from +inf to -inf.
    """
    if not (rbar[0] > 0 > rbar[-1]):
        return np.nan, False
    sgn = np.where(np.diff(np.sign(rbar)) != 0)[0]
    if len(sgn) == 0:
        return np.nan, False
    k = sgn[0]
    if require_interior and (k == 0 or k >= len(logc) - 2):
        return np.nan, False
    f = PchipInterpolator(logc, rbar)
    try:
        return float(np.exp(brentq(f, logc[k], logc[k + 1], xtol=1e-13))), True
    except ValueError:
        return np.nan, False


def crossover_by_c(z, *, per_state=False):
    """c*(d) from the c-collapsed grid. Primary E1a estimator."""
    pw, zo = err_only(z)
    n_s, n_o = len(z["sigmas"]), len(z["omegas"])
    r = log_ratio_by_c(pw, zo, n_s, n_o)                    # (states, n_c)
    logc = np.log(c_grid(z))
    if per_state:
        return np.array([solve_crossover(logc, r[i])[0] for i in range(r.shape[0])])
    return solve_crossover(logc, r.mean(0))


def crossover_by_column(z):
    """c* solved independently in each sigma column. The collapse check.

    Registered threshold: sd of log(sigma*omega*) about log c*(d) below 0.05.
    """
    pw, zo = err_only(z)
    r = np.log(np.maximum(zo, 1e-300)) - np.log(np.maximum(pw, 1e-300))
    rbar = r.mean(0)                                        # (n_sigma, n_omega)
    logom = np.log(z["omegas"])
    out = []
    for i, sg in enumerate(z["sigmas"]):
        om_star, ok = solve_crossover(logom, rbar[i])
        out.append(sg * om_star if ok else np.nan)
    return np.array(out)


def fit_p(ds, cstars):
    m = np.isfinite(cstars)
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(np.asarray(ds)[m]), np.log(np.asarray(cstars)[m]), 1)[0])


def bootstrap_p(zs, ds, *, nboot=NBOOT, seed=BOOT_RNG, level=None):
    """Hierarchical bootstrap over states, then batches within each drawn state.

    `level` freezes one level ("states" or "batches") so the variance decomposition by
    level can be reported. A CI driven entirely by the batch level is an artefact of N,
    a knob, not evidence about p -- the report must say which dominates.
    """
    rng = np.random.default_rng(seed)
    ps = []
    for _ in range(nboot):
        cs = []
        for z in zs:
            p3, z3 = z["s3_pw"], z["s3_zo"]
            n_st, n_b = p3.shape[0], p3.shape[1]
            si = np.arange(n_st) if level == "batches" else \
                rng.integers(0, n_st, n_st)
            pw = np.empty((n_st, *p3.shape[2:])); zo = np.empty_like(pw)
            for a, s in enumerate(si):
                bi = np.arange(n_b) if level == "states" else rng.integers(0, n_b, n_b)
                pw[a] = p3[s, bi].mean(0); zo[a] = z3[s, bi].mean(0)
            r = log_ratio_by_c(pw, zo, len(z["sigmas"]), len(z["omegas"]))
            cs.append(solve_crossover(np.log(c_grid(z)), r.mean(0))[0])
        ps.append(fit_p(ds, np.array(cs)))
    ps = np.array(ps); ps = ps[np.isfinite(ps)]
    if len(ps) < 2:
        # Fewer than two d values (or no bracketed crossover) -- no slope to fit. Report
        # it as undefined rather than emitting a number from an empty resample set.
        return dict(p_mean=np.nan, p_sd=np.nan, ci=(np.nan, np.nan), nboot=len(ps))
    return dict(p_mean=float(ps.mean()), p_sd=float(ps.std(ddof=1)),
                ci=(float(np.percentile(ps, 2.5)), float(np.percentile(ps, 97.5))),
                nboot=len(ps))
