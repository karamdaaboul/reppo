"""Exact closed forms for the error-only estimator errors, and the crossover they imply.

This module is not in the spec. It is here because the whole of E1a turns out to be
analytically solvable, which converts every measured number in the sweep into a
checkable one (gate G7) and makes the pre-registered exponent a KNOWN quantity rather
than a measured one. Stating that in advance is the difference between "we verified
the estimator identities in closed form" and "we tested sqrt(d) and it passed"; a
reviewer re-derives the latter in ten minutes.

DERIVATION (rank-one field; c = sigma*omega, theta = omega v^T mu + phi).
Decompose the whitened draw as u_i = t_i v + w_i with w_i orthogonal to v, so t_i and
w_i are independent and w_i has d-1 components.

Pathwise. grad_a e is always a scalar multiple of v, so

    dg_PW - estimand = (eps*omega) [ (1/M) sum_i cos(theta + c t_i) - E cos ] v
    Var[g_PW]_e = (eps/sigma)^2 * c^2 * Vcos(c, theta) / M          <- d-INDEPENDENT

Zeroth-order, after the M/(M-1) correction. The component along w contributes, per
orthogonal coordinate, (M/(M-1))^2 M^-2 E[sum_i (f_i - fbar)^2] = Vsin/(M-1), and there
are d-1 of them. The component along v is (M/(M-1)) times the sample covariance of
(f, t) divided by (M-1) -- that is, exactly the UNBIASED sample covariance C of
f = sin(theta + c t) with t. Hence

    Var[g_ZO]_e = (eps/sigma)^2 * [ (d-1) Vsin(c, theta)/(M-1) + Lambda(c, theta, M) ]
    Lambda = Var(C) = (mu22 - mu11^2)/M + (mu20 mu02 - mu11^2)/(M(M-1))

with central moments of (a, b) = (f - E f, t). Lambda is exactly d-INDEPENDENT.

CONSEQUENCES. Both terms carry the same (eps/sigma)^2, so the ratio depends only on
(c, d, M, theta): the E1a contour is exactly the hyperbola sigma*omega = c*(d), and it
is independent of eps. As c grows, Vcos, Vsin -> 1/2 and Lambda -> 1/(2(M-1)), so

    c^2/(2M) = d/(2(M-1))   =>   c* -> sqrt(d M/(M-1))

i.e. the exponent p in log c* = const + p log d tends to 1/2 BY CONSTRUCTION, because
the pathwise error variance is d-independent by construction and the zeroth-order one
is (d-1)*const + const. This is D1 of the plan and Rule A of the prereg.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def vcos(c, th):
    """Var(cos(theta + c t)), t ~ N(0,1)."""
    c, th = np.asarray(c, float), np.asarray(th, float)
    return 0.5 + 0.5 * np.exp(-2 * c * c) * np.cos(2 * th) - np.exp(-c * c) * np.cos(th) ** 2


def vsin(c, th):
    """Var(sin(theta + c t)), t ~ N(0,1)."""
    c, th = np.asarray(c, float), np.asarray(th, float)
    return 0.5 - 0.5 * np.exp(-2 * c * c) * np.cos(2 * th) - np.exp(-c * c) * np.sin(th) ** 2


def _moments(c, th):
    """Central moments of (a, b) = (sin(theta + c t) - E sin, t).

    All from Gaussian identities: E[t g(t)] = E[g'(t)] and E[t^2 g] = E[g] + E[g''].
    """
    c, th = np.asarray(c, float), np.asarray(th, float)
    e1, e2 = np.exp(-0.5 * c * c), np.exp(-2 * c * c)
    m_f = e1 * np.sin(th)                                   # E[f]
    m_ft = c * e1 * np.cos(th)                              # E[f t]
    m_f2 = 0.5 - 0.5 * e2 * np.cos(2 * th)                  # E[f^2]
    m_ft2 = (1.0 - c * c) * e1 * np.sin(th)                 # E[f t^2]
    m_f2t2 = 0.5 * (1.0 - (1.0 - 4 * c * c) * e2 * np.cos(2 * th))   # E[f^2 t^2]

    mu11 = m_ft                       # E[ab], since E[t] = 0
    mu20 = m_f2 - m_f * m_f           # Var(f) == vsin
    mu02 = np.ones_like(mu20)         # Var(t)
    mu22 = m_f2t2 - 2.0 * m_f * m_ft2 + m_f * m_f
    return mu11, mu20, mu02, mu22


def lam_along_v(c, th, M):
    """Lambda: the along-v error energy of the corrected ZO estimator. d-independent.

    Equals Var of the unbiased sample covariance of (f, t) over M draws.
    """
    mu11, mu20, mu02, mu22 = _moments(c, th)
    M = float(M)
    return (mu22 - mu11 ** 2) / M + (mu20 * mu02 - mu11 ** 2) / (M * (M - 1.0))


def var_pw_error_only(c, th, M):
    """Var[g_PW]_e in units of (eps/sigma)^2. Exactly d-independent."""
    c = np.asarray(c, float)
    return c * c * vcos(c, th) / float(M)


def var_zo_error_only(c, th, M, d):
    """Var[g_ZO]_e in units of (eps/sigma)^2, after the M/(M-1) correction."""
    return (d - 1.0) * vsin(c, th) / (float(M) - 1.0) + lam_along_v(c, th, M)


def log_ratio(c, th, M, d):
    """log Var_ZO_e - log Var_PW_e. Monotone decreasing in c; zero at the crossover."""
    return np.log(var_zo_error_only(c, th, M, d)) - np.log(var_pw_error_only(c, th, M))


def crossover_c_star(d, th, M, *, lo=1e-3, hi=1e4):
    """Solve log_ratio = 0 in log c, at fixed (d, theta, M)."""
    f = lambda lc: float(log_ratio(np.exp(lc), th, M, d))
    a, b = np.log(lo), np.log(hi)
    fa, fb = f(a), f(b)
    if not (fa > 0 > fb):
        raise ValueError(f"no interior bracket for d={d}, theta={th}: f={fa:.3g},{fb:.3g}")
    return float(np.exp(brentq(f, a, b, xtol=1e-13, rtol=1e-14)))


def c_star_asymptote(d, M):
    """The large-c limit sqrt(d M/(M-1)), i.e. exactly sqrt(d) up to the M correction."""
    return np.sqrt(d * float(M) / (float(M) - 1.0))


def mse_zo_smooth_linear(g_norm2, d, M):
    """MSE of the corrected ZO estimator on a LINEAR critic: ||g||^2 (d+1)/(M-1).

    This is the Nesterov-Spokoiny dimension factor, and it is what E0a asserts. The
    spec's "MSE_ZO/MSE_PW grows linearly in d" does NOT hold here, because MSE_PW is
    nonzero and itself grows with d through tr(H^2); that ratio tends to a constant.
    """
    return np.asarray(g_norm2) * (d + 1.0) / (float(M) - 1.0)


def mse_pw_smooth(tr_H2, sigma, M):
    """MSE of the pathwise estimator on the smooth part: 4 sigma^2 tr(H^2)/M.

    Exact, because grad_a Q^pi is affine in a: g_PW = g* - (2 sigma/M) H sum_i u_i.
    """
    return 4.0 * np.asarray(sigma) ** 2 * tr_H2 / float(M)
