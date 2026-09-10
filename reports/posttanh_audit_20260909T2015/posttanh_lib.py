"""Post-tanh action-distribution geometry.

AMENDMENT 1: two-panel Gauss-Legendre split at the tanh transition, replacing
Gauss-Hermite, which does not converge for tanh moments at large sigma.

With z = mu + sigma t, t ~ N(0,1), panels are [-T, c] and [c, T] where
c = clip(-mu/sigma, -T, T) is the point z = 0. Each panel uses N-node Gauss-Legendre
against the standard normal density. float64 throughout, fully deterministic.
"""
import numpy as np
from scipy.special import erf

T_TRUNC = 15.0
_GL = {}


def gl(n):
    if n not in _GL:
        x, w = np.polynomial.legendre.leggauss(n)
        _GL[n] = (x.astype(np.float64), w.astype(np.float64))
    return _GL[n]


def _phi(t):
    return np.exp(-0.5 * t * t) / np.sqrt(2.0 * np.pi)


def tanh_moments(mu, sg, n=128, T=T_TRUNC, chunk=200000):
    """m1 = E[tanh z], m2 = E[tanh^2 z], elementwise. Chunked over flattened elements."""
    mu = np.asarray(mu, np.float64); sg = np.asarray(sg, np.float64)
    shape = np.broadcast(mu, sg).shape
    muf = np.broadcast_to(mu, shape).reshape(-1)
    sgf = np.broadcast_to(sg, shape).reshape(-1)
    xg, wg = gl(n)
    m1 = np.empty(muf.size, np.float64); m2 = np.empty(muf.size, np.float64)
    for i in range(0, muf.size, chunk):
        m_, s_ = muf[i:i + chunk], sgf[i:i + chunk]
        c = np.clip(-m_ / s_, -T, T)
        a1 = np.zeros_like(m_); a2 = np.zeros_like(m_)
        for lo, hi in ((np.full_like(c, -T), c), (c, np.full_like(c, T))):
            half = (hi - lo) / 2.0; mid = (hi + lo) / 2.0
            t = mid[:, None] + half[:, None] * xg
            z = m_[:, None] + s_[:, None] * t
            tt = np.tanh(z)
            ww = wg * half[:, None] * _phi(t)
            a1 += np.sum(ww * tt, axis=1)
            a2 += np.sum(ww * tt * tt, axis=1)
        m1[i:i + chunk] = a1; m2[i:i + chunk] = a2
    return m1.reshape(shape), m2.reshape(shape)


def posttanh_var(mu, sg, n=128):
    """v = m2 - m1^2 per (state, dim). Returns v, n_clamped, max_clamp_magnitude."""
    m1, m2 = tanh_moments(mu, sg, n)
    v = m2 - m1 * m1
    neg = v < 0.0
    n_neg = int(neg.sum())
    max_neg = float(np.abs(v[neg]).max()) if n_neg else 0.0   # F1: was -v[neg].max(), the SMALLEST magnitude
    return np.where(neg, 0.0, v), n_neg, max_neg


def sat_prob(mu, sg, thresh=0.95):
    """Analytic P(|tanh z| > thresh) = P(z > c) + P(z < -c), c = atanh(thresh). Exact."""
    mu = np.asarray(mu, np.float64); sg = np.asarray(sg, np.float64)
    c = float(np.arctanh(thresh))
    Phi = lambda t: 0.5 * (1.0 + erf(t / np.sqrt(2.0)))
    return Phi((-c - mu) / sg) + 1.0 - Phi((c - mu) / sg)


def V(mu, sg, n=128):
    v, n_neg, max_neg = posttanh_var(mu, sg, n)
    return float(v.mean()), n_neg, max_neg


def S(mu, sg, thresh=0.95):
    return float(sat_prob(mu, sg, thresh).mean())


def decompose(muA, sgA, muB, sgB, F):
    """Symmetric two-factor OUTPUT-LEVEL decomposition on common states."""
    f_AA = F(muA, sgA); f_AB = F(muA, sgB); f_BA = F(muB, sgA); f_BB = F(muB, sgB)
    C_sigma = 0.5 * ((f_AA - f_AB) + (f_BA - f_BB))
    C_mu = 0.5 * ((f_AB - f_BB) + (f_AA - f_BA))
    total = f_AA - f_BB
    return dict(F_AA=f_AA, F_AB=f_AB, F_BA=f_BA, F_BB=f_BB,
                C_sigma=C_sigma, C_mu=C_mu, total=total,
                resid=total - (C_sigma + C_mu))


def paired_bootstrap(logratios, rng, nboot=100000, level=95.0, stat=np.mean):
    a = np.asarray(logratios, np.float64)
    idx = rng.integers(0, a.size, size=(nboot, a.size))
    b = stat(a[idx], axis=1)
    lo, hi = np.percentile(b, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(stat(a)), float(lo), float(hi)
