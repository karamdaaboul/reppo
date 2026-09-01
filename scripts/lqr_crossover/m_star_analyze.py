"""Tracks A and B: extract M*, fit the surface, and test whether ESS is blind.

    python scripts/lqr_crossover/m_star_analyze.py

Adjudicated against docs/prereg_m_star.md as committed. Sec. 2 of that document records
that b ~ 1 for the centred ZO arm is DETERMINED by E0a and is not a finding; every report
line that touches it says so.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import scripts.lqr_crossover  # noqa: F401,E402

import numpy as np  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from scripts.lqr_crossover import OUT  # noqa: E402
from scripts.lqr_crossover.m_star import DS, EPS_ES, MS, TAUS  # noqa: E402

HEADLINE_TAU = 0.95            # registered
REGIMES = ("low", "high")


def load(d, regime):
    p = os.path.join(OUT, f"mstar_d{d}_{regime}.npz")
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def m_star(cos_by_M, tau):
    """Smallest M reaching tau, interpolated on log M. Returns (M*, censored)."""
    c = np.asarray(cos_by_M, float)
    lm = np.log(np.asarray(MS, float))
    hit = np.where(c >= tau)[0]
    if len(hit) == 0:
        return np.nan, True                      # right-censored at M = 2048
    i = hit[0]
    if i == 0:
        return float(MS[0]), False               # already there at the smallest M
    c0, c1 = c[i - 1], c[i]
    if c1 <= c0:
        return float(MS[i]), False
    t = (tau - c0) / (c1 - c0)
    return float(np.exp(lm[i - 1] + t * (lm[i] - lm[i - 1]))), False


def collect(regime, tau):
    """M* tables. Returns (es[d, eps], es_cens, zo[d], zo_cens, pw[d])."""
    es = np.full((len(DS), len(EPS_ES)), np.nan)
    esc = np.zeros_like(es, bool)
    zo = np.full(len(DS), np.nan); zoc = np.zeros(len(DS), bool)
    pw = np.full(len(DS), np.nan)
    for i, d in enumerate(DS):
        z = load(d, regime)
        if z is None:
            continue
        zo[i], zoc[i] = m_star(z["cos_zo"].mean(-1), tau)
        pw[i], _ = m_star(z["cos_pw"].mean(-1), tau)
        for j in range(len(EPS_ES)):
            es[i, j], esc[i, j] = m_star(z["cos_es"][:, j, :].mean(-1), tau)
    return es, esc, zo, zoc, pw


def fit_surface(es, esc):
    """log M* = a*eps_E + b*log d + c, unweighted OLS, censored cells excluded."""
    X, y = [], []
    for i, d in enumerate(DS):
        for j, e in enumerate(EPS_ES):
            if esc[i, j] or not np.isfinite(es[i, j]):
                continue
            X.append([e, np.log(d), 1.0]); y.append(np.log(es[i, j]))
    if len(y) < 4:
        return None
    X, y = np.array(X), np.array(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ss = 1.0 - resid.var() / max(y.var(), 1e-300)
    return dict(a=float(coef[0]), b=float(coef[1]), c=float(coef[2]),
                r2=float(ss), n=len(y), resid_sd=float(resid.std(ddof=1)))


def main():
    W = print
    W("=" * 78)
    W("TRACK A -- the M* surface")
    W("=" * 78)
    for regime in REGIMES:
        z0 = load(DS[0], regime)
        if z0 is None:
            W(f"\n[{regime}] no data"); continue
        W(f"\n### regime {regime}  (c = sigma*omega = {float(z0['c']):.2f})")
        for tau in TAUS:
            es, esc, zo, zoc, pw = collect(regime, tau)
            head = " HEADLINE" if tau == HEADLINE_TAU else ""
            W(f"\n-- tau = {tau}{head}")
            W(f"  {'d':>4} {'PW':>9} {'ZO':>9} | " +
              " ".join(f"eps={e:<5}" for e in EPS_ES))
            for i, d in enumerate(DS):
                row = f"  {d:4d} {pw[i]:9.1f} " + \
                      (f"{zo[i]:9.1f}" if not zoc[i] else f"{'>2048':>9}") + " | "
                row += " ".join((f"{es[i,j]:9.1f}" if not esc[i, j]
                                 else f"{'>2048':>9}") for j in range(len(EPS_ES)))
                W(row)
            nc = int(esc.sum())
            W(f"  censored ESTEP cells: {nc}/{esc.size}")
            if tau == HEADLINE_TAU:
                f = fit_surface(es, esc)
                if f:
                    W(f"\n  FIT log M* = a*eps_E + b*log d + c   (unweighted OLS, "
                      f"n={f['n']}, R2={f['r2']:.4f}, resid sd={f['resid_sd']:.4f})")
                    W(f"    a = {f['a']:+.4f}   (CD null band [-0.2, 0.2]; "
                      f"CD supported band [0.7, 1.3])")
                    W(f"    b = {f['b']:+.4f}   (dimension band [0.7, 1.3])")
                # monotonicity on eps_E, registered guard against a cancelling `a`
                W("\n  eps_E monotonicity (prereg Sec. 3):")
                nonmono = 0
                for i, d in enumerate(DS):
                    v = es[i]; m = np.isfinite(v) & ~esc[i]
                    if m.sum() < 3:
                        continue
                    rho = spearmanr(np.array(EPS_ES)[m], v[m]).statistic
                    amin = int(np.nanargmin(np.where(m, v, np.inf)))
                    interior = 0 < amin < len(EPS_ES) - 1
                    nonmono += interior
                    W(f"    d={d:3d}: spearman(M*, eps_E) = {rho:+.3f}, "
                      f"argmin at eps_E={EPS_ES[amin]}"
                      f"{'  INTERIOR -> non-monotone' if interior else ''}")
                W(f"    -> non-monotone at {nonmono}/{len(DS)} dimensions")
                # ratio ESTEP/ZO at production eps_E = 0.5 (Q1)
                j5 = EPS_ES.index(0.5)
                W("\n  Q1  M*_ESTEP / M*_ZO at eps_E = 0.5 (does the gap grow with d?):")
                for i, d in enumerate(DS):
                    if esc[i, j5] or zoc[i] or not np.isfinite(es[i, j5] * zo[i]):
                        W(f"    d={d:3d}: censored"); continue
                    W(f"    d={d:3d}: {es[i,j5]/zo[i]:6.2f}x   "
                      f"(ESTEP {es[i,j5]:7.1f} vs ZO {zo[i]:7.1f})")
                # linearity of log M* in log d, registered check
                m = np.isfinite(es[:, j5]) & ~esc[:, j5]
                if m.sum() >= 4:
                    dd, vv = np.log(np.array(DS)[m]), np.log(es[m, j5])
                    p1 = np.polyfit(dd, vv, 1)
                    r1 = vv - np.polyval(p1, dd)
                    W(f"\n  linearity of log M* in log d at eps_E=0.5: slope "
                      f"{p1[0]:+.4f}, resid sd {r1.std(ddof=1):.4f}")
                    W("    (for the ZO arm a slope of ~1 is DETERMINED by E0a, "
                      "prereg Sec. 2 -- not a finding)")

    W("\n" + "=" * 78)
    W("TRACK B -- is ESS blind to the deficit?")
    W("=" * 78)
    rows = {r: [] for r in REGIMES}
    for regime in REGIMES:
        for d in DS:
            z = load(d, regime)
            if z is None:
                continue
            for im, M in enumerate(MS):
                for ie, e in enumerate(EPS_ES):
                    rows[regime].append(dict(
                        d=d, M=M, eps=e,
                        deficit=float(z["cos_pw"][im].mean() - z["cos_es"][im, ie].mean()),
                        ess=float(z["ess"][im, ie].mean()),
                        ess_M=float(z["ess"][im, ie].mean() / M),
                        ess_d=float(z["ess"][im, ie].mean() / d),
                        kl=float(z["kl_unif"][im, ie].mean()),
                        logit_sd=float(z["logit_sd"][im, ie].mean())))
    diag = ("ess", "ess_M", "ess_d", "kl", "logit_sd")
    for regime in REGIMES:
        R = rows[regime]
        if not R:
            continue
        dfc = np.array([r["deficit"] for r in R])
        W(f"\n### regime {regime}: {len(R)} cells, deficit range "
          f"[{dfc.min():.4f}, {dfc.max():.4f}] (factor "
          f"{dfc.max()/max(dfc.min(),1e-9):.1f})")
        W(f"  {'diagnostic':>10} {'pearson':>9} {'spearman':>9} "
          f"{'range in worst-deficit decile':>32}")
        k = max(1, len(R) // 10)
        worst = np.argsort(-dfc)[:k]
        for nm in diag:
            v = np.array([r[nm] for r in R])
            pr = pearsonr(v, dfc).statistic; sr = spearmanr(v, dfc).statistic
            w = v[worst]
            W(f"  {nm:>10} {pr:+9.3f} {sr:+9.3f} "
              f"{f'[{w.min():.3f}, {w.max():.3f}]  ({w.max()/max(w.min(),1e-9):.2f}x)':>32}")

    # registered out-of-sample test: fit on low, evaluate on high
    W("\n  Registered OUT-OF-SAMPLE test (fit on LOW regime, evaluate on HIGH):")
    if rows["low"] and rows["high"]:
        ylo = np.array([r["deficit"] for r in rows["low"]])
        yhi = np.array([r["deficit"] for r in rows["high"]])
        W(f"    {'diagnostic':>10} {'in-sample R2':>13} {'OUT-OF-SAMPLE R2':>18}")
        for nm in diag:
            xlo = np.log(np.maximum([r[nm] for r in rows["low"]], 1e-12))
            xhi = np.log(np.maximum([r[nm] for r in rows["high"]], 1e-12))
            A = np.c_[xlo, np.ones_like(xlo)]
            co, *_ = np.linalg.lstsq(A, ylo, rcond=None)
            r2in = 1 - ((ylo - A @ co) ** 2).sum() / max(((ylo - ylo.mean()) ** 2).sum(), 1e-300)
            pred = np.c_[xhi, np.ones_like(xhi)] @ co
            r2out = 1 - ((yhi - pred) ** 2).sum() / max(((yhi - yhi.mean()) ** 2).sum(), 1e-300)
            W(f"    {nm:>10} {r2in:13.3f} {r2out:18.3f}")
        W("    (a diagnostic that only separates in-sample, or only in one regime,")
        W("     is reported as FAILING -- prereg Sec. 5)")


if __name__ == "__main__":
    main()
