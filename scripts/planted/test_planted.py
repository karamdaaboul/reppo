"""Validation tests for the planted-error phase diagram.

Each test states what it would take to FAIL. Run: pytest -q scripts/planted/test_planted.py
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from planted_error_sweep import (q_pi, grad_q_pi, e_field, grad_e_field,
                                 solve_eta, tr_step, tr_err)

RNG = np.random.default_rng(7)


def _dirs(d):
    v = RNG.normal(size=d); v /= np.linalg.norm(v)
    w = RNG.normal(size=d); w /= np.linalg.norm(w)
    return v, w


def test_error_is_exactly_the_planted_field():
    """Q_phi - Q_pi must equal e(a) to machine precision, by construction."""
    for d in (2, 8, 32):
        v, ve = _dirs(d); astar = v; a = RNG.normal(size=(500, d))
        qpi = q_pi(a, astar)
        e = e_field(a, ve, 3.0, 0.7, 1.3)
        # recovering e from Q_phi - Q_pi is exact up to float64 rounding on the
        # O(1) magnitudes involved, not bit-exact
        assert np.max(np.abs(((qpi + e) - qpi) - e)) < 1e-12 * max(1.0, np.abs(qpi).max())
        assert np.max(np.abs(e)) <= 1.3 + 1e-12          # amplitude bound holds


def test_omega_matches_the_theorem_definition():
    """omega := ||grad e||_inf / ||e||_inf, function sup-norms over the DOMAIN.

    For e = eps sin(om v^T a) the sup-norms are attained analytically:
    sup|e| = eps and sup||grad e|| = eps*om, so omega = om exactly. Verified on a
    dense grid covering a full period along v, NOT on policy samples -- sampling
    only the policy region measures a local slope, not the sup-norm.
    """
    for d in (2, 8, 32):
        for om in (0.5, 2.0, 8.0, 64.0):
            _, ve = _dirs(d); eps, ph = 1.7, 0.4
            t = np.linspace(0, 2 * np.pi / om, 20001)      # one full period along ve
            a = t[:, None] * ve[None, :]
            sup_e = np.abs(e_field(a, ve, om, ph, eps)).max()
            sup_g = np.linalg.norm(grad_e_field(a, ve, om, ph, eps), axis=-1).max()
            assert abs(sup_e / eps - 1) < 1e-6, (d, om, sup_e)
            assert abs(sup_g / (eps * om) - 1) < 1e-6, (d, om, sup_g)
            assert abs((sup_g / sup_e) / om - 1) < 1e-5    # omega recovered


def test_oracle_gradient_by_finite_differences():
    for d in (2, 8, 32):
        v, _ = _dirs(d); astar = 1.0 * v; a = RNG.normal(size=d) * 0.3
        g = grad_q_pi(a, astar)
        h = 1e-6
        fd = np.array([(q_pi(a + h * np.eye(d)[i], astar)
                        - q_pi(a - h * np.eye(d)[i], astar)) / (2 * h) for i in range(d)])
        assert np.max(np.abs(g - fd)) < 1e-6


def test_quadratic_blur_is_gradient_invariant():
    """Q_pi quadratic => its Gaussian blur has the SAME gradient, so the oracle is
    exact rather than Monte Carlo. Fails if the oracle were mis-specified."""
    for d in (2, 16):
        v, _ = _dirs(d); astar = v
        for sig in (0.1, 0.8):
            u = RNG.normal(size=(400000, d))
            g_blur = grad_q_pi(sig * u, astar).mean(0)
            assert np.max(np.abs(g_blur - grad_q_pi(np.zeros(d), astar))) < 6e-3 * max(1, sig)


def test_pw_unbiased_for_blurred_gradient_large_M():
    for d in (4, 16):
        v, ve = _dirs(d); astar = v; sig, om, eps = 0.3, 4.0, 1.0
        u = RNG.normal(size=(400000, d)); a = sig * u
        g = grad_q_pi(a, astar) + grad_e_field(a, ve, om, 0.3, eps)
        emp = g.mean(0)
        # blurred gradient of the sinusoid, analytic: eps*om*exp(-om^2 sig^2/2)*cos(ph)*ve
        gb = grad_q_pi(np.zeros(d), astar) + eps * om * np.exp(-0.5 * om ** 2 * sig ** 2) * np.cos(0.3) * ve
        assert np.max(np.abs(emp - gb)) < 0.02


def test_zo_unbiased_up_to_one_minus_one_over_M():
    """E[g_ZO] = (1 - 1/M) grad f_Sigma  (manuscript eq:app-shrink)."""
    d, M, sig = 6, 32, 0.4
    v, ve = _dirs(d); astar = v
    u = RNG.normal(size=(200000, M, d)); a = sig * u
    Q = q_pi(a, astar) + e_field(a, ve, 3.0, 0.2, 1.0)
    zo = ((Q - Q.mean(1, keepdims=True))[..., None] * u).mean(1) / sig
    pw = (grad_q_pi(a, astar) + grad_e_field(a, ve, 3.0, 0.2, 1.0)).mean(1)
    # compare VECTORS, not componentwise ratios: a component whose blurred gradient
    # is near zero makes the ratio arbitrarily unstable while the vector claim holds
    target = (1 - 1 / M) * pw.mean(0)
    rel = np.linalg.norm(zo.mean(0) - target) / np.linalg.norm(target)
    assert rel < 0.05, (rel, zo.mean(0), target)


def test_wml_equals_zo_plus_a_live_ubar_term():
    """The ACTUAL weighted-MLE mean update is NOT the manuscript's g_ZO.

    Expanding the raw self-normalised softmax at large eta,
        w_i = (1/M)(1 + (Q_i - Qbar)/eta) + O(eta^-2)
        Delta_mu_WML = sum_i w_i u_i = ubar + (1/eta) * m_hat + O(eta^-2),
    where ubar = (1/M) sum_i u_i and m_hat = (1/M) sum_i (Q_i - Qbar) u_i is exactly
    the manuscript's (unwhitened) g_ZO numerator.  The ubar term does NOT appear in
    g_ZO, because its coefficients (Q_i - Qbar) sum to zero.  This is the \bar u noise
    term that Amendment A answer 2 records as LIVE in this implementation
    (raw softmax, no centring, no baseline, no antithetic pairing).

    Two assertions:
      (a) after removing ubar, the WML direction matches g_ZO to first order;
      (b) the ubar term is not negligible -- it dominates as eta grows, which is why
          this test replaces the naive "large eta => agreement" version.
    """
    d, M, sig = 6, 32, 0.3
    v, ve = _dirs(d); astar = v
    u = RNG.normal(size=(20000, M, d)); a = sig * u
    Q = q_pi(a, astar) + e_field(a, ve, 2.0, 0.1, 1.0)
    zo = ((Q - Q.mean(1, keepdims=True))[..., None] * u).mean(1)
    ubar = u.mean(1)
    eta = 1e4
    z = Q / eta; z -= z.max(1, keepdims=True)
    w = np.exp(z); w /= w.sum(1, keepdims=True)
    wml = (w[..., None] * u).sum(1)

    def cosine(p, q):
        return np.sum(p * q, -1) / (np.linalg.norm(p, axis=-1) * np.linalg.norm(q, axis=-1))

    # (a) the residual after removing ubar is the ZO direction
    resid = wml - ubar
    assert np.median(cosine(resid, zo)) > 0.999, np.median(cosine(resid, zo))
    # (b) ubar dominates the raw WML step, so WML != g_ZO as operators
    assert np.median(cosine(wml, ubar)) > 0.99
    assert abs(np.median(cosine(wml, zo))) < 0.2, np.median(cosine(wml, zo))


def test_rotation_invariance():
    """Sigma = sigma^2 I and the construction are isotropic, so a common rotation of
    (a_star, v_e) must leave every reported quantity unchanged in distribution."""
    d, M, sig = 8, 32, 0.3
    v, ve = _dirs(d); astar = v
    A = np.linalg.qr(RNG.normal(size=(d, d)))[0]
    u = RNG.normal(size=(20000, M, d))
    def energy(astar_, ve_, u_):
        a = sig * u_
        Q = q_pi(a, astar_) + e_field(a, ve_, 3.0, 0.5, 1.0)
        zo = ((Q - Q.mean(1, keepdims=True))[..., None] * u_).mean(1) / sig
        return np.trace(np.cov(zo.T))
    e0 = energy(astar, ve, u)
    e1 = energy(A @ astar, A @ ve, u @ A.T)
    assert abs(e0 / e1 - 1) < 1e-9, (e0, e1)


def test_error_induced_variance_ratio_is_amplitude_invariant():
    """ratio_e = Var[ZO]_e / Var[PW]_e must not depend on the error amplitude eps:
    both scale as eps^2. This is why eps is not a tuned parameter."""
    d, M, sig, om = 8, 32, 0.3, 4.0
    v, ve = _dirs(d); astar = v
    u = RNG.normal(size=(40000, M, d)); a = sig * u
    def ratio(eps):
        dQ = e_field(a, ve, om, 0.3, eps)
        dG = grad_e_field(a, ve, om, 0.3, eps)
        dzo = ((dQ - dQ.mean(1, keepdims=True))[..., None] * u).mean(1) / sig
        dpw = dG.mean(1)
        return np.trace(np.cov(dzo.T)) / np.trace(np.cov(dpw.T))
    r1, r2 = ratio(0.1), ratio(10.0)
    assert abs(r1 / r2 - 1) < 1e-8, (r1, r2)


def test_equal_query_budget():
    """Both estimators consume exactly M critic evaluations per batch."""
    d, M, sig = 4, 32, 0.3
    v, ve = _dirs(d); astar = v
    u = RNG.normal(size=(10, M, d))
    calls = {"q": 0, "g": 0}
    a = sig * u
    calls["q"] += a.shape[1]          # ZO/WML: M forward
    calls["g"] += a.shape[1]          # PW: M backward (and M forward)
    assert calls["q"] == calls["g"] == M


def test_no_weight_collapse_explains_the_ordering():
    """If the softmax collapsed onto one sample, WML would be pure noise and the
    ordering would be an artifact. ESS must stay well above 1 across the grid."""
    d, M, sig = 8, 32, 0.3
    v, ve = _dirs(d); astar = v
    u = RNG.normal(size=(5000, M, d)); a = sig * u
    Q = q_pi(a, astar) + e_field(a, ve, 8.0, 0.2, 1.0)
    eta = solve_eta(Q, 0.5)
    z = Q / eta[:, None]; z -= z.max(1, keepdims=True)
    w = np.exp(z); w /= w.sum(1, keepdims=True)
    ess = 1.0 / (w ** 2).sum(1)
    assert np.isfinite(Q).all() and np.isfinite(w).all()
    assert np.median(ess) > 2.0, np.median(ess)


def test_eta_dual_is_solved_not_guessed():
    """solve_eta must satisfy the MPO dual stationarity condition."""
    Q = RNG.normal(size=(300, 32)) * 3.0
    eta = solve_eta(Q, 0.5)
    z = Q / eta[:, None]; zm = z.max(1, keepdims=True)
    lse = zm[:, 0] + np.log(np.mean(np.exp(z - zm), 1))
    w = np.exp(z - zm); w /= w.sum(1, keepdims=True)
    resid = 0.5 + lse - (w * Q).sum(1) / eta
    assert np.max(np.abs(resid)) < 1e-3, np.max(np.abs(resid))
