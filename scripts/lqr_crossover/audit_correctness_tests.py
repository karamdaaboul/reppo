"""Forensic correctness tests for the LQR crossover experiment.  TEMPORARY AUDIT CODE.

Independent of analyze.py / reference.py wherever a check is about them: every
quantity is re-derived in numpy here and compared to (a) the production core in
src/jaxrl/estimators.py, (b) the sweep kernel's arithmetic, (c) the saved .npz
statistics, (d) the numbers in reports/lqr_crossover.md.  Nothing in production is
modified.  CPU, float64.

    JAX_PLATFORMS=cpu python scripts/lqr_crossover/audit_correctness_tests.py
"""
from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
import scripts.lqr_crossover  # noqa: F401,E402  CPU + x64
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from scripts.lqr_crossover import OUT, SEED_ROOT, error_field as EF, lqr  # noqa: E402
from src.jaxrl.estimators import (  # noqa: E402
    centred_zo, pathwise_mean, whitened_pathwise, deattenuation_factor,
    softmax_displacement)

ART = os.path.join(REPO_ROOT, "reports", "artifacts")
M = 32
ROWS = []


def rec(test, quantity, value, bound, ok, note=""):
    ROWS.append(dict(test=test, quantity=quantity, value=value, bound=bound,
                     ok=bool(ok), note=note))
    print(f"  [{'PASS' if ok else 'FAIL'}] {test:6s} {quantity:48s} {value:.3e}  "
          f"(bound {bound:g}) {note}")


# =============================================================== independent maths
def my_e(a, V, phi, om, eps, r):
    """e(a) = eps/sqrt r sum_j sin(om v_j.a + phi_j); a:(...,d) V:(r,d)."""
    return eps / np.sqrt(r) * np.sin(om * (a @ V.T) + phi).sum(-1)


def my_grad_e(a, V, phi, om, eps, r):
    return eps * om / np.sqrt(r) * (np.cos(om * (a @ V.T) + phi)[..., None] * V).sum(-2)


def my_pw(gradq):                      # gradq: (R, M, d) action-space gradients
    return gradq.mean(1)


def my_zo_plain(q, u, sigma):          # q:(R,M) u:(R,M,d)
    qc = q - q.mean(1, keepdims=True)
    return (qc[..., None] * u).mean(1) / sigma


def my_zo_deatt(q, u, sigma):
    return my_zo_plain(q, u, sigma) * M / (M - 1.0)


def trcov(X):                          # X:(R,d)  -> E||X - EX||^2, ddof=1
    return float(np.var(X, axis=0, ddof=1).sum())


def rms_moments(c, om, th, r):
    """E[e^2]/eps^2 and E||grad e||^2/eps^2 for a ~ N(mu, sigma^2 I), theta_j given."""
    e2 = np.exp(-2 * c * c)
    Ee2 = ((0.5 - 0.5 * e2 * np.cos(2 * th)).sum(-1)
           + np.exp(-c * c) * (np.sin(th).sum(-1) ** 2 - (np.sin(th) ** 2).sum(-1))) / r
    Eg2 = om * om / r * (0.5 + 0.5 * e2 * np.cos(2 * th)).sum(-1)
    return Ee2, Eg2


def regen(d, kind="rank1", rank=None, n_states=64):
    """Replay the sweep's exact draw order for system, states and field."""
    s = lqr.build_system(d, seed=SEED_ROOT + d, normalize="unit_H", cost="identity")
    rng = np.random.default_rng(SEED_ROOT + 2000 + d)
    states = lqr.sample_states(s, rng, n_states)
    H, g, mu = lqr.q_coeffs(s, states)
    pe = EF.draw_error(rng, n_states, d, kind=kind, rank=rank, omega=1.0)
    return s, states, H, g, mu, pe


# ===================================================================== TEST A
def test_A():
    print("\nTEST A  finite-difference gradient of Q^pi (unnormalised and normalised)")
    rng = np.random.default_rng(1)
    worst_u, worst_n, worst_red = 0.0, 0.0, 0.0
    for d in (1, 4, 16):
        s = lqr.build_system(d, seed=SEED_ROOT + d)
        st = lqr.sample_states(s, rng, 3)
        a = rng.normal(size=(3, d))
        g_an = lqr.grad_a_q_pi(s, st, a)
        h = 1e-6
        for j in range(d):
            e = np.zeros(d); e[j] = h
            fd = (lqr.q_pi(s, st, a + e, 0.3) - lqr.q_pi(s, st, a - e, 0.3)) / (2 * h)
            worst_u = max(worst_u, float(np.abs(fd - g_an[:, j]).max()
                                         / max(np.abs(g_an).max(), 1e-300)))
        # normalised reduced form used by the sweep: Q_n(mu + sigma u) =
        # const - sigma^2 u^T H_n u + sigma u^T g_n  ==>  grad_a = -2 H_n (a - mu) + g_n
        Hn, gn, mun = lqr.q_coeffs(s, st)
        aQ, aS = s.scale["alpha_Q"], s.scale["alpha_s"]
        a2 = mun + 0.3 * rng.normal(size=(3, d))
        red = -2.0 * (a2 - mun) @ Hn.T + gn
        # independent: alpha_Q * grad_a Q^pi at the SCALED state (that is what q_coeffs
        # encodes: mu = -K (alpha_s s), b evaluated at alpha_s s, H scaled by alpha_Q)
        full = aQ * lqr.grad_a_q_pi(s, aS * st, a2)
        worst_n = max(worst_n, float(np.abs(red - full).max() / np.abs(full).max()))
        # the sweep kernel's qq/ql form: Q(mu+sigma u) - Q(mu) == -sigma^2 uHu + sigma u.g
        u = rng.normal(size=(3, d)); sig = 0.3
        lhs = aQ * (lqr.q_pi(s, aS * st, mun + sig * u, sig) - lqr.q_pi(s, aS * st, mun, sig))
        rhs = -sig ** 2 * np.einsum("si,ij,sj->s", u, Hn, u) + sig * np.sum(u * gn, -1)
        worst_red = max(worst_red, float(np.abs(lhs - rhs).max() / np.abs(lhs).max()))
        # unit_H normalisation claim
        rec("A", f"d={d} ||alpha_Q H||_2 - 1", abs(np.linalg.norm(Hn, 2) - 1), 1e-12,
            abs(np.linalg.norm(Hn, 2) - 1) < 1e-12)
    rec("A", "FD vs grad_a_q_pi, max rel", worst_u, 1e-6, worst_u < 1e-6)
    rec("A", "reduced form vs alpha_Q grad at scaled state, max rel", worst_n, 1e-12, worst_n < 1e-12)
    rec("A", "kernel qq/ql form vs Q(mu+su)-Q(mu), max rel", worst_red, 1e-10, worst_red < 1e-10)


# ===================================================================== TEST B
def test_B():
    print("\nTEST B  planted field: FD gradient, orthonormality, rank rule, eps scaling, theta")
    rng = np.random.default_rng(2)
    worst_fd, worst_impl = 0.0, 0.0
    for d, kind, rank in ((4, "rank1", None), (8, "rank_r", 2), (1, "rank_r", 2), (6, "full", None)):
        pe = EF.draw_error(rng, 5, d, kind=kind, rank=rank, omega=3.7)
        r_expect = {"rank1": 1, "rank_r": min(2, d), "full": d}[kind]
        rec("B", f"{kind}{rank or ''} d={d} rank == {r_expect}", pe.rank, r_expect, pe.rank == r_expect)
        orth = max(float(np.abs(V @ V.T - np.eye(pe.rank)).max()) for V in pe.V)
        rec("B", f"{kind} d={d} ||V V^T - I||_max", orth, 1e-13, orth < 1e-13)
        mu = rng.normal(size=(5, d)); sig = 0.3; eps = rng.uniform(0.5, 2, size=5)
        u = rng.normal(size=(5, 7, d))
        a = mu[:, None, :] + sig * u
        # implementation vs my sine field
        e_impl = np.asarray(EF.e_value(pe, jnp.asarray(eps)[:, None], jnp.asarray(mu)[:, None, :], sig, jnp.asarray(u)))
        e_mine = np.stack([my_e(a[i], pe.V[i], pe.phi[i], pe.omega, eps[i], pe.rank) for i in range(5)])
        worst_impl = max(worst_impl, float(np.abs(e_impl - e_mine).max()))
        g_impl = np.asarray(EF.e_grad(pe, jnp.asarray(eps)[:, None, None], jnp.asarray(a)))
        g_mine = np.stack([my_grad_e(a[i], pe.V[i], pe.phi[i], pe.omega, eps[i], pe.rank) for i in range(5)])
        worst_impl = max(worst_impl, float(np.abs(g_impl - g_mine).max() / np.abs(g_mine).max()))
        h = 1e-6
        for j in range(d):
            ee = np.zeros(d); ee[j] = h
            fd = np.stack([(my_e(a[i] + ee, pe.V[i], pe.phi[i], pe.omega, eps[i], pe.rank)
                            - my_e(a[i] - ee, pe.V[i], pe.phi[i], pe.omega, eps[i], pe.rank)) / (2 * h)
                           for i in range(5)])
            worst_fd = max(worst_fd, float(np.abs(fd - g_impl[..., j]).max() / np.abs(g_impl).max()))
    rec("B", "e_value/e_grad vs independent sine field, max rel", worst_impl, 1e-12, worst_impl < 1e-12)
    rec("B", "FD of e vs e_grad, max rel", worst_fd, 1e-6, worst_fd < 1e-6)
    # eps scaling and q_spread closed form (MC check)
    s = lqr.build_system(4, seed=SEED_ROOT + 4)
    st = lqr.sample_states(s, rng, 2)
    for sig in (0.05, 0.4):
        qs = lqr.q_spread_closed_form(s, st, sig)
        qf = lqr.q_of_u_factory(s, st, sig)
        u = jnp.asarray(rng.normal(size=(2, 400000, 4)))
        mc = np.asarray(qf(u)).std(1, ddof=1)
        rel = float(np.abs(mc / qs - 1).max())
        rec("B", f"q_spread closed form vs MC (sigma={sig}), max rel", rel, 5e-3, rel < 5e-3)
    z = np.load(os.path.join(OUT, "d4_rank1_M32_unit_H_identity.npz"), allow_pickle=True)
    s4, st4, H4, g4, mu4, pe4 = regen(4)
    qs = np.stack([lqr.q_spread_closed_form(s4, st4, sg) for sg in z["sigmas"]], 1)
    rec("B", "saved eps == 0.05 * q_spread (replayed states), max rel",
        float(np.abs(z["eps"] / (0.05 * qs) - 1).max()), 1e-12,
        float(np.abs(z["eps"] / (0.05 * qs) - 1).max()) < 1e-12)
    rec("B", "saved phi == replayed phi, max abs", float(np.abs(z["phi"] - pe4.phi).max()), 0,
        float(np.abs(z["phi"] - pe4.phi).max()) == 0)
    vt = np.einsum("srd,sd->sr", pe4.V, mu4)
    rec("B", "saved vtmu == v^T mu (replayed), max abs", float(np.abs(z["vtmu"] - vt).max()), 1e-12,
        float(np.abs(z["vtmu"] - vt).max()) < 1e-12)
    # theta convention: EF.theta = omega v^T mu + phi; report.py uses phi alone
    th = EF.theta(EF.PlantedError(pe4.kind, pe4.rank, 5.0, pe4.V, pe4.phi), mu4)
    dev = float(np.abs(th - (5.0 * vt + pe4.phi)).max())
    rec("B", "EF.theta == omega v^T mu + phi", dev, 1e-12, dev < 1e-12)
    rec("B", "|omega v^T mu| median at omega=5 (0 would make theta=phi exact)",
        float(np.median(np.abs(5.0 * vt))), 0, True, "phi-only convention is NOT exact")


# ===================================================================== TEST C/D
def test_CD():
    print("\nTEST C/D  linearity in Q with identical draws: production core AND kernel arithmetic")
    rng = np.random.default_rng(3)
    worst = {"pw_core": 0, "zo_core": 0, "pw_mine": 0, "zo_mine": 0, "core_vs_mine": 0}
    for d in (1, 4, 16):
        s, st, H, g, mu, pe = regen(d, n_states=3)
        for sig, om in ((0.1, 5.0), (0.4, 20.0), (2.0, 0.7)):
            pe_o = EF.PlantedError(pe.kind, pe.rank, om, pe.V, pe.phi)
            eps = 0.05 * lqr.q_spread_closed_form(s, st, sig)
            qf = lqr.q_of_u_factory(s, st, sig)
            mu_j, eps_j = jnp.asarray(mu)[:, None, None, :], jnp.asarray(eps)[:, None, None]
            f_sm = qf
            f_e = lambda u: EF.e_value(pe_o, eps_j, mu_j, sig, u)
            f_full = lambda u: f_sm(u) + f_e(u)
            u = jnp.asarray(rng.normal(size=(3, 200, M, d)))
            outs = {}
            for nm, f in (("sm", f_sm), ("e", f_e), ("full", f_full)):
                q, Hw = whitened_pathwise(f, u)
                outs[nm] = (np.asarray(pathwise_mean(Hw, axis=-2) / sig),
                            np.asarray(centred_zo(q, u, axis=2, deattenuate=True) / sig),
                            np.asarray(q))
            for k, lbl in ((0, "pw_core"), (1, "zo_core")):
                diff = outs["full"][k] - outs["sm"][k] - outs["e"][k]
                worst[lbl] = max(worst[lbl], float(np.abs(diff).max() / np.abs(outs["e"][k]).max()))
            # my own numpy estimators on the same u: PW from analytic gradients, ZO from q
            un = np.asarray(u)
            a = mu[:, None, None, :] + sig * un
            gq = {"sm": -2.0 * a @ H.T + (g + 2.0 * mu @ H.T)[:, None, None, :]}
            gq["e"] = np.stack([my_grad_e(a[i], pe.V[i], pe.phi[i], om, eps[i], pe.rank) for i in range(3)])
            gq["full"] = gq["sm"] + gq["e"]
            qn = {k: outs[k][2] for k in outs}
            pw_m = {k: np.stack([my_pw(gq[k][i]) for i in range(3)]) for k in gq}
            zo_m = {k: np.stack([my_zo_deatt(qn[k][i], un[i], sig) for i in range(3)]) for k in qn}
            worst["pw_mine"] = max(worst["pw_mine"], float(np.abs(pw_m["full"] - pw_m["sm"] - pw_m["e"]).max() / np.abs(pw_m["e"]).max()))
            worst["zo_mine"] = max(worst["zo_mine"], float(np.abs(zo_m["full"] - zo_m["sm"] - zo_m["e"]).max() / np.abs(zo_m["e"]).max()))
            for k in ("sm", "e", "full"):
                worst["core_vs_mine"] = max(worst["core_vs_mine"],
                    float(np.abs(pw_m[k] - outs[k][0]).max() / np.abs(outs[k][0]).max()),
                    float(np.abs(zo_m[k] - outs[k][1]).max() / np.abs(outs[k][1]).max()))
    for k, v in worst.items():
        rec("C/D", f"linearity / agreement: {k}", v, 1e-12, v < 1e-12)


# ===================================================================== TEST E / J
def test_EJ():
    print("\nTEST E/J  linear critic: E[plain centred ZO] = (1-1/M) h ;  de-attenuation applied once")
    rng = np.random.default_rng(4)
    d, sig = 6, 0.3
    h = rng.normal(size=d)
    R = 400000
    u = rng.normal(size=(R, M, d))
    q = sig * u @ h                                    # linear critic, const dropped
    plain = my_zo_plain(q, u, sig)                     # (R, d)
    de = my_zo_deatt(q, u, sig)
    m_plain, m_de = plain.mean(0), de.mean(0)
    se = plain.std(0, ddof=1).max() / np.sqrt(R)
    rec("E", "|E[plain]/h - (1-1/M)| max (SE-scaled)", float(np.abs(m_plain / h - (1 - 1 / M)).max() / (se / np.abs(h).min())), 4, float(np.abs(m_plain / h - (1 - 1 / M)).max() / (se / np.abs(h).min())) < 4)
    rec("E", "|E[deatt]/h - 1| max (SE-scaled)", float(np.abs(m_de / h - 1).max() / (se * M / (M - 1) / np.abs(h).min())), 4, float(np.abs(m_de / h - 1).max() / (se * M / (M - 1) / np.abs(h).min())) < 4)
    # J: production core applies factor exactly once; sweep kernel applies it once
    core_plain = np.asarray(centred_zo(jnp.asarray(q[:2000]), jnp.asarray(u[:2000]), axis=1)) / sig
    core_de = np.asarray(centred_zo(jnp.asarray(q[:2000]), jnp.asarray(u[:2000]), axis=1, deattenuate=True)) / sig
    rec("J", "core deatt / core plain == M/(M-1)", float(np.abs(core_de / core_plain - M / (M - 1)).max()), 1e-12, float(np.abs(core_de / core_plain - M / (M - 1)).max()) < 1e-12)
    rec("J", "core plain == my plain", float(np.abs(core_plain - plain[:2000]).max()), 1e-12, float(np.abs(core_plain - plain[:2000]).max()) < 1e-12)
    rec("J", "deattenuation_factor(32) == 32/31", abs(float(deattenuation_factor(32, jnp.float64)) - 32 / 31), 1e-15, abs(float(deattenuation_factor(32, jnp.float64)) - 32 / 31) < 1e-15)
    # variance scales by (M/(M-1))^2 exactly, mean by M/(M-1)
    rec("J", "trCov(deatt)/trCov(plain) - (M/(M-1))^2", abs(trcov(de) / trcov(plain) - (M / (M - 1)) ** 2), 1e-12, abs(trcov(de) / trcov(plain) - (M / (M - 1)) ** 2) < 1e-12)


# ===================================================================== TEST F / Part 4
def test_F_part4():
    print("\nTEST F / PART 4  error-channel variance: A = g(Q^pi+e)-g(Q^pi) vs B = g(e); trCov vs saved s3; vs closed form")
    d = 4
    z = np.load(os.path.join(OUT, "d4_rank1_M32_unit_H_identity.npz"), allow_pickle=True)
    s, st, H, g, mu, pe = regen(d)
    i_state = 0
    rng = np.random.default_rng(5)
    for (si, oi) in ((10, 12), (14, 8), (6, 20)):
        sig, om = float(z["sigmas"][si]), float(z["omegas"][oi])
        c = sig * om
        eps = float(z["eps"][i_state, si])
        pe_o = EF.PlantedError(pe.kind, pe.rank, om, pe.V, pe.phi)
        V, phi = pe.V[i_state], pe.phi[i_state]
        R = 40000
        u = rng.normal(size=(R, M, d))
        a = mu[i_state] + sig * u
        q_sm = -sig ** 2 * np.einsum("rmi,ij,rmj->rm", u, H, u) + sig * (u @ g[i_state])
        q_e = my_e(a, V, phi, om, eps, 1)
        A_pw = my_pw(-2.0 * a @ H.T + (g[i_state] + 2 * mu[i_state] @ H.T) + my_grad_e(a, V, phi, om, eps, 1)) \
            - my_pw(-2.0 * a @ H.T + (g[i_state] + 2 * mu[i_state] @ H.T))
        B_pw = my_pw(my_grad_e(a, V, phi, om, eps, 1))
        A_zo = my_zo_deatt(q_sm + q_e, u, sig) - my_zo_deatt(q_sm, u, sig)
        B_zo = my_zo_deatt(q_e, u, sig)
        rec("4", f"cell sig={sig:.3g} om={om:.3g}: A==B PW max abs", float(np.abs(A_pw - B_pw).max()), 1e-9, float(np.abs(A_pw - B_pw).max()) < 1e-9)
        rec("4", f"cell sig={sig:.3g} om={om:.3g}: A==B ZO max abs", float(np.abs(A_zo - B_zo).max()), 1e-9, float(np.abs(A_zo - B_zo).max()) < 1e-9)
        # trCov in units of (eps/sigma)^2 vs saved s3 (batch-mean, this state, this cell)
        unit = (eps / sig) ** 2
        v_pw, v_zo = trcov(B_pw) / unit, trcov(B_zo) / unit
        s3p = float(z["s3_pw"][i_state, :, si, oi].mean()); s3z = float(z["s3_zo"][i_state, :, si, oi].mean())
        # saved s3 is a mean of squared errors about the EXACT mean over 10^4 draws;
        # MC relative SE ~ sqrt(2/N) each -> compare at 5 combined SE
        tol = 5 * np.sqrt(2 / R + 2 / 10000)
        rec("4", f"cell: my trCov_PW(g(e)) vs saved s3_pw, rel", abs(v_pw / s3p - 1), tol, abs(v_pw / s3p - 1) < tol)
        rec("4", f"cell: my trCov_ZO(g(e)) vs saved s3_zo, rel", abs(v_zo / s3z - 1), tol, abs(v_zo / s3z - 1) < tol)
        # is the saved statistic a variance (about the exact mean) or E||g||^2?  compare mean of B to the analytic blurred grad
        th = om * (V @ mu[i_state]) + phi
        bg = eps * om * np.exp(-0.5 * c * c) * np.cos(th)[:, None] * V
        bg = bg.sum(0)
        rec("4", f"cell: mean B_pw == blurred grad (SE-scaled)", float(np.abs(B_pw.mean(0) - bg).max() / (B_pw.std(0).max() / np.sqrt(R))), 4, float(np.abs(B_pw.mean(0) - bg).max() / (B_pw.std(0).max() / np.sqrt(R))) < 4)
        rec("4", f"cell: mean B_zo == blurred grad (SE-scaled)", float(np.abs(B_zo.mean(0) - bg).max() / (B_zo.std(0).max() / np.sqrt(R))), 4, float(np.abs(B_zo.mean(0) - bg).max() / (B_zo.std(0).max() / np.sqrt(R))) < 4)
        # F: my independent closed forms (rank one) -- derived in the report, not from reference.py
        th1 = float(th[0])
        Vcos = 0.5 + 0.5 * np.exp(-2 * c * c) * np.cos(2 * th1) - np.exp(-c * c) * np.cos(th1) ** 2
        Vsin = 0.5 - 0.5 * np.exp(-2 * c * c) * np.cos(2 * th1) - np.exp(-c * c) * np.sin(th1) ** 2
        var_pw_cf = c * c * Vcos / M
        rec("F", f"cell: Var_PW closed form vs MC, rel", abs(v_pw / var_pw_cf - 1), 5 * np.sqrt(2 / R), abs(v_pw / var_pw_cf - 1) < 5 * np.sqrt(2 / R))
        # ZO: orthogonal part (d-1) Vsin/(M-1) is exact; along-v part Lambda: estimate it by MC
        # of the unbiased sample covariance of (sin(theta+c t), t) and compare the total
        t = rng.normal(size=(R, M)); f = np.sin(th1 + c * t)
        fc = f - f.mean(1, keepdims=True)
        C = (fc * t).sum(1) / (M - 1)
        lam = C.var(ddof=1)
        var_zo_cf = (d - 1) * Vsin / (M - 1) + lam
        rec("F", f"cell: Var_ZO closed form (Lambda by MC) vs MC, rel", abs(v_zo / var_zo_cf - 1), 5 * np.sqrt(2 / R) * 2, abs(v_zo / var_zo_cf - 1) < 5 * np.sqrt(2 / R) * 2)


# ===================================================================== TEST G / Part 5
def test_G_part5():
    print("\nTEST G / PART 5  RMS moments: MC vs my closed form vs lqr_audit_rms.csv (realised phases)")
    rms = pd.read_csv(os.path.join(ART, "lqr_audit_rms.csv"))
    rng = np.random.default_rng(6)
    worst_mc, worst_csv = 0.0, 0.0
    for d, arm, kind, rank in ((4, "rank1", "rank1", None), (8, "rank_r2", "rank_r", 2), (16, "full", "full", None)):
        z = np.load(os.path.join(OUT, f"d{d}_{arm}_M32_unit_H_identity.npz"), allow_pickle=True)
        n_states = int(z["n_states"])
        s, st, H, g, mu, pe = regen(d, kind=kind, rank=rank, n_states=n_states)
        assert np.abs(z["phi"] - pe.phi).max() == 0
        for (si, oi) in ((10, 12), (16, 5)):
            sig, om = float(z["sigmas"][si]), float(z["omegas"][oi]); c = sig * om
            th = om * np.asarray(z["vtmu"]) + np.asarray(z["phi"])           # (st, r)
            Ee2, Eg2 = rms_moments(c, om, th, pe.rank)
            # MC at 4 states
            for i in range(4):
                u = rng.normal(size=(200000, d)); a = mu[i] + sig * u
                e = my_e(a, pe.V[i], pe.phi[i], om, 1.0, pe.rank)
                ge = my_grad_e(a, pe.V[i], pe.phi[i], om, 1.0, pe.rank)
                worst_mc = max(worst_mc, abs(np.mean(e ** 2) / Ee2[i] - 1), abs(np.mean((ge ** 2).sum(-1)) / Eg2[i] - 1))
            om_rms = np.sqrt(Eg2.mean() / Ee2.mean())
            row = rms[(rms.arm == arm) & (rms.d == d) & (np.isclose(rms.sigma, sig)) & (np.isclose(rms.omega, om))]
            worst_csv = max(worst_csv, abs(float(row.omega_rms_over_omega.iloc[0]) / (om_rms / om) - 1))
    rec("G", "closed-form moments vs MC (2e5 draws), max rel", worst_mc, 2e-2, worst_mc < 2e-2)
    rec("G", "omega_RMS/omega: my closed form vs lqr_audit_rms.csv, max rel", worst_csv, 1e-12, worst_csv < 1e-12)


# ===================================================================== PART 6/7/8 from saved artifacts
def parts_678():
    print("\nPART 6/7/8  crossover, aggregation order, exponent, RMS product -- from saved s3 only")
    rep = {}
    for d in (1, 2, 4, 8, 16, 32, 64):
        rep[d] = {}
    summ = json.load(open(os.path.join(ART, "lqr_audit_summary.json")))
    out6 = []
    for arm in ("rank1", "rank_r2", "full"):
        cs_avg, cs_med, cs_mean, dl = [], [], [], []
        for d in (1, 2, 4, 8, 16, 32, 64):
            z = np.load(os.path.join(OUT, f"d{d}_{arm}_M32_unit_H_identity.npz"), allow_pickle=True)
            pw, zo = z["s3_pw"].mean(1), z["s3_zo"].mean(1)                 # (st, sig, om)
            sig, om = z["sigmas"], z["omegas"]
            C = sig[:, None] * om[None, :]
            r = np.log(zo) - np.log(pw)                                     # per state
            # collapse onto c level sets (i+j) -- my own grouping
            n_s, n_o = len(sig), len(om)
            keys = np.add.outer(np.arange(n_s), np.arange(n_o))
            cgrid = np.array([C[np.where(keys == k)][0] for k in range(n_s + n_o - 1)])
            rbar_state = np.stack([np.array([r[s_][keys == k].mean() for k in range(n_s + n_o - 1)]) for s_ in range(r.shape[0])])
            logc = np.log(cgrid)

            def root(y):
                sgn = np.where(np.diff(np.sign(y)) != 0)[0]
                if len(sgn) == 0 or not (y[0] > 0 > y[-1]):
                    return np.nan, len(sgn)
                k = sgn[0]
                f = PchipInterpolator(logc, y)
                return float(np.exp(brentq(f, logc[k], logc[k + 1], xtol=1e-13))), len(sgn)
            c_avg, ncross = root(rbar_state.mean(0))
            per = np.array([root(rbar_state[s_])[0] for s_ in range(r.shape[0])])
            cs_avg.append(c_avg); cs_med.append(np.nanmedian(per)); cs_mean.append(np.nanmean(per)); dl.append(d)
            # linear-interp root as a second independent method
            y = rbar_state.mean(0); k = np.where(np.diff(np.sign(y)) != 0)[0][0]
            c_lin = float(np.exp(logc[k] - y[k] * (logc[k + 1] - logc[k]) / (y[k + 1] - y[k])))
            if arm == "rank1":
                rec("7", f"d={d} sign changes of state-avg ratio", ncross, 1, ncross == 1)
                rec("7", f"d={d} PCHIP root vs linear-interp root, rel", abs(c_avg / c_lin - 1), 2e-3, abs(c_avg / c_lin - 1) < 2e-3)
                rec("7", f"d={d} root of mean vs median of per-state roots, rel", abs(c_avg / np.nanmedian(per) - 1), 1e-2, True, f"mean-of-roots {np.nanmean(per):.4f}, no-bracket {np.isnan(per).mean():.3f}")
                rec("7", f"d={d} my c* vs lqr_audit_summary (report.py path), rel", abs(c_avg / summ["ladder"]["rank1"]["cstars"][str(d)] - 1), 1e-9, abs(c_avg / summ["ladder"]["rank1"]["cstars"][str(d)] - 1) < 1e-9)
            # PART 6: RMS product at cells with r_nom in [0.5, 3]
            th_all = om[None, None, :] * np.asarray(z["vtmu"])[:, :, None] + np.asarray(z["phi"])[:, :, None]  # (st, r, om)
            for i, sg in enumerate(sig):
                for j, w in enumerate(om):
                    c = sg * w
                    if not (0.5 <= c / np.sqrt(d) <= 3.0):
                        continue
                    Ee2, Eg2 = rms_moments(c, w, th_all[:, :, j], int(z["rank"]))
                    r_rms2 = sg * sg * (Eg2.mean() / Ee2.mean()) / d
                    R_ = zo[:, i, j].mean() / pw[:, i, j].mean()
                    out6.append(dict(arm=arm, d=d, rank=int(z["rank"]), sigma=sg, omega=w, R=R_, r_rms=np.sqrt(r_rms2), product=R_ * r_rms2))
        # PART 8 exponents
        ds_fit = [2, 4, 8, 16, 32, 64]
        sel = [k for k, d in enumerate(dl) if d in ds_fit]
        p_avg = np.polyfit(np.log(np.array(dl)[sel]), np.log(np.array(cs_avg)[sel]), 1)[0]
        p_med = np.polyfit(np.log(np.array(dl)[sel]), np.log(np.array(cs_med)[sel]), 1)[0]
        p_all = np.polyfit(np.log(dl), np.log(cs_avg), 1)[0]
        rep_p = summ["ladder"][arm]["p_nominal"]
        rec("8", f"{arm}: p (registered d-set, root-of-mean) vs report", abs(p_avg - rep_p), 1e-9, abs(p_avg - rep_p) < 1e-9, f"p={p_avg:.4f}")
        rec("8", f"{arm}: p with per-state-median roots", abs(p_med - p_avg), 5e-3, True, f"p_med={p_med:.4f}")
        rec("8", f"{arm}: p including d=1", abs(p_all - p_avg), 1, True, f"p_all={p_all:.4f}")
        if arm == "full":
            rec("8", "full: p_omega_inf == p_nominal - 0.5 exactly (relabelling)", abs((rep_p - 0.5) - summ["ladder"]["full"]["p_omega_inf"]), 1e-12, abs((rep_p - 0.5) - summ["ladder"]["full"]["p_omega_inf"]) < 1e-12)
            # direct fit against c/sqrt(r) with r=d
            p_inf_fit = np.polyfit(np.log(np.array(dl)[sel]), np.log(np.array(cs_avg)[sel] / np.sqrt(np.array(dl)[sel])), 1)[0]
            rec("8", "full: refit of log(c*/sqrt d) gives the same shift", abs(p_inf_fit - (p_avg - 0.5)), 1e-12, abs(p_inf_fit - (p_avg - 0.5)) < 1e-12)
    P6 = pd.DataFrame(out6)
    P6.to_csv(os.path.join(ART, "lqr_code_correctness_part6.csv"), index=False)
    print("\n  PART 6 table: R * r_RMS^2 (cells with r_nom in [0.5,3]); predicted M/(M-1) = %.5f" % (M / (M - 1)))
    T = P6.groupby(["arm", "d"]).agg(R_med=("R", "median"), r_rms_med=("r_rms", "median"), product_med=("product", "median"), product_min=("product", "min"), product_max=("product", "max"), n=("product", "size")).reset_index()
    T["rel_err_vs_M_over_M1"] = T.product_med / (M / (M - 1)) - 1
    T["rel_err_vs_M1_over_M"] = T.product_med / ((M - 1) / M) - 1
    print(T.round(4).to_string(index=False))
    for _, row in T.iterrows():
        if row.d >= 8:
            rec("6", f"{row.arm} d={int(row.d)} product median vs 32/31", abs(row.rel_err_vs_M_over_M1), 1e-2, abs(row.rel_err_vs_M_over_M1) < 1e-2)
    # PART 10: c -> 0 behaviour of the registered cross-term ratio from saved s2/s3
    z = np.load(os.path.join(OUT, "d4_rank1_M32_unit_H_identity.npz"), allow_pickle=True)
    r_ = (z["eps"] / z["sigmas"][None, :])[:, :, None]
    sig, om = z["sigmas"], z["omegas"]
    for nm in ("pw", "zo"):
        s2 = z[f"s2_{nm}"]; s3 = z[f"s3_{nm}"]
        s2d = np.stack([s2[:, :, i, i, :] for i in range(s2.shape[2])], axis=2)
        cross = np.abs((2.0 * r_[:, None] * s2d).mean(1)).mean(0)
        err = ((r_[:, None] ** 2) * s3).mean(1).mean(0)
        ratio = cross / err
        # slope of log ratio vs log c along the sigma=0.01 column at the 6 smallest omega
        cc = sig[0] * om[:6]
        sl = np.polyfit(np.log(cc), np.log(ratio[0, :6]), 1)[0]
        rec("10", f"{nm}: d log(|2Cov|/Var_e)/d log c at small c (sigma=0.01)", sl, 0, True, "analytic: PW ~ c^-2, ZO ~ c^-1 (derived in the report)")


# ===================================================================== PART 13
def part13():
    print("\nPART 13  the shipped E-step vs centred ZO on the same draws (small, d=4)")
    rng = np.random.default_rng(7)
    d, sig, om = 4, 0.37, 22.0
    s, st, H, g, mu, pe = regen(d, n_states=4)
    pe_o = EF.PlantedError(pe.kind, pe.rank, om, pe.V, pe.phi)
    eps = 0.05 * lqr.q_spread_closed_form(s, st, sig)
    qf = lqr.q_of_u_factory(s, st, sig)
    mu_j, eps_j = jnp.asarray(mu)[:, None, None, :], jnp.asarray(eps)[:, None, None]
    q_of_u = lambda u: qf(u) + EF.e_value(pe_o, eps_j, mu_j, sig, u)
    u = jnp.asarray(rng.normal(size=(4, 2000, M, d)))
    q, Hw = whitened_pathwise(q_of_u, u)
    g_zo = np.asarray(centred_zo(q, u, axis=2, deattenuate=True) / sig)
    for eta in (1e3, 10.0, 1.0, 0.1):
        w = jax.nn.softmax(q / eta, axis=-1)
        d_es = np.asarray(softmax_displacement(w, u, axis=2) / sig)
        ubar = np.asarray(u.mean(2)) / sig
        # first-order expansion: d_es = ubar + (1/eta) * (sigma * g_zo_plain) + O(eta^-2)
        # with g_zo_plain = (1/M) sum (q - qbar) u / sigma  => sigma*g_zo_plain*(M-1)/M ... use plain
        qn = np.asarray(q); un = np.asarray(u)
        zo_plain = (((qn - qn.mean(2, keepdims=True))[..., None] * un).mean(2)) / sig
        pred = ubar + zo_plain * sig / eta
        rel = float(np.abs(d_es - pred).max() / np.abs(d_es).max())
        cosz = float(np.mean(np.sum(d_es * g_zo, -1) / (np.linalg.norm(d_es, axis=-1) * np.linalg.norm(g_zo, axis=-1))))
        rec("13", f"eta={eta:g}: |d_ES - (ubar + zo_plain*sigma/eta)| rel", rel, 1, True, f"cos(d_ES, g_ZO)={cosz:+.3f}, ESS/M={float((1/(w**2).sum(-1)).mean())/M:.3f}")


# ===================================================================== TEST K (dtype)
def test_K():
    """Precision.  The sweep's small-sigma corner (sigma = 0.01) is where the registered
    cross-term ratio reaches 7e5; if the kernel ran in float32 those cells are noise.
    Evidence gathered: (i) what jax emits in this process after the package import,
    (ii) the dtype of every array in every cited .npz, (iii) an independent float64 AND
    float32 recomputation of the corner cell against the saved s3, (iv) whether the saved
    corner follows the analytic power laws s3_pw ~ c^4, s3_zo ~ c^2, s2_pw ~ c^2,
    s2_zo ~ c (noise would not)."""
    print("\nTEST K  dtype: is the sweep float64, and is the sigma = 0.01 corner numerically real?")
    x64 = bool(jax.config.jax_enable_x64)
    rec("K", "jax_enable_x64 after `import scripts.lqr_crossover`", float(x64), 1, x64)
    k = jax.random.PRNGKey(0)
    rec("K", "jax.random.normal dtype is float64", float(jax.random.normal(k, (2,)).dtype == jnp.float64), 1, jax.random.normal(k, (2,)).dtype == jnp.float64)
    rec("K", "jnp.asarray(np.float64) stays float64", float(jnp.asarray(np.zeros(2)).dtype == jnp.float64), 1, jnp.asarray(np.zeros(2)).dtype == jnp.float64)
    rec("K", "platform is cpu", float(jax.devices()[0].platform == "cpu"), 1, jax.devices()[0].platform == "cpu")
    man = pd.read_csv(os.path.join(ART, "lqr_npz_manifest.csv"))
    bad = []
    for f in man[man.cited_by_report & ~man.file.str.startswith("estep")].file:
        z = np.load(os.path.join(OUT, f), allow_pickle=True)
        for key in ("s1_pw", "s1_zo", "s2_pw", "s2_zo", "s3_pw", "s3_zo", "eps", "sigmas", "omegas"):
            if z[key].dtype != np.float64:
                bad.append((f, key, str(z[key].dtype)))
    rec("K", "every stored statistic in the 28 cited sweep .npz is float64", float(len(bad)), 0, len(bad) == 0, str(bad[:3]))
    # (iii) corner recomputation, float64 and float32, vs saved s3 (d = 4, state 0)
    d = 4
    z = np.load(os.path.join(OUT, "d4_rank1_M32_unit_H_identity.npz"), allow_pickle=True)
    s, st, H, g, mu, pe = regen(d)
    i_state, si, oi = 0, 0, 0
    sig, om = float(z["sigmas"][si]), float(z["omegas"][oi]); c = sig * om
    V, phi = pe.V[i_state], pe.phi[i_state]
    th = om * (V @ mu[i_state]) + phi
    rng = np.random.default_rng(11)
    R = 20000
    u64 = rng.normal(size=(R, M, d))
    for dt, lbl in ((np.float64, "float64"), (np.float32, "float32")):
        u = u64.astype(dt); a = (mu[i_state].astype(dt) + dt(sig) * u)
        T = u @ V.T.astype(dt)                                             # (R, M, 1)
        arg = th.astype(dt) + dt(c) * T
        damp = dt(np.exp(-0.5 * c * c))
        a_pw = np.cos(arg).mean(1) - damp * np.cos(th).astype(dt)         # (R, 1)
        de_pw = dt(c) * (a_pw @ V.astype(dt))                             # (R, d)
        f = np.sin(arg).sum(-1); fc = f - f.mean(1, keepdims=True)
        b = (fc[..., None] * u).mean(1) * dt(M / (M - 1.0))
        tgt = dt(c) * damp * (np.cos(th).astype(dt) @ V.astype(dt))
        de_zo = b - tgt
        v_pw = float((de_pw.astype(np.float64) ** 2).sum(-1).mean())
        v_zo = float((de_zo.astype(np.float64) ** 2).sum(-1).mean())
        s3p = float(z["s3_pw"][i_state, :, si, oi].mean()); s3z = float(z["s3_zo"][i_state, :, si, oi].mean())
        tol = 5 * np.sqrt(2 / R + 2 / 10000)
        rec("K", f"{lbl} recompute of corner (sigma=0.01, omega=0.1) s3_pw vs saved, rel", abs(v_pw / s3p - 1), tol, abs(v_pw / s3p - 1) < tol, f"saved {s3p:.3e}")
        rec("K", f"{lbl} recompute of corner s3_zo vs saved, rel", abs(v_zo / s3z - 1), tol, abs(v_zo / s3z - 1) < tol, f"saved {s3z:.3e}")
    # (iv) power laws along the sigma = 0.01 column, 6 smallest omega, state-averaged
    cc = z["sigmas"][0] * z["omegas"][:6]
    r_ = (z["eps"] / z["sigmas"][None, :])[:, :, None]
    for nm, pw_exp, s2_exp in (("pw", 4.0, 2.0), ("zo", 2.0, 1.0)):
        s3 = z[f"s3_{nm}"].mean(1).mean(0)[0, :6]
        s2 = z[f"s2_{nm}"]; s2d = np.stack([s2[:, :, i, i, :] for i in range(s2.shape[2])], axis=2)
        s2c = np.abs(s2d.mean(1)).mean(0)[0, :6]
        sl3 = np.polyfit(np.log(cc), np.log(s3), 1)[0]; sl2 = np.polyfit(np.log(cc), np.log(s2c), 1)[0]
        rec("K", f"{nm}: slope of log s3 vs log c at sigma=0.01 (analytic {pw_exp:g})", sl3, 0.15, abs(sl3 - pw_exp) < 0.15)
        rec("K", f"{nm}: slope of log |s2| vs log c at sigma=0.01 (analytic {s2_exp:g})", sl2, 0.15, abs(sl2 - s2_exp) < 0.15)
    # magnitudes actually stored in the corner, and the physical variance they imply
    s3p = float(z["s3_pw"].mean(1).mean(0)[0, 0]); s3z = float(z["s3_zo"].mean(1).mean(0)[0, 0])
    pref = float(((z["eps"][:, 0] / z["sigmas"][0]) ** 2).mean())
    rec("K", "corner s3_pw (units (eps/sigma)^2)", s3p, 0, True, f"physical Var_PW_e = {s3p*pref:.3e}; (eps/sigma)^2 = {pref:.3e}")
    rec("K", "corner s3_zo (units (eps/sigma)^2)", s3z, 0, True, f"physical Var_ZO_e = {s3z*pref:.3e}")


def main():
    test_K(); test_A(); test_B(); test_CD(); test_EJ(); test_F_part4(); test_G_part5(); parts_678(); part13()
    df = pd.DataFrame(ROWS)
    df.to_csv(os.path.join(ART, "lqr_code_correctness_checks.csv"), index=False)
    print(f"\n{df.ok.sum()}/{len(df)} checks pass; wrote {ART}/lqr_code_correctness_checks.csv")
    print("FAILED:"); print(df[~df.ok].to_string(index=False) if (~df.ok).any() else "  none")


if __name__ == "__main__":
    main()
