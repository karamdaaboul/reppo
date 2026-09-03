#!/usr/bin/env python
"""Multi-mode planted-error experiment: J = 1 vs J = 4.

Registered in docs/prereg_planted_multimode.md (committed 44c86a9, before this ran).
Construction, normalisation, grid, budget, seeds and statistics come from that
document and are not chosen here.

    e(a) = sum_{j<=J} c_j sin(omega v_j^T a + phi_j),  c_j = A_0/J,  {v_j} orthonormal

so A_eff = ||e||_inf = A_0 exactly and omega_eff = ||grad e||_inf/||e||_inf
= omega/sqrt(J) exactly.  Cells are matched on r_eff = sigma omega_eff/sqrt(d),
hence omega = r_eff sqrt(J d)/sigma.

Primary statistic is the CONDITIONAL systematic displacement: the action-average is
taken inside each fixed error-field block, the norm is taken per block, and only then
is the average over blocks formed.

    ./.venv/bin/python scripts/planted/multimode_sweep.py reports/artifacts
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planted_error_sweep as P
from planted_error_sweep import q_pi, grad_q_pi, solve_eta
from decompose_sweep import tr_cov, tr_cross, op_err, sphere_decomp

# ------------------------------------------------------------- prereg Sec. 3
AMP = 1.0                       # A_0, single amplitude
SIGMA = 0.4                     # single sigma
DS = [4, 16, 64]
R_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0]
JS = [1, 4]
JMAX = max(JS)
N_DIR, N_BATCH = 64, 512        # 32,768 batches per cell
SEED = 20260904
M = 32


# ------------------------------------------------------------- the field
def e_mm(a, V, ph, c, om):
    """e(a) = sum_j c_j sin(om v_j^T a + phi_j).  V is (J, d) orthonormal rows."""
    return (c * np.sin(om * (a @ V.T) + ph)).sum(-1)


def grad_e_mm(a, V, ph, c, om):
    return ((c * om * np.cos(om * (a @ V.T) + ph))[..., None] * V).sum(-2)


def sup_norms_analytic(c, om):
    """Exact for orthonormal v_j: the J phase arguments are independently settable."""
    return float(np.abs(c).sum()), float(om * np.sqrt((c ** 2).sum()))


def sup_norms_numeric(V, ph, c, om, d, rng, n_start=20, n_probe=200000):
    """Check on the exact formula (prereg Sec. 5.4), not the source of the value.
    Multistart L-BFGS plus a dense probe; both give LOWER bounds on the sup."""
    from scipy.optimize import minimize
    best_e = best_g = 0.0
    scale = 2 * np.pi / om
    for _ in range(n_start):
        x0 = rng.normal(size=d) * scale
        r1 = minimize(lambda x: -abs(e_mm(x, V, ph, c, om)), x0, method="L-BFGS-B")
        best_e = max(best_e, float(-r1.fun))
        r2 = minimize(lambda x: -np.linalg.norm(grad_e_mm(x, V, ph, c, om)), x0,
                      method="L-BFGS-B")
        best_g = max(best_g, float(-r2.fun))
    Pr = rng.uniform(-20 * scale, 20 * scale, size=(n_probe, d))
    best_e = max(best_e, float(np.abs(e_mm(Pr, V, ph, c, om)).max()))
    best_g = max(best_g, float(np.linalg.norm(grad_e_mm(Pr, V, ph, c, om), axis=-1).max()))
    return best_e, best_g


# ------------------------------------------------------------- one block
def measure_block(u, sig, astar, V, ph, c, om):
    """All statistics for ONE fixed error field.  `u` is (R, M, d): the R axis is
    action-sampling noise at fixed field, so every mean(0) below is E_action[. | e_b]."""
    a = sig * u
    d = a.shape[-1]
    out = {}

    Q0 = q_pi(a, astar)
    gr0 = grad_q_pi(a, astar)
    eA = e_mm(a, V, ph, c, om)
    geA = grad_e_mm(a, V, ph, c, om)
    Q1, gr1 = Q0 + eA, gr0 + geA

    def zo(Qx):
        return ((Qx - Qx.mean(1, keepdims=True))[..., None] * u).mean(1) / sig

    def wml(Qx):
        eta = solve_eta(Qx, P.EPS_E)
        z = Qx / eta[:, None]
        z = z - z.max(1, keepdims=True)
        w = np.exp(z)
        w /= w.sum(1, keepdims=True)
        return (w[..., None] * a).sum(1), eta, 1.0 / (w ** 2).sum(1)

    pw0, pw1 = gr0.mean(1), gr1.mean(1)
    zo0, zo1 = zo(Q0), zo(Q1)
    wml0, eta0, ess0 = wml(Q0)
    wml1, eta1, ess1 = wml(Q1)
    dpw, dzo, dwml = pw1 - pw0, zo1 - zo0, wml1 - wml0

    gstar = grad_q_pi(np.zeros(d), astar)

    # ---- PRIMARY: conditional systematic displacement (prereg Sec. 4.1)
    delta_b = dwml.mean(0)                 # E_action[Delta_err | e_b]
    s_b = wml0.mean(0)                     # E_action[Delta_clean | e_b]
    ns = float(np.linalg.norm(s_b))
    out["delta_norm"] = float(np.linalg.norm(delta_b))
    out["s_norm"] = ns
    out["B_b"] = out["delta_norm"] / max(ns, 1e-300)
    # ---- SECONDARY: conditional error-channel noise (prereg Sec. 4.2)
    out["N_b"] = tr_cov(dwml) / max(ns ** 2, 1e-300)
    out["delta_dot_vsig"] = float(delta_b @ (astar / np.linalg.norm(astar)))

    # ---- centred ZO/PW error channel, paired CRN definition (prereg Sec. 4.6)
    for nm, g0, dg in (("pw", pw0, dpw), ("zo", zo0, dzo), ("wml", wml0, dwml)):
        out[f"var_{nm}_e"] = tr_cov(dg)
        out[f"var_{nm}_clean"] = tr_cov(g0)
        out[f"cross_{nm}"] = tr_cross(g0, dg)
    out["pre_pw_clean_meannorm"] = float(np.linalg.norm(pw0.mean(0)))
    out["pre_zo_clean_meannorm"] = float(np.linalg.norm(zo0.mean(0)))
    out["pre_wml_clean_meannorm"] = ns

    # ---- operational (prereg Sec. 4.3)
    for nm, X in (("pw", pw1), ("zo", zo1), ("wml", wml1)):
        tot, b2, v, cos = sphere_decomp(X, gstar)
        out[f"mse_{nm}"] = float(op_err(X, gstar).mean())
        out[f"cos_{nm}"] = cos
        out[f"post_{nm}_bias2"] = b2
        out[f"post_{nm}_var"] = v
    out["G"] = out["mse_wml"] - out["mse_pw"]

    # ---- counterfactual decomposition, existing procedure (prereg Sec. 4.4)
    for nm, g0, dg in (("pw", pw0, dpw), ("zo", zo0, dzo), ("wml", wml0, dwml)):
        out[f"err_{nm}_total"] = float(op_err(g0 + dg, gstar).mean())
        out[f"err_{nm}_cleanoff"] = float(op_err(g0.mean(0) + dg, gstar).mean())
        out[f"err_{nm}_erroroff"] = float(op_err(g0 + dg.mean(0), gstar).mean())
        out[f"err_{nm}_baseline"] = float(op_err(g0, gstar).mean())

    # ---- pathology (prereg Sec. 5)
    out["ess_med"] = float(np.median(ess1))
    out["ess_min"] = float(np.min(ess1))
    out["eta_at_bound"] = float(np.mean((eta1 <= 1.0001e-4) | (eta1 >= 0.9999e4)))
    out["nonfinite"] = float(not (np.isfinite(pw1).all() and np.isfinite(zo1).all()
                                  and np.isfinite(wml1).all()))
    out["wml0_checksum"] = float(wml0.sum())   # CRN check: identical across arms
    return out, delta_b


# ------------------------------------------------------------- sweep
def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "reports/artifacts"
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    chk = np.random.default_rng(SEED + 1)

    reps, cells = [], []
    n_cfg = len(DS) * len(R_GRID)
    for ci, d in enumerate(DS):
        for r_eff in R_GRID:
            acc = {J: [] for J in JS}
            dsum = {J: np.zeros(d) for J in JS}
            om_of = {J: r_eff * np.sqrt(J * d) / SIGMA for J in JS}
            frame0 = {}
            for k in range(N_DIR):
                # one draw per block, shared by both arms (prereg Sec. 3.1)
                v_sig = rng.normal(size=d); v_sig /= np.linalg.norm(v_sig)
                V = np.linalg.qr(rng.normal(size=(d, JMAX)))[0].T   # (JMAX, d)
                ph = rng.uniform(0, 2 * np.pi, size=JMAX)
                astar = P.G_SIG * v_sig
                u = rng.normal(size=(N_BATCH, M, d))
                orth = float(np.abs(V @ V.T - np.eye(JMAX)).max())

                for J in JS:
                    c = np.full(J, AMP / J)
                    om = om_of[J]
                    b, delta = measure_block(u, SIGMA, astar, V[:J], ph[:J], c, om)
                    A_an, L_an = sup_norms_analytic(c, om)
                    b.update(d=d, sigma=SIGMA, r_eff=r_eff, J=J, omega_nominal=om,
                             direction=k, A_eff=A_an, L_eff=L_an,
                             omega_eff=L_an / A_an, orth_resid=orth)
                    acc[J].append(b)
                    dsum[J] += delta
                    reps.append(b)
                    if k == 0:
                        frame0[J] = (V[:J].copy(), ph[:J].copy(), c.copy(), om)

            for J in JS:
                row = dict(d=d, sigma=SIGMA, r_eff=r_eff, J=J, omega_nominal=om_of[J])
                for key in acc[J][0]:
                    if key not in ("d", "sigma", "r_eff", "J", "omega_nominal",
                                   "direction"):
                        row[key] = float(np.mean([b[key] for b in acc[J]]))
                # cross-field cancellation (prereg Sec. 4.5)
                row["delta_norm_conditional"] = row["delta_norm"]
                row["delta_norm_pooled"] = float(np.linalg.norm(dsum[J] / N_DIR))
                row["cancellation_ratio"] = (row["delta_norm_pooled"]
                                             / max(row["delta_norm"], 1e-300))
                # numeric sup-norm check on the first block's field (prereg Sec. 5.4)
                V0, ph0, c0, om0 = frame0[J]
                ne, ng = sup_norms_numeric(V0, ph0, c0, om0, d, chk)
                A_an, L_an = sup_norms_analytic(c0, om0)
                row["A_eff_numeric_ratio"] = ne / A_an
                row["L_eff_numeric_ratio"] = ng / L_an
                cells.append(row)
            print(f"  [{ci * len(R_GRID) + R_GRID.index(r_eff) + 1}/{n_cfg}] "
                  f"d={d} r_eff={r_eff}  omega J1={om_of[1]:.4g} J4={om_of[4]:.4g}",
                  flush=True)

    import pandas as pd
    rep = pd.DataFrame(reps)
    cell = pd.DataFrame(cells)
    rp = f"{outdir}/planted_multimode_replicates.csv"
    cp = f"{outdir}/planted_multimode.csv"
    rep.to_csv(rp, index=False)
    cell.to_csv(cp, index=False)
    print(f"\nwrote {rp}  ({len(rep)} rows)")
    print(f"wrote {cp}  ({len(cell)} cells)")

    # ---------------- validation (prereg Sec. 5), printed regardless of outcome
    print("\n=== validation ===")
    key = ["d", "r_eff", "direction"]
    piv = rep.pivot_table(index=key, columns="J", values="wml0_checksum")
    crn = float(np.abs(piv[1] - piv[4]).max())
    print(f"  CRN: max |wml0 checksum J1 - J4| = {crn:.3e}  "
          f"{'OK (clean channel identical across arms)' if crn == 0 else 'FAILED'}")
    print(f"  frame orthonormality: max ||V V^T - I|| = {rep.orth_resid.max():.3e}")
    print(f"  sup-norm numeric/analytic: A_eff in "
          f"[{cell.A_eff_numeric_ratio.min():.6f}, {cell.A_eff_numeric_ratio.max():.6f}]"
          f"   L_eff in [{cell.L_eff_numeric_ratio.min():.6f}, "
          f"{cell.L_eff_numeric_ratio.max():.6f}]  (<=1 by construction)")
    for J, g in cell.groupby("J"):
        print(f"  J={J}: min median ESS {g.ess_med.min():6.2f}  min ESS "
              f"{g.ess_min.min():6.3f}  eta-at-bound {g.eta_at_bound.max():.4f}  "
              f"nonfinite {g.nonfinite.sum():.0f}")
    # consistency with the amplitude experiment's A_0 arm
    try:
        amp = pd.read_csv(f"{outdir}/planted_amplitude.csv")
        amp = amp[(amp.amplitude == 1.0) & (amp.sigma == 0.4)]
        j1 = cell[cell.J == 1].merge(amp, left_on=["d", "r_eff"],
                                     right_on=["d", "r"], suffixes=("", "_amp"))
        rel = np.abs(j1.mse_wml / j1.mse_wml_amp - 1)
        print(f"  J=1 vs planted_amplitude.csv (A=1): max rel dev in mse_wml = "
              f"{rel.max():.3e}, median {np.median(rel):.3e}  (different seed)")
    except Exception as exc:
        print(f"  consistency check skipped: {exc}")


if __name__ == "__main__":
    main()
