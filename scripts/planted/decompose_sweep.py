#!/usr/bin/env python
"""Instrumented re-run of the committed 240-cell planted-error grid.

Registered in docs/prereg_planted_amplitude.md Sec. 7.  Two jobs:

1. REPRODUCE.  Same seed, same 8 x 4096 partition, same RNG call order as
   scripts/planted/planted_error_sweep.py -- including the discarded `om_meas`
   draw, which must be kept or the stream diverges.  The `ratio_e` column must
   come back equal to the committed one to floating-point tolerance.

2. INSTRUMENT.  Everything the committed sweep did not measure:
     - the error-channel variance of the ACTUAL E-step (WML).  The committed
       sweep measured it for PW and ZO only, so the operator the paper is about
       had no error-channel number at all;
     - both variance definitions (paired / subtraction) and their difference,
       the cross term 2 Cov(g(Q^pi), g(e));
     - pre-normalisation bias / variance / MSE against the exact target;
     - post-normalisation bias^2 + variance on the unit sphere;
     - channel-attribution counterfactuals (prereg Sec. 2.7).

Nothing here tunes anything.  Usage:

    ./.venv/bin/python scripts/planted/decompose_sweep.py \
        scripts/planted/planted_sweep_config.json reports/artifacts
"""
import json, os, sys, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planted_error_sweep as P
from planted_error_sweep import (q_pi, grad_q_pi, e_field, grad_e_field,
                                 solve_eta, tr_step, tr_err)

EPS_TR = P.EPS_TR
G_SIG = P.G_SIG


# ------------------------------------------------------------------ helpers
def tr_cov(X):
    """tr Cov(X) = E||X - EX||^2 for X of shape (R, d).  The scalar variance
    measure fixed in prereg Sec. 2.1."""
    return float(np.trace(np.cov(X.T))) if X.shape[1] > 1 else float(np.var(X))


def tr_cross(X, Y):
    """2 * tr Cov(X, Y), the cross term of prereg Sec. 2.2."""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    return float(2.0 * np.sum(Xc * Yc) / (X.shape[0] - 1))


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-300)


def sphere_decomp(X, gstar):
    """E||uhat - uhat*||^2 = ||E uhat - uhat*||^2 + E||uhat - E uhat||^2.

    Exact on the unit sphere (prereg Sec. 2.5).  Returns (total, bias2, var, cos).
    The operational error is 2 * EPS_TR * total; verified against tr_err below.
    """
    uh, us = unit(X), gstar / np.linalg.norm(gstar)
    mu = uh.mean(0)
    total = float(np.mean(np.sum((uh - us) ** 2, axis=1)))
    bias2 = float(np.sum((mu - us) ** 2))
    var = float(np.mean(np.sum((uh - mu) ** 2, axis=1)))
    cos = float(np.mean(uh @ us))
    return total, bias2, var, cos


def op_err(X, gstar):
    """The committed 'better update' metric, = 4 eps_TR (1 - cos), per batch."""
    us = gstar / np.linalg.norm(gstar)
    return 4.0 * EPS_TR * (1.0 - unit(X) @ us)


def grad_decomp(X, target):
    """Pre-normalisation bias / variance / MSE against `target` (prereg Sec. 2.6)."""
    mu = X.mean(0)
    bias2 = float(np.sum((mu - target) ** 2))
    var = tr_cov(X)
    return bias2, var, bias2 + var, float(np.linalg.norm(mu))


def omega_supnorm(ve, om, ph, d):
    """omega := ||grad e||_inf / ||e||_inf on a dense grid over one full period
    along v_e -- the definition in the theorem.  NOT the policy-region sampling
    of the committed sweep's om_meas column, which its own report records as
    invalid (prereg Sec. 6.4)."""
    t = np.linspace(0, 2 * np.pi / om, 20001)
    a = t[:, None] * ve[None, :]
    sup_e = np.abs(e_field(a, ve, om, ph, 1.0)).max()
    sup_g = np.linalg.norm(grad_e_field(a, ve, om, ph, 1.0), axis=-1).max()
    return float(sup_g / max(sup_e, 1e-300))


# ------------------------------------------------------------------ one block
def measure_block(u, sig, astar, ve, om, ph, amp):
    """All statistics for one direction block.  `u` is (R, M, d)."""
    a = sig * u
    out = {}

    # ---- critic pieces.  e is the planted field at amplitude `amp`.
    Q0 = q_pi(a, astar)
    gr0 = grad_q_pi(a, astar)
    eA = e_field(a, ve, om, ph, amp)
    geA = grad_e_field(a, ve, om, ph, amp)
    Q1, gr1 = Q0 + eA, gr0 + geA

    # ---- estimators on Q^pi (clean), Q^pi + e (contaminated), e alone
    def zo(Qx):
        return ((Qx - Qx.mean(1, keepdims=True))[..., None] * u).mean(1) / sig

    def wml(Qx):
        eta = solve_eta(Qx, P.EPS_E)
        z = Qx / eta[:, None]
        z = z - z.max(1, keepdims=True)
        w = np.exp(z)
        w /= w.sum(1, keepdims=True)
        ess = 1.0 / (w ** 2).sum(1)
        return (w[..., None] * a).sum(1), eta, ess

    pw0, pw1, pw_e = gr0.mean(1), gr1.mean(1), geA.mean(1)
    zo0, zo1, zo_e = zo(Q0), zo(Q1), zo(eA)
    wml0, eta0, ess0 = wml(Q0)
    wml1, eta1, ess1 = wml(Q1)
    wml_e, eta_e, _ = wml(eA)

    # paired error-channel component (prereg Sec. 2.2, primary definition)
    dpw, dzo, dwml = pw1 - pw0, zo1 - zo0, wml1 - wml0

    # linearity of PW and ZO: the paired difference IS g(e), exactly.  Measured,
    # not assumed -- it is the check that licenses the A-invariance statement.
    out["lin_resid_pw"] = float(np.abs(dpw - pw_e).max())
    out["lin_resid_zo"] = float(np.abs(dzo - zo_e).max())
    out["lin_resid_wml"] = float(np.abs(dwml - wml_e).max())

    gstar = grad_q_pi(np.zeros(a.shape[-1]), astar)
    out["gstar_norm"] = float(np.linalg.norm(gstar))

    # ---------------- error-channel variance, both definitions + cross term
    for nm, g0, g1, dg in (("pw", pw0, pw1, dpw), ("zo", zo0, zo1, dzo),
                           ("wml", wml0, wml1, dwml)):
        out[f"var_{nm}_e"] = tr_cov(dg)                       # paired (primary)
        out[f"var_{nm}_clean"] = tr_cov(g0)                   # smooth channel
        out[f"var_{nm}_tot"] = tr_cov(g1)
        out[f"var_{nm}_e_sub"] = tr_cov(g1) - tr_cov(g0)      # subtraction
        out[f"cross_{nm}"] = tr_cross(g0, dg)                 # 2 Cov(g0, dg)

    # ---------------- pre-normalisation decomposition (Sec. 2.6)
    # PW and ZO are gradient estimators: target g* for the clean and total
    # channels, target 0 for the error channel (its ideal contribution is none).
    for nm, g0, g1, ge in (("pw", pw0, pw1, pw_e), ("zo", zo0, zo1, zo_e)):
        for ch, X, tgt in (("clean", g0, gstar), ("tot", g1, gstar),
                           ("err", ge, np.zeros_like(gstar))):
            b2, v, m, nrm = grad_decomp(X, tgt)
            out[f"pre_{nm}_{ch}_bias2"] = b2
            out[f"pre_{nm}_{ch}_var"] = v
            out[f"pre_{nm}_{ch}_mse"] = m
            out[f"pre_{nm}_{ch}_meannorm"] = nrm
    # WML is a displacement, not a gradient: no scale convention, so no bias
    # against g* is reported.  First two moments + angle instead (Sec. 2.6).
    for ch, X in (("clean", wml0), ("tot", wml1), ("err", dwml)):
        mu = X.mean(0)
        nrm = float(np.linalg.norm(mu))
        out[f"pre_wml_{ch}_meannorm"] = nrm
        out[f"pre_wml_{ch}_var"] = tr_cov(X)
        out[f"pre_wml_{ch}_nsr"] = tr_cov(X) / max(nrm ** 2, 1e-300)
        out[f"pre_wml_{ch}_cosmean"] = float(
            mu @ gstar / max(nrm * np.linalg.norm(gstar), 1e-300))

    # ---------------- post-normalisation, on the unit sphere (Sec. 2.5)
    for nm, X in (("pw", pw1), ("zo", zo1), ("wml", wml1)):
        tot, b2, v, cos = sphere_decomp(X, gstar)
        out[f"post_{nm}_sphere"] = tot
        out[f"post_{nm}_bias2"] = b2
        out[f"post_{nm}_var"] = v
        out[f"cos_{nm}"] = cos
        out[f"mse_{nm}"] = float(op_err(X, gstar).mean())

    # committed metric, recomputed the committed way, as a cross-check
    dmu_star = tr_step(gstar[None], sig)
    for nm, X in (("pw", pw1), ("zo", zo1)):
        out[f"mse_{nm}_committed"] = float(
            tr_err(tr_step(X, sig), dmu_star, sig).mean())
    dm_w = np.sqrt(2 * EPS_TR) * wml1 / np.maximum(
        np.linalg.norm(wml1, axis=-1, keepdims=True) / sig, 1e-300)
    out["mse_wml_committed"] = float(tr_err(dm_w, dmu_star, sig).mean())

    e_pw = op_err(pw1, gstar)
    out["win_wml_vs_pw"] = float(np.mean(op_err(wml1, gstar) < e_pw))
    out["win_zo_vs_pw"] = float(np.mean(op_err(zo1, gstar) < e_pw))

    # ---------------- channel attribution counterfactuals (Sec. 2.7)
    for nm, g0, dg in (("pw", pw0, dpw), ("zo", zo0, dzo), ("wml", wml0, dwml)):
        out[f"err_{nm}_total"] = float(op_err(g0 + dg, gstar).mean())
        out[f"err_{nm}_cleanoff"] = float(op_err(g0.mean(0) + dg, gstar).mean())
        out[f"err_{nm}_erroroff"] = float(op_err(g0 + dg.mean(0), gstar).mean())
        out[f"err_{nm}_baseline"] = float(op_err(g0, gstar).mean())

    # ---------------- pathology diagnostics (Sec. 6)
    out["ess_med"] = float(np.median(ess1))
    out["ess_min"] = float(np.min(ess1))
    out["eta_at_bound"] = float(np.mean((eta1 <= 1.0001e-4) | (eta1 >= 0.9999e4)))
    out["nonfinite"] = float(not (np.isfinite(pw1).all() and np.isfinite(zo1).all()
                                  and np.isfinite(wml1).all()))
    return out


# ------------------------------------------------------------------ sweep
def main():
    cfg = json.load(open(sys.argv[1]))
    outdir = sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    G = cfg["grid"]
    amp = cfg["planted_error"]["eps"]
    M = G["M"]

    rng = np.random.default_rng(G["seed"])
    rows = []
    combos = list(itertools.product(G["d"], G["sigma"], G["omega"]))
    for ci, (d, sig, om) in enumerate(combos):
        blocks = []
        for _ in range(G["n_directions"]):
            # RNG call order fixed by the committed sweep -- do not reorder.
            v_sig = rng.normal(size=d); v_sig /= np.linalg.norm(v_sig)
            ve = rng.normal(size=d); ve /= np.linalg.norm(ve)
            ph = rng.uniform(0, 2 * np.pi)
            astar = G_SIG * v_sig
            u = rng.normal(size=(G["n_batches"], M, d))

            b = measure_block(u, sig, astar, ve, om, ph, amp)
            b["om_supnorm"] = omega_supnorm(ve, om, ph, d)
            blocks.append(b)

            # the committed sweep's discarded om_meas draw.  Kept ONLY to hold
            # the RNG stream identical; its value is not used (invalid, see
            # reports/planted_error_phase_diagram.md Sec. 6).
            _ = rng.normal(size=(20000, d)) * sig

        row = dict(d=d, sigma=sig, omega=om, r=sig * om / np.sqrt(d), amplitude=amp)
        for k in blocks[0]:
            row[k] = float(np.mean([b[k] for b in blocks]))
        row["ratio_e"] = row["var_zo_e"] / row["var_pw_e"]
        row["ratio_e_wml"] = row["var_wml_e"] / row["var_pw_e"]
        row["mse_ratio_zo_pw"] = row["mse_zo"] / row["mse_pw"]
        row["mse_ratio_wml_pw"] = row["mse_wml"] / row["mse_pw"]
        rows.append(row)
        if ci % 40 == 0:
            print(f"  [{ci+1}/{len(combos)}] d={d} sig={sig} om={om} "
                  f"r={row['r']:.3g} ratio_e={row['ratio_e']:.4g}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    path = f"{outdir}/planted_error_decomposition.csv"
    df.to_csv(path, index=False)
    print(f"\nwrote {path}  ({len(df)} cells, {len(df.columns)} columns)")

    # ---- blocking reproduction check (prereg Sec. 7)
    ref = pd.read_csv(f"{outdir}/planted_sweep.csv")
    key = ["d", "sigma", "omega"]
    m = df.merge(ref[key + ["ratio_e", "mse_pw", "mse_wml"]], on=key,
                 suffixes=("", "_ref"))
    for col in ("ratio_e", "mse_pw", "mse_wml"):
        rel = np.abs(m[col] / m[f"{col}_ref"] - 1).max()
        print(f"  reproduction {col:9s}: max rel dev = {rel:.3e}"
              f"   {'OK' if rel < 1e-9 else 'MISMATCH'}")
    print(f"  linearity resid PW/ZO (must be ~0): "
          f"{df.lin_resid_pw.max():.2e} / {df.lin_resid_zo.max():.2e}")
    print(f"  linearity resid WML  (must NOT be 0): {df.lin_resid_wml.max():.2e}")
    print(f"  omega sup-norm calibration: max |meas/nom - 1| = "
          f"{np.abs(df.om_supnorm / df.omega - 1).max():.3e}")
    print(f"  ESS median min over cells: {df.ess_med.min():.2f}"
          f"   eta-at-bound max: {df.eta_at_bound.max():.3f}"
          f"   nonfinite: {df.nonfinite.sum():.0f}")


if __name__ == "__main__":
    main()
