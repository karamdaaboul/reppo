#!/usr/bin/env python
"""Planted-error amplitude experiment.

Registered in docs/prereg_planted_amplitude.md (committed 4b88f03, before this ran).
Design, amplitudes, grid, budget, seeds and the CRN scheme are taken from that
document and are not chosen here.

  A in {0.25, 1.0, 4.0} = {A_0/4, A_0, 4 A_0}
  d in {4, 16, 64} at sigma = 0.4, plus a sigma = 0.2 check at d = 16
  r in {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0},  omega = r sqrt(d) / sigma
  64 direction blocks x 512 batches = 32,768 batches per cell, the committed
  sweep's budget re-partitioned so the bootstrap has a resamplable unit.

Common random numbers: v_signal, v_e, phase and u are drawn ONCE per
(config, direction) and reused across all three amplitudes, so the smooth channel
is bit-identical across amplitudes.  That identity is asserted, not assumed.

Usage:
    ./.venv/bin/python scripts/planted/amplitude_sweep.py reports/artifacts
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decompose_sweep import measure_block, omega_supnorm
import planted_error_sweep as P

# ------------------------------------------------------------- prereg Sec. 3
AMPS = [0.25, 1.0, 4.0]
R_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0]
CONFIGS = ([(d, 0.4) for d in (4, 16, 64)]      # primary
           + [(16, 0.2)])                        # sigma confound check
N_DIR, N_BATCH = 64, 512
SEED = 20260903
M = 32
G_SIG = P.G_SIG


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "reports/artifacts"
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    reps = []
    n_cfg = len(CONFIGS) * len(R_GRID)
    for ci, (d, sig) in enumerate(CONFIGS):
        for r in R_GRID:
            om = r * np.sqrt(d) / sig
            for k in range(N_DIR):
                # one draw, reused across all amplitudes (CRN, prereg Sec. 3.5)
                v_sig = rng.normal(size=d); v_sig /= np.linalg.norm(v_sig)
                ve = rng.normal(size=d); ve /= np.linalg.norm(ve)
                ph = rng.uniform(0, 2 * np.pi)
                astar = G_SIG * v_sig
                u = rng.normal(size=(N_BATCH, M, d))
                om_sup = omega_supnorm(ve, om, ph, d)

                for A in AMPS:
                    b = measure_block(u, sig, astar, ve, om, ph, A)
                    b.update(d=d, sigma=sig, omega=om, r=r, amplitude=A,
                             direction=k, om_supnorm=om_sup)
                    reps.append(b)
            print(f"  [{ci * len(R_GRID) + R_GRID.index(r) + 1}/{n_cfg}] "
                  f"d={d} sig={sig} r={r} om={om:.3g}", flush=True)

    import pandas as pd
    rep = pd.DataFrame(reps)
    rpath = f"{outdir}/planted_amplitude_replicates.csv"
    rep.to_csv(rpath, index=False)
    print(f"\nwrote {rpath}  ({len(rep)} rows = "
          f"{n_cfg} configs x {len(AMPS)} amplitudes x {N_DIR} blocks)")

    # ---- CRN assertion: the smooth channel must not move with amplitude
    key = ["d", "sigma", "r", "direction"]
    worst = 0.0
    for col in ("var_pw_clean", "var_zo_clean", "var_wml_clean", "err_pw_baseline"):
        spread = rep.groupby(key)[col].agg(lambda s: float(np.ptp(s.values)))
        scale = rep.groupby(key)[col].agg(lambda s: float(np.abs(s.values).mean()))
        worst = max(worst, float((spread / np.maximum(scale, 1e-300)).max()))
    print(f"  CRN check: max relative spread of a clean-channel column across "
          f"amplitudes = {worst:.3e}  {'OK' if worst < 1e-12 else 'FAILED'}")

    # ---- cell-level aggregate (the deliverable)
    gk = ["d", "sigma", "omega", "r", "amplitude"]
    cell = rep.groupby(gk, as_index=False).mean(numeric_only=True).drop(
        columns=["direction"])
    cell["ratio_e"] = cell.var_zo_e / cell.var_pw_e
    cell["ratio_e_wml"] = cell.var_wml_e / cell.var_pw_e
    cell["mse_ratio_zo_pw"] = cell.mse_zo / cell.mse_pw
    cell["mse_ratio_wml_pw"] = cell.mse_wml / cell.mse_pw
    cpath = f"{outdir}/planted_amplitude.csv"
    cell.to_csv(cpath, index=False)
    print(f"wrote {cpath}  ({len(cell)} cells)")

    # ---- pathology report (prereg Sec. 6), printed regardless of outcome
    print("\n=== pathology diagnostics, by amplitude (prereg Sec. 6) ===")
    for A, g in cell.groupby("amplitude"):
        print(f"  A={A:<5g} min median ESS={g.ess_med.min():6.2f}  "
              f"min ESS={g.ess_min.min():6.3f}  "
              f"max eta-at-bound={g.eta_at_bound.max():.4f}  "
              f"nonfinite={g.nonfinite.sum():.0f}")
    print(f"  omega sup-norm calibration: max |meas/nom - 1| = "
          f"{np.abs(cell.om_supnorm / cell.omega - 1).max():.3e}")
    print(f"  linearity resid PW/ZO: {rep.lin_resid_pw.max():.2e} / "
          f"{rep.lin_resid_zo.max():.2e}   WML: {rep.lin_resid_wml.max():.2e}")

    # ---- the A-invariance identity for the linear pair (prereg Sec. 1)
    piv = cell.pivot_table(index=["d", "sigma", "r"], columns="amplitude",
                           values="ratio_e")
    dev = float(np.abs(piv.div(piv[1.0], axis=0) - 1).max().max())
    print(f"  ZO/PW error-channel ratio, max relative deviation across "
          f"amplitudes = {dev:.3e}  (implementation check, not a finding)")


if __name__ == "__main__":
    main()
