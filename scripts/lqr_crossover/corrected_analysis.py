"""Recoverable statistics for reports/lqr_crossover_corrected.md.  Post-hoc audit
computations on the PREREGISTERED saved experimental data; no sweep is rerun, no .npz
is modified, no estimator is touched.

    JAX_PLATFORMS=cpu python scripts/lqr_crossover/corrected_analysis.py            # tables
    JAX_PLATFORMS=cpu python scripts/lqr_crossover/corrected_analysis.py --bootstrap # + 4 x 10^4 resamples

Outputs (reports/artifacts/):
  lqr_corrected_ruleA.csv        measured c* (root-of-mean and per-state roots), the
                                 preregistered per-state analytic roots at the realised
                                 theta, the measurement-matched analytic aggregate, the
                                 old report's theta = phi column, bootstrap SE, dev/SE
  lqr_corrected_crossterm.csv    registered / post-hoc cross-term readings and
                                 Var_e / MSE_total at the crossover, PW and ZO, per d
  lqr_corrected_bootstrap.json   hierarchical bootstraps at the registered 10 000:
                                 rank_r2 both-level, full both-level, rank1 states-only,
                                 rank1 batches-only  (rank1 both-level: the committed
                                 out/bootstrap_p.json, 10 000, reused)
"""
from __future__ import annotations

import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
import scripts.lqr_crossover  # noqa: F401,E402   CPU, float64 asserted
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scripts.lqr_crossover import OUT, analyze as A, reference as R  # noqa: E402
from scripts.lqr_crossover.audit import cstar_bootstrap  # noqa: E402

ART = os.path.join(REPO_ROOT, "reports", "artifacts")
DS_ALL = (1, 2, 4, 8, 16, 32, 64)
FIT_DS = (2, 4, 8, 16, 32, 64)
M = 32
SIGMA_MIN_POLICY = 0.1           # src/networks/jax_models.py:336, effective min_std


def load(d, arm="rank1"):
    return A.load(os.path.join(OUT, f"d{d}_{arm}_M32_unit_H_identity.npz"))


def root_logc(logc, y):
    """Interior root of a decreasing curve y(logc); mirrors analyze.solve_crossover."""
    if not (y[0] > 0 > y[-1]):
        return np.nan
    k = np.where(np.diff(np.sign(y)) != 0)[0]
    if len(k) == 0 or k[0] == 0 or k[0] >= len(logc) - 2:
        return np.nan
    k = k[0]
    return float(np.exp(brentq(PchipInterpolator(logc, y), logc[k], logc[k + 1], xtol=1e-13)))


def analytic_logratio_by_c(z, d):
    """Per-state analytic log(Var_ZO_e/Var_PW_e) on the c level sets of the SWEPT grid,
    with the realised theta_s(cell) = omega_j v^T mu_s + phi_s.  Shape (states, n_c).
    Mirrors analyze.log_ratio_by_c cell for cell, with the closed forms of reference.py
    in place of the sampled s3."""
    sig, om = z["sigmas"], z["omegas"]
    phi, vt = np.asarray(z["phi"])[:, 0], np.asarray(z["vtmu"])[:, 0]
    n_s, n_o = len(sig), len(om)
    groups = {}
    for i in range(n_s):
        for j in range(n_o):
            groups.setdefault(i + j, []).append((i, j))
    out = np.zeros((len(phi), n_s + n_o - 1))
    for k in sorted(groups):
        vals = []
        for i, j in groups[k]:
            c, w = sig[i] * om[j], om[j]
            vals.append(R.log_ratio(c, w * vt + phi, M, d))       # (states,)
        out[:, k] = np.mean(vals, axis=0)
    return out


def part_ruleA():
    rows = []
    for d in DS_ALL:
        z = load(d)
        logc = np.log(A.c_grid(z))
        # measured
        c_meas, ok = A.crossover_by_c(z)                          # root of state-mean
        per = A.crossover_by_c(z, per_state=True)                 # per-state roots
        se, _ = cstar_bootstrap(z, nboot=1000)
        # analytic, realised theta, on the swept grid
        L = analytic_logratio_by_c(z, d)
        an_B = root_logc(logc, L.mean(0))                         # measurement-matched
        an_A_roots = np.array([root_logc(logc, L[s]) for s in range(L.shape[0])])
        # the old report's column: per-state roots at theta = phi on an unbounded c axis
        th = np.asarray(z["phi"])[:, 0]
        old = float(np.median([R.crossover_c_star(d, float(t), M) for t in th]))
        rows.append(dict(
            d=d, c_meas_root_of_mean=c_meas, c_meas_per_state_median=float(np.nanmedian(per)),
            c_meas_per_state_mean=float(np.nanmean(per)), frac_no_bracket=float(np.isnan(per).mean()),
            c_an_B_measurement_matched=an_B,
            c_an_A_per_state_median=float(np.nanmedian(an_A_roots)),
            c_an_A_per_state_mean=float(np.nanmean(an_A_roots)),
            c_an_old_theta_eq_phi=old,
            boot_se=se,
            dev_over_se_B=abs(c_meas - an_B) / se,
            dev_over_se_A_median_vs_measured_median=abs(float(np.nanmedian(per)) - float(np.nanmedian(an_A_roots))) / se,
            dev_over_se_old=abs(c_meas - old) / se,
            registered_fit_d=d in FIT_DS))
        print(f"  d={d:2d} meas {c_meas:.4f} | per-state med {np.nanmedian(per):.4f} | "
              f"an B {an_B:.4f} ({abs(c_meas-an_B)/se:.2f} SE) | an A med {np.nanmedian(an_A_roots):.4f} | "
              f"old {old:.4f} ({abs(c_meas-old)/se:.2f} SE)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ART, "lqr_corrected_ruleA.csv"), index=False)
    return df


def part_crossterm(ruleA):
    rows = []
    for d in DS_ALL:
        z = load(d)
        c_star = float(ruleA.loc[ruleA.d == d, "c_meas_root_of_mean"].iloc[0])
        r = A.eps_over_sigma(z)[:, :, None]
        sig, om = z["sigmas"], z["omegas"]
        reach = sig >= SIGMA_MIN_POLICY
        C = sig[:, None] * om[None, :]
        k = np.unravel_index(np.argmin(np.abs(np.log(C) - np.log(c_star))), C.shape)
        row = dict(d=d, c_star=c_star, crossover_sigma=float(sig[k[0]]), crossover_omega=float(om[k[1]]),
                   crossover_in_policy_reachable=bool(sig[k[0]] >= SIGMA_MIN_POLICY),
                   n_cells_sigma_lt_0p1=int((~reach).sum() * len(om)), n_cells=int(len(sig) * len(om)))
        for nm, pw in (("pw", True), ("zo", False)):
            s1 = z["s1_pw" if pw else "s1_zo"]; s2 = z["s2_pw" if pw else "s2_zo"]; s3 = z["s3_pw" if pw else "s3_zo"]
            s2d = np.stack([s2[:, :, i, i, :] for i in range(s2.shape[2])], axis=2)
            cross_s = (2.0 * r[:, None] * s2d).mean(1)                            # (states, sig, om)
            err_s = ((r[:, None] ** 2) * s3).mean(1)
            smooth_s = (s1[..., None] * np.ones_like(s3)).mean(1)
            tot_s = smooth_s + cross_s + err_s
            # per-state reading, as report.py / analyze.cross_share and the registered rule:
            # mean_s |2Cov_s| against mean_s Var_e,s (Claim 4 is a per-state statement)
            cross = np.abs(cross_s).mean(0); err = err_s.mean(0)
            smooth = smooth_s.mean(0); tot = np.abs(tot_s).mean(0)
            ratio_e = cross / np.maximum(err, 1e-300)
            ratio_t = cross / np.maximum(tot, 1e-300)
            row[f"{nm}_pooled_2Cov_over_VarE_at_crossover"] = float(np.abs(cross_s.mean(0))[k] / err[k])
            row[f"{nm}_A_registered_allgrid_max"] = float(ratio_e.max())
            i, j = np.unravel_index(np.argmax(ratio_e), ratio_e.shape)
            row[f"{nm}_A_argmax_sigma"], row[f"{nm}_A_argmax_omega"] = float(sig[i]), float(om[j])
            row[f"{nm}_A_argmax_in_policy_reachable"] = bool(sig[i] >= SIGMA_MIN_POLICY)
            row[f"{nm}_B_at_crossover"] = float(ratio_e[k])
            row[f"{nm}_C_reachable_max"] = float(ratio_e[reach].max())
            row[f"{nm}_D_frac_total_reachable_max"] = float(ratio_t[reach].max())
            row[f"{nm}_D_frac_total_at_crossover"] = float(ratio_t[k])
            row[f"{nm}_VarE_over_MSEtotal_at_crossover"] = float(err[k] / tot[k])
            row[f"{nm}_smooth_over_MSEtotal_at_crossover"] = float(smooth[k] / tot[k])
        rows.append(row)
        print(f"  d={d:2d} cross cell sig={row['crossover_sigma']:.3g} reachable={row['crossover_in_policy_reachable']} | "
              f"Var_e/MSE PW {row['pw_VarE_over_MSEtotal_at_crossover']:.3f} ZO {row['zo_VarE_over_MSEtotal_at_crossover']:.3f} | "
              f"|2Cov|/Var_e PW {row['pw_B_at_crossover']:.3f} ZO {row['zo_B_at_crossover']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ART, "lqr_corrected_crossterm.csv"), index=False)
    return df


def part_bootstrap():
    """Registered hierarchical bootstrap (analyze.bootstrap_p: states with replacement,
    then batches within each drawn state; c* re-rooted and p re-fit per resample;
    np.random.default_rng(20260902)), at the registered 10 000 resamples."""
    out = {}
    for arm in ("rank_r2", "full"):
        zs = [load(d, arm) for d in FIT_DS]
        t0 = time.time()
        res = A.bootstrap_p(zs, list(FIT_DS), nboot=10000, seed=A.BOOT_RNG, level=None)
        res.update(arm=arm, level="states+batches", d_fit=list(FIT_DS), n_states=int(zs[0]["n_states"]),
                   n_batch=int(zs[0]["n_batch"]), r_batch=int(zs[0]["r_batch"]), seconds=time.time() - t0,
                   provenance="computed during post-hoc audit from preregistered saved experimental data")
        out[f"{arm}_both"] = res
        print(f"  {arm} both-level: p={res['p_mean']:.4f} sd={res['p_sd']:.5f} CI={res['ci']} ({res['seconds']:.0f}s)", flush=True)
    zs = [load(d, "rank1") for d in FIT_DS]
    for level in ("states", "batches"):
        t0 = time.time()
        res = A.bootstrap_p(zs, list(FIT_DS), nboot=10000, seed=A.BOOT_RNG, level=level)
        res.update(arm="rank1", level=f"{level}-only", d_fit=list(FIT_DS), n_states=64, n_batch=40, r_batch=250,
                   seconds=time.time() - t0,
                   provenance="rerun at the registered 10 000 during post-hoc audit; out/bootstrap_p.json holds the 3 000-resample original")
        out[f"rank1_{level}_only"] = res
        print(f"  rank1 {level}-only: p={res['p_mean']:.4f} sd={res['p_sd']:.5f} CI={res['ci']} ({res['seconds']:.0f}s)", flush=True)
    with open(os.path.join(ART, "lqr_corrected_bootstrap.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    return out


if __name__ == "__main__":
    print("Rule A comparators"); ra = part_ruleA()
    print("cross term + Var_e/MSE_total at the crossover"); part_crossterm(ra)
    if "--bootstrap" in sys.argv:
        print("bootstraps at the registered 10 000"); part_bootstrap()
