"""Empirical variance-scaling law from pilot 1, used to size pilot 2.

This is a POWER CALCULATION on completed, already-reported pilot-1 data. It chooses
the rollout count R for pilot 2 and nothing else; every other design constant is
carried over unchanged.

Method: recompute the preregistered estimators using only R' of the 8 rollouts per
group, for R' in {1, 2, 4, 8}, averaging over the disjoint sub-blocks available at
each R' (8 at R'=1, 4 at R'=2, 2 at R'=4, 1 at R'=8). Bootstrap the sampling standard
error at each R', then fit

    se^2(R) = v_state + v_noise / R

by weighted least squares. ``v_state`` is the irreducible state-heterogeneity floor
that more rollouts can never cross; ``v_noise`` is what rollouts buy. Extrapolating
that fit gives the R needed to satisfy criteria A, B and E.

Usage: mc_oracle_power.py <pw_pilot.npz> <wml_pilot.npz> <out.json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scripts.analysis.mc_oracle_analyse import (  # noqa: E402
    C_STEPS, boot_indices, fd_index, load_run, r_rms, stat_D, stat_N,
)

N_BOOT = 2000
SEED = 20260902
RPRIME = (1, 2, 4, 8)
# criterion targets, from prereg 1 section 11 (unchanged in prereg 2)
Z = 1.96
E_REL_WIDTH = 0.60          # width(CI) <= 0.60 * point  =>  se <= 0.153 * point


def est_from(q, qphi, fdp, d, sl):
    """Estimators using rollout slice ``sl`` of each group."""
    eA = qphi - q[0][sl].mean(0)
    eB = qphi - q[1][sl].mean(0)
    return eA, eB


def se_at(r, rprime, n_boot=N_BOOT):
    """Bootstrap SE of D, N_c and r_c using rprime rollouts per group."""
    q, qphi = r["q_oracle"], r["q_phi"]
    d = int(r["mu"].shape[1])
    K = qphi.shape[1]
    fdp = {c: fd_index(r, c) for c in C_STEPS}
    sidx, kidx = boot_indices(r["source"], K, n_boot, SEED)

    keys = ["D"] + ["N_%.2f" % c for c in C_STEPS] + ["r_%.2f" % c for c in C_STEPS]
    acc = {k: [] for k in keys}
    point = {k: [] for k in keys}
    n_block = 8 // rprime
    for b in range(n_block):
        sl = slice(b * rprime, (b + 1) * rprime)
        eA, eB = est_from(q, qphi, fdp, d, sl)
        pt = {"D": stat_D(eA, eB)}
        for c in C_STEPS:
            p, m = fdp[c]
            pt["N_%.2f" % c] = stat_N(eA, eB, p, m, c)
            pt["r_%.2f" % c] = r_rms(pt["N_%.2f" % c], pt["D"], d)
        for k in keys:
            point[k].append(pt[k])

        draws = {k: np.empty(n_boot) for k in keys}
        for i in range(n_boot):
            si = sidx[i]
            ki = kidx[i]
            aa = np.take_along_axis(eA[si], ki[..., None], axis=1)
            bb = np.take_along_axis(eB[si], ki[..., None], axis=1)
            D = stat_D(aa, bb)
            draws["D"][i] = D
            for c in C_STEPS:
                p, m = fdp[c]
                N = stat_N(aa, bb, p, m, c)
                draws["N_%.2f" % c][i] = N
                draws["r_%.2f" % c][i] = r_rms(N, D, d)
        for k in keys:
            v = draws[k][np.isfinite(draws[k])]
            acc[k].append(float(v.std(ddof=1)) if v.size > 2 else np.nan)
    return ({k: float(np.nanmean(v)) for k, v in acc.items()},
            {k: float(np.nanmean(v)) for k, v in point.items()})


def fit_scaling(rs, ses):
    """Least squares for se^2(R) = v_state + v_noise / R, both constrained >= 0."""
    rs = np.asarray(rs, float); y = np.asarray(ses, float) ** 2
    ok = np.isfinite(y)
    rs, y = rs[ok], y[ok]
    if len(rs) < 2:
        return np.nan, np.nan
    A = np.stack([np.ones_like(rs), 1.0 / rs], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    v_state, v_noise = float(coef[0]), float(coef[1])
    if v_state < 0:                      # noise-dominated: refit through the origin
        v_noise = float((y * (1.0 / rs)).sum() / ((1.0 / rs) ** 2).sum())
        v_state = 0.0
    if v_noise < 0:
        v_noise = 0.0
        v_state = float(y.mean())
    return v_state, v_noise


def r_needed(v_state, v_noise, target_se):
    """Smallest R with sqrt(v_state + v_noise/R) <= target_se, or None if unreachable."""
    if not np.isfinite(target_se) or target_se <= 0:
        return None
    if v_state >= target_se ** 2:
        return None                       # state-heterogeneity floor blocks it
    if v_noise <= 0:
        return 1
    return v_noise / (target_se ** 2 - v_state)


def main(pw, wml, out):
    res = {}
    for tag, path in (("PW", pw), ("WML", wml)):
        r = load_run(path)
        d = int(r["mu"].shape[1])
        curves, points = {}, {}
        for rp in RPRIME:
            s, p = se_at(r, rp)
            curves[rp], points[rp] = s, p
            print("%-4s R'=%d  " % (tag, rp) + "  ".join(
                "%s se=%.4g pt=%.4g" % (k, s[k], p[k]) for k in ("D", "N_0.10", "r_0.10")),
                flush=True)
        entry = {"se_by_R": {str(k): v for k, v in curves.items()},
                 "point_by_R": {str(k): v for k, v in points.items()},
                 "fit": {}, "R_needed": {}}
        for k in ("D", "N_0.10", "N_0.05", "r_0.10", "r_0.05"):
            vs, vn = fit_scaling(RPRIME, [curves[rp][k] for rp in RPRIME])
            entry["fit"][k] = {"v_state": vs, "v_noise": vn,
                               "se_inf": float(np.sqrt(vs)),
                               "se_at_8": curves[8][k]}
            pt8 = points[8][k]
            if k.startswith("r"):
                target = E_REL_WIDTH * pt8 / (2 * Z)        # criterion E
                crit = "E"
            else:
                target = abs(pt8) / Z                       # criteria A / B
                crit = "A" if k == "D" else "B"
            entry["R_needed"][k] = {
                "criterion": crit, "point_at_R8": pt8, "target_se": target,
                "R": r_needed(vs, vn, target),
                "blocked_by_state_floor": bool(vs >= target ** 2)
                if np.isfinite(target) and target > 0 else None}
        res[tag] = entry

    print()
    for tag in ("PW", "WML"):
        print("=====", tag)
        for k, v in res[tag]["fit"].items():
            n = res[tag]["R_needed"][k]
            print("  %-8s se(8)=%-10.4g se(inf)=%-10.4g  crit %s target_se=%-10.4g "
                  "R_needed=%s%s" % (
                      k, v["se_at_8"], v["se_inf"], n["criterion"], n["target_se"],
                      ("%.1f" % n["R"]) if n["R"] else "UNREACHABLE",
                      "  <- state floor" if n["blocked_by_state_floor"] else ""))
    with open(out, "w") as f:
        json.dump(res, f, indent=1,
                  default=lambda o: None if o is None or not np.isfinite(o) else float(o))


if __name__ == "__main__":
    main(*sys.argv[1:4])
