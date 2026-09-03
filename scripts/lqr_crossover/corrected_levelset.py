"""Cross-term and Var_e / MSE_total ALONG the crossover level set c = c*(d).

analyze.cross_share picks the cell nearest c* with np.argmin over the flattened grid.
Every cell on the level set c = sigma*omega = c* is equally near, so argmin returns the
first in row-major order: the smallest-sigma cell.  The report's "at crossover" values
are therefore single-cell values at sigma ~ 0.01, outside the policy-reachable region
sigma >= 0.1.  This script evaluates the same ratios at every cell of the level set,
and reports the reachable and unreachable parts separately.

    JAX_PLATFORMS=cpu python scripts/lqr_crossover/corrected_levelset.py
Writes reports/artifacts/lqr_corrected_levelset.csv (one row per d x level-set cell).
"""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
import scripts.lqr_crossover  # noqa: F401,E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scripts.lqr_crossover import OUT, analyze as A  # noqa: E402

ART = os.path.join(REPO_ROOT, "reports", "artifacts")
SIGMA_MIN_POLICY = 0.1

rows = []
for d in (1, 2, 4, 8, 16, 32, 64):
    z = A.load(os.path.join(OUT, f"d{d}_rank1_M32_unit_H_identity.npz"))
    c_star, _ = A.crossover_by_c(z)
    r = A.eps_over_sigma(z)[:, :, None]
    sig, om = z["sigmas"], z["omegas"]
    C = sig[:, None] * om[None, :]
    dist = np.abs(np.log(C) - np.log(c_star))
    # the level set: cells at the minimal distance (all share the same c)
    level = np.argwhere(np.isclose(dist, dist.min(), atol=1e-9))
    first = tuple(np.unravel_index(np.argmin(dist), C.shape))
    stats = {}
    for nm, pw in (("pw", True), ("zo", False)):
        s1 = z["s1_pw" if pw else "s1_zo"]; s2 = z["s2_pw" if pw else "s2_zo"]; s3 = z["s3_pw" if pw else "s3_zo"]
        s2d = np.stack([s2[:, :, i, i, :] for i in range(s2.shape[2])], axis=2)
        cross_s = (2.0 * r[:, None] * s2d).mean(1)                 # (states, sig, om), batch mean
        err_s = ((r[:, None] ** 2) * s3).mean(1)
        smooth_s = (s1[..., None] * np.ones_like(s3)).mean(1)
        tot_s = smooth_s + cross_s + err_s
        # per-state reading (Claim 4 is a per-state statement; this is what report.py and
        # the registered rule's |2 Cov| mean): mean_s |2Cov_s| / mean_s Var_e,s
        stats[nm] = (np.abs(cross_s).mean(0), err_s.mean(0), smooth_s.mean(0), np.abs(tot_s).mean(0),
                     np.abs(cross_s.mean(0)))                       # last: state-pooled |mean_s 2Cov_s|
    for (i, j) in level:
        row = dict(d=d, c_star=c_star, c_cell=float(C[i, j]), sigma=float(sig[i]), omega=float(om[j]),
                   policy_reachable=bool(sig[i] >= SIGMA_MIN_POLICY),
                   is_argmin_cell=bool((i, j) == first))
        for nm in ("pw", "zo"):
            cross, err, smooth, tot, cross_pooled = stats[nm]
            row[f"{nm}_2Cov_over_VarE"] = float(cross[i, j] / err[i, j])              # per-state reading
            row[f"{nm}_2Cov_over_MSE"] = float(cross[i, j] / tot[i, j])
            row[f"{nm}_VarE_over_MSE"] = float(err[i, j] / tot[i, j])
            row[f"{nm}_smooth_over_MSE"] = float(smooth[i, j] / tot[i, j])
            row[f"{nm}_2Cov_pooled_over_VarE"] = float(cross_pooled[i, j] / err[i, j])  # |mean_s 2Cov_s|
        rows.append(row)
    L = pd.DataFrame([x for x in rows if x["d"] == d])
    rc = L[L.policy_reachable]
    print(f"d={d:2d} c*={c_star:.3f} level set {len(L)} cells ({len(rc)} reachable); argmin cell sigma={sig[first[0]]:.4f} "
          f"| ZO |2Cov|/Var_e: argmin {L[L.is_argmin_cell].zo_2Cov_over_VarE.iloc[0]:.3f}, reachable med {rc.zo_2Cov_over_VarE.median():.3f} max {rc.zo_2Cov_over_VarE.max():.3f} "
          f"| PW: argmin {L[L.is_argmin_cell].pw_2Cov_over_VarE.iloc[0]:.3f}, reachable med {rc.pw_2Cov_over_VarE.median():.3f} max {rc.pw_2Cov_over_VarE.max():.3f} "
          f"| Var_e/MSE reachable med PW {rc.pw_VarE_over_MSE.median():.3f} ZO {rc.zo_VarE_over_MSE.median():.3f}", flush=True)

pd.DataFrame(rows).to_csv(os.path.join(ART, "lqr_corrected_levelset.csv"), index=False)
print("wrote", os.path.join(ART, "lqr_corrected_levelset.csv"))
