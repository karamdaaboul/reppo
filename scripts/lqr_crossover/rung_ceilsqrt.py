"""The fourth Rule B rung, r(d) = ceil(sqrt d), and the complete four-rung adjudication.

    JAX_PLATFORMS=cpu python scripts/lqr_crossover/rung_ceilsqrt.py

Reads out/d{d}_rank_r{r}_M32_unit_H_identity_csd.npz (run note:
reports/lqr_rank_ceilsqrt_runnote.md), computes c*(d) by the registered estimator,
p_nominal (direct fit), p_RMS (direct fit of c* rescaled by omega_RMS/omega at the
crossover cell, from the realised field moments), p_omega_inf (algebraic relabelling
c*/sqrt r(d)), the registered 10 000-resample hierarchical bootstrap, the contamination
guards, and then adjudicates Rule B on all four rungs against the registered wording.
Writes reports/artifacts/lqr_ruleB_fourrungs.{csv,json}.
"""
from __future__ import annotations

import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
import scripts.lqr_crossover  # noqa: F401,E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scripts.lqr_crossover import OUT, analyze as A  # noqa: E402
from scripts.lqr_crossover.audit import rms_product, sha  # noqa: E402

ART = os.path.join(REPO_ROOT, "reports", "artifacts")
DS_ALL = (1, 2, 4, 8, 16, 32, 64)
FIT_DS = (2, 4, 8, 16, 32, 64)
R_OF = {d: math.ceil(math.sqrt(d)) for d in DS_ALL}


def path(d):
    return os.path.join(OUT, f"d{d}_rank_r{R_OF[d]}_M32_unit_H_identity_csd.npz")


def fit(ds, ys):
    return float(np.polyfit(np.log(ds), np.log(ys), 1)[0])


def main():
    rows, zs = [], {}
    for d in DS_ALL:
        z = A.load(path(d)); zs[d] = z
        c, ok = A.crossover_by_c(z)
        per = A.crossover_by_c(z, per_state=True)
        col = A.crossover_by_column(z)
        rm = rms_product(z)
        k = (np.log(rm.c) - np.log(c)).abs().idxmin()
        w_ratio = float(rm.loc[k, "omega_rms_over_omega"])
        rows.append(dict(
            d=d, rank=int(z["rank"]), rank_expected=R_OF[d], npz=os.path.basename(path(d)),
            sha256_12=sha(path(d)), git_sha=str(z["git_sha"])[:10], prereg_sha=str(z["prereg_sha"])[:10],
            n_states=int(z["n_states"]), N=int(z["n_batch"]) * int(z["r_batch"]),
            rho_closed=float(z["rho_closed"]), cond_H=float(z["cond_H"]), retries=int(z["retries"]),
            eps_frac=float(z["eps_frac"]),
            c_star=c, bracketed=bool(ok), frac_no_bracket=float(np.isnan(per).mean()),
            collapse_sd_log=float(np.nanstd(np.log(col))), cols_bracketed=int(np.isfinite(col).sum()),
            omega_rms_over_omega_at_crossover=w_ratio,
            c_star_rms=c * w_ratio, c_star_omega_inf=c / math.sqrt(R_OF[d]),
            product_rms_median_r05_3=float(rm[(rm.r_nom >= 0.5) & (rm.r_nom <= 3)].product_rms.median())))
        print(f"  d={d:2d} r={int(z['rank'])} c*={c:.4f} (bracketed {ok}, no-bracket {np.isnan(per).mean():.3f}, "
              f"collapse sd {np.nanstd(np.log(col)):.4f}) omega_RMS/omega={w_ratio:.4f} "
              f"rho={float(z['rho_closed']):.3f} cond_H={float(z['cond_H']):.2f} retries={int(z['retries'])}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ART, "lqr_ruleB_fourrungs_rung_ceilsqrt.csv"), index=False)

    # bit-identity with the existing rank_r2 arm where ceil(sqrt d) = 2 (d = 2, 4) and
    # with rank_r2's r = 1 case at d = 1 -- a provenance check, not a result
    for d in (1, 2, 4):
        zr = A.load(os.path.join(OUT, f"d{d}_rank_r2_M32_unit_H_identity.npz"))
        same = all(np.array_equal(zs[d][k], zr[k]) for k in ("s3_pw", "s3_zo", "s1_pw", "s1_zo"))
        print(f"  d={d}: identical to existing rank_r2 file: {same}")

    sel = [d for d in FIT_DS]
    p_nom = fit(sel, df.set_index("d").loc[sel, "c_star"])
    p_rms = fit(sel, df.set_index("d").loc[sel, "c_star_rms"])
    p_inf = fit(sel, df.set_index("d").loc[sel, "c_star_omega_inf"])
    shift = fit(sel, [1.0 / math.sqrt(R_OF[d]) for d in sel])      # OLS slope of -(1/2) ln r(d)
    print(f"\n  p_nominal = {p_nom:.4f}  (direct fit)")
    print(f"  p_RMS     = {p_rms:.4f}  (direct fit of c* x omega_RMS/omega)")
    print(f"  p_omega_inf = {p_inf:.4f}  = p_nominal + {shift:+.4f}  (algebraic: slope of -1/2 ln ceil(sqrt d) on ln d; "
          f"check {p_nom + shift:.4f})")

    # registered hierarchical bootstrap, 10 000, seed 20260902, FIT_DS
    B = A.bootstrap_p([zs[d] for d in sel], sel, nboot=10000, seed=A.BOOT_RNG, level=None)
    print(f"  bootstrap: p_nominal mean {B['p_mean']:.4f} sd {B['p_sd']:.5f} CI [{B['ci'][0]:.4f}, {B['ci'][1]:.4f}] n={B['nboot']}")
    ci_inf = (B["ci"][0] + shift, B["ci"][1] + shift)          # exact: the relabelling is a fixed per-d shift
    print(f"  p_omega_inf CI (rigid shift of the nominal CI): [{ci_inf[0]:.4f}, {ci_inf[1]:.4f}]")

    # ---- joint Rule B on all four rungs (registered sup-norm coordinate)
    prior = json.load(open(os.path.join(ART, "lqr_corrected_bootstrap.json")))
    orig = json.load(open(os.path.join(OUT, "bootstrap_p.json")))["both"]
    summ = json.load(open(os.path.join(ART, "lqr_audit_summary.json")))["ladder"]
    rungs = [
        dict(rung="r = 1", p_nominal=summ["rank1"]["p_nominal"], shift=0.0,
             ci_nominal=orig["ci"], cstars=summ["rank1"]["cstars"]),
        dict(rung="r = 2", p_nominal=summ["rank_r2"]["p_nominal"], shift=0.0,
             ci_nominal=prior["rank_r2_both"]["ci"], cstars=summ["rank_r2"]["cstars"]),
        dict(rung="r = ceil(sqrt d)", p_nominal=p_nom, shift=shift, ci_nominal=list(B["ci"]),
             cstars={str(d): float(df.set_index("d").loc[d, "c_star"]) for d in DS_ALL}),
        dict(rung="r = d", p_nominal=summ["full"]["p_nominal"], shift=-0.5,
             ci_nominal=prior["full_both"]["ci"], cstars=summ["full"]["cstars"]),
    ]
    verdict_rows = []
    for r in rungs:
        p = r["p_nominal"] + r["shift"]; lo, hi = r["ci_nominal"][0] + r["shift"], r["ci_nominal"][1] + r["shift"]
        cs = [r["cstars"][str(d)] for d in FIT_DS]
        mono = bool(np.all(np.diff(cs) > 0))
        verdict_rows.append(dict(rung=r["rung"], p_nominal=r["p_nominal"], p_omega_inf=p, ci_lo=lo, ci_hi=hi,
                                 in_confirm_band=bool(0.35 <= p <= 0.65), ci_excludes_0_and_1=bool(lo > 0 and hi < 1),
                                 in_linear_band=bool(0.8 <= p <= 1.2), c_star_monotone_in_d=mono))
    V = pd.DataFrame(verdict_rows)
    V.to_csv(os.path.join(ART, "lqr_ruleB_fourrungs.csv"), index=False)
    print("\n  Rule B, four rungs, registered sup-norm coordinate:")
    print(V.round(4).to_string(index=False))
    confirmed = bool(V.in_confirm_band.all() and V.ci_excludes_0_and_1.all())
    linear = bool(V.in_linear_band.all())
    ranges_disjoint = bool((V.ci_hi.min() < V.ci_lo.max()))
    third = bool((~V.in_confirm_band & ~V.in_linear_band).any() or (~V.c_star_monotone_in_d).any() or ranges_disjoint)
    verdict = "CONFIRMED" if confirmed else ("LINEAR" if linear else ("REFUTED" if third else "NONE OF THE BRANCHES"))
    print(f"\n  RULE B (joint, four rungs): {verdict}")
    out = dict(rung_ceilsqrt=dict(p_nominal=p_nom, p_rms=p_rms, p_omega_inf=p_inf, shift=shift,
                                  bootstrap=B, ci_omega_inf=ci_inf, fit_d=sel, n_states=32, n_batch=20, r_batch=100),
               verdict=verdict, all_in_confirm_band=confirmed, all_in_linear_band=linear,
               some_rung_outside_both=bool((~V.in_confirm_band & ~V.in_linear_band).any()),
               p_differs_across_rungs_beyond_cis=ranges_disjoint, any_nonmonotone=bool((~V.c_star_monotone_in_d).any()))
    with open(os.path.join(ART, "lqr_ruleB_fourrungs.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
