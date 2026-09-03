"""Read-only audit of the LQR crossover study against docs/prereg_lqr_crossover.md.

Recomputes, from the .npz files the report cites, every statistic a registered rule
adjudicates on, plus the two things the report never printed: the registered
"anywhere on the grid" cross-term rule and the RMS-convention product
(Var_ZO_e / Var_PW_e) * r_RMS^2.  Nothing here modifies an estimator or an artifact.

    JAX_PLATFORMS=cpu python scripts/lqr_crossover/audit.py

Writes reports/artifacts/lqr_audit_{rules,crossterm,rms}.csv.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
import scripts.lqr_crossover  # noqa: F401,E402  (CPU + float64, asserted)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scripts.lqr_crossover import OUT, analyze as A, reference as R  # noqa: E402

ART = os.path.join(REPO_ROOT, "reports", "artifacts")
DS = (1, 2, 4, 6, 8, 16, 32, 64)
FIT_DS = (2, 4, 8, 16, 32, 64)          # registered primary d-set (prereg 5.4)
M = 32


def sha(path, n=12):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def path_of(d, arm):
    return os.path.join(OUT, f"d{d}_{arm}_M32_unit_H_identity.npz")


# ------------------------------------------------------------------ crossovers
def cstar_bootstrap(z, nboot=1000, seed=A.BOOT_RNG):
    """Hierarchical bootstrap (states, then batches) of c* for ONE d.  Gives the
    Monte-Carlo SE that Rule A's '4 SE of the closed form' clause needs and the report
    never printed."""
    rng = np.random.default_rng(seed)
    p3, z3 = z["s3_pw"], z["s3_zo"]
    WT = A.group_matrix(len(z["sigmas"]), len(z["omegas"])).T
    logc = np.log(A.c_grid(z))
    n_st, n_b = p3.shape[:2]
    out = []
    for _ in range(nboot):
        si = rng.integers(0, n_st, n_st)
        bi = rng.integers(0, n_b, (n_st, n_b))
        pw = p3[si[:, None], bi].mean(1)
        zo = z3[si[:, None], bi].mean(1)
        r = np.log(zo) - np.log(pw)
        out.append(A.solve_crossover(logc, (r.reshape(n_st, -1) @ WT).mean(0))[0])
    out = np.array(out)
    return float(np.nanstd(out, ddof=1)), int(np.isfinite(out).sum())


def closed_form_exact_aggregation(z, d):
    """Rule A's closed form aggregated the way the measurement is: theta_s(cell) =
    omega_j v^T mu_s + phi_s in every grid cell, the analytic log-ratio averaged over
    states and over the cells sharing c, then the root.  report.py instead takes the
    median over states of per-state analytic roots with theta = phi -- a different
    aggregate and a different theta convention."""
    from scipy.optimize import brentq
    phi, vt = np.asarray(z["phi"])[:, 0], np.asarray(z["vtmu"])[:, 0]
    sig, om = z["sigmas"], z["omegas"]

    def f(lc):
        c, vals = np.exp(lc), []
        for s_ in sig:
            w = c / s_
            if om[0] * 0.999 <= w <= om[-1] * 1.001:
                vals.append(np.mean(R.log_ratio(c, w * vt + phi, M, d)))
        return float(np.mean(vals))
    return float(np.exp(brentq(f, np.log(0.3), np.log(50), xtol=1e-12)))


def per_state_roots(z):
    """The registered wording: root-find per state BEFORE aggregation."""
    cs = A.crossover_by_c(z, per_state=True)
    return cs


# ------------------------------------------------------------------ cross term
def cross_term_all(z, c_star):
    """Both denominators, both restrictions.

    registered  : max over grid cells of |2Cov| / Var_e        (prereg 5.4, verbatim)
    at_crossover: |2Cov| / Var_e in the c-cell nearest c*      (what report.py prints)
    frac_total  : max over cells of |2Cov| / total MSE         (analyze.cross_share,
                                                                computed, never printed)
    """
    r = A.eps_over_sigma(z)[:, :, None]
    sig, om = z["sigmas"], z["omegas"]
    keep = sig >= 0.1
    C = sig[:, None] * om[None, :]
    k = np.unravel_index(np.argmin(np.abs(np.log(C) - np.log(c_star))), C.shape)
    out = {}
    for nm, pw in (("pw", True), ("zo", False)):
        s1 = z["s1_pw" if pw else "s1_zo"]
        s2 = z["s2_pw" if pw else "s2_zo"]
        s3 = z["s3_pw" if pw else "s3_zo"]
        s2d = np.stack([s2[:, :, i, i, :] for i in range(s2.shape[2])], axis=2)
        cross = (2.0 * r[:, None] * s2d).mean(1)               # (st, sig, om)
        err = ((r[:, None] ** 2) * s3).mean(1)
        tot = (s1[..., None] + 2.0 * r[:, None] * s2d + (r[:, None] ** 2) * s3).mean(1)
        # state-averaged numerator / state-averaged denominator, as cross_share does
        ratio_e = np.abs(cross).mean(0) / np.maximum(np.abs(err).mean(0), 1e-300)
        ratio_t = np.abs(cross).mean(0) / np.maximum(np.abs(tot).mean(0), 1e-300)
        out[f"{nm}_registered_allgrid"] = float(ratio_e.max())
        out[f"{nm}_registered_sig01"] = float(ratio_e[keep].max())
        out[f"{nm}_at_crossover"] = float(ratio_e[k])
        out[f"{nm}_frac_total_allgrid"] = float(ratio_t.max())
        out[f"{nm}_frac_total_sig01"] = float(ratio_t[keep].max())
        # where the registered maximum lives
        i, j = np.unravel_index(np.argmax(ratio_e), ratio_e.shape)
        out[f"{nm}_argmax_sigma"], out[f"{nm}_argmax_omega"] = float(sig[i]), float(om[j])
    out["crossover_cell_sigma"], out["crossover_cell_omega"] = float(sig[k[0]]), float(om[k[1]])
    return out


# ------------------------------------------------------------------ RMS product
def rms_product(z):
    """(Var_ZO_e / Var_PW_e) * r_RMS^2 per grid cell, with omega_RMS from the population
    moments of the planted field.  The sweep stores phi and v^T mu per state (theta_j =
    omega v^T mu_j + phi_j) but not the sampled u, so the moments are the exact
    population values over a ~ N(mu, sigma^2 I) given the realised phases, evaluated in
    closed form -- the same object the s3 statistics estimate against.

        E[e^2]/eps^2      = (1/r)[ sum_j (1/2 - 1/2 e^{-2c^2} cos 2th_j)
                                   + sum_{i!=j} e^{-c^2} sin th_i sin th_j ]
        E||grad e||^2/eps^2 = (omega^2/r) sum_j (1/2 + 1/2 e^{-2c^2} cos 2th_j)

    (orthonormal V, independent projections; eps cancels from every ratio below.)
    """
    sig, om = z["sigmas"], z["omegas"]
    d, rank = int(z["d"]), int(z["rank"])
    phi, vtmu = np.asarray(z["phi"]), np.asarray(z["vtmu"])       # (st, r)
    s3p, s3z = z["s3_pw"].mean(1), z["s3_zo"].mean(1)             # (st, sig, om)
    rows = []
    for i, sg in enumerate(sig):
        for j, w in enumerate(om):
            c = sg * w
            th = w * vtmu + phi                                     # (st, r)
            e2 = 2 * c * c
            s2 = (0.5 - 0.5 * np.exp(-e2) * np.cos(2 * th)).sum(-1)
            sn = np.sin(th)
            cross = np.exp(-c * c) * (sn.sum(-1) ** 2 - (sn ** 2).sum(-1))
            Ee2 = (s2 + cross) / rank                               # (st,)
            Eg2 = (w * w / rank) * (0.5 + 0.5 * np.exp(-e2) * np.cos(2 * th)).sum(-1)
            om_rms2 = Eg2.mean() / Ee2.mean()
            r_rms2 = sg * sg * om_rms2 / d
            ratio = s3z[:, i, j].mean() / s3p[:, i, j].mean()
            rows.append(dict(d=d, rank=rank, sigma=float(sg), omega=float(w), c=float(c),
                             r_nom=float(c / np.sqrt(d)), r_rms=float(np.sqrt(r_rms2)),
                             omega_rms_over_omega=float(np.sqrt(om_rms2) / w),
                             ratio=float(ratio), product_rms=float(ratio * r_rms2),
                             product_nom=float(ratio * c * c / d)))
    return pd.DataFrame(rows)


def main():
    rules, xrows, rms = [], [], []
    cst, cana, cse, cpsm, nobr, colsd, cb = {}, {}, {}, {}, {}, {}, {}
    for d in DS:
        p = path_of(d, "rank1")
        if not os.path.exists(p):
            continue
        z = A.load(p)
        h = sha(p)
        c_meas, ok = A.crossover_by_c(z)
        th = np.asarray(z["phi"])[:, 0]
        c_an = float(np.median([R.crossover_c_star(d, float(t), M) for t in th]))
        se, nb = cstar_bootstrap(z)
        ps = per_state_roots(z)
        col = A.crossover_by_column(z)
        pw, zo = A.full_mse(z)
        rr = A.log_ratio_by_c(pw, zo, len(z["sigmas"]), len(z["omegas"]))
        c_b, okb = A.solve_crossover(np.log(A.c_grid(z)), rr.mean(0))
        c_ex = closed_form_exact_aggregation(z, d)
        cst[d], cana[d], cse[d], cb[d] = c_meas, c_an, se, c_b
        cpsm[d] = float(np.nanmedian(ps)); nobr[d] = float(np.isnan(ps).mean())
        colsd[d] = float(np.nanstd(np.log(col)))
        rules.append(dict(d=d, npz=os.path.basename(p), sha256_12=h,
                          n_states=int(z["n_states"]),
                          N=int(z["n_batch"]) * int(z["r_batch"]),
                          c_star_state_avg=c_meas, c_star_per_state_median=cpsm[d],
                          c_star_per_state_mean=float(np.nanmean(ps)),
                          frac_states_no_bracket=nobr[d],
                          c_star_closed_form=c_an, c_star_boot_se=se,
                          abs_dev_over_se=abs(c_meas - c_an) / se,
                          c_closed_exact_agg=c_ex,
                          abs_dev_over_se_exact_agg=abs(c_meas - c_ex) / se,
                          collapse_sd_log=colsd[d],
                          cols_bracketed=int(np.isfinite(col).sum()),
                          c_star_E1b=c_b, ratio_E1b_E1a=c_b / c_meas))
        xt = cross_term_all(z, c_meas); xt.update(d=d, npz=os.path.basename(p), sha256_12=h)
        xrows.append(xt)
        rf = rms_product(z); rf["arm"] = "rank1"; rf["npz"] = os.path.basename(p); rms.append(rf)
        print(f"d={d:2d} c*={c_meas:.4f} closed={c_an:.4f} se={se:.2e} "
              f"|dev|/se={abs(c_meas-c_an)/se:.2f} per-state med={cpsm[d]:.4f} "
              f"nobracket={nobr[d]:.3f} E1b={c_b:.3f}", flush=True)

    # rank ladder arms: exponents as report.py computes them, plus RMS product
    ladder = {}
    for arm in ("rank1", "rank_r2", "full"):
        cs, dl = [], []
        for d in (1, 2, 4, 8, 16, 32, 64):
            p = path_of(d, arm)
            if not os.path.exists(p):
                continue
            z = A.load(p)
            c, ok = A.crossover_by_c(z)
            cs.append(c if ok else np.nan); dl.append(d)
            if arm != "rank1":
                rf = rms_product(z); rf["arm"] = arm; rf["npz"] = os.path.basename(p); rms.append(rf)
        sel = [i for i, d in enumerate(dl) if d in FIT_DS]
        pn = A.fit_p([dl[i] for i in sel], np.array(cs)[sel])
        # the omega_inf column in report.py is pn + (-0.5 if arm == "full" else 0.0)
        ladder[arm] = dict(p_nominal=pn, p_omega_inf=pn + (-0.5 if arm == "full" else 0.0),
                           cstars=dict(zip(dl, cs)))
        print(f"ladder {arm:8s} p_nom={pn:.4f}  p_inf={ladder[arm]['p_omega_inf']:.4f}")

    fit = lambda ds_: A.fit_p(ds_, np.array([cst[d] for d in ds_]))
    summary = dict(
        p_E1a_fitds=fit([d for d in FIT_DS if d in cst]),
        p_E1a_alld=fit([d for d in DS if d in cst and d != 6]),
        p_E1a_closed=A.fit_p([d for d in FIT_DS if d in cana], np.array([cana[d] for d in FIT_DS if d in cana])),
        p_E1b_fitds=A.fit_p([d for d in FIT_DS if d in cb], np.array([cb[d] for d in FIT_DS if d in cb])),
        ladder=ladder,
        predicted_product_deattenuated=M / (M - 1.0),
        predicted_product_plain=(M - 1.0) / M,
        predicted_rstar_rms=float(np.sqrt(M / (M - 1.0))),
    )
    RU = pd.DataFrame(rules)
    RU.to_csv(os.path.join(ART, "lqr_audit_rules.csv"), index=False)
    RU[RU.d != 6][["d", "c_star_state_avg", "c_closed_exact_agg", "c_star_boot_se",
                   "abs_dev_over_se_exact_agg"]].rename(columns={
        "c_star_state_avg": "c_meas", "c_star_boot_se": "se",
        "abs_dev_over_se_exact_agg": "dev_over_se"}).assign(
        rel=lambda x: (x.c_meas / x.c_closed_exact_agg - 1).abs())[
        ["d", "c_meas", "c_closed_exact_agg", "rel", "se", "dev_over_se"]].to_csv(
        os.path.join(ART, "lqr_audit_ruleA.csv"), index=False)
    pd.DataFrame(xrows).to_csv(os.path.join(ART, "lqr_audit_crossterm.csv"), index=False)
    RM = pd.concat(rms, ignore_index=True)
    RM.to_csv(os.path.join(ART, "lqr_audit_rms.csv"), index=False)
    with open(os.path.join(ART, "lqr_audit_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)

    print("\n=== exponents ===")
    for k, v in summary.items():
        if k != "ladder":
            print(f"  {k}: {v}")
    print("\n=== cross term ===")
    X = pd.DataFrame(xrows)
    print(X[["d", "pw_registered_allgrid", "zo_registered_allgrid", "pw_registered_sig01",
             "zo_registered_sig01", "pw_at_crossover", "zo_at_crossover",
             "pw_frac_total_sig01", "zo_frac_total_sig01"]].round(4).to_string(index=False))
    print("\n=== RMS product (Var_ZO/Var_PW) * r_RMS^2, cells with r_nom in [0.5, 3] ===")
    S = RM[(RM.r_nom >= 0.5) & (RM.r_nom <= 3.0)]
    print(S.groupby(["arm", "d"]).product_rms.agg(["min", "median", "max", "count"]).round(4).to_string())
    print("\n  all sigma>=0.1 cells:")
    S2 = RM[RM.sigma >= 0.1]
    print(S2.groupby("arm").product_rms.agg(["min", "median", "max"]).round(4).to_string())
    print(f"\n  predicted: M/(M-1) = {M/(M-1):.5f} (deattenuated ZO, this study); "
          f"(M-1)/M = {(M-1)/M:.5f} (plain centred ZO, planted sweep)")
    print("  omega_RMS/omega at each d's crossover cell (rank1):")
    for d in DS:
        if d in cst:
            g = RM[(RM.arm == "rank1") & (RM.d == d)]
            k = (np.log(g.c) - np.log(cst[d])).abs().idxmin()
            print(f"    d={d:2d}: {g.loc[k, 'omega_rms_over_omega']:.4f} at c={g.loc[k,'c']:.3f}")


if __name__ == "__main__":
    main()
