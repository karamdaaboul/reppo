#!/usr/bin/env python
"""Analysis for the J=4 independent replication and the J=8 extension.

Registered in docs/prereg_planted_j4_replication_j8.md (committed e0fdc06, before
either run).  Statistics and decision rules come from that document.

    ./.venv/bin/python scripts/planted/analyse_j4rep_j8.py
"""
import numpy as np, pandas as pd

ART = "reports/artifacts"
NB = 10000
W = 96

RUNS = {
    "old":  ("planted_multimode",       20260904),   # committed J=1/J=4
    "rep":  ("planted_j4_replication",  20260905),   # experiment A
    "j8":   ("planted_j8",              20260906),   # experiment B
}


def hdr(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


def load(tag):
    pre, seed = RUNS[tag]
    return (pd.read_csv(f"{ART}/{pre}_replicates.csv"),
            pd.read_csv(f"{ART}/{pre}.csv"), seed)


def boot_mean(x, seed):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    bs = x[rng.integers(0, len(x), size=(NB, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def cellwise_B(rep, J, seed):
    """Per-cell B with bootstrap CI over the 64 error-field blocks."""
    rows = []
    for (d, r), g in rep[rep.J == J].groupby(["d", "r_eff"]):
        m, lo, hi = boot_mean(g.B_b.to_numpy(), seed)
        rows.append(dict(d=d, r_eff=r, B=m, lo=lo, hi=hi,
                         B_med=float(np.median(g.B_b))))
    return pd.DataFrame(rows)


def decomp(cell, J, nm, ds=None):
    g = cell[cell.J == J] if ds is None else cell[(cell.J == J) & (cell.d.isin(ds))]
    fl = g[f"err_{nm}_baseline"].median()
    eo = g[f"err_{nm}_erroroff"].median()
    tt = g[f"err_{nm}_total"].median()
    sys_, noi = eo - fl, tt - eo
    return fl, sys_, noi, tt, (sys_ / noi if abs(noi) > 1e-12 else np.inf)


def ols(x, y):
    sl, ic = np.polyfit(x, y, 1)
    r2 = 1 - np.var(y - (ic + sl * x)) / np.var(y) if np.var(y) > 0 else np.nan
    return float(sl), float(r2), float(np.exp(-ic / sl))


def main():
    rep_o, cell_o, s_o = load("old")
    rep_r, cell_r, s_r = load("rep")
    rep_8, cell_8, s_8 = load("j8")

    # ------------------------------------------------------------ checks
    hdr("0. CODE-CORRECTNESS CHECKS (prereg Sec. 6) -- all must pass")
    ok = True
    for tag, (rp, cl, sd) in (("rep(J=4)", (rep_r, cell_r, s_r)),
                              ("j8 (J=8)", (rep_8, cell_8, s_8))):
        js = sorted(rp.J.unique())
        piv = rp.pivot_table(index=["d", "r_eff", "direction"], columns="J",
                             values="wml0_checksum")
        crn = float(np.abs(piv[js[0]] - piv[js[-1]]).max())
        orth = float(rp.orth_resid.max())
        an = (float(cl.A_eff_numeric_ratio.min()), float(cl.A_eff_numeric_ratio.max()))
        ln = (float(cl.L_eff_numeric_ratio.min()), float(cl.L_eff_numeric_ratio.max()))
        nf = float(cl.nonfinite.sum())
        eb = float(cl.eta_at_bound.max())
        ess = float(cl.ess_med.min())
        aeff = sorted(cl.A_eff.round(12).unique())
        good = (crn == 0 and orth < 1e-12 and an[0] > 0.999999 and ln[0] > 0.999999
                and nf == 0 and eb == 0 and ess > 2)
        ok &= good
        print(f"  {tag}:  J arms {js}   d {sorted(cl.d.unique())}")
        print(f"    1 orthonormality max|VV^T-I| = {orth:.2e}")
        print(f"    2 ||e||_inf numeric/analytic in [{an[0]:.6f}, {an[1]:.6f}]"
              f"   (analytic A_eff = {aeff})")
        print(f"    3 ||grad e||_inf numeric/analytic in [{ln[0]:.6f}, {ln[1]:.6f}]")
        print(f"    4 non-finite values = {nf:.0f}")
        print(f"    5 eta bracket hits = {eb:.4f}")
        print(f"    6 min median ESS = {ess:.2f} (floor 2); min ESS = {cl.ess_min.min():.2f}")
        print(f"    7 clean Q^pi independent of J (CRN checksum) = {crn:.2e}")
        print(f"    => {'PASS' if good else 'FAIL'}")
    print(f"\n  ALL CORRECTNESS CHECKS: {'PASS' if ok else 'FAIL -- stop here'}")

    # ------------------------------------------------------------ A
    hdr("1. EXPERIMENT A -- J=4 independent replication (seed 20260905 vs 20260904)")
    Bo = cellwise_B(rep_o, 4, s_o)
    Br = cellwise_B(rep_r, 4, s_r)
    mrg = Bo.merge(Br, on=["d", "r_eff"], suffixes=("_old", "_new"))
    print(f"  {'d':>3} {'r_eff':>6} | {'B old':>7} {'95% CI':>16} | {'B new':>7} "
          f"{'95% CI':>16} | {'% diff':>7} {'CIs overlap':>12}")
    for _, x in mrg.iterrows():
        pct = 100 * (x.B_new / x.B_old - 1)
        ov = "yes" if (x.lo_new <= x.hi_old and x.lo_old <= x.hi_new) else "no"
        print(f"  {int(x.d):>3} {x.r_eff:>6.2f} | {x.B_old:>7.4f} "
              f"[{x.lo_old:>6.4f},{x.hi_old:>6.4f}] | {x.B_new:>7.4f} "
              f"[{x.lo_new:>6.4f},{x.hi_new:>6.4f}] | {pct:>+6.1f}% {ov:>12}")
    b_old, b_new = Bo.B.median(), Br.B.median()
    print(f"\n  HEADLINE median across cells:  old {b_old:.4f}   new {b_new:.4f}   "
          f"diff {100*(b_new/b_old-1):+.1f}%")
    print(f"  registered band: new B_J4 in [0.138, 0.206]  ->  "
          f"{'IN BAND' if 0.138 <= b_new <= 0.206 else 'OUT OF BAND'}")
    print(f"  per-cell CI overlap: "
          f"{sum((mrg.lo_new<=mrg.hi_old)&(mrg.lo_old<=mrg.hi_new))}/{len(mrg)} cells")

    print("\n  decomposition, systematic/noise (medians over cells):")
    print(f"  {'run':>10} {'PW':>32} | {'E-step':>32}")
    print(f"  {'':>10} {'floor':>8}{'+sys':>8}{'+noise':>8}{'s/n':>8} | "
          f"{'floor':>8}{'+sys':>8}{'+noise':>8}{'s/n':>8}")
    qual = {}
    for tag, cl in (("old J=4", cell_o), ("new J=4", cell_r)):
        p = decomp(cl, 4, "pw"); w = decomp(cl, 4, "wml")
        qual[tag] = (p[4], w[4])
        print(f"  {tag:>10} {p[0]:>8.4f}{p[1]:>+8.4f}{p[2]:>+8.4f}{p[4]:>8.2f} | "
              f"{w[0]:>8.4f}{w[1]:>+8.4f}{w[2]:>+8.4f}{w[4]:>8.2f}")
    pw_ok = qual["new J=4"][0] < 1
    es_ok = qual["new J=4"][1] > 1
    print(f"\n  registered qualitative test: PW systematic/noise < 1 -> {pw_ok}; "
          f"E-step systematic/noise > 1 -> {es_ok}")
    replicated = pw_ok and es_ok and (0.138 <= b_new <= 0.206)
    print(f"  => J=4 {'INDEPENDENTLY REPLICATED' if replicated else 'see report'}")

    # ------------------------------------------------------------ B
    hdr("2. EXPERIMENT B -- J = 8")
    B8 = cellwise_B(rep_8, 8, s_8)
    g8 = rep_8[rep_8.J == 8]
    m, lo, hi = boot_mean(g8.B_b.to_numpy(), s_8)
    print(f"  B_J8 pooled over all blocks: mean {m:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
          f"median {np.median(g8.B_b):.4f}")
    print(f"  across-cell: mean {B8.B.mean():.4f}  median {B8.B.median():.4f}")
    print("\n  by d:")
    for d, g in g8.groupby("d"):
        mm, ll, hh = boot_mean(g.B_b.to_numpy(), s_8)
        print(f"    d={d:<3} B = {mm:.4f}  95% CI [{ll:.4f}, {hh:.4f}]  "
              f"median {np.median(g.B_b):.4f}")
    print("\n  by r_eff:")
    for r, g in g8.groupby("r_eff"):
        mm, ll, hh = boot_mean(g.B_b.to_numpy(), s_8)
        print(f"    r_eff={r:<5} B = {mm:.4f}  95% CI [{ll:.4f}, {hh:.4f}]")

    # ------------------------------------------------------------ trend
    hdr("3. TREND  J = 1 -> 4 -> 8   (restricted to d in {16,64}, the J=8 grid)")
    DJ8 = [16, 64]
    def restr(rep_, J):
        return rep_[(rep_.J == J) & (rep_.d.isin(DJ8))].B_b.to_numpy()
    tab = [("J=1 (old run)", restr(rep_o, 1), s_o),
           ("J=1 (rep run)", restr(rep_r, 1), s_r),
           ("J=1 (j8 run)", restr(rep_8, 1), s_8),
           ("J=4 (old run)", restr(rep_o, 4), s_o),
           ("J=4 (rep run)", restr(rep_r, 4), s_r),
           ("J=8", restr(rep_8, 8), s_8)]
    print(f"  {'arm':>15} {'B':>8} {'95% CI':>18} {'median':>8}")
    vals = {}
    for nm, v, sd in tab:
        mm, ll, hh = boot_mean(v, sd)
        vals[nm] = mm
        print(f"  {nm:>15} {mm:>8.4f} [{ll:>7.4f},{hh:>7.4f}] {np.median(v):>8.4f}")
    b1 = np.mean([vals["J=1 (old run)"], vals["J=1 (rep run)"], vals["J=1 (j8 run)"]])
    b4 = np.mean([vals["J=4 (old run)"], vals["J=4 (rep run)"]])
    b8 = vals["J=8"]
    print(f"\n  TREND (d in {{16,64}}):  J=1 {b1:.4f}  ->  J=4 {b4:.4f}  ->  J=8 {b8:.4f}")
    print(f"    B_J4/B_J1 = {b4/b1:.4f}   B_J8/B_J1 = {b8/b1:.4f}   "
          f"B_J8/B_J4 = {b8/b4:.4f}")

    # ---- how much is just the weaker RMS field? (prereg Sec. 5.2)
    print("\n  Under c_j = A0/J the RMS field falls as 1/sqrt(J):")
    print("    RMS_4/RMS_1 = 0.5000   RMS_8/RMS_1 = 0.3536   RMS_8/RMS_4 = 0.7071")
    try:
        amp = pd.read_csv(f"{ART}/planted_amplitude_replicates.csv")
        amp = amp[(amp.sigma == 0.4) & (amp.d.isin(DJ8))].copy()
        amp["B_b"] = amp.pre_wml_err_meannorm / amp.pre_wml_clean_meannorm
        mA = amp.groupby("amplitude").B_b.mean()
        p = float(np.polyfit(np.log(mA.index.values), np.log(mA.values), 1)[0])
        print(f"    committed amplitude experiment (d in {{16,64}}): B ~ A^p, "
              f"p = {p:.4f}")
        for J, meas in ((4, b4 / b1), (8, b8 / b1)):
            exp = (1 / np.sqrt(J)) ** p
            print(f"    J={J}: expected from weaker field alone {exp:.4f}, "
                  f"measured {meas:.4f}, residual {meas/exp:.4f}")
        print("    (residual 1.0 = fully explained by the weaker field; < 1 = an extra")
        print("     reduction attributable to mode structure)")
    except Exception as exc:
        print(f"    amplitude calibration skipped: {exc}")

    # ------------------------------------------------------------ decomposition J=8
    hdr("4. J = 8 DECOMPOSITION -- what hurts pathwise, and what hurts the E-step?")
    print(f"  {'op':>7} {'floor':>9} {'+systematic':>12} {'+noise':>10} {'= total':>9} "
          f"{'sys/noise':>10} {'verdict':>12}")
    print("  restricted to d in {16, 64}, the J=8 grid, so the arms are comparable")
    for J, cl, lbl in ((1, cell_o, "J=1 (old)"), (4, cell_o, "J=4 (old)"),
                       (4, cell_r, "J=4 (rep)"), (8, cell_8, "J=8")):
        print(f"  -- {lbl} --")
        for nm in ("pw", "zo", "wml"):
            fl, sy, no, tt, rt = decomp(cl, J, nm, ds=DJ8)
            verdict = "SYSTEMATIC" if rt > 1 else "NOISE"
            print(f"  {nm.upper():>7} {fl:>9.4f} {sy:>+12.4f} {no:>+10.4f} {tt:>9.4f} "
                  f"{rt:>10.2f} {verdict:>12}")

    # ------------------------------------------------------------ update quality
    hdr("5. UPDATE QUALITY  G = Err[E-step] - Err[PW]   (no crossover fitted)")
    print(f"  {'r_eff':>6} | {'G (J=4 old)':>12} {'G (J=4 rep)':>12} {'G (J=8)':>12} "
          f"{'who is better at J=8':>22}")
    for r in sorted(cell_8.r_eff.unique()):
        go = cell_o[(cell_o.J == 4) & (cell_o.r_eff == r) & (cell_o.d.isin(DJ8))].G.mean()
        gr = cell_r[(cell_r.J == 4) & (cell_r.r_eff == r) & (cell_r.d.isin(DJ8))].G.mean()
        g8v = cell_8[(cell_8.J == 8) & (cell_8.r_eff == r)].G.mean()
        who = "E-step" if g8v < 0 else "pathwise"
        print(f"  {r:>6.2f} | {go:>12.4f} {gr:>12.4f} {g8v:>12.4f} {who:>22}")

    # ------------------------------------------------------------ alignment
    hdr("6. WHAT THE SYSTEMATIC EFFECT DOES:  cos(delta_err, v_signal)")
    for tag, rp, J in (("J=4 old", rep_o, 4), ("J=4 rep", rep_r, 4), ("J=8", rep_8, 8)):
        g = rp[(rp.J == J) & (rp.d.isin(DJ8))]
        cs = (g.delta_dot_vsig / g.delta_norm)
        print(f"  {tag:>8}  median {cs.median():+.4f}  mean {cs.mean():+.4f}  "
              f"q05 {cs.quantile(.05):+.4f}  q95 {cs.quantile(.95):+.4f}  "
              f"frac<-0.9 {(cs < -0.9).mean():.3f}")
    print("\n  close to -1 => critic error mainly SHRINKS the useful E-step signal")

    # ------------------------------------------------------------ theory
    hdr("7. CENTRED ZO/PW THEORY CHECK -- r_eff vs r_nom  (appendix only)")
    print(f"  {'run':>10} {'J':>2} {'x-axis':>7} {'beta':>8} {'R^2':>8} {'crossing':>9}")
    for tag, cl in (("old", cell_o), ("rep", cell_r), ("j8", cell_8)):
        for J in sorted(cl.J.unique()):
            g = cl[cl.J == J].copy()
            g["R_var"] = g.var_zo_e / g.var_pw_e
            for xn, xv in (("r_eff", g.r_eff), ("r_nom", g.r_eff * np.sqrt(J))):
                sl, r2, xs = ols(np.log(xv.to_numpy()), np.log(g.R_var.to_numpy()))
                print(f"  {tag:>10} {J:>2} {xn:>7} {sl:>+8.4f} {r2:>8.4f} {xs:>9.4f}")
    print("\n  Claim 4 supplies bounds; this is consistent with their ratio, not implied")
    print("  by it.  Appendix only -- Claim 4 is not revised on this evidence.")


if __name__ == "__main__":
    main()
