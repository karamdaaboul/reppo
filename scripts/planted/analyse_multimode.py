#!/usr/bin/env python
"""Analysis for the J = 1 vs J = 4 multi-mode planted-error experiment.

Statistics, bootstrap unit and interpretation rules are fixed in
docs/prereg_planted_multimode.md (committed 44c86a9, before the run).

    ./.venv/bin/python scripts/planted/analyse_multimode.py
"""
import numpy as np, pandas as pd

ART = "reports/artifacts"
B = 10000
BOOT_SEED = 20260904
W = 96


def hdr(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


def boot_ci(x, stat=np.mean, seed=BOOT_SEED):
    """Bootstrap over the block axis (axis 0)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    bs = stat(x[idx], axis=1)
    return float(stat(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired_boot_ci(num, den, seed=BOOT_SEED):
    """Paired bootstrap of mean(num/den) resampling BLOCKS (shared index)."""
    rng = np.random.default_rng(seed)
    ratio = np.asarray(num) / np.asarray(den)
    idx = rng.integers(0, len(ratio), size=(B, len(ratio)))
    bs = ratio[idx].mean(1)
    return float(ratio.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def ols(x, y):
    sl, ic = np.polyfit(x, y, 1)
    pred = ic + sl * x
    r2 = 1 - np.var(y - pred) / np.var(y) if np.var(y) > 0 else np.nan
    return float(sl), float(ic), float(r2), float(np.exp(-ic / sl))


def main():
    rep = pd.read_csv(f"{ART}/planted_multimode_replicates.csv")
    cell = pd.read_csv(f"{ART}/planted_multimode.csv")
    cells = sorted(cell.groupby(["d", "r_eff"]).groups.keys())

    # ---------------------------------------------------------------- 0
    hdr("0. VALIDATION (prereg Sec. 5)")
    piv = rep.pivot_table(index=["d", "r_eff", "direction"], columns="J",
                          values="wml0_checksum")
    print(f"  CRN  max |wml0 checksum J1-J4| = {np.abs(piv[1]-piv[4]).max():.3e}"
          f"   (0 = clean channel identical across arms)")
    print(f"  frame orthonormality max ||VV^T - I|| = {rep.orth_resid.max():.3e}")
    print(f"  sup-norm numeric/analytic  A_eff [{cell.A_eff_numeric_ratio.min():.6f}, "
          f"{cell.A_eff_numeric_ratio.max():.6f}]   L_eff "
          f"[{cell.L_eff_numeric_ratio.min():.6f}, {cell.L_eff_numeric_ratio.max():.6f}]")
    for J, g in cell.groupby("J"):
        print(f"  J={J}  min median ESS {g.ess_med.min():6.2f}   min ESS "
              f"{g.ess_min.min():6.3f}   eta-at-bound {g.eta_at_bound.max():.4f}   "
              f"nonfinite {g.nonfinite.sum():.0f}   A_eff {g.A_eff.unique()}")

    # ---------------------------------------------------------------- 1
    hdr("1. PRIMARY -- conditional systematic E-step displacement  B = ||delta||/||s||")
    print(f"  {'d':>3} {'r_eff':>6} | {'B_J1':>8} {'95% CI':>18} | {'B_J4':>8} "
          f"{'95% CI':>18} | {'B_J4/B_J1':>10} {'paired 95% CI':>18}")
    rows = []
    for (d, r) in cells:
        s1 = rep[(rep.d == d) & (rep.r_eff == r) & (rep.J == 1)].sort_values("direction")
        s4 = rep[(rep.d == d) & (rep.r_eff == r) & (rep.J == 4)].sort_values("direction")
        b1, l1, h1 = boot_ci(s1.B_b.to_numpy())
        b4, l4, h4 = boot_ci(s4.B_b.to_numpy())
        rt, lr, hr = paired_boot_ci(s4.B_b.to_numpy(), s1.B_b.to_numpy())
        rows.append(dict(d=d, r_eff=r, B1=b1, B1_lo=l1, B1_hi=h1, B4=b4, B4_lo=l4,
                         B4_hi=h4, ratio=rt, ratio_lo=lr, ratio_hi=hr,
                         B1_med=float(np.median(s1.B_b)), B4_med=float(np.median(s4.B_b)),
                         B1_q25=float(np.percentile(s1.B_b, 25)),
                         B1_q75=float(np.percentile(s1.B_b, 75)),
                         B4_q25=float(np.percentile(s4.B_b, 25)),
                         B4_q75=float(np.percentile(s4.B_b, 75))))
        print(f"  {d:>3} {r:>6.2f} | {b1:>8.4f} [{l1:>7.4f},{h1:>7.4f}] | "
              f"{b4:>8.4f} [{l4:>7.4f},{h4:>7.4f}] | {rt:>10.4f} "
              f"[{lr:>7.4f},{hr:>7.4f}]")
    P = pd.DataFrame(rows)
    P.to_csv(f"{ART}/planted_multimode_primary.csv", index=False)
    print(f"\n  HEADLINE  median across the {len(P)} cells:")
    print(f"    B_J1 = {P.B1.median():.4f}   B_J4 = {P.B4.median():.4f}   "
          f"B_J4/B_J1 = {P.ratio.median():.4f}")
    print(f"    mean across cells: B_J1 = {P.B1.mean():.4f}  B_J4 = {P.B4.mean():.4f}  "
          f"ratio = {P.ratio.mean():.4f}")
    print(f"    per-cell ratio range [{P.ratio.min():.4f}, {P.ratio.max():.4f}]; "
          f"cells whose paired CI excludes 1: "
          f"{int(((P.ratio_hi < 1) | (P.ratio_lo > 1)).sum())}/{len(P)}")
    print("\n  across-block distribution of B_b (pooled over cells):")
    for J in (1, 4):
        s = rep[rep.J == J].B_b
        print(f"    J={J}  mean {s.mean():.4f}  median {s.median():.4f}  "
              f"q25 {s.quantile(.25):.4f}  q75 {s.quantile(.75):.4f}  "
              f"min {s.min():.4f}  max {s.max():.4f}")

    # ---------------------------------------------------------------- 2
    hdr("2. SECONDARY -- conditional error-channel noise  N = trCov_action/||s||^2")
    print(f"  {'d':>3} {'r_eff':>6} | {'N_J1':>10} | {'N_J4':>10} | {'N_J4/N_J1':>10}")
    nrows = []
    for (d, r) in cells:
        s1 = rep[(rep.d == d) & (rep.r_eff == r) & (rep.J == 1)].sort_values("direction")
        s4 = rep[(rep.d == d) & (rep.r_eff == r) & (rep.J == 4)].sort_values("direction")
        n1, n4 = s1.N_b.mean(), s4.N_b.mean()
        rt, lr, hr = paired_boot_ci(s4.N_b.to_numpy(), s1.N_b.to_numpy())
        nrows.append(dict(d=d, r_eff=r, N1=n1, N4=n4, ratio=rt, lo=lr, hi=hr))
        print(f"  {d:>3} {r:>6.2f} | {n1:>10.4f} | {n4:>10.4f} | {rt:>10.4f}")
    N = pd.DataFrame(nrows)
    print(f"\n  HEADLINE  median N_J1 = {N.N1.median():.4f}   N_J4 = {N.N4.median():.4f}"
          f"   N_J4/N_J1 = {N.ratio.median():.4f}")
    print(f"  Does J=4 change systematic displacement, noise, or both?")
    print(f"    B ratio (systematic) median = {P.ratio.median():.4f}")
    print(f"    N ratio (noise)      median = {N.ratio.median():.4f}")

    # ---------------------------------------------------------------- 3
    hdr("3. OPERATIONAL -- G = Err[E-step] - Err[PW] vs r_eff  (no crossover fitted)")
    print(f"  {'r_eff':>6} | {'G(J=1)':>9} {'95% CI':>18} | {'G(J=4)':>9} "
          f"{'95% CI':>18} | {'shrinks?':>9}")
    grows = []
    for r in sorted(cell.r_eff.unique()):
        s1 = rep[(rep.r_eff == r) & (rep.J == 1)]
        s4 = rep[(rep.r_eff == r) & (rep.J == 4)]
        g1, a1, b1_ = boot_ci(s1.G.to_numpy())
        g4, a4, b4_ = boot_ci(s4.G.to_numpy())
        tag = "yes" if b4_ < a1 else ("grows" if a4 > b1_ else "overlap")
        grows.append(tag)
        print(f"  {r:>6.2f} | {g1:>9.4f} [{a1:>7.4f},{b1_:>7.4f}] | {g4:>9.4f} "
              f"[{a4:>7.4f},{b4_:>7.4f}] | {tag:>9}")
    print(f"\n  summary: shrinks in {grows.count('yes')}/{len(grows)} r_eff values, "
          f"grows in {grows.count('grows')}, overlapping in {grows.count('overlap')}")

    # ---------------------------------------------------------------- 4
    hdr("4. COUNTERFACTUAL DECOMPOSITION (existing procedure, unchanged)")
    for J in (1, 4):
        g = cell[cell.J == J]
        print(f"\n  J = {J}   (medians over the 24 cells)")
        print(f"  {'op':5s} {'floor':>9} {'+err MEAN':>11} {'+err NOISE':>11} "
              f"{'= total':>9} {'sys/noise':>10}")
        for nm in ("pw", "zo", "wml"):
            fl = g[f"err_{nm}_baseline"].median()
            eo = g[f"err_{nm}_erroroff"].median()
            tt = g[f"err_{nm}_total"].median()
            mean_c, noise_c = eo - fl, tt - eo
            ratio = mean_c / noise_c if abs(noise_c) > 1e-12 else np.inf
            print(f"  {nm.upper():5s} {fl:>9.4f} {mean_c:>+11.4f} {noise_c:>+11.4f} "
                  f"{tt:>9.4f} {ratio:>10.2f}")
    print("\n  the J=1 pattern was: PW dominated by error-channel NOISE,")
    print("  E-step dominated by systematic error-channel MEAN.  Does it survive J=4?")

    # ---------------------------------------------------------------- 5
    hdr("5. CROSS-FIELD CANCELLATION (prereg Sec. 4.5)")
    print(f"  {'J':>2} {'E_field||delta||':>17} {'||E_field delta||':>18} "
          f"{'pooled/conditional':>19} {'E[delta.v_sig]':>15}")
    for J in (1, 4):
        g = cell[cell.J == J]
        print(f"  {J:>2} {g.delta_norm_conditional.median():>17.5f} "
              f"{g.delta_norm_pooled.median():>18.5f} "
              f"{g.cancellation_ratio.median():>19.4f} "
              f"{g.delta_dot_vsig.median():>15.5f}")
    print("\n  A small pooled/conditional ratio means independent error fields cancel at")
    print("  the population level while each fixed field still displaces the update.")

    # ---------------------------------------------------------------- 6
    hdr("6. CENTRED ZO/PW CHECK vs r_eff (secondary)")
    for J in (1, 4):
        g = cell[cell.J == J].copy()
        g["R_var"] = g.var_zo_e / g.var_pw_e
        if g.r_eff.nunique() < 3:
            print(f"  J={J}: too few distinct r_eff values to fit ({g.r_eff.nunique()})")
            continue
        sl, ic, r2, xs = ols(np.log(g.r_eff), np.log(g.R_var))
        rng = np.random.default_rng(BOOT_SEED + 5)
        x, y = np.log(g.r_eff.to_numpy()), np.log(g.R_var.to_numpy())
        bs = np.array([np.polyfit(x[i], y[i], 1)[0]
                       for i in rng.integers(0, len(x), size=(2000, len(x)))])
        lo_ = np.where(g.r_eff < 1, g.R_var < 1, False).sum()
        hi_ = np.where(g.r_eff > 1, g.R_var < 1, False).sum()
        print(f"  J={J}: beta = {sl:+.4f}  95% CI [{np.percentile(bs,2.5):+.4f}, "
              f"{np.percentile(bs,97.5):+.4f}]  R^2 = {r2:.4f}  crossing r_eff* = {xs:.4f}")
        print(f"        ZO better below r_eff=1: {lo_}/{(g.r_eff<1).sum()}   "
              f"above: {hi_}/{(g.r_eff>1).sum()}")
    print("\n  Language: this is consistent with the ratio of the two bounds Claim 4")
    print("  supplies.  Claim 4 does not imply the measured ratio must equal r^-2.")
    print("\n  -- POST-HOC (not registered): which frequency convention organises it? --")
    print("  r_eff uses Claim 4's SUP-NORM omega = ||grad e||_inf/||e||_inf.  The")
    print("  realised variances depend on RMS field statistics instead:")
    print("    Var_e[PW] ~ (1/M) E||grad e||^2 = omega^2 A0^2/(2J M)")
    print("    Var_e[ZO] ~ (d/M) E[e^2]/sigma^2 = d A0^2/(2J M sigma^2)")
    print("  so the A0^2/(2J) cancels and the ratio is r_nom^-2 with r_nom = sigma")
    print("  omega/sqrt(d), the NOMINAL r -- independent of J.  Prediction: the")
    print("  crossing sits at r_eff = 1/sqrt(J), i.e. 1.0 at J=1 and 0.5 at J=4.")
    for J in (1, 4):
        g = cell[cell.J == J].copy()
        g["R_var"] = g.var_zo_e / g.var_pw_e
        g["r_nom"] = g.r_eff * np.sqrt(J)
        _, _, r2n, xn = ols(np.log(g.r_nom), np.log(g.R_var))
        _, _, _, xe = ols(np.log(g.r_eff), np.log(g.R_var))
        print(f"    J={J}: crossing in r_eff = {xe:.4f} (predicted {1/np.sqrt(J):.4f})"
              f"   crossing in r_nom = {xn:.4f} (predicted 1.0)  R^2 = {r2n:.4f}")

    # ---------------------------------------------------------------- 6b
    hdr("6b. POST-HOC -- is the B attenuation mode competition, or just weaker field?")
    print("  The sup-norm normalisation c_j = A0/J holds ||e||_inf = A0 fixed but NOT")
    print("  the RMS field strength: E_phi E_a[e^2] = sum_j c_j^2/2 = A0^2/(2J), so the")
    print("  typical field strength falls by exactly sqrt(J) = 2 at J = 4.")
    try:
        amp = pd.read_csv(f"{ART}/planted_amplitude_replicates.csv")
        amp = amp[amp.sigma == 0.4].copy()
        amp["B_b"] = amp.pre_wml_err_meannorm / amp.pre_wml_clean_meannorm
        print("\n  B vs amplitude at J=1, from the COMMITTED amplitude experiment:")
        print(f"  {'d':>3} {'p (B ~ A^p)':>12} {'expected 0.5^p':>15} "
              f"{'measured B4/B1':>15} {'residual':>10}")
        res = []
        for d, g in amp.groupby("d"):
            m = g.groupby("amplitude").B_b.mean()
            p = float(np.polyfit(np.log(m.index.values), np.log(m.values), 1)[0])
            meas = float(P[P.d == d].ratio.median())
            exp = 0.5 ** p
            res.append(meas / exp)
            print(f"  {d:>3} {p:>12.4f} {exp:>15.4f} {meas:>15.4f} {meas/exp:>10.4f}")
        print(f"\n  residual (1.0 = attenuation fully explained by the weaker field):")
        print(f"    median across d = {np.median(res):.4f}")
        print(f"  So most of the {1/P.ratio.median():.2f}x attenuation is the amplitude")
        print(f"  drop the sup-norm normalisation implies, and only a further "
              f"{1/np.median(res):.2f}x is")
        print("  attributable to competition between modes.")
    except Exception as exc:
        print(f"  skipped: {exc}")

    # ---------------------------------------------------------------- 7
    hdr("7. INTERPRETATION (prereg Sec. 6 rules, applied as registered)")
    ratio = P.ratio.median()
    g4 = cell[cell.J == 4]
    sysc = g4.err_wml_erroroff.median() - g4.err_wml_baseline.median()
    noic = g4.err_wml_total.median() - g4.err_wml_erroroff.median()
    print(f"  headline B_J4/B_J1 = {ratio:.4f}")
    print(f"  at J=4 the E-step's systematic-mean contribution = {sysc:+.4f} "
          f"vs noise contribution = {noic:+.4f}   "
          f"({'systematic dominates' if sysc > noic else 'noise dominates'})")
    if ratio >= 0.5 and sysc > noic:
        case = "CASE A -- multi-mode bias PERSISTS"
    elif ratio >= 0.1:
        case = "CASE B -- PARTIAL ATTENUATION"
    else:
        case = "CASE C -- STRONG CANCELLATION (check operational gap)"
    print(f"  => {case}")
    print(f"     attenuation factor B_J1/B_J4 = {1/ratio:.2f}x")


if __name__ == "__main__":
    main()
