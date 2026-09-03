#!/usr/bin/env python
"""Analysis for the planted-error mechanism study.

Decision bands, crossover estimator and bootstrap are fixed in
docs/prereg_planted_amplitude.md (committed 4b88f03, before the runs).  Nothing
here selects a rule after seeing an outcome.

    ./.venv/bin/python scripts/planted/analyse_mechanism.py
"""
import numpy as np, pandas as pd

ART = "reports/artifacts"
B = 10000                 # prereg Sec. 5
BOOT_SEED = 20260903      # prereg Sec. 5
R_LO, R_HI = 0.5, 3.0     # sampled range; outside it r* is flagged extrapolated
W = 96


def hdr(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


def ols_crossover(logr, logratio):
    """r* = exp(-intercept/slope) from OLS of log ratio on log r, with R^2."""
    sl, ic = np.polyfit(logr, logratio, 1)
    pred = ic + sl * logr
    ss = np.var(logratio)
    r2 = 1.0 - np.var(logratio - pred) / ss if ss > 0 else np.nan
    return float(np.exp(-ic / sl)), float(sl), float(r2)


def flag(rstar):
    if not np.isfinite(rstar):
        return "n/a"
    if rstar < R_LO:
        return f"<{R_LO} (extrapolated)"
    if rstar > R_HI:
        return f">{R_HI} (extrapolated)"
    return f"{rstar:.3f}"


# =========================================================== part 1: the audit
def part1_audit():
    hdr("1. AUDIT of the committed sweep, and the reproduction check")
    ref = pd.read_csv(f"{ART}/planted_sweep.csv")
    new = pd.read_csv(f"{ART}/planted_error_decomposition.csv")
    key = ["d", "sigma", "omega"]
    m = new.merge(ref[key + ["ratio_e", "mse_pw", "mse_zo", "mse_wml",
                             "var_pw_e", "var_zo_e"]], on=key, suffixes=("", "_ref"))
    print(f"  cells: {len(m)}")
    for c in ("ratio_e", "var_pw_e", "var_zo_e", "mse_pw", "mse_zo", "mse_wml"):
        rel = np.abs(m[c] / m[f"{c}_ref"] - 1).max()
        print(f"    reproduce {c:10s} max rel dev {rel:.3e}  "
              f"{'OK' if rel < 1e-9 else 'MISMATCH'}")

    print("\n  -- the two variance definitions (prereg Sec. 2.2) --")
    for nm in ("pw", "zo", "wml"):
        rel = np.abs(new[f"var_{nm}_e_sub"] / new[f"var_{nm}_e"] - 1)
        idc = np.abs(new[f"cross_{nm}"]) / new[f"var_{nm}_e"]
        print(f"    {nm.upper():3s}  |sub/paired - 1|: median {rel.median():.3e}  "
              f"max {rel.max():.3e}   |cross|/paired: median {idc.median():.3e}  "
              f"max {idc.max():.3e}")
    ident = np.abs((new.var_pw_e + new.cross_pw) - new.var_pw_e_sub).max()
    print(f"    identity  V_sub - V_paired = cross  holds to {ident:.2e}")

    print("\n  -- does the r~1 conclusion survive the paired definition? --")
    for lbl, col in (("paired (primary)", "ratio_e"),):
        lo, hi = new[new.r < 1], new[new.r > 1]
        print(f"    {lbl}: ZO better in {(lo[col]<1).sum()}/{len(lo)} cells r<1, "
              f"{(hi[col]<1).sum()}/{len(hi)} cells r>1")
    sub_ratio = new.var_zo_e_sub / new.var_pw_e_sub
    lo, hi = new[new.r < 1], new[new.r > 1]
    print(f"    subtraction     : ZO better in "
          f"{(sub_ratio[new.r<1]<1).sum()}/{len(lo)} cells r<1, "
          f"{(sub_ratio[new.r>1]<1).sum()}/{len(hi)} cells r>1")

    print("\n  -- 'better update' is exactly a cosine (prereg Sec. 2.4) --")
    for nm in ("pw", "zo", "wml"):
        dev = np.abs(new[f"mse_{nm}"] - new[f"mse_{nm}_committed"]).max()
        dev2 = np.abs(new[f"mse_{nm}"] - 4 * 0.1 * (1 - new[f"cos_{nm}"])).max()
        print(f"    {nm.upper():3s}  |recomputed - committed formula| = {dev:.2e}   "
              f"|mse - 4 eps_TR (1-cos)| = {dev2:.2e}")

    print("\n  -- linearity: the paired difference IS g(e) for PW and ZO --")
    print(f"    max |paired - g(e)|  PW {new.lin_resid_pw.max():.2e}   "
          f"ZO {new.lin_resid_zo.max():.2e}   "
          f"WML {new.lin_resid_wml.max():.2e} (nonlinear, as expected)")

    print("\n  -- the committed operational crossover, refitted --")
    for nm, col in (("error-channel ZO/PW", "ratio_e"),
                    ("error-channel WML/PW", "ratio_e_wml"),
                    ("operational ZO/PW", "mse_ratio_zo_pw"),
                    ("operational WML/PW", "mse_ratio_wml_pw")):
        rs, sl, r2 = ols_crossover(np.log(new.r), np.log(new[col]))
        print(f"    {nm:22s} slope {sl:+.3f}  r* {rs:6.3f}  R2 {r2:.4f}")
    return new


# ================================================== part 2: the decomposition
def part2_decomposition(new, band=(1.0, 1.67)):
    hdr(f"2. DECOMPOSITION -- is the {band[0]} < r < {band[1]} band as hypothesised?")
    b = new[(new.r > band[0]) & (new.r < band[1])]
    print(f"  cells in band: {len(b)}   (d = {sorted(b.d.unique())})")

    c1 = (b.var_wml_e < b.var_pw_e)
    c2 = (b.mse_wml > b.mse_pw)
    print(f"\n  H(1) E-step has LOWER variance on the planted-error channel: "
          f"{c1.sum()}/{len(b)} cells")
    print(f"  H(2) E-step still has LARGER total update error:              "
          f"{c2.sum()}/{len(b)} cells")
    print(f"       both together (the mismatch region):                    "
          f"{(c1 & c2).sum()}/{len(b)} cells")
    print(f"       median Var_e[WML]/Var_e[PW] = {(b.var_wml_e/b.var_pw_e).median():.4f}"
          f"    median Err[WML]/Err[PW] = {(b.mse_wml/b.mse_pw).median():.4f}")

    print("\n  H(3) where does the residual error come from? "
          "(counterfactuals, prereg Sec. 2.7)")
    print(f"  {'op':4s} {'Err_total':>10s} {'Err_cleanoff':>13s} {'Err_erroroff':>13s} "
          f"{'Err_baseline':>13s} {'clean share':>12s}")
    for nm in ("pw", "zo", "wml"):
        t = b[f"err_{nm}_total"].median()
        co = b[f"err_{nm}_cleanoff"].median()
        eo = b[f"err_{nm}_erroroff"].median()
        bl = b[f"err_{nm}_baseline"].median()
        # how much of the total error is removed by perfecting each channel
        red_clean = (b[f"err_{nm}_total"] - b[f"err_{nm}_cleanoff"]) / b[f"err_{nm}_total"]
        print(f"  {nm.upper():4s} {t:10.4f} {co:13.4f} {eo:13.4f} {bl:13.4f} "
              f"{red_clean.median():11.1%}")
    print("\n  reading: Err_cleanoff removes the SMOOTH channel's noise, Err_erroroff")
    print("  removes the ERROR channel's noise.  Whichever is lower identifies the")
    print("  dominant source.  Err_baseline is the error floor with the field off.")

    print("\n  -- pre-normalisation channel magnitudes in the band (medians) --")
    print(f"  {'op':4s} {'var_clean':>11s} {'var_err':>11s} {'bias2_err':>11s} "
          f"{'|E g_clean|':>12s}")
    for nm in ("pw", "zo"):
        print(f"  {nm.upper():4s} {b[f'pre_{nm}_clean_var'].median():11.4g} "
              f"{b[f'pre_{nm}_err_var'].median():11.4g} "
              f"{b[f'pre_{nm}_err_bias2'].median():11.4g} "
              f"{b[f'pre_{nm}_clean_meannorm'].median():12.4g}")
    print(f"  WML  {b['pre_wml_clean_var'].median():11.4g} "
          f"{b['pre_wml_err_var'].median():11.4g} "
          f"{'n/a':>11s} {b['pre_wml_clean_meannorm'].median():12.4g}"
          f"   (displacement, no gradient-scale bias)")

    print("\n  -- post-normalisation, on the unit sphere (bias^2 + variance) --")
    print(f"  {'op':4s} {'E|u-u*|^2':>11s} {'bias^2':>11s} {'variance':>11s} "
          f"{'var share':>10s}")
    for nm in ("pw", "zo", "wml"):
        tot = b[f"post_{nm}_sphere"].median()
        b2 = b[f"post_{nm}_bias2"].median()
        v = b[f"post_{nm}_var"].median()
        print(f"  {nm.upper():4s} {tot:11.4f} {b2:11.4f} {v:11.4f} {v/tot:9.1%}")


# ================================================ part 3: amplitude + bootstrap
def boot_crossovers(rep, amps, cfg_filter, num, den, label):
    """Bootstrap r* over direction blocks (prereg Sec. 5).  Resampling is
    independent within each cell, because blocks in different cells are
    independent draws."""
    rng = np.random.default_rng(BOOT_SEED)
    out = {}
    for A in amps:
        sub = rep[(rep.amplitude == A) & cfg_filter(rep)]
        cells = sorted(sub.groupby(["d", "sigma", "r"]).groups.keys())
        nb = sub.groupby(["d", "sigma", "r"]).size().unique()
        assert len(nb) == 1, nb
        nb = int(nb[0])
        Vn = np.stack([sub[(sub.d == c[0]) & (sub.sigma == c[1]) & (sub.r == c[2])][num]
                       .to_numpy() for c in cells])           # (ncell, nblock)
        Vd = np.stack([sub[(sub.d == c[0]) & (sub.sigma == c[1]) & (sub.r == c[2])][den]
                       .to_numpy() for c in cells])
        logr = np.log(np.array([c[2] for c in cells]))
        point, sl, r2 = ols_crossover(logr, np.log(Vn.mean(1) / Vd.mean(1)))

        rs = np.empty(B)
        step = 500
        for s in range(0, B, step):
            n = min(step, B - s)
            idx = rng.integers(0, nb, size=(n, len(cells), nb))
            num_b = np.take_along_axis(Vn[None], idx, axis=2).mean(2)   # (n, ncell)
            den_b = np.take_along_axis(Vd[None], idx, axis=2).mean(2)
            y = np.log(num_b / den_b)
            x = logr
            xm, ym = x.mean(), y.mean(1, keepdims=True)
            slb = ((x - xm) * (y - ym)).sum(1) / ((x - xm) ** 2).sum()
            icb = ym[:, 0] - slb * xm
            rs[s:s + n] = np.exp(-icb / slb)
        lo, hi = np.percentile(rs, [2.5, 97.5])
        out[A] = dict(point=point, slope=sl, r2=r2, lo=float(lo), hi=float(hi))
        print(f"    {label:22s} A={A:<5g} r* = {flag(point):>18s}  "
              f"95% CI [{lo:.3f}, {hi:.3f}]  slope {sl:+.3f}  R2 {r2:.4f}")
    return out


def part3_amplitude():
    hdr("3. AMPLITUDE EXPERIMENT -- H1, H2, and the mismatch region")
    rep = pd.read_csv(f"{ART}/planted_amplitude_replicates.csv")
    cell = pd.read_csv(f"{ART}/planted_amplitude.csv")
    amps = sorted(cell.amplitude.unique())
    primary = lambda df: (df.sigma == 0.4)
    sigchk = lambda df: (df.sigma == 0.2)

    print("  pathology diagnostics (prereg Sec. 6):")
    for A, g in cell.groupby("amplitude"):
        print(f"    A={A:<5g} min median ESS {g.ess_med.min():6.2f}  "
              f"min ESS {g.ess_min.min():6.3f}  eta-at-bound {g.eta_at_bound.max():.4f}  "
              f"nonfinite {g.nonfinite.sum():.0f}")

    print("\n  A. ERROR-CHANNEL crossover r*_var  (primary configs, sigma=0.4, d in 4/16/64)")
    print("     -- ZO/PW is the IMPLEMENTATION CHECK (invariant by construction) --")
    zo_var = boot_crossovers(rep, amps, primary, "var_zo_e", "var_pw_e", "r*_var ZO/PW")
    print("     -- WML/PW is the actual H1 test --")
    wml_var = boot_crossovers(rep, amps, primary, "var_wml_e", "var_pw_e", "r*_var WML/PW")

    # ---- POST-HOC, flagged as such: the registered r*_var for the E-step
    # compares a DISPLACEMENT variance with a GRADIENT variance, which are not
    # in the same units, so the absolute location of that crossover carries an
    # arbitrary scale (measured: the raw ratio picks up ~sigma^2).  The
    # dimensionless form divides each operator's error-channel variance by its
    # OWN clean-signal magnitude.  Reported alongside the registered quantity,
    # not in place of it.
    for df_ in (rep,):
        df_["nsr_wml"] = df_.var_wml_e / df_.pre_wml_clean_meannorm ** 2
        df_["nsr_zo"] = df_.var_zo_e / df_.pre_zo_clean_meannorm ** 2
        df_["nsr_pw"] = df_.var_pw_e / df_.pre_pw_clean_meannorm ** 2
    print("\n     -- POST-HOC (not registered): dimensionless form, each operator's")
    print("        error-channel variance relative to its own clean signal --")
    wml_var_nd = boot_crossovers(rep, amps, primary, "nsr_wml", "nsr_pw",
                                 "r*_var WML/PW (dimensionless)")

    print("\n  B. OPERATIONAL crossover r*_op")
    zo_op = boot_crossovers(rep, amps, primary, "mse_zo", "mse_pw", "r*_op ZO/PW")
    wml_op = boot_crossovers(rep, amps, primary, "mse_wml", "mse_pw", "r*_op WML/PW")

    print("\n  sigma = 0.2 confound check at d = 16:")
    boot_crossovers(rep, amps, sigchk, "var_wml_e", "var_pw_e", "r*_var WML/PW s.2")
    boot_crossovers(rep, amps, sigchk, "mse_wml", "mse_pw", "r*_op  WML/PW s.2")

    # ---------------- decision bands, applied exactly as registered
    hdr("4. DECISION BANDS (prereg Sec. 4)")
    print("  H1: r*_var(A) in [0.8, 1.25] for all three A, for the WML/PW error channel")
    ok1 = True
    for A in amps:
        v = wml_var[A]["point"]
        good = 0.8 <= v <= 1.25
        ok1 &= good
        print(f"    A={A:<5g} r*_var = {v:.4f}  CI [{wml_var[A]['lo']:.3f}, "
              f"{wml_var[A]['hi']:.3f}]  {'in band' if good else 'OUT OF BAND'}")
    print(f"  => H1 {'SUPPORTED' if ok1 else 'NOT SUPPORTED'}")
    print("  same verdict in the dimensionless form (post-hoc, correct units):")
    for A in amps:
        v = wml_var_nd[A]
        print(f"    A={A:<5g} r*_var = {flag(v['point']):>18s}  "
              f"CI [{v['lo']:.3f}, {v['hi']:.3f}]  "
              f"{'in band' if 0.8 <= v['point'] <= 1.25 else 'OUT OF BAND'}")
    print(f"    (check) ZO/PW r*_var across A: "
          f"{[round(zo_var[A]['point'], 4) for A in amps]}  "
          f"max rel spread {np.ptp([zo_var[A]['point'] for A in amps])/np.mean([zo_var[A]['point'] for A in amps]):.2e}")

    print("  in-range evidence, no extrapolation needed: at the LOWEST sampled r = 0.5,")
    print("  well below the theoretical boundary, is the E-step already ahead on the")
    print("  error channel?")
    for A in amps:
        g = cell[(cell.amplitude == A) & (cell.sigma == 0.4) & (cell.r == 0.5)]
        print(f"    A={A:<5g} median Var_e[WML]/Var_e[PW] at r=0.5 = "
              f"{g.ratio_e_wml.median():.4f}  ({'already < 1' if g.ratio_e_wml.median() < 1 else 'not yet < 1'})")

    print("\n  H2: r*_op(4 A0) < r*_op(A0/4) with NON-OVERLAPPING 95% intervals (WML/PW)")
    hiA, loA = max(amps), min(amps)
    a, c = wml_op[hiA], wml_op[loA]
    lt = a["point"] < c["point"]
    disj = a["hi"] < c["lo"]
    extrap = [A for A in (hiA, loA)
              if not (R_LO <= wml_op[A]["point"] <= R_HI)]
    print(f"    r*_op(A={hiA}) = {flag(a['point']):>18s}  CI [{a['lo']:.3f}, {a['hi']:.3f}]")
    print(f"    r*_op(A={loA}) = {flag(c['point']):>18s}  CI [{c['lo']:.3f}, {c['hi']:.3f}]")
    print(f"    ordered: {lt}   intervals disjoint: {disj}")
    if extrap:
        print(f"    BUT r*_op is outside the sampled range [{R_LO}, {R_HI}] at A = {[float(x) for x in extrap]}.")
        print("    prereg Sec. 4: 'an extrapolated bound may not be used to satisfy a")
        print("    decision band'.  The band therefore CANNOT be applied.")
        print("  => H2 NOT SUPPORTED AS REGISTERED -- reported descriptively below")
    else:
        print(f"  => H2 {'SUPPORTED' if (lt and disj) else 'NOT SUPPORTED -- descriptive'}")
    print(f"    (also) ZO/PW r*_op: {[flag(zo_op[A]['point']) for A in amps]}")

    # Descriptive statistic the registered rule DOES permit: it is entirely
    # in-range, and it tests the same direction H2 asserts without locating a
    # crossover.  Not a registered test; reported as description.
    print("\n  DESCRIPTIVE (in-range, no extrapolation): does the operational gap")
    print("  Err[WML] - Err[PW] shrink with amplitude at each sampled r?")
    rng2 = np.random.default_rng(BOOT_SEED + 3)
    print(f"    {'r':>5s} {'A0/4':>18s} {'4A0':>18s}  {'gap shrinks':>12s}")
    for rv in sorted(cell.r.unique()):
        out = {}
        for A in (loA, hiA):
            s = rep[(rep.amplitude == A) & (rep.sigma == 0.4) & (rep.r == rv)]
            cells = sorted(s.groupby(["d", "sigma", "r"]).groups.keys())
            Dm = np.stack([(s[(s.d == k[0]) & (s.r == k[2])].mse_wml
                            - s[(s.d == k[0]) & (s.r == k[2])].mse_pw).to_numpy()
                           for k in cells])
            nb = Dm.shape[1]
            idx = rng2.integers(0, nb, size=(B, len(cells), nb))
            bs = np.take_along_axis(Dm[None], idx, axis=2).mean(2).mean(1)
            out[A] = (float(Dm.mean()), *np.percentile(bs, [2.5, 97.5]))
        lo_, hi_ = out[loA], out[hiA]
        ok = "yes" if hi_[2] < lo_[1] else "overlapping"
        print(f"    {rv:5.2f} {lo_[0]:8.4f} [{lo_[1]:.4f},{lo_[2]:.4f}] "
              f"{hi_[0]:8.4f} [{hi_[1]:.4f},{hi_[2]:.4f}]  {ok:>12s}")

    # ---------------- mismatch region
    hdr("5. MISMATCH REGION (prereg Sec. 4C)")
    rng = np.random.default_rng(BOOT_SEED + 1)
    print(f"  cells with Var_e[WML] < Var_e[PW] AND Err[WML] > Err[PW], sigma=0.4")
    for A in amps:
        g = cell[(cell.amplitude == A) & (cell.sigma == 0.4)]
        mask = (g.var_wml_e < g.var_pw_e) & (g.mse_wml > g.mse_pw)
        rs = g[mask].r
        sub = rep[(rep.amplitude == A) & (rep.sigma == 0.4)]
        # bootstrap the fraction over blocks
        cells = sorted(sub.groupby(["d", "sigma", "r"]).groups.keys())
        arr = {c: sub[(sub.d == c[0]) & (sub.sigma == c[1]) & (sub.r == c[2])] for c in cells}
        nb = len(next(iter(arr.values())))
        fr = np.empty(B)
        cols = ["var_wml_e", "var_pw_e", "mse_wml", "mse_pw"]
        Vs = np.stack([arr[c][cols].to_numpy() for c in cells])   # (ncell, nb, 4)
        for s in range(0, B, 500):
            n = min(500, B - s)
            idx = rng.integers(0, nb, size=(n, len(cells), nb))
            mb = np.stack([np.take_along_axis(Vs[None, :, :, j], idx, axis=2).mean(2)
                           for j in range(4)], axis=-1)           # (n, ncell, 4)
            fr[s:s + n] = ((mb[..., 0] < mb[..., 1]) & (mb[..., 2] > mb[..., 3])).mean(1)
        lo, hi = np.percentile(fr, [2.5, 97.5])
        span = f"r in [{rs.min():.2f}, {rs.max():.2f}]" if len(rs) else "empty"
        print(f"    A={A:<5g} {mask.sum():2d}/{len(g)} cells = {mask.mean():5.1%}  "
              f"95% CI [{lo:.1%}, {hi:.1%}]   {span}")


# =================================================== part 4: the r^-2 statement
def part4_slope(new):
    hdr("6. THE r^-2 STATEMENT (prereg / deliverable 8)")
    print("  Population: the committed 240-cell grid, error-channel ratio ZO/PW.")
    rng = np.random.default_rng(BOOT_SEED + 2)
    for lbl, sub in (("all 240 cells", new),
                     ("r in [0.5, 2]", new[(new.r >= 0.5) & (new.r <= 2)]),
                     ("r in [0.25, 4]", new[(new.r >= 0.25) & (new.r <= 4)]),
                     ("r in [0.8, 1.25]", new[(new.r >= 0.8) & (new.r <= 1.25)])):
        if len(sub) < 3:
            print(f"    {lbl:18s} n={len(sub)}  too few cells")
            continue
        x, y = np.log(sub.r.to_numpy()), np.log(sub.ratio_e.to_numpy())
        rs, sl, r2 = ols_crossover(x, y)
        # bootstrap over CELLS here: the replicate unit available in this artifact
        bs = np.empty(2000)
        for i in range(2000):
            j = rng.integers(0, len(x), len(x))
            bs[i] = np.polyfit(x[j], y[j], 1)[0]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"    {lbl:18s} n={len(sub):3d}  beta = {sl:+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  R2 = {r2:.4f}  r* = {rs:.4f}")
    print("\n  Reference value -2 comes from the RATIO OF TWO UPPER BOUNDS in Claim 4")
    print("  ((1/M)||grad e||_inf^2 and (d/M)||e||_inf^2/sigma^2).  A ratio of bounds")
    print("  does not imply the ratio of the quantities they bound; agreement with -2")
    print("  is an empirical finding about this construction, not an implication.")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    new = part1_audit()
    part2_decomposition(new)
    part3_amplitude()
    part4_slope(new)
