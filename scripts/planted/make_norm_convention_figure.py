#!/usr/bin/env python
"""Appendix figure: which frequency convention organises the realised ZO/PW
error-channel variance ratio for a multi-mode planted critic error.

Panel A -- Claim 4's SUP-NORM convention, r_inf = sigma omega_inf / sqrt(d) with
           omega_inf = ||grad e||_inf / ||e||_inf.
Panel B -- the NOMINAL (RMS) convention, r_nom = sigma omega / sqrt(d).

Same y values in both panels; only the x-axis convention differs.

Built from committed artifacts only -- no experiment is run:
  J = 1, 4  ->  reports/artifacts/planted_multimode{,_replicates}.csv   (d 4/16/64)
  J = 8     ->  reports/artifacts/planted_j8{,_replicates}.csv          (d 16/64)

Honesty constraints enforced in the drawing, not just the caption:
  * fitted lines are SOLID only over the sampled range; the extension to a
    crossing that lies outside it is thin and dotted, and its crossing marker is
    hollow.  Only J = 1 has an in-range crossing in either panel.
  * J = 8 never samples below r_nom = 1 (its range is [1.414, 8.485]), so
    nothing is drawn there except the explicitly-marked extrapolation.

    ./.venv/bin/python scripts/planted/make_norm_convention_figure.py
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ART = "reports/artifacts"
NB = 10000
BOOT_SEED = 20260907

SOURCES = {1: ("planted_multimode", 20260904),
           4: ("planted_multimode", 20260904),
           8: ("planted_j8", 20260906)}

# J is ordered -> one-hue ordinal ramp (steps 300/500/700), validated light-mode
# --ordinal --pairs all.  Colour is REDUNDANT: marker shape and line style carry
# the same information so the figure survives grayscale printing.
J_COLOR = {1: "#6da7ec", 4: "#256abf", 8: "#0d366b"}
J_MARKER = {1: "o", 4: "s", 8: "^"}
J_LS = {1: "-", 4: "--", 8: "-."}
INK, INK2, MUTED = "#111111", "#444444", "#9a9a9a"
RULE = "#b0b0b0"
SURFACE = "#ffffff"


def ratio_with_ci(rep, J, seed):
    """Per-cell Var_e[ZO]/Var_e[PW] with a 95% bootstrap CI over field blocks."""
    rng = np.random.default_rng(seed)
    rows = []
    for (d, r), g in rep[rep.J == J].groupby(["d", "r_eff"]):
        n, m = g.var_zo_e.to_numpy(), g.var_pw_e.to_numpy()
        idx = rng.integers(0, len(n), size=(NB, len(n)))
        bs = n[idx].mean(1) / m[idx].mean(1)
        rows.append(dict(J=J, d=int(d), r_eff=float(r),
                         r_inf=float(r), r_nom=float(r * np.sqrt(J)),
                         ratio=float(n.mean() / m.mean()),
                         ci_lo=float(np.percentile(bs, 2.5)),
                         ci_hi=float(np.percentile(bs, 97.5)),
                         n_blocks=len(n)))
    return pd.DataFrame(rows)


def fit(x, y):
    """OLS of log y on log x; returns slope, R^2 and the crossing y = 1."""
    lx, ly = np.log(x), np.log(y)
    sl, ic = np.polyfit(lx, ly, 1)
    r2 = 1 - np.var(ly - (ic + sl * lx)) / np.var(ly)
    return float(sl), float(r2), float(np.exp(-ic / sl))


def main():
    # ---------------------------------------------------------------- load
    cache, frames = {}, []
    for J, (pre, seed) in SOURCES.items():
        if pre not in cache:
            cache[pre] = (pd.read_csv(f"{ART}/{pre}_replicates.csv"),
                          pd.read_csv(f"{ART}/{pre}.csv"))
        frames.append(ratio_with_ci(cache[pre][0], J, seed + J))
    D = pd.concat(frames, ignore_index=True)

    # ------------------------------------------------- verification vs source
    print("=" * 78)
    print("VERIFICATION -- plotted values recomputed from *_replicates.csv and")
    print("checked against the committed cell-level *.csv")
    print("=" * 78)
    worst = 0.0
    for J, (pre, _) in SOURCES.items():
        cell = cache[pre][1]
        c = cell[cell.J == J].copy()
        c["ref"] = c.var_zo_e / c.var_pw_e
        m = D[D.J == J].merge(c[["d", "r_eff", "ref"]], on=["d", "r_eff"])
        rel = np.abs(m.ratio / m.ref - 1)
        worst = max(worst, float(rel.max()))
        rn = c.sigma * c.omega_nominal / np.sqrt(c.d)
        xdev = float(np.abs(m.merge(c.assign(rn=rn)[["d", "r_eff", "rn"]],
                                    on=["d", "r_eff"]).eval("r_nom - rn")).max())
        print(f"  J={J}  source {pre}.csv  cells {len(m)}  d {sorted(c.d.unique())}")
        print(f"        max |recomputed/committed - 1| = {rel.max():.3e}")
        print(f"        max |r_nom - sigma*omega/sqrt(d)| = {xdev:.3e}")
        print(f"        sampled r_inf [{m.r_inf.min():.4f}, {m.r_inf.max():.4f}]   "
              f"r_nom [{m.r_nom.min():.4f}, {m.r_nom.max():.4f}]")
    print(f"\n  worst deviation over all series: {worst:.3e}  "
          f"{'OK' if worst < 1e-9 else 'MISMATCH -- do not save'}")
    assert worst < 1e-9, "plotted values disagree with the committed artifacts"

    print("\n" + "=" * 78)
    print("FITS USED IN THE FIGURE   (log-log OLS, crossing = where the fit hits 1)")
    print("=" * 78)
    fits = {}
    print(f"  {'panel':>6} {'J':>2} {'x':>6} {'slope':>9} {'R^2':>8} {'crossing':>9} "
          f"{'sampled x range':>22} {'in range?':>10}")
    for panel, xcol in (("A", "r_inf"), ("B", "r_nom")):
        for J in (1, 4, 8):
            g = D[D.J == J]
            sl, r2, xs = fit(g[xcol].to_numpy(), g.ratio.to_numpy())
            lo, hi = g[xcol].min(), g[xcol].max()
            inr = lo <= xs <= hi
            fits[(panel, J)] = (sl, r2, xs, lo, hi, inr)
            print(f"  {panel:>6} {J:>2} {xcol:>6} {sl:>+9.4f} {r2:>8.4f} {xs:>9.4f} "
                  f"{f'[{lo:.4f}, {hi:.4f}]':>22} {str(inr):>10}")
    print("\n  Panel A predicted crossing under the sup-norm convention is 1/sqrt(J):")
    for J in (1, 4, 8):
        print(f"    J={J}: predicted {1/np.sqrt(J):.4f}   measured "
              f"{fits[('A', J)][2]:.4f}")
    print("  Panel B predicted crossing under the nominal convention is 1.0 for all J.")
    print("\n  NOTE: only J=1 has an in-range crossing.  J=4 and J=8 crossings are")
    print("  extrapolations below their sampled ranges and are drawn as such.")

    D.to_csv(f"{ART}/fig_planted_norm_convention_data.csv", index=False)
    print(f"\nwrote {ART}/fig_planted_norm_convention_data.csv  ({len(D)} rows)")

    # ---------------------------------------------------------------- draw
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.edgecolor": INK2, "axes.linewidth": 0.7,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42, "ps.fonttype": 42,      # embedded, selectable text
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)

    for ax, panel, xcol, xlab, title in (
            (axes[0], "A", "r_inf",
             r"$r_\infty=\sigma\,\omega_\infty/\sqrt{d}$",
             r"A   Sup-norm convention"),
            (axes[1], "B", "r_nom",
             r"$r_{\mathrm{nom}}=\sigma\,\omega/\sqrt{d}$",
             r"B   Nominal (RMS) convention")):
        ax.axhline(1.0, color=RULE, lw=0.8, zorder=1)
        ax.axvline(1.0, color=RULE, lw=0.8, zorder=1)
        for J in (1, 4, 8):
            g = D[D.J == J].sort_values(xcol)
            col, mk = J_COLOR[J], J_MARKER[J]
            sl, r2, xs, lo, hi, inr = fits[(panel, J)]
            # fit over the SAMPLED range only
            xf = np.logspace(np.log10(lo), np.log10(hi), 100)
            ax.plot(xf, np.exp(np.log(xf) * sl + (np.log(g.ratio.to_numpy()).mean()
                    - sl * np.log(g[xcol].to_numpy()).mean())),
                    J_LS[J], color=col, lw=1.5, zorder=3)
            # extrapolation to the crossing, if it lies outside the data
            if not inr:
                xe = np.logspace(np.log10(min(xs, lo)), np.log10(lo), 40)
                ax.plot(xe, np.exp(np.log(xe) * sl + (np.log(g.ratio.to_numpy()).mean()
                        - sl * np.log(g[xcol].to_numpy()).mean())),
                        ls=(0, (1, 2)), color=col, lw=1.1, zorder=3)
            ax.errorbar(g[xcol], g.ratio,
                        yerr=[g.ratio - g.ci_lo, g.ci_hi - g.ratio],
                        fmt=mk, ms=4.2, color=col, ecolor=col, elinewidth=0.9,
                        capsize=1.6, mfc=col, mec=SURFACE, mew=0.6, ls="none",
                        zorder=5)
            ax.plot([xs], [1.0], marker="v", ms=6.5, mfc=(col if inr else "none"),
                    mec=col, mew=1.2, ls="none", zorder=6, clip_on=False)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(xlab, fontsize=9.0)
        ax.set_title(title, fontsize=9.5, loc="left", pad=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ticks = [0.35, 0.5, 0.75, 1, 2, 3] if panel == "A" else [0.5, 1, 2, 3, 5, 8]
        ax.set_xticks(ticks)
        ax.set_xticklabels([("%g" % t) for t in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[0].set_ylabel(r"$\mathrm{Var}_e[\hat g_{\mathrm{ZO}}]\,/\,"
                       r"\mathrm{Var}_e[\hat g_{\mathrm{PW}}]$", fontsize=9.0)

    # crossing labels, offset so they never sit on a data point
    for ax, panel in ((axes[0], "A"), (axes[1], "B")):
        # In A the three crossings are far apart in x, so one height suffices; in
        # B they nearly coincide, so they are stacked.
        offs = {1: 14, 4: 14, 8: 14} if panel == "A" else {1: 14, 4: 28, 8: 42}
        for J in (1, 4, 8):
            dy = offs[J]
            sl, r2, xs, lo, hi, inr = fits[(panel, J)]
            ax.annotate(f"{xs:.3f}" + ("" if inr else "*"), (xs, 1.0),
                        textcoords="offset points", xytext=(0, dy), ha="center",
                        fontsize=7.8, color=J_COLOR[J], fontweight="bold", zorder=7,
                        bbox=dict(fc=SURFACE, ec="none", pad=0.9))

    handles = [Line2D([], [], color=J_COLOR[J], marker=J_MARKER[J], ls=J_LS[J],
                      lw=1.5, ms=4.2, mec=SURFACE, mew=0.6, label=f"$J={J}$")
               for J in (1, 4, 8)]
    handles += [Line2D([], [], color=MUTED, marker="v", ls="none", ms=6.5,
                       mfc="none", mec=MUTED, mew=1.2,
                       label="crossing (* extrapolated)")]
    axes[1].legend(handles=handles, fontsize=8.0, loc="center left",
                   bbox_to_anchor=(1.02, 0.5), frameon=False, handlelength=2.2,
                   borderaxespad=0.0)

    fig.tight_layout(w_pad=1.6)
    fig.savefig(f"{ART}/fig_planted_norm_convention.pdf", bbox_inches="tight")
    fig.savefig(f"{ART}/fig_planted_norm_convention.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {ART}/fig_planted_norm_convention.pdf (vector, fonttype 42) "
          f"and .png (300 dpi)")


if __name__ == "__main__":
    main()
