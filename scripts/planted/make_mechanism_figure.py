#!/usr/bin/env python
"""Main controlled-evidence figure for the planted-error section.

Panel A: the ESTIMATOR-level result -- error-channel variance ratio against r,
         with the theoretical boundary r = 1.  Two operators are shown because
         the contrast between them IS the result: the manuscript's centred g_ZO,
         which is linear in Q and whose ratio is therefore exactly amplitude-
         invariant and crosses 1 at r = 1; and the actual softmax E-step, whose
         ratio is neither, one curve per amplitude.
Panel B: the OPERATIONAL result -- difference in normalised update quality
         against r, same amplitude series, with any zero crossings marked.

Emits the figure and its source data.  Uncertainty is the 95% bootstrap interval
over direction blocks, the replicate unit fixed in
docs/prereg_planted_amplitude.md Sec. 5.

    ./.venv/bin/python scripts/planted/make_mechanism_figure.py
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ART = "reports/artifacts"
B = 10000
BOOT_SEED = 20260903

# Ordinal blue ramp, steps 250 / 450 / 650 -- validated light-mode, --ordinal
# --pairs all: monotone lightness, adjacent dL >= 0.06, light end 2.06:1 on the
# chart surface, single hue (spread 3 deg).  Amplitude is an ORDERED quantity, so
# it takes a one-hue ramp rather than categorical slots.
AMP_COLOR = {0.25: "#86b6ef", 1.0: "#2a78d6", 4.0: "#104281"}
# g_ZO is a different entity, not a step of the amplitude ramp, so it takes a
# categorical slot.  blue<->orange all-pairs light: CVD dE 24.7, normal 33.6.
ZO_COLOR = "#eb6834"
D_MARKER = {4: "o", 16: "s", 64: "^"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURFACE = "#fcfcfb"
RULE = "#c0392b"          # the theoretical boundary; a reference line, not a series


def boot_cells(rep, num, den, mode):
    """Point estimate and 95% interval per cell, resampling direction blocks."""
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for (d, sig, r, A), g in rep.groupby(["d", "sigma", "r", "amplitude"]):
        a, b = g[num].to_numpy(), g[den].to_numpy()
        nb = len(a)
        idx = rng.integers(0, nb, size=(B, nb))
        an, bn = a[idx].mean(1), b[idx].mean(1)
        if mode == "logratio":
            pt = np.log10(a.mean() / b.mean())
            bs = np.log10(an / bn)
        else:
            pt = a.mean() - b.mean()
            bs = an - bn
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(d=d, sigma=sig, r=r, amplitude=A, value=pt,
                         ci_lo=float(lo), ci_hi=float(hi)))
    return pd.DataFrame(rows)


def zero_crossing(x, y):
    """Linear interpolation in log r of the first sign change of y."""
    lx = np.log(x)
    for i in range(len(y) - 1):
        if y[i] > 0 >= y[i + 1] or y[i] < 0 <= y[i + 1]:
            t = -y[i] / (y[i + 1] - y[i])
            return float(np.exp(lx[i] + t * (lx[i + 1] - lx[i])))
    return np.nan


def main():
    rep = pd.read_csv(f"{ART}/planted_amplitude_replicates.csv")
    rep = rep[rep.sigma == 0.4]                      # primary configs
    amps = sorted(rep.amplitude.unique())

    # Panel A uses the DIMENSIONLESS error-channel measure: each operator's
    # error-channel variance divided by its own clean-signal magnitude.  The raw
    # ratio Var_e[WML]/Var_e[PW] compares a displacement variance with a gradient
    # variance, so where it crosses 1 carries an arbitrary unit scale (~sigma^2);
    # dividing each by its own signal removes it and makes the crossing mean
    # something.  See reports/planted_error_mechanism.md Sec. 2.
    for nm in ("wml", "zo", "pw"):
        rep[f"nsr_{nm}"] = rep[f"var_{nm}_e"] / rep[f"pre_{nm}_clean_meannorm"] ** 2

    pa = boot_cells(rep, "nsr_wml", "nsr_pw", "logratio")
    pz = boot_cells(rep, "nsr_zo", "nsr_pw", "logratio")
    pb = boot_cells(rep, "mse_wml", "mse_pw", "diff")
    pa["panel"] = "A_dimensionless_err_channel_log10_WML_over_PW"
    pz["panel"] = "A_dimensionless_err_channel_log10_ZO_over_PW"
    pb["panel"] = "B_update_err_WML_minus_PW"
    src = pd.concat([pa, pz, pb], ignore_index=True)
    src.to_csv(f"{ART}/fig_planted_mechanism_data.csv", index=False)
    print(f"wrote {ART}/fig_planted_mechanism_data.csv  ({len(src)} rows)")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))

    for ax, sub, ylab, title in (
            (axes[0], pa,
             "$\\log_{10}$  error-channel variance rel. to own signal,\n"
             r"$\mathrm{E\text{-}step}$ vs $\mathrm{PW}$",
             "A  Estimator level: the critic-error channel"),
            (axes[1], pb,
             r"$\mathrm{Err}[\mathrm{E\text{-}step}] - \mathrm{Err}[\mathrm{PW}]$",
             "B  Operational level: the whole policy update")):
        ax.axhline(0, color=MUTED, lw=0.7, ls=":", zorder=1)
        ax.axvline(1.0, color=RULE, lw=1.3, zorder=1)
        for A in amps:
            c = AMP_COLOR[A]
            g = sub[sub.amplitude == A]
            m = g.groupby("r", as_index=False).agg(
                value=("value", "mean"), ci_lo=("ci_lo", "mean"), ci_hi=("ci_hi", "mean"))
            m = m.sort_values("r")
            ax.fill_between(m.r, m.ci_lo, m.ci_hi, color=c, alpha=0.16, lw=0, zorder=2)
            ax.plot(m.r, m.value, "-", color=c, lw=2.0, zorder=4,
                    solid_capstyle="round")
            for d in sorted(g.d.unique()):
                gd = g[g.d == d].sort_values("r")
                ax.plot(gd.r, gd.value, D_MARKER[d], color=c, ms=4.2, zorder=5,
                        markeredgecolor=SURFACE, markeredgewidth=0.7)
            x0 = zero_crossing(m.r.to_numpy(), m.value.to_numpy())
            if np.isfinite(x0):
                ax.plot([x0], [0], "v", color=c, ms=7, zorder=6,
                        markeredgecolor=SURFACE, markeredgewidth=0.8, clip_on=False)
                ax.annotate(f"{x0:.2f}", (x0, 0), textcoords="offset points",
                            xytext=(0, 10 + 12 * amps.index(A)), ha="center",
                            fontsize=7.4, color=c, fontweight="bold", zorder=7,
                            bbox=dict(fc=SURFACE, ec="none", pad=0.8))
        if ax is axes[0]:
            # the manuscript's g_ZO: linear in Q, so this ratio is EXACTLY
            # amplitude-invariant -- one curve, not three.
            mz = pz.groupby("r", as_index=False).agg(
                value=("value", "mean"), ci_lo=("ci_lo", "mean"),
                ci_hi=("ci_hi", "mean")).sort_values("r")
            ax.fill_between(mz.r, mz.ci_lo, mz.ci_hi, color=ZO_COLOR, alpha=0.16,
                            lw=0, zorder=2)
            ax.plot(mz.r, mz.value, "--", color=ZO_COLOR, lw=2.0, zorder=4)
            xz = zero_crossing(mz.r.to_numpy(), mz.value.to_numpy())
            if np.isfinite(xz):
                ax.plot([xz], [0], "v", color=ZO_COLOR, ms=7, zorder=6,
                        markeredgecolor=SURFACE, markeredgewidth=0.8)
                ax.annotate(f"{xz:.2f}", (xz, 0), textcoords="offset points",
                            xytext=(0, -15), ha="center", fontsize=7.4,
                            color=ZO_COLOR, fontweight="bold", zorder=7,
                            bbox=dict(fc=SURFACE, ec="none", pad=0.8))
            ax.annotate(r"manuscript $\hat g_{ZO}$ (amplitude-invariant)",
                        (mz.r.iloc[4], mz.value.iloc[4]), textcoords="offset points",
                        xytext=(2, 13), fontsize=7.4, color=ZO_COLOR,
                        fontweight="bold", zorder=7,
                        bbox=dict(fc=SURFACE, ec="none", pad=1.5))
            ax.annotate("actual E-step", (2.0, pa[pa.r == 2.0].value.mean()),
                        textcoords="offset points", xytext=(4, 14), fontsize=7.4,
                        color=AMP_COLOR[1.0], fontweight="bold", zorder=7)
        ax.set_xscale("log")
        ax.set_xlabel(r"$r=\sigma\omega/\sqrt{d}$")
        ax.set_ylabel(ylab, fontsize=8.2)
        ax.set_title(title, fontsize=9.2, loc="left", pad=8, color=INK)
        ax.grid(True, which="major", axis="both", color="#e8e7e3", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3])
        ax.set_xticklabels(["0.5", "0.75", "1", "1.5", "2", "3"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    axes[0].set_ylim(top=axes[0].get_ylim()[1] + 0.45)
    axes[0].text(1.06, axes[0].get_ylim()[1], "theory: $r=1$", color=RULE,
                 fontsize=7.6, va="top", fontweight="bold")

    amp_h = [Line2D([], [], color=AMP_COLOR[A], lw=2.2,
                    label=(r"$A_0/4$" if A == 0.25 else r"$A_0$" if A == 1.0 else r"$4A_0$"))
             for A in amps]
    d_h = [Line2D([], [], color=MUTED, marker=D_MARKER[d], ls="none", ms=4.2,
                  label=f"$d={d}$") for d in sorted(D_MARKER)]
    leg1 = axes[0].legend(handles=amp_h, title="error amplitude", fontsize=7.6,
                          title_fontsize=7.6, loc="lower left", frameon=False,
                          handlelength=1.6, borderaxespad=0.9)
    axes[0].add_artist(leg1)
    axes[1].legend(handles=d_h, title="dimension", fontsize=7.6, title_fontsize=7.6,
                   loc="upper right", frameon=False, handlelength=1.2,
                   borderaxespad=0.9)

    fig.tight_layout(w_pad=2.4)
    for ext in ("pdf", "png"):
        fig.savefig(f"{ART}/fig_planted_mechanism.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {ART}/fig_planted_mechanism.pdf and .png")

    print("\n  zero crossings (across-d mean curve, linear interpolation in log r):")
    for nm, sub in (("panel A  Var_e ratio", pa), ("panel B  update error", pb)):
        for A in amps:
            m = sub[sub.amplitude == A].groupby("r", as_index=False).value.mean()
            m = m.sort_values("r")
            print(f"    {nm:22s} A={A:<5g} crossing r = "
                  f"{zero_crossing(m.r.to_numpy(), m.value.to_numpy()):.4f}")


if __name__ == "__main__":
    main()
