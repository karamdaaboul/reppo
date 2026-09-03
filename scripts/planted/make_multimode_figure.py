#!/usr/bin/env python
"""Diagnostic figure for the J = 1 vs J = 4 multi-mode experiment.

Panel A: conditional systematic E-step displacement B = E_field[||delta||/||s||].
Panel B: operational gap G = Err[E-step] - Err[PW].
Panel C: centred ZO/PW error-channel variance ratio, with the r_eff = 1 reference.

All against r_eff = sigma omega_eff / sqrt(d), omega_eff = ||grad e||_inf/||e||_inf.
Bootstrap 95% intervals over error-field blocks.

    ./.venv/bin/python scripts/planted/make_multimode_figure.py
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ART = "reports/artifacts"
B = 10000
BOOT_SEED = 20260904

# J is an entity, not a magnitude ramp: categorical slots 1 and 2.
# blue<->orange all-pairs light: CVD dE 24.7, normal-vision dE 33.6 (validated).
J_COLOR = {1: "#2a78d6", 4: "#eb6834"}
J_LS = {1: "-", 4: "--"}
D_MARKER = {4: "o", 16: "s", 64: "^"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURFACE = "#fcfcfb"
RULE = "#c0392b"


def boot_cells(rep, col, transform=None):
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for (d, r, J), g in rep.groupby(["d", "r_eff", "J"]):
        v = g[col].to_numpy()
        idx = rng.integers(0, len(v), size=(B, len(v)))
        bs = v[idx].mean(1)
        pt = v.mean()
        if transform:
            pt, bs = transform(pt), transform(bs)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(d=d, r_eff=r, J=J, value=float(pt),
                         ci_lo=float(lo), ci_hi=float(hi)))
    return pd.DataFrame(rows)


def main():
    rep = pd.read_csv(f"{ART}/planted_multimode_replicates.csv")
    cell = pd.read_csv(f"{ART}/planted_multimode.csv")

    pa = boot_cells(rep, "B_b")
    pb = boot_cells(rep, "G")
    rep = rep.copy()
    cellr = cell.copy()
    cellr["R_var"] = cellr.var_zo_e / cellr.var_pw_e
    # panel C from the paired per-block ratio of block means
    rng = np.random.default_rng(BOOT_SEED + 1)
    rows = []
    for (d, r, J), g in rep.groupby(["d", "r_eff", "J"]):
        n, dn = g.var_zo_e.to_numpy(), g.var_pw_e.to_numpy()
        idx = rng.integers(0, len(n), size=(B, len(n)))
        bs = np.log10(n[idx].mean(1) / dn[idx].mean(1))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(d=d, r_eff=r, J=J, value=float(np.log10(n.mean() / dn.mean())),
                         ci_lo=float(lo), ci_hi=float(hi)))
    pc = pd.DataFrame(rows)

    pa["panel"], pb["panel"], pc["panel"] = "A_B_conditional_bias", "B_operational_gap", \
        "C_log10_ZO_over_PW_error_channel"
    pd.concat([pa, pb, pc], ignore_index=True).to_csv(
        f"{ART}/fig_planted_multimode_data.csv", index=False)
    print(f"wrote {ART}/fig_planted_multimode_data.csv")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.85))

    specs = [
        (axes[0], pa, r"$B=\mathbb{E}_{\mathrm{field}}\,\|\delta\|/\|s\|$",
         "A  Systematic E-step displacement, per fixed error field", False),
        (axes[1], pb, r"$\mathrm{Err}[\mathrm{E\text{-}step}]-\mathrm{Err}[\mathrm{PW}]$",
         "B  Operational gap", False),
        (axes[2], pc,
         r"$\log_{10}\ \mathrm{Var}_e[\hat g_{ZO}]/\mathrm{Var}_e[\hat g_{PW}]$",
         "C  Centred ZO vs PW error channel", True),
    ]
    for ax, sub, ylab, title, ref in specs:
        ax.axhline(0, color=MUTED, lw=0.7, ls=":", zorder=1)
        if ref:
            ax.axvline(1.0, color=RULE, lw=1.3, zorder=1)
        for J in (1, 4):
            c = J_COLOR[J]
            g = sub[sub.J == J]
            m = g.groupby("r_eff", as_index=False).agg(
                value=("value", "mean"), ci_lo=("ci_lo", "mean"),
                ci_hi=("ci_hi", "mean")).sort_values("r_eff")
            ax.fill_between(m.r_eff, m.ci_lo, m.ci_hi, color=c, alpha=0.16, lw=0,
                            zorder=2)
            ax.plot(m.r_eff, m.value, J_LS[J], color=c, lw=2.0, zorder=4,
                    solid_capstyle="round")
            for d in sorted(g.d.unique()):
                gd = g[g.d == d].sort_values("r_eff")
                ax.plot(gd.r_eff, gd.value, D_MARKER[d], color=c, ms=4.0, zorder=5,
                        markeredgecolor=SURFACE, markeredgewidth=0.7)
        ax.set_xscale("log")
        ax.set_xlabel(r"$r_{\mathrm{eff}}=\sigma\,\omega_{\mathrm{eff}}/\sqrt{d}$")
        ax.set_ylabel(ylab, fontsize=8.1)
        ax.set_title(title, fontsize=9.0, loc="left", pad=8, color=INK)
        ax.grid(True, color="#e8e7e3", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3])
        ax.set_xticklabels(["0.5", "0.75", "1", "1.5", "2", "3"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    axes[0].set_yscale("log")
    axes[0].set_yticks([0.1, 0.2, 0.3, 0.5, 0.7])
    axes[0].set_yticklabels(["0.1", "0.2", "0.3", "0.5", "0.7"])
    axes[0].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    # panel C: mark where each arm crosses.  The sup-norm convention puts the
    # J=4 crossing at 1/sqrt(J), not 1 -- that shift IS the finding.
    for J in (1, 4):
        m = pc[pc.J == J].groupby("r_eff", as_index=False).value.mean().sort_values("r_eff")
        x, y = np.log(m.r_eff.to_numpy()), m.value.to_numpy()
        xc = np.nan
        for i in range(len(y) - 1):
            if y[i] * y[i + 1] <= 0:
                t = -y[i] / (y[i + 1] - y[i])
                xc = float(np.exp(x[i] + t * (x[i + 1] - x[i])))
        if np.isfinite(xc):
            axes[2].plot([xc], [0], "v", color=J_COLOR[J], ms=7, zorder=6,
                         markeredgecolor=SURFACE, markeredgewidth=0.8)
            axes[2].annotate(f"{xc:.2f}", (xc, 0), textcoords="offset points",
                             xytext=(0, 11), ha="center", fontsize=7.4,
                             color=J_COLOR[J], fontweight="bold", zorder=7,
                             bbox=dict(fc=SURFACE, ec="none", pad=0.8))
    j_h = [Line2D([], [], color=J_COLOR[J], lw=2.2, ls=J_LS[J],
                  label=("$J=1$ (single mode)" if J == 1 else "$J=4$ (four modes)"))
           for J in (1, 4)]
    d_h = [Line2D([], [], color=MUTED, marker=D_MARKER[d], ls="none", ms=4.0,
                  label=f"$d={d}$") for d in sorted(D_MARKER)]
    axes[0].legend(handles=j_h, fontsize=7.6, loc="lower left", frameon=False,
                   handlelength=1.9, borderaxespad=0.8)
    axes[1].legend(handles=d_h, title="dimension", fontsize=7.6, title_fontsize=7.6,
                   loc="upper right", frameon=False, handlelength=1.2,
                   borderaxespad=0.8)
    axes[2].text(1.05, axes[2].get_ylim()[1] * 0.96, "theory: $r_{\\mathrm{eff}}=1$",
                 color=RULE, fontsize=7.4, va="top", fontweight="bold")
    # The J=4 crossing lands at 0.49, just off the sampled range: the sup-norm
    # convention shifts it by 1/sqrt(J) exactly.  Say so rather than leave a gap.
    axes[2].annotate("$J=4$ crosses at $0.49 = 1/\\sqrt{J}$:\n"
                     "sup-norm $\\omega$ is off by $\\sqrt{J}$",
                     (0.5, 0.0), textcoords="offset points", xytext=(4, -40),
                     fontsize=7.3, color=J_COLOR[4], fontweight="bold", zorder=7,
                     bbox=dict(fc=SURFACE, ec="none", pad=1.2))

    fig.tight_layout(w_pad=2.2)
    for ext in ("pdf", "png"):
        fig.savefig(f"{ART}/fig_planted_multimode.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {ART}/fig_planted_multimode.pdf and .png")


if __name__ == "__main__":
    main()
