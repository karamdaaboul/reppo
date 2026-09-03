#!/usr/bin/env python
"""Appendix figure: what happens as the critic error goes 1 -> 4 -> 8 waves.

Panel A: E-step systematic displacement B, for J = 1, 4 (both seeds) and 8.
Panel B: total update error for pathwise and the E-step, J = 4 and J = 8.
Panel C: systematic vs noise contribution, per operator, at J = 1, 4, 8.

All panels are restricted to d in {16, 64}, the dimensions the J = 8 arm can use
(eight orthonormal directions do not exist in d = 4), so the three J values are
compared on the same grid.

    ./.venv/bin/python scripts/planted/make_j1_j4_j8_figure.py
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ART = "reports/artifacts"
NB = 10000
DJ8 = [16, 64]

# J is an ordered quantity -> one-hue ordinal ramp, steps 250/450/650 (validated
# light-mode --ordinal: monotone L, adjacent dL >= 0.06, light end 2.06:1).
J_COLOR = {1: "#86b6ef", 4: "#2a78d6", 8: "#104281"}
# Panel C components are two entities -> categorical slots 1 and 2
# (blue<->orange all-pairs light: CVD dE 24.7, normal dE 33.6).
C_SYS, C_NOI = "#eb6834", "#2a78d6"
D_MARKER = {16: "s", 64: "^"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURFACE = "#fcfcfb"

RUNS = {"old": ("planted_multimode", 20260904),
        "rep": ("planted_j4_replication", 20260905),
        "j8": ("planted_j8", 20260906)}


def load(tag):
    pre, seed = RUNS[tag]
    return (pd.read_csv(f"{ART}/{pre}_replicates.csv"),
            pd.read_csv(f"{ART}/{pre}.csv"), seed)


def curve(rep, J, seed, col="B_b"):
    """Per-r_eff mean with bootstrap CI over blocks, restricted to d in {16,64}."""
    rng = np.random.default_rng(seed)
    rows = []
    g0 = rep[(rep.J == J) & (rep.d.isin(DJ8))]
    for r, g in g0.groupby("r_eff"):
        v = g[col].to_numpy()
        bs = v[rng.integers(0, len(v), size=(NB, len(v)))].mean(1)
        rows.append(dict(r_eff=r, value=float(v.mean()),
                         ci_lo=float(np.percentile(bs, 2.5)),
                         ci_hi=float(np.percentile(bs, 97.5))))
    return pd.DataFrame(rows).sort_values("r_eff")


def decomp(cell, J, nm):
    g = cell[(cell.J == J) & (cell.d.isin(DJ8))]
    fl = g[f"err_{nm}_baseline"].median()
    eo = g[f"err_{nm}_erroroff"].median()
    tt = g[f"err_{nm}_total"].median()
    return eo - fl, tt - eo


def main():
    rep_o, cell_o, s_o = load("old")
    rep_r, cell_r, s_r = load("rep")
    rep_8, cell_8, s_8 = load("j8")

    series = [("$J=1$", 1, rep_o, s_o, "-", J_COLOR[1]),
              ("$J=4$", 4, rep_o, s_o, "-", J_COLOR[4]),
              ("$J=4$ (replication)", 4, rep_r, s_r, ":", J_COLOR[4]),
              ("$J=8$", 8, rep_8, s_8, "-", J_COLOR[8])]

    src = []
    for lbl, J, rp, sd, _, _ in series:
        c = curve(rp, J, sd)
        c["series"], c["panel"] = lbl, "A_B_systematic"
        src.append(c)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.9))

    # ---------------- Panel A
    ax = axes[0]
    for lbl, J, rp, sd, ls, col in series:
        c = curve(rp, J, sd)
        ax.fill_between(c.r_eff, c.ci_lo, c.ci_hi, color=col, alpha=0.15, lw=0, zorder=2)
        ax.plot(c.r_eff, c.value, ls, color=col, lw=2.0, zorder=4)
        for d in DJ8:
            g = rp[(rp.J == J) & (rp.d == d)].groupby("r_eff").B_b.mean()
            ax.plot(g.index, g.values, D_MARKER[d], color=col, ms=4.0, zorder=5,
                    markeredgecolor=SURFACE, markeredgewidth=0.7)
    ax.set_yscale("log")
    ax.set_yticks([0.05, 0.1, 0.2, 0.3, 0.5])
    ax.set_yticklabels(["0.05", "0.1", "0.2", "0.3", "0.5"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylabel(r"$B=\mathbb{E}_{\mathrm{field}}\ \|\delta\|/\|s\|$", fontsize=8.1)
    ax.set_title("A  E-step systematic displacement", fontsize=9.0, loc="left", pad=8)

    # ---------------- Panel B
    ax = axes[1]
    OP = {"wml": ("E-step", "#eb6834"), "pw": ("pathwise", "#2a78d6")}
    for J, rp, sd, ls in ((4, rep_o, s_o, "-"), (8, rep_8, s_8, "--")):
        for nm, (onm, col) in OP.items():
            c = curve(rp, J, sd, col=f"mse_{nm}")
            c["series"], c["panel"] = f"{onm} J={J}", "B_update_error"
            src.append(c)
            ax.fill_between(c.r_eff, c.ci_lo, c.ci_hi, color=col, alpha=0.13, lw=0,
                            zorder=2)
            ax.plot(c.r_eff, c.value, ls, color=col, lw=2.0, zorder=4)
    ax.set_ylabel(r"update error $\mathrm{Err}$", fontsize=8.1)
    ax.set_title("B  Total update error", fontsize=9.0, loc="left", pad=8)

    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_xlabel(r"$r_{\mathrm{eff}}=\sigma\,\omega_{\mathrm{eff}}/\sqrt{d}$")
        ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3])
        ax.set_xticklabels(["0.5", "0.75", "1", "1.5", "2", "3"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    # ---------------- Panel C
    ax = axes[2]
    groups = [(1, cell_o, "$J{=}1$"), (4, cell_o, "$J{=}4$"), (8, cell_8, "$J{=}8$")]
    xs, labels, width = [], [], 0.36
    pos = 0.0
    for J, cl, jl in groups:
        for nm, onm in (("pw", "PW"), ("wml", "E-step")):
            sy, no = decomp(cl, J, nm)
            ax.bar(pos - width / 2, sy, width, color=C_SYS, zorder=3,
                   edgecolor=SURFACE, linewidth=1.4)
            ax.bar(pos + width / 2, no, width, color=C_NOI, zorder=3,
                   edgecolor=SURFACE, linewidth=1.4)
            src.append(pd.DataFrame([dict(panel="C_decomposition", series=f"{onm} J={J}",
                                          systematic=sy, noise=no)]))
            labels.append(f"{onm}\n{jl}")
            xs.append(pos)
            pos += 1.0
        pos += 0.35
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_ylabel("contribution to update error", fontsize=8.1)
    ax.set_title("C  Systematic vs noise, by operator", fontsize=9.0, loc="left", pad=8)
    ax.legend(handles=[Line2D([], [], color=C_SYS, lw=6, label="systematic"),
                       Line2D([], [], color=C_NOI, lw=6, label="noise")],
              fontsize=7.6, loc="upper left", frameon=False, handlelength=1.1)

    for ax in axes:
        ax.grid(True, axis="y", color="#e8e7e3", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    j_h = [Line2D([], [], color=c, lw=2.2, ls=ls, label=lbl)
           for lbl, _, _, _, ls, c in series]
    axes[0].legend(handles=j_h, fontsize=7.3, loc="upper right", frameon=False,
                   handlelength=1.9, borderaxespad=0.7)
    axes[1].legend(handles=[Line2D([], [], color=OP[k][1], lw=2.2, label=OP[k][0])
                            for k in ("pw", "wml")]
                   + [Line2D([], [], color=MUTED, lw=1.6, ls=l, label=f"$J={J}$")
                      for J, l in ((4, "-"), (8, "--"))],
                   fontsize=7.3, loc="upper left", frameon=False, handlelength=1.9)

    fig.tight_layout(w_pad=2.2)
    for ext in ("pdf", "png"):
        fig.savefig(f"{ART}/fig_planted_j1_j4_j8.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.concat(src, ignore_index=True).to_csv(
        f"{ART}/fig_planted_j1_j4_j8_data.csv", index=False)
    print(f"wrote {ART}/fig_planted_j1_j4_j8.pdf/.png and _data.csv")


if __name__ == "__main__":
    main()
