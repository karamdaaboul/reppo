"""Four-panel figure for the MC oracle WalkerRun feasibility pilot.

Panels, fixed by docs/prereg_mc_oracle_walker_pilot.md deliverables:
  A  centered critic-error RMS, PW- and WML-trained checkpoints
  B  whitened gradient-energy RMS
  C  r_RMS with 95% interval and a PRE-MARKED horizontal line at 1
  D  sensitivity: c = 0.10 vs c = 0.05, and H = 500 vs H = 1000 on the fixed subset

Return is deliberately not plotted anywhere in this figure.

Usage: mc_oracle_figure.py <results.json> <boot.npz> <out_prefix>
"""

from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ARMS = ("PW", "WML")
COL = {"PW": "#1f4e79", "WML": "#c0392b"}
LBL = {"PW": "PW-trained critic", "WML": "WML-trained critic"}


def band(ax, x, pt, lo, hi, c, w=0.30):
    ax.plot([x - w, x + w], [pt, pt], color=c, lw=2.4, solid_capstyle="butt")
    ax.plot([x, x], [lo, hi], color=c, lw=1.4)
    for y in (lo, hi):
        ax.plot([x - w / 2.4, x + w / 2.4], [y, y], color=c, lw=1.2)


def main(res_path, boot_path, out):
    R = json.load(open(res_path))
    B = np.load(boot_path)
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.0))

    # ---- A: centered error RMS (sqrt of the debiased D) ----------------------
    ax = axes[0]
    for i, a in enumerate(ARMS):
        d = R[a]["point"]["D"]
        bd = B["%s_D" % a]
        pos = bd[np.isfinite(bd)]
        lo, hi = np.percentile(pos, [2.5, 97.5])
        f = lambda v: np.sqrt(v) if v > 0 else np.nan
        band(ax, i, f(d), f(lo), f(hi), COL[a])
        ax.annotate("%.3f" % f(d), (i, f(d)), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color=COL[a])
    ax.set_xticks(range(2)); ax.set_xticklabels([LBL[a] for a in ARMS], fontsize=8.5)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(r"centred error RMS  $\sqrt{D}$  (value units)")
    ax.set_title("A  centred critic error", fontsize=10, loc="left")

    # ---- B: whitened gradient-energy RMS ------------------------------------
    ax = axes[1]
    for i, a in enumerate(ARMS):
        n = R[a]["point"]["N_0.10"]
        bn = B["%s_N_0.10" % a]
        lo, hi = np.percentile(bn[np.isfinite(bn)], [2.5, 97.5])
        f = lambda v: np.sqrt(v) if v > 0 else np.nan
        band(ax, i, f(n), f(lo), f(hi), COL[a])
        ax.annotate("%.3f" % f(n), (i, f(n)), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color=COL[a])
    ax.set_xticks(range(2)); ax.set_xticklabels([LBL[a] for a in ARMS], fontsize=8.5)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(r"whitened gradient RMS  $\sqrt{N_{c=0.10}}$")
    ax.set_title(r"B  $\|\Sigma^{1/2}\nabla_y e\|$", fontsize=10, loc="left")

    # ---- C: r_RMS against the pre-marked boundary ---------------------------
    ax = axes[2]
    ax.axhline(1.0, color="0.25", ls="--", lw=1.3, zorder=0)
    ax.annotate(r"$r_{\rm RMS}=1$", (1.62, 1.0), xytext=(0, 4),
                textcoords="offset points", fontsize=8.5, color="0.25", ha="right")
    for i, a in enumerate(ARMS):
        p = R[a]["point"]["r_0.10"]
        lo, hi = R[a]["ci"]["r_0.10"]
        band(ax, i, p, lo, hi, COL[a])
        ax.annotate("%.2f\n[%.2f, %.2f]" % (p, lo, hi), (i, hi),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=8, color=COL[a])
    ax.set_xticks(range(2)); ax.set_xticklabels([LBL[a] for a in ARMS], fontsize=8.5)
    ax.set_xlim(-0.6, 1.7)
    ax.set_yscale("log")
    ax.set_ylabel(r"$r_{\rm RMS}$  (95% bootstrap interval)")
    ax.set_title("C  RMS error-frequency analogue", fontsize=10, loc="left")

    # ---- D: step-size and horizon sensitivity -------------------------------
    ax = axes[3]
    ax.axhline(1.0, color="0.6", ls=":", lw=1.1, zorder=0)
    ax.axhspan(0.8, 1.25, color="0.90", zorder=0)
    xs, labels = [], []
    for i, a in enumerate(ARMS):
        p = R[a]["point"]
        h = R[a]["horizon"]
        rows = [
            (r"$r_{0.05}/r_{0.10}$", p["r_0.05"] / p["r_0.10"] if p["r_0.10"] else np.nan),
            (r"$N_{0.05}/N_{0.10}$", p["N_0.05"] / p["N_0.10"] if p["N_0.10"] else np.nan),
            (r"$N_{H1000}/N_{H500}$", h["N_ratio_1000_over_500"]),
        ]
        for j, (lab, v) in enumerate(rows):
            x = i * 3.4 + j
            ax.plot([x], [v], "o", color=COL[a], ms=7,
                    mfc=COL[a] if 0.8 <= v <= 1.25 else "white",
                    mec=COL[a], mew=1.6)
            xs.append(x); labels.append(lab)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.5, rotation=35, ha="right")
    ax.set_ylabel("ratio (shaded band = pass region)")
    ax.set_title("D  step-size and horizon stability", fontsize=10, loc="left")
    for i, a in enumerate(ARMS):
        ax.annotate(LBL[a], (i * 3.4 + 1, ax.get_ylim()[1]), xytext=(0, -11),
                    textcoords="offset points", ha="center", fontsize=8, color=COL[a])

    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="0.92", lw=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("MC $Q^\\pi$ oracle feasibility pilot — WalkerRun seed 301, "
                 "$H=500$, $S=64$, $K=16$, 8+8 rollouts   (%s)" % R["overall"],
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig("%s.%s" % (out, ext), bbox_inches="tight", dpi=170)
    print("wrote", out + ".pdf/.png")


if __name__ == "__main__":
    main(*sys.argv[1:4])
