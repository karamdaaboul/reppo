"""Four-panel figure for the MC oracle WalkerRun feasibility pilot.

Panels, fixed by docs/prereg_mc_oracle_walker_pilot.md deliverables:
  A  centred critic-error power D (RMS annotated where defined)
  B  whitened gradient energy N at c = 0.10
  C  r_RMS with 95% interval and a PRE-MARKED horizontal line at 1
  D  sensitivity: c = 0.10 vs c = 0.05, and H = 500 vs H = 1000 on the fixed subset

A and B are plotted as the SIGNED debiased cross-products on a symmetric-log axis
rather than as square roots. The cross-product estimator can and does come out
negative when Monte-Carlo noise dominates, and prereg section 8 forbids clamping such
a value to zero; plotting a square root would hide exactly the failure the pilot is
reporting. The RMS is annotated where the quantity is positive.

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
LBL = {"PW": "PW-trained\ncritic", "WML": "WML-trained\ncritic"}


def band(ax, x, pt, lo, hi, c, w=0.28, hollow=False):
    ax.plot([x - w, x + w], [pt, pt], color=c, lw=2.6, solid_capstyle="butt",
            zorder=4)
    ax.plot([x, x], [lo, hi], color=c, lw=1.4, zorder=3)
    for y in (lo, hi):
        ax.plot([x - w / 2.2, x + w / 2.2], [y, y], color=c, lw=1.2, zorder=3)


def main(res_path, boot_path, out):
    R = json.load(open(res_path))
    B = np.load(boot_path)
    fig, axes = plt.subplots(1, 4, figsize=(15.4, 4.3))

    # ---- A: debiased centred error power -----------------------------------
    ax = axes[0]
    ax.axhline(0, color="0.35", lw=1.0, zorder=1)
    for i, a in enumerate(ARMS):
        d = R[a]["point"]["D"]
        lo, hi = R[a]["ci"]["D"]
        band(ax, i, d, lo, hi, COL[a])
        txt = "D = %.3g\n$\\sqrt{D}$ = %.3g" % (d, np.sqrt(d)) if d > 0 else "D = %.3g" % d
        ax.annotate(txt, (i, hi), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8, color=COL[a])
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xticks(range(2)); ax.set_xticklabels([LBL[a] for a in ARMS], fontsize=8.5)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(r"debiased centred error power  $D$")
    ax.set_ylim(-0.6, 3.0e4)
    ax.set_title("A  centred critic error", fontsize=10, loc="left")

    # ---- B: whitened gradient energy ----------------------------------------
    ax = axes[1]
    ax.axhline(0, color="0.35", lw=1.0, zorder=1)
    for i, a in enumerate(ARMS):
        n = R[a]["point"]["N_0.10"]
        lo, hi = R[a]["ci"]["N_0.10"]
        band(ax, i, n, lo, hi, COL[a])
        ax.annotate("N = %.3g\n(interval crosses 0)" % n, (i, hi),
                    textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=8, color=COL[a])
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xticks(range(2)); ax.set_xticklabels([LBL[a] for a in ARMS], fontsize=8.5)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(r"debiased gradient energy  $N_{c=0.10}$")
    ax.set_ylim(-4.0e3, 2.0e6)
    ax.set_title(r"B  $\|\Sigma^{1/2}\nabla_y e\|^2$ — both intervals cross 0",
                 fontsize=9.5, loc="left")

    # ---- C: r_RMS against the pre-marked boundary ---------------------------
    ax = axes[2]
    ax.axhline(1.0, color="0.25", ls="--", lw=1.4, zorder=2)
    ax.annotate(r"$r_{\rm RMS}=1$  (pre-marked)", (1.62, 1.0), xytext=(0, 5),
                textcoords="offset points", fontsize=8, color="0.25", ha="right")
    for i, a in enumerate(ARMS):
        p = R[a]["point"]["r_0.10"]
        frac = float(np.isfinite(B["%s_r_0.10" % a]).mean())
        if p is None or not np.isfinite(p):
            ax.annotate("UNRESOLVED\n$N<0$; defined in only\n%.0f%% of replicates"
                        % (100 * frac), (i, 1.0), ha="center", va="center",
                        fontsize=8.5, color=COL[a],
                        bbox=dict(fc="white", ec=COL[a], lw=1.0, boxstyle="round,pad=0.4"))
        else:
            lo, hi = R[a]["ci"]["r_0.10"]
            band(ax, i, p, lo, hi, COL[a])
            ax.annotate("%.2f\n[%.2f, %.2f]\n(%.0f%% of replicates defined)"
                        % (p, lo, hi, 100 * frac), (i, hi),
                        textcoords="offset points", xytext=(0, 7), ha="center",
                        fontsize=8, color=COL[a])
    ax.set_xticks(range(2)); ax.set_xticklabels([LBL[a] for a in ARMS], fontsize=8.5)
    ax.set_xlim(-0.6, 1.7); ax.set_ylim(0.0, 3.6)
    ax.set_ylabel(r"$r_{\rm RMS}$  at $c=0.10$")
    ax.set_title("C  RMS error-frequency analogue", fontsize=10, loc="left")

    # ---- D: step-size and horizon sensitivity -------------------------------
    ax = axes[3]
    ax.axhspan(0.8, 1.25, color="0.90", zorder=0)
    ax.axhline(1.0, color="0.6", ls=":", lw=1.1, zorder=1)
    xs, labels = [], []
    for i, a in enumerate(ARMS):
        p, h = R[a]["point"], R[a]["horizon"]
        def rat(x, y):
            return x / y if (x is not None and y not in (None, 0)
                             and np.isfinite(x) and np.isfinite(y)) else np.nan
        rows = [(r"$r_{0.05}/r_{0.10}$", rat(p["r_0.05"], p["r_0.10"])),
                (r"$N_{0.05}/N_{0.10}$", rat(p["N_0.05"], p["N_0.10"])),
                (r"$N_{H1000}/N_{H500}$", h["N_ratio_1000_over_500"])]
        for j, (lab, v) in enumerate(rows):
            x = i * 3.5 + j
            if np.isfinite(v):
                ax.plot([x], [v], "o", color=COL[a], ms=8,
                        mfc=COL[a] if 0.8 <= v <= 1.25 else "white",
                        mec=COL[a], mew=1.7, zorder=4)
                ax.annotate("%.2f" % v, (x, v), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=7.5, color=COL[a])
            else:
                ax.annotate("n/a\n($r$ undefined)", (x, 1.0), ha="center",
                            va="center", fontsize=7.5, color=COL[a],
                            bbox=dict(fc="white", ec="none", pad=1.0))
            xs.append(x); labels.append(lab)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.5, rotation=35, ha="right")
    ax.set_ylim(0.55, 2.05)
    ax.set_ylabel("ratio (shaded band = pass region)")
    ax.set_title("D  step-size and horizon stability", fontsize=10, loc="left")
    for i, a in enumerate(ARMS):
        ax.annotate(LBL[a].replace("\n", " "), (i * 3.5 + 1, 1.98), ha="center",
                    fontsize=8, color=COL[a])

    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="0.93", lw=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("MC $Q^\\pi$ oracle feasibility pilot — WalkerRun seed 301, "
                 "$H{=}500$, $S{=}64$, $K{=}16$, 8+8 rollouts   —   %s"
                 % R["overall"], fontsize=10.5, y=1.03)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig("%s.%s" % (out, ext), bbox_inches="tight", dpi=170)
    print("wrote", out + ".pdf/.png")


if __name__ == "__main__":
    main(*sys.argv[1:4])
