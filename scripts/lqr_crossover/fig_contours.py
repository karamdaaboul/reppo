"""The figure docs/prereg_lqr_crossover.md line 239 commits to.

sigma x omega crossover contours per d -- the locus where the error-only variances of
the zeroth-order and pathwise operators are equal (log Var_ZO_e - log Var_PW_e = 0),
drawn on the swept grid from the batch-averaged, state-averaged s3 statistics of the
rank-one .npz files -- with the band sigma < 0.1 shaded because the DMC arms cannot
reach it: `min_std = 0.1` is hardcoded at src/networks/jax_models.py:336 and
`ReppoConfig.actor_min_std` is dead config (prereg Sec. 6).

Regenerable from the .npz files listed in reports/artifacts/lqr_npz_manifest.csv (they
are git-ignored; the manifest carries their sha256).  The per-cell grid this script
contours is also written to reports/figures/fig_lqr_crossover_contours_data.csv so the
drawing can be reproduced without the 538 MB inputs.

    JAX_PLATFORMS=cpu python scripts/lqr_crossover/fig_contours.py
"""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
import scripts.lqr_crossover  # noqa: F401,E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scripts.lqr_crossover import OUT, analyze as A  # noqa: E402

FIG_DIR = os.path.join(REPO_ROOT, "reports", "figures")
DS_REGISTERED = (1, 2, 4, 8, 16, 32, 64)
D_POSTHOC = 6                     # prereg addendum A1
MIN_STD = 0.1                     # src/networks/jax_models.py:336

# Direct labels carry identity; colour does not.  Registered arms in one ink, the
# post-hoc arm in the second categorical slot and dashed so it reads in grayscale.
INK, INK2, MUTED = "#111111", "#444444", "#9a9a9a"
ACCENT = "#eb6834"
SURFACE = "#ffffff"


def grid_logratio(z):
    pw, zo = A.err_only(z)                                    # (states, sig, om)
    r = np.log(np.maximum(zo, 1e-300)) - np.log(np.maximum(pw, 1e-300))
    return r.mean(0)                                          # (sig, om)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    rows, series = [], []
    for d in sorted(DS_REGISTERED + (D_POSTHOC,)):
        p = os.path.join(OUT, f"d{d}_rank1_M32_unit_H_identity.npz")
        if not os.path.exists(p):
            continue
        z = A.load(p)
        R = grid_logratio(z)
        c_star, ok = A.crossover_by_c(z)
        series.append((d, z["sigmas"], z["omegas"], R, c_star))
        for i, s in enumerate(z["sigmas"]):
            for j, w in enumerate(z["omegas"]):
                rows.append(dict(d=d, sigma=float(s), omega=float(w),
                                 log_ratio_zo_over_pw=float(R[i, j]), c_star=c_star))
    pd.DataFrame(rows).to_csv(
        os.path.join(FIG_DIR, "fig_lqr_crossover_contours_data.csv"), index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.edgecolor": INK2, "axes.linewidth": 0.7,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        # deterministic output so regeneration is byte-identical
        "svg.hashsalt": "lqr", "pdf.compression": 6,
    })
    fig, ax = plt.subplots(figsize=(6.6, 4.4))

    sig0, om0 = series[0][1], series[0][2]
    ax.axhspan(sig0.min() / 1.5, MIN_STD, color="#e6e6e6", zorder=0)
    ax.axhline(MIN_STD, color=MUTED, lw=0.8, zorder=1)
    ax.text(om0.min() * 1.15, MIN_STD * 0.86,
            r"$\sigma<0.1$: unreachable for the DMC arms"
            "\n(min_std = 0.1 hardcoded, jax_models.py:336)",
            fontsize=7.4, color=INK2, va="top", zorder=6)

    # Labels ride on their own contour at staggered heights, rotated along the line,
    # so eight closely spaced hyperbolae stay legible; no gaps are cut in the lines.
    top = float(sig0.max())
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(om0.min(), om0.max()); ax.set_ylim(sig0.min(), sig0.max())
    fig.canvas.draw()
    p0 = ax.transData.transform((1.0, 1.0)); p1 = ax.transData.transform((10.0, 0.1))
    rot = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
    for k, (d, sig, om, R, c_star) in enumerate(series):
        post = d == D_POSTHOC
        ax.contour(om, sig, R, levels=[0.0],
                   colors=[ACCENT if post else INK],
                   linestyles=["--" if post else "-"],
                   linewidths=[1.5 if post else 1.2], zorder=4)
        y_lab = 2.2 * 0.60 ** k
        ax.annotate(f"$d={d}$" + ("*" if post else ""), (c_star / y_lab, y_lab),
                    ha="center", va="center", fontsize=7.2, rotation=rot,
                    rotation_mode="anchor", color=ACCENT if post else INK, zorder=7,
                    bbox=dict(fc=SURFACE, ec="none", pad=0.6))
    ax.text(0.985, 0.60, "zeroth-order wins\nabove / right of each contour",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.4, color=INK2)

    ax.set_xlabel(r"error frequency $\omega$")
    ax.set_ylabel(r"policy width $\sigma$")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(which="minor", length=2, color=MUTED)

    handles = [
        Line2D([], [], color=INK, lw=1.2, label=r"$\mathrm{Var}_e[\hat g_{ZO}]=\mathrm{Var}_e[\hat g_{PW}]$, registered $d$"),
        Line2D([], [], color=ACCENT, lw=1.5, ls="--", label=r"same, $d=6^*$ (post hoc, addendum A1)"),
        Patch(facecolor="#e6e6e6", edgecolor="none", label=r"$\sigma<0.1$, unreachable"),
    ]
    ax.legend(handles=handles, fontsize=7.4, loc="upper right", frameon=False,
              handlelength=2.0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_lqr_crossover_contours.{ext}"),
                    dpi=300, bbox_inches="tight", metadata={"CreationDate": None}
                    if ext == "pdf" else {"Software": None})
    plt.close(fig)
    print("wrote", FIG_DIR, "fig_lqr_crossover_contours.{pdf,png,_data.csv};",
          "d =", [s[0] for s in series])


if __name__ == "__main__":
    main()
