#!/usr/bin/env python3
"""Compact figure: every seed plus task-level bootstrap intervals. READ-ONLY."""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(OUT, "task_summary.csv"))))
TASKS = ["walker", "g1", "leap"]
CONTRASTS = ["WML05/PW", "WML01/WML05", "WML01/PW"]
LABEL = {"WML05/PW": "WML$_{0.5}$ / PW", "WML01/WML05": "WML$_{0.1}$ / WML$_{0.5}$",
         "WML01/PW": "WML$_{0.1}$ / PW"}
MK = {"walker": ("o", "#17694A"), "g1": ("s", "#1F5F8B"), "leap": ("^", "#7A5A12")}

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ypos, ylabels = [], []
y = 0
for ci, c in enumerate(CONTRASTS):
    for t in TASKS:
        r = [x for x in rows if x["task"] == t and x["bank"] == "neutral"
             and x["metric"] == "V_ratio" and x["contrast"] == c][0]
        est, lo, hi = float(r["est"]), float(r["lo"]), float(r["hi"])
        seeds = [float(v) for v in r["seed_values"].split(";")]
        mk, col = MK[t]
        ax.scatter(seeds, np.full(len(seeds), y) + np.linspace(-.16, .16, len(seeds)),
                   marker=mk, s=17, facecolors="none", edgecolors=col, linewidths=.9, zorder=3)
        ax.plot([lo, hi], [y, y], color=col, lw=2.4, solid_capstyle="round", zorder=4)
        ax.plot([est], [y], marker=mk, ms=8, color=col, zorder=5)
        excl = (lo - 1) * (hi - 1) > 0
        ax.text(hi * 1.10, y, "%.2f [%.2f, %.2f]%s" % (est, lo, hi, "" if excl else "  n.d."),
                va="center", fontsize=7.4, color=col, family="monospace")
        ypos.append(y); ylabels.append("%-6s %s" % (t, LABEL[c])); y += 1
    if ci < len(CONTRASTS) - 1:
        ax.axhline(y - 0.5, color="#CCCCCC", lw=.8); y += 0.6

ax.axvline(1.0, color="#444444", ls="--", lw=1.2, zorder=2)
ax.set_xscale("log")
ax.set_xlim(0.15, 60)
ax.set_yticks(ypos); ax.set_yticklabels(ylabels, fontsize=8, family="monospace")
ax.invert_yaxis()
ax.set_xlabel("post-tanh action variance ratio  (neutral bank, log scale)", fontsize=9)
ax.set_title("Post-tanh action-distribution geometry: all 8 seeds and paired bootstrap 95% CI",
             fontsize=9.5)
ax.grid(axis="x", color="#EEEEEE", lw=.7)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.text(0.99, 0.02, "open markers = individual seeds; filled = mean log-ratio; dashed = no change",
        transform=ax.transAxes, ha="right", fontsize=6.8, color="#666666")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, "figure_posttanh.%s" % ext), dpi=200)
print("  wrote figure_posttanh.pdf / .png")
