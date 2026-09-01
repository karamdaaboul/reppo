"""Publication figure for the ubar-ratio audit. Four preregistered panels."""
import numpy as np, pandas as pd, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT, FIG = "reports/artifacts", "reports/figures"
os.makedirs(FIG, exist_ok=True)
df = pd.read_csv(f"{OUT}/ubar_per_checkpoint.csv")
ps = pd.read_csv(f"{OUT}/ubar_per_state.csv")
COL = {"A": "#2b6cb0", "B": "#c05621"}
order = df.sort_values(["d", "pad"]).condition.unique().tolist()
lab = {c: c for c in order}

fig = plt.figure(figsize=(13.6, 9.0))
gs = fig.add_gridspec(2, 2, hspace=.34, wspace=.26)

# --- P1: R2_exact by condition and arm (NO sqrt(d/M) line here) --------------
ax = fig.add_subplot(gs[0, 0])
for i, cond in enumerate(order):
    for j, arm in enumerate(("A", "B")):
        s = df[(df.condition == cond) & (df.arm == arm)]
        if not len(s): continue
        x = np.full(len(s), i + (j - .5) * .32)
        ax.scatter(x + np.random.default_rng(i * 2 + j).normal(0, .035, len(s)),
                   s.R2_exact, s=42, color=COL[arm], edgecolor="k", linewidth=.35, zorder=3)
        ax.hlines(s.R2_exact.median(), i + (j - .5) * .32 - .12,
                  i + (j - .5) * .32 + .12, color="k", lw=2.1, zorder=4)
ax.set_yscale("log")
ax.axhline(1.0, color="grey", ls=":", lw=1.1)
ax.text(len(order) - .45, 1.06, r"$R_2=1$: equal energy", fontsize=7.5, color="grey", ha="right")
ax.set_xticks(range(len(order)))
ax.set_xticklabels(["%s\nd=%d" % (c, df[df.condition == c].d.iloc[0]) for c in order], fontsize=8)
ax.set_ylabel(r"$R_2^{\mathrm{exact}}=\sqrt{\sum\|\bar u\|^2/\sum\|c\|^2}$")
ax.set_title("(a) Uniform vs centered energy, per checkpoint\n"
             "seed is the unit; bars are condition medians", fontsize=9.5)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=COL[a], label="arm %s" % a)
                   for a in ("A", "B")], fontsize=8, loc="best")

# --- P2: direction change ----------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
for i, cond in enumerate(order):
    for j, arm in enumerate(("A", "B")):
        s = ps[(ps.condition == cond) & (ps.arm == arm)]
        if not len(s): continue
        p = ax.violinplot([s.cos_v_c.values], positions=[i + (j - .5) * .32],
                          widths=.28, showmedians=True)
        for b in p["bodies"]:
            b.set_facecolor(COL[arm]); b.set_alpha(.6); b.set_edgecolor("k"); b.set_linewidth(.4)
        for k in ("cmedians", "cbars", "cmins", "cmaxes"):
            if k in p: p[k].set_color("k"); p[k].set_linewidth(.9)
ax.axhline(1.0, color="grey", ls=":", lw=1.1)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(["%s\nd=%d" % (c, df[df.condition == c].d.iloc[0]) for c in order], fontsize=8)
ax.set_ylabel(r"$\cos(v,\,c)$")
ax.set_title(r"(b) Does $\bar u$ move the direction? state-level $\cos(v,c)$"
             "\n" r"$\cos=1$ would mean $\bar u$ changes nothing", fontsize=9.5)

# --- P3: raw-Gaussian RMS sanity check vs sqrt(d/M) --------------------------
ax = fig.add_subplot(gs[1, 0])
dd = np.array(sorted(df.d.unique()))
ax.plot(dd, np.sqrt(dd / 32), "k--", lw=1.4, zorder=1, label=r"$\sqrt{d/M}$ (exact)")
for arm in ("A", "B"):
    s = df[df.arm == arm]
    ax.scatter(s.d + (0.25 if arm == "B" else -0.25), s.ubar_raw_rms, s=42,
               color=COL[arm], edgecolor="k", linewidth=.35, zorder=3, label="arm %s" % arm)
ax.set_xlabel("estimator-visible $d$")
ax.set_ylabel(r"RMS $\|\bar u_{\mathrm{raw}}\|$")
ax.set_title("(c) Raw-Gaussian probe sanity check\n"
             r"RMS (not median) is the quantity $\sqrt{d/M}$ predicts", fontsize=9.5)
ax.legend(fontsize=8)

# --- P4: linearization residual vs logit spread ------------------------------
ax = fig.add_subplot(gs[1, 1])
for arm in ("A", "B"):
    s = ps[ps.arm == arm]
    ax.scatter(s.logit_spread, s.residual_linear, s=4, alpha=.16,
               color=COL[arm], linewidths=0, rasterized=True)
q = ps.copy()
q["bin"] = pd.cut(np.log10(np.maximum(q.logit_spread, 1e-6)), 24)
m = q.groupby("bin", observed=True).agg(x=("logit_spread", "median"),
                                        y=("residual_linear", "median")).dropna()
ax.plot(m.x, m.y, "k-", lw=1.8, zorder=5, label="binned median")
ax.axhline(0.25, color="crimson", lw=1.5, zorder=6)
ax.text(0.02, 0.30, "preregistered P1 threshold 0.25", color="crimson", fontsize=8)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-4, 2e3)
ax.set_xlabel(r"logit spread  $\mathrm{sd}_i(Q_i/\eta)$")
ax.set_ylabel(r"$\|c-\hat m/\eta\|\,/\,\|c\|$")
ax.set_title("(d) First-order expansion degrades as the logits spread\n"
             "points are states; both arms", fontsize=9.5)
ax.legend(fontsize=8, loc="upper left")

fig.suptitle("The uniform empirical-mean term in the implemented E-step "
             "(frozen checkpoints, read-only)", fontsize=11.5, y=.985)
for ext in ("png", "pdf"):
    fig.savefig(f"{FIG}/fig_ubar_ratio.{ext}", dpi=170, bbox_inches="tight")
print("wrote %s/fig_ubar_ratio.{png,pdf}" % FIG)
