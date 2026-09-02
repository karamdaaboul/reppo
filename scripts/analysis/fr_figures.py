import numpy as np, pandas as pd, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT, FIG = "reports/artifacts", "reports/figures"
os.makedirs(FIG, exist_ok=True)
R = pd.read_csv(f"{OUT}/corrected_runs.csv")
P = pd.read_csv(f"{OUT}/corrected_paired_results.csv")
D = pd.read_csv(f"{OUT}/corrected_diagnostics.csv")
COL = {"PW-1-faithful-repair": "#2b6cb0", "WML-32-faithful-repair": "#c05621"}
SH = {"PW-1-faithful-repair": "PW-1", "WML-32-faithful-repair": "WML-32"}
TASKS = [("walker", 6, "WalkerRun"), ("g1", 29, "G1JoystickFlatTerrain")]

# ---------------- learning curves ----------------
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4),
                         gridspec_kw={"width_ratios": [1, 1, 0.85]})
for ax, (tk, dd, nm) in zip(axes[:2], TASKS):
    for arm in COL:
        sub = R[(R.task == tk) & (R.arm == arm)]
        cur = np.array([[float(x) for x in c.split(";")] for c in sub.ret_curve])
        xs = np.arange(cur.shape[1])
        for row in cur:
            ax.plot(xs, row, color=COL[arm], alpha=.30, lw=.9)
        ax.plot(xs, np.median(cur, 0), color=COL[arm], lw=2.6, label=SH[arm])
    ax.axvspan(17.5, 20.5, color="grey", alpha=.15, lw=0)
    ax.text(18.0, ax.get_ylim()[0], " final\n window", fontsize=7, color="grey", va="bottom")
    ax.set_title("%s  (d=%d)" % (nm, dd), fontsize=10)
    ax.set_xlabel("evaluation index"); ax.set_ylabel("episode return")
    ax.legend(fontsize=8, loc="lower right")
ax = axes[2]
for i, (tk, dd, nm) in enumerate(TASKS):
    p = P[P.task == tk].iloc[0]
    dif = np.array([float(x) for x in p.diffs.split(";")])
    ax.scatter(np.full(len(dif), i) + np.random.default_rng(i).normal(0, .05, len(dif)),
               dif, s=48, color="#2b6cb0", edgecolor="k", linewidth=.4, zorder=3)
    ax.hlines(p.median_diff, i - .22, i + .22, color="k", lw=2.4, zorder=4)
    ax.vlines(i, p.ci_lo, p.ci_hi, color="k", lw=1.4, zorder=4)
    ax.text(i + .27, p.median_diff, "%+.1f\n[%+.1f, %+.1f]\n%d/8, p=%.4f"
            % (p.median_diff, p.ci_lo, p.ci_hi, p.n_pos, p.p_exact), fontsize=7.5, va="center")
ax.axhline(0, color="crimson", lw=1.6)
ax.set_xticks(range(2)); ax.set_xticklabels(["%s\nd=%d" % (t[0], t[1]) for t in TASKS])
ax.set_xlim(-.5, 1.9); ax.set_yscale("symlog", linthresh=1)
ax.set_ylabel(r"paired difference  PW-1 $-$ WML-32")
ax.set_title("Paired seed differences\n(positive = pathwise higher)", fontsize=10)
fig.suptitle("Faithful-repair replication: 2 tasks x 2 arms x 8 paired seeds (301-308). "
             "Old and corrected experiments are never pooled.", fontsize=11, y=1.02)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"{FIG}/fig_corrected_learning_curves.{e}", dpi=170, bbox_inches="tight")
plt.close(fig)

# ---------------- operator diagnostics ----------------
fig, axes = plt.subplots(1, 4, figsize=(16, 4.0))
ax = axes[0]
for i, (tk, dd, nm) in enumerate(TASKS):
    for j, arm in enumerate(("PW-1", "WML-32")):
        s = D[(D.task == tk) & (D.arm == arm)]
        ax.scatter(np.full(len(s), i + (j - .5) * .3), s.gate_op, s=40,
                   color=list(COL.values())[j], edgecolor="k", linewidth=.35, zorder=3)
        ax.hlines(s.gate_op.median(), i + (j - .5) * .3 - .1, i + (j - .5) * .3 + .1,
                  color="k", lw=2, zorder=4)
ax.set_xticks(range(2)); ax.set_xticklabels([t[0] for t in TASKS]); ax.set_ylim(0, 1)
ax.set_ylabel("fraction of states on the operator branch")
ax.set_title("(a) Published gate: how often the\noperator objective survives", fontsize=9)
ax = axes[1]
ax.scatter(D.kl_analytic, D.kl_sampled, c=[COL["PW-1-faithful-repair"] if a == "PW-1"
           else COL["WML-32-faithful-repair"] for a in D.arm], s=42, edgecolor="k", linewidth=.35)
lim = [min(D.kl_analytic.min(), D.kl_sampled.min()) * .95,
       max(D.kl_analytic.max(), D.kl_sampled.max()) * 1.05]
ax.plot(lim, lim, "k--", lw=1.2); ax.axhline(0.1, color="crimson", lw=1, ls=":")
ax.axvline(0.1, color="crimson", lw=1, ls=":")
ax.set_xlabel("analytic Gaussian KL"); ax.set_ylabel("sampled KL (M=32)")
ax.set_title("(b) The repair: sampled KL now tracks\nthe true KL (corr 0.975-1.000)", fontsize=9)
ax = axes[2]
for j, arm in enumerate(("PW-1", "WML-32")):
    s = D[D.arm == arm]
    ax.scatter(s.sigma, s.kl_analytic, s=42, color=list(COL.values())[j],
               edgecolor="k", linewidth=.35, label=arm)
ax.set_xlabel(r"policy $\sigma$ (mean)"); ax.set_ylabel("analytic KL")
ax.set_title("(c) Policy width vs trust-region load", fontsize=9); ax.legend(fontsize=8)
ax = axes[3]
for j, arm in enumerate(("PW-1", "WML-32")):
    s = D[D.arm == arm]
    ax.scatter(np.arange(len(s)) + j * .0, s.lag, s=42, color=list(COL.values())[j],
               edgecolor="k", linewidth=.35, label=arm)
ax.set_yscale("log"); ax.set_xlabel("run"); ax.set_ylabel(r"effective KL multiplier $\lambda$")
ax.set_title("(d) Published exponential multiplier\n(unbounded by design, logged)", fontsize=9)
ax.legend(fontsize=8)
fig.suptitle("Corrected-implementation operator diagnostics. Diagnostic only: none of "
             "this entered training, the gate, the dual, or any exclusion.", fontsize=10, y=1.03)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"{FIG}/fig_corrected_operator_diagnostics.{e}", dpi=170, bbox_inches="tight")
plt.close(fig)
print("wrote fig_corrected_learning_curves.{png,pdf} and fig_corrected_operator_diagnostics.{png,pdf}")
