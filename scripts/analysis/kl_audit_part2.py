#!/usr/bin/env python
"""Part 2 of the read-only KL audit: within-arm associations and figures.

The pooled (n=16) correlations in part 1 mix the two arms and therefore mostly
re-express the arm contrast itself.  Everything here is computed WITHIN arm
(n=8) so that the arm difference cannot manufacture the association.  All of it
is descriptive: n=8, no causal claim, no mediation.
"""
import numpy as np, pandas as pd, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT = sys.argv[1] if len(sys.argv) > 1 else "reports/artifacts"
df = pd.read_csv(f"{OUT}/kl_per_run.csv")
KL_BOUND = 0.1

print("=== WITHIN-ARM Spearman, n=8 (descriptive; |rho|>=0.738 is p<0.05 two-sided) ===")
recs = []
for t in ["hopper", "walker", "leap", "g1"]:
    for arm in ["A", "B"]:
        s = df[(df.task == t) & (df.arm == arm)]
        for col in ["kl_iqr", "F_run", "overshoot_run", "lag_med", "sigma_mean_last",
                    "abs_pred_action_last", "value_loss_last"]:
            rho, p = stats.spearmanr(s[col], s["final_return"])
            recs.append(dict(task=t, d=s.d.iloc[0], arm=arm, x=col, rho=rho, p=p, n=len(s)))
w = pd.DataFrame(recs)
w.to_csv(f"{OUT}/within_arm_assoc.csv", index=False)
piv = w.pivot_table(index=["d", "task", "arm"], columns="x", values="rho").round(3)
print(piv.to_string())

print("\n=== The one that matters: g1 arm B, KL dispersion vs return ===")
g = df[(df.task == "g1") & (df.arm == "B")].sort_values("kl_iqr")
print(g[["seed", "kl_iqr", "kl_q75", "overshoot_run", "F_run", "lag_med",
         "final_return"]].to_string(index=False))
rho, p = stats.spearmanr(g.kl_iqr, g.final_return)
print(f"spearman(kl_iqr, final_return | g1, arm B, n=8) = {rho:+.3f}  p={p:.3f}")
rho2, p2 = stats.spearmanr(g.overshoot_run, g.final_return)
print(f"spearman(overshoot, final_return | g1, arm B, n=8) = {rho2:+.3f}  p={p2:.3f}")

# ---------------- figures ----------------
tasks = [("hopper", 4), ("walker", 6), ("leap", 16), ("g1", 29)]
COL = {"A": "#2b6cb0", "B": "#c05621"}

# Fig 1: KL curves, all 64 runs
fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), sharex=True)
for j, (t, d) in enumerate(tasks):
    for i, arm in enumerate(["A", "B"]):
        ax = axes[i, j]
        for _, r in df[(df.task == t) & (df.arm == arm)].iterrows():
            k = np.load(f"{OUT}/kl_curve_{r.run}.npy")
            ax.plot(k, color=COL[arm], alpha=.55, lw=1.1)
        ax.axhline(KL_BOUND, color="k", ls="--", lw=1)
        ax.set_title(f"{t} (d={d}) arm {arm}", fontsize=9)
        ax.set_ylim(0, .32)
        if j == 0: ax.set_ylabel("logged KL (batch mean)")
        if i == 1: ax.set_xlabel("evaluation index")
fig.suptitle("Logged actor KL vs the 0.1 bound — all 64 confirmatory runs\n"
             "dashed line = cfg.kl_bound; a point on or above it PROVES the defective "
             "branch fired for at least one state", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_kl_curves.png", dpi=160); plt.close(fig)

# Fig 2: KL multiplier trajectories (log scale)
fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
for j, (t, d) in enumerate(tasks):
    ax = axes[j]
    for arm in ["A", "B"]:
        for _, r in df[(df.task == t) & (df.arm == arm)].iterrows():
            ax.plot(np.load(f"{OUT}/lag_curve_{r.run}.npy"), color=COL[arm], alpha=.55, lw=1.1)
    ax.set_yscale("log"); ax.set_title(f"{t} (d={d})", fontsize=9)
    ax.set_xlabel("evaluation index")
axes[0].set_ylabel(r"KL multiplier $\lambda$")
fig.suptitle(r"Recovered KL-multiplier trajectory (blue = arm A pathwise, orange = arm B weighted-MLE)."
             "\n"r"$\lambda$ is 35–112$\times$ larger in arm B on EVERY task, so a large $\lambda$ is not g1-specific.",
             fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_lagrangian.png", dpi=160); plt.close(fig)

# Fig 3: the g1-specific dispersion contrast + within-arm scatter
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
ax = axes[0]
xs = np.arange(4)
for i, arm in enumerate(["A", "B"]):
    vals = [df[(df.task == t) & (df.arm == arm)].kl_iqr.values for t, _ in tasks]
    bp = ax.boxplot(vals, positions=xs + (i - .5) * .3, widths=.26, patch_artist=True,
                    medianprops=dict(color="k"))
    for b in bp["boxes"]: b.set_facecolor(COL[arm]); b.set_alpha(.65)
ax.set_xticks(xs); ax.set_xticklabels([f"{t}\nd={d}" for t, d in tasks])
ax.set_ylabel("KL IQR over evaluations"); ax.set_yscale("log")
ax.set_title("KL dispersion: B looser ONLY on g1\n(blue A, orange B)", fontsize=9)

ax = axes[1]
for i, arm in enumerate(["A", "B"]):
    vals = [df[(df.task == t) & (df.arm == arm)].overshoot_run.values for t, _ in tasks]
    bp = ax.boxplot(vals, positions=xs + (i - .5) * .3, widths=.26, patch_artist=True,
                    medianprops=dict(color="k"))
    for b in bp["boxes"]: b.set_facecolor(COL[arm]); b.set_alpha(.65)
ax.set_xticks(xs); ax.set_xticklabels([f"{t}\nd={d}" for t, d in tasks])
ax.set_ylabel(r"mean$_t\,\max(KL_t-0.1,0)$"); ax.set_yscale("log")
ax.set_title("Overshoot above the bound", fontsize=9)

ax = axes[2]
for arm in ["A", "B"]:
    s = df[(df.task == "g1") & (df.arm == arm)]
    ax.scatter(s.kl_iqr, s.final_return, color=COL[arm], s=55, label=f"arm {arm}")
rho, _ = stats.spearmanr(g.kl_iqr, g.final_return)
ax.set_xlabel("KL IQR"); ax.set_ylabel("final eval return")
ax.set_title(f"g1 only. Within arm B, dispersion and return move\n"
             rf"TOGETHER ($\rho$={rho:+.2f}, n=8) — opposite to the pooled sign", fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_g1_dispersion.png", dpi=160); plt.close(fig)
print(f"\nfigures -> {OUT}/fig_kl_curves.png, fig_lagrangian.png, fig_g1_dispersion.png")
