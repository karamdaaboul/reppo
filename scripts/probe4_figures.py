import numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = "reports/artifacts"
CELLS = ["A_PW", "A_WML", "B_PW", "B_WML"]
COL = {"A_PW": "#2b6cb0", "A_WML": "#63b3ed", "B_PW": "#9c4221", "B_WML": "#ed8936"}
seeds = [0, 1, 2, 3, 4]
D = {law: {s: np.load(f"{OUT}/probe4_s{s}_{law}.npz", allow_pickle=True) for s in seeds}
     for law in ("ckpt", "std")}

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

# --- primary: paired seed-level differences, both laws
ax = axes[0]
for j, law in enumerate(("ckpt", "std")):
    for critic, off in (("A", -0.18), ("B", 0.18)):
        d = [np.median(D[law][s][f"{critic}_WML_D"]) - np.median(D[law][s][f"{critic}_PW_D"])
             for s in seeds]
        x = np.full(5, j + off) + np.random.default_rng(1).normal(0, .025, 5)
        ax.scatter(x, d, s=55, color=COL[f"{critic}_WML"], edgecolor="k", linewidth=.4,
                   zorder=3, label=f"{critic}-trained critic" if j == 0 else None)
        ax.hlines(np.median(d), j + off - .11, j + off + .11, color="k", lw=2.2, zorder=4)
ax.axhline(0, color="crimson", lw=1.6)
ax.set_xticks([0, 1]); ax.set_xticklabels(["checkpoint law\n(primary)", "standardized law\n(sensitivity)"])
ax.set_ylabel(r"paired $D_{\rm WML}-D_{\rm PW}$  (seed medians)")
ax.set_title("PRIMARY. Committed rule: falsified if median $\\leq$ 0.\n"
             "A-trained: 0/5 positive (falsifies). B-trained: 5/5 (supports).", fontsize=9)
ax.legend(fontsize=8)

# --- D per cell
ax = axes[1]
for i, c in enumerate(CELLS):
    v = [np.median(D["ckpt"][s][f"{c}_D"]) for s in seeds]
    ax.scatter(np.full(5, i) + np.random.default_rng(2).normal(0, .04, 5), v,
               s=50, color=COL[c], edgecolor="k", linewidth=.4, zorder=3)
    ax.hlines(np.median(v), i - .18, i + .18, color="k", lw=2.2, zorder=4)
ax.set_xticks(range(4)); ax.set_xticklabels(CELLS); ax.set_yscale("log")
ax.set_ylabel(r"$D$ = $\|\Delta\mu_x[Q_\phi]-\Delta\mu_x[\bar Q_\phi]\|^2_{\Sigma_x^{-1}}$")
ax.set_title("Crossed same-critic table (checkpoint law).\n"
             "Critic source moves D far more than operator does.", fontsize=9)

# --- L: fraction of the trust-region step spent in the INERT padded block
ax = axes[2]
for i, c in enumerate(CELLS):
    v = [np.median(D["ckpt"][s][f"{c}_L"]) for s in seeds]
    ax.scatter(np.full(5, i) + np.random.default_rng(3).normal(0, .04, 5), v,
               s=50, color=COL[c], edgecolor="k", linewidth=.4, zorder=3)
    ax.hlines(np.median(v), i - .18, i + .18, color="k", lw=2.2, zorder=4)
ax.axhline(16 / 22, color="grey", ls=":", lw=1.2)
ax.text(0.05, 16 / 22 + .02, "k/d = 16/22: an isotropic step", fontsize=7, color="grey")
ax.set_xticks(range(4)); ax.set_xticklabels(CELLS); ax.set_ylim(0, 1)
ax.set_ylabel(r"$L$ = padded fraction of the step")
ax.set_title("SECONDARY (exploratory reading). WML spends 46-68% of its\n"
             "trust region on coordinates the simulator discards.", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_probe4.png", dpi=160); plt.close(fig)
print("wrote fig_probe4.png")
for law in ("ckpt", "std"):
    for c in CELLS:
        v = [float(np.median(D[law][s][f"{c}_L"])) for s in seeds]
        print(f"  {law} {c:6s} L median = {np.median(v):.4f}")
