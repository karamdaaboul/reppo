"""Walker per-seed pre-tanh sigma, WML and PW, from the same artifact as 0a10971.

Source field: train/pi_sigma_mean in
  $HPCWORK/reppo_runs/outputs/faithful_repair/walker_{WML32,PW1}_s{seed}/metrics.npz
which is pi.distribution.scale.mean() (src/jaxrl/reppo.py:785, :1267) -- the
PRE-TANH Normal scale, sigma = exp(log_std) + min_std with min_std = 0.1
(src/networks/jax_models.py:425, :336), averaged over the minibatch AND the action
coordinates.

Per-seed lines only, no across-seed mean. Read-only.
"""
from __future__ import annotations
import json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
ROOT = os.environ.get("HPCWORK", "/hpcwork/qzi10910") + "/reppo_runs/outputs/faithful_repair"
SEEDS = list(range(301, 309))
ARMS = [("WML", "walker_WML32_s%d"), ("PW", "walker_PW1_s%d")]
FIELD = "train/pi_sigma_mean"
RET = "eval/episode_return"
# 399 outer iterations, 21 evaluations -> eval k at iteration 19k; 1024 envs x 128 steps
ITERS_PER_EVAL, STATES_PER_ITER = 19, 1024 * 128

def load(arm_pat, seed):
    p = os.path.join(ROOT, arm_pat % seed, "metrics.npz")
    z = np.load(p)
    s = np.asarray(z[FIELD], float).reshape(-1)
    r = np.asarray(z[RET], float).reshape(-1)
    steps = np.arange(1, len(s) + 1) * ITERS_PER_EVAL * STATES_PER_ITER
    return s, r, steps, p

def main():
    data, rows = {}, []
    for arm, pat in ARMS:
        data[arm] = {}
        for sd in SEEDS:
            s, r, steps, p = load(pat, sd)
            i = int(np.argmax(s))
            data[arm][sd] = dict(sigma=s, steps=steps, ret=r, path=p)
            rows.append(dict(arm=arm, seed=sd, max_sigma=float(s.max()),
                             step_at_max=int(steps[i]), eval_index_at_max=i + 1,
                             final_sigma=float(s[-1]), final_return=float(r[-1]),
                             n_points=len(s)))
    lo = min(d["sigma"].min() for a in data for d in data[a].values())
    hi = max(d["sigma"].max() for a in data for d in data[a].values())
    ylim = (lo * 0.7, hi * 1.5)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, (arm, _) in zip(axes, ARMS):
        for j, sd in enumerate(SEEDS):
            d = data[arm][sd]
            ax.plot(d["steps"], d["sigma"], lw=1.6, color=cmap(j / (len(SEEDS) - 1)),
                    label="s%d" % sd)
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.set_xlabel("environment steps")
        ax.set_title("WalkerRun  %s-faithful-repair  (n=%d seeds, drawn individually)"
                     % ("WML-32" if arm == "WML" else "PW-1", len(SEEDS)),
                     fontsize=10, loc="left")
        ax.grid(which="both", color="0.93", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=7.5, ncol=2, loc="upper left", frameon=False)
    axes[0].set_ylabel("train/pi_sigma_mean\n"
                       r"pre-tanh $\sigma$, mean over minibatch and action coordinates")
    fig.tight_layout()
    out_png = "reports/figures/fig_walker_sigma_seeds.png"
    out_pdf = "reports/figures/fig_walker_sigma_seeds.pdf"
    os.makedirs("reports/figures", exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_png, "and", out_pdf)

    json.dump(dict(field=FIELD, return_field=RET,
                   reduction="pi.distribution.scale.mean(): mean over minibatch AND action coordinates",
                   sigma_definition="pre-tanh Normal scale = exp(log_std) + min_std, min_std=0.1",
                   source_paths={"%s_s%d" % (r["arm"], r["seed"]):
                                 data[r["arm"]][r["seed"]]["path"] for r in rows},
                   iters_per_eval=ITERS_PER_EVAL, states_per_iter=STATES_PER_ITER,
                   shared_ylim=list(ylim), rows=rows),
              open("reports/artifacts/walker_sigma_seeds.json", "w"), indent=1)
    print("wrote reports/artifacts/walker_sigma_seeds.json\n")

    for arm, _ in ARMS:
        print("=== WalkerRun %s ===" % ("WML-32" if arm == "WML" else "PW-1"))
        print("  %-6s %12s %16s %13s %14s" %
              ("seed", "max sigma", "step at max", "final sigma", "final return"))
        for r in [x for x in rows if x["arm"] == arm]:
            print("  %-6d %12.4f %16s %13.4f %14.3f" %
                  (r["seed"], r["max_sigma"], "{:,}".format(r["step_at_max"]),
                   r["final_sigma"], r["final_return"]))
        print()

if __name__ == "__main__":
    main()
