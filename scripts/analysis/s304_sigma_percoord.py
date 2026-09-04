"""Per-action-coordinate pre-tanh sigma for Walker WML seed 304, from existing
checkpoints only. Read-only. No training."""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
import jax.numpy as jnp                                              # noqa: E402
from scripts.load_ckpt import load                                   # noqa: E402

BANK = "reports/artifacts/cd_bank_walker_corrected.npz"
CKPTS = [("p25  (12.45M steps)", "exports/WalkerRun_weighted_mle_s304_p25"),
         ("p50  (24.90M steps)", "exports/WalkerRun_weighted_mle_s304_p50"),
         ("final(52.30M steps)", "exports/WalkerRun_weighted_mle_s304_final")]
MIN_STD = 0.1

def main():
    z = np.load(BANK, allow_pickle=True)
    obs = np.asarray(z["obs"], np.float32)[:1024]        # 1024 states = one minibatch
    out = {"bank": BANK, "n_states": int(obs.shape[0]), "min_std": MIN_STD, "checkpoints": {}}
    percoord_for_fig = None
    for label, ck in CKPTS:
        if not os.path.isdir(ck):
            print("  MISSING", ck); continue
        c = load(ck)
        mu, sg = c.policy_dist(jnp.asarray(obs))
        sg = np.asarray(sg, np.float64)
        d = sg.shape[-1]
        log_std = np.log(np.maximum(sg - MIN_STD, 1e-300))
        gm = float(np.exp(np.log(sg).mean()))
        ent = dict(numel=int(sg.size), shape=list(sg.shape),
                   mean=float(sg.mean()), geometric_mean=gm, median=float(np.median(sg)),
                   minimum=float(sg.min()), maximum=float(sg.max()),
                   p75=float(np.percentile(sg,75)), p90=float(np.percentile(sg,90)),
                   p95=float(np.percentile(sg,95)), p99=float(np.percentile(sg,99)),
                   max_over_mean=float(sg.max()/sg.mean()),
                   frac_within_1pct_of_floor=float(np.mean(sg <= MIN_STD*1.01)),
                   per_coordinate=[dict(coord=j, median=float(np.median(sg[:,j])),
                                        mean=float(sg[:,j].mean()),
                                        p95=float(np.percentile(sg[:,j],95)),
                                        maximum=float(sg[:,j].max()),
                                        median_log_std=float(np.median(log_std[:,j])),
                                        max_log_std=float(log_std[:,j].max()))
                                   for j in range(d)])
        out["checkpoints"][ck] = ent
        print("\n=== %s   %s" % (label, ck))
        print("   tensor shape %s  numel %d" % (ent["shape"], ent["numel"]))
        print("   mean %.4f  geo-mean %.4f  median %.4f  min %.4f  max %.4f  max/mean %.1f"
              % (ent["mean"], gm, ent["median"], ent["minimum"], ent["maximum"], ent["max_over_mean"]))
        print("   quantiles  p75 %.4f  p90 %.4f  p95 %.4f  p99 %.4f" %
              (ent["p75"], ent["p90"], ent["p95"], ent["p99"]))
        print("   fraction within 1%% of the %.1f floor: %.4f" % (MIN_STD, ent["frac_within_1pct_of_floor"]))
        print("   %-6s %10s %10s %10s %12s %12s %12s" %
              ("coord","median","mean","p95","maximum","med log_std","max log_std"))
        for r in ent["per_coordinate"]:
            print("   %-6d %10.4f %10.4f %10.4f %12.4f %12.4f %12.4f" %
                  (r["coord"], r["median"], r["mean"], r["p95"], r["maximum"],
                   r["median_log_std"], r["max_log_std"]))
        if ck.endswith("_final"):
            percoord_for_fig = sg
    # ---- figure: distribution across states, per coordinate, final checkpoint --
    if percoord_for_fig is not None:
        sg = percoord_for_fig; d = sg.shape[-1]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.boxplot([sg[:, j] for j in range(d)], positions=range(d), widths=0.6,
                   showfliers=True, flierprops=dict(marker=".", markersize=2, alpha=0.35))
        ax.set_yscale("log")
        ax.axhline(MIN_STD, color="crimson", lw=1.2, ls="--")
        ax.annotate("min_std = 0.1 (additive floor)", (d-1, MIN_STD), xytext=(0, -14),
                    textcoords="offset points", ha="right", fontsize=8, color="crimson")
        ax.set_xlabel("action coordinate"); ax.set_ylabel(r"pre-tanh $\sigma$ (log scale)")
        ax.set_title("WalkerRun WML-32 seed 304, FINAL checkpoint (52.3M steps, post-collapse)\n"
                     "distribution across 1024 states, per coordinate", fontsize=9.5, loc="left")
        ax.grid(which="both", color="0.93", lw=0.7); ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        os.makedirs("reports/figures", exist_ok=True)
        p = "reports/figures/fig_s304_sigma_percoord.png"
        fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
        out["figure"] = p; print("\nwrote", p)
    json.dump(out, open("reports/artifacts/s304_sigma_percoord.json", "w"), indent=1)
    print("wrote reports/artifacts/s304_sigma_percoord.json")

if __name__ == "__main__":
    main()
