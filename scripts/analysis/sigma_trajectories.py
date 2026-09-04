"""Descriptive sigma trajectories for the corrected Walker/G1/LEAP runs.

sigma is the PRE-TANH Gaussian scale, sigma = exp(log_std) + min_std with
min_std = 0.1 (src/networks/jax_models.py:336, :425), state-dependent and
per-action-coordinate. The logged scalar train/pi_sigma_mean is
pi.distribution.scale.mean() (src/jaxrl/reppo.py:785, :1267), i.e. a mean over
BOTH the minibatch and the action coordinates.

Seed IDs are DISCOVERED from the run directories, not assumed.
Read-only. Emits reports/artifacts/sigma_trajectories.json and figures.
"""
from __future__ import annotations
import glob, json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
ROOT = os.environ.get("HPCWORK", "/hpcwork/qzi10910") + "/reppo_runs/outputs"
GROUPS = [("walker", "PW",  ROOT + "/faithful_repair/walker_PW1_s*"),
          ("walker", "WML", ROOT + "/faithful_repair/walker_WML32_s*"),
          ("g1",     "PW",  ROOT + "/faithful_repair/g1_PW1_s*"),
          ("g1",     "WML", ROOT + "/faithful_repair/g1_WML32_s*"),
          ("leap",   "PW",  ROOT + "/leap_corrected/leap_PW_s*"),
          ("leap",   "WML", ROOT + "/leap_corrected/leap_WML_s*")]
FIELD = "train/pi_sigma_mean"

def load_group(pat):
    runs = {}
    for d in sorted(glob.glob(pat)):
        m = re.search(r"_s(\d+)$", d)
        p = os.path.join(d, "metrics.npz")
        if not (m and os.path.exists(p)):
            continue
        z = np.load(p)
        if FIELD not in z.files:
            continue
        v = np.asarray(z[FIELD], float).reshape(-1)
        runs[int(m.group(1))] = v[np.isfinite(v)]
    return runs

def summarise(v):
    n = len(v)
    it = np.arange(1, n + 1) * (399.0 / n)
    l20 = v[it > 0.8 * 399.0]
    return dict(initial=float(v[0]), final=float(v[-1]),
                ratio_final_initial=float(v[-1] / v[0]),
                maximum=float(v.max()), argmax_frac=float((np.argmax(v) + 1) / n),
                final20_mean=float(l20.mean()), n_points=n, n_final20=int(len(l20)))

def main():
    out, fig_paths = {}, []
    for task, arm, pat in GROUPS:
        runs = load_group(pat)
        key = "%s_%s" % (task, arm)
        out[key] = dict(pattern=pat, seeds=sorted(runs), n_runs=len(runs), per_seed={})
        for s, v in sorted(runs.items()):
            out[key]["per_seed"][str(s)] = summarise(v)
        if runs:
            arr = np.stack([runs[s] for s in sorted(runs)])
            out[key]["across_seed"] = dict(
                initial_mean=float(arr[:, 0].mean()), final_mean=float(arr[:, -1].mean()),
                ratio_mean=float((arr[:, -1] / arr[:, 0]).mean()),
                ratio_min=float((arr[:, -1] / arr[:, 0]).min()),
                ratio_max=float((arr[:, -1] / arr[:, 0]).max()),
                max_mean=float(arr.max(axis=1).mean()),
                final20_mean=float(np.mean([summarise(runs[s])["final20_mean"] for s in sorted(runs)])))
    # ---- figures: one per task, raw and normalised ---------------------------
    for task in ("walker", "g1", "leap"):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
        for arm, col in (("PW", "#1f4e79"), ("WML", "#c0392b")):
            runs = load_group(dict((f"{t}_{a}", p) for t, a, p in GROUPS)[f"{task}_{arm}"])
            if not runs:
                continue
            arr = np.stack([runs[s] for s in sorted(runs)])
            x = np.arange(1, arr.shape[1] + 1) * (399.0 / arr.shape[1])
            for r in arr:
                axes[0].plot(x, r, color=col, alpha=0.30, lw=0.9)
                axes[1].plot(x, r / r[0], color=col, alpha=0.30, lw=0.9)
            axes[0].plot(x, arr.mean(0), color=col, lw=2.4, label="%s (n=%d)" % (arm, len(runs)))
            axes[1].plot(x, (arr / arr[:, :1]).mean(0), color=col, lw=2.4, label=arm)
        axes[0].set_ylabel(r"pre-tanh $\sigma$  (mean over batch and coordinates)")
        axes[1].set_ylabel(r"$\sigma(t)\,/\,\sigma(\mathrm{first\ eval})$")
        axes[1].axhline(1.0, color="0.5", lw=0.8, ls=":")
        for ax, ttl in zip(axes, ("raw", "normalised to each run's first evaluation")):
            ax.set_xlabel("training iteration"); ax.set_title("%s — %s" % (task, ttl), fontsize=10, loc="left")
            ax.legend(fontsize=8); ax.grid(color="0.93", lw=0.7); ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        p = "reports/figures/fig_sigma_%s.png" % task
        os.makedirs("reports/figures", exist_ok=True)
        fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig); fig_paths.append(p)
        print("wrote", p)
    out["_meta"] = dict(field=FIELD, reduction="mean over minibatch AND action coordinates",
                        sigma_definition="pre-tanh Normal scale = exp(log_std) + min_std, "
                                         "min_std=0.1, state-dependent, per-coordinate",
                        source="src/networks/jax_models.py:336,:425,:436; src/jaxrl/reppo.py:785,:1267",
                        note="initial = FIRST LOGGED EVALUATION, not initialisation",
                        figures=fig_paths)
    json.dump(out, open("reports/artifacts/sigma_trajectories.json", "w"), indent=1)
    print("\n%-8s %-4s %-28s %9s %9s %9s %9s %9s" %
          ("task","arm","seeds","initial","final","fin/init","max","fin20"))
    for task, arm, _ in GROUPS:
        k = "%s_%s" % (task, arm); g = out[k]
        if not g["n_runs"]:
            print("%-8s %-4s NONE FOUND" % (task, arm)); continue
        a = g["across_seed"]
        print("%-8s %-4s %-28s %9.4f %9.4f %9.4f %9.4f %9.4f" %
              (task, arm, ",".join(map(str, g["seeds"])), a["initial_mean"], a["final_mean"],
               a["ratio_mean"], a["max_mean"], a["final20_mean"]))
        print("%-8s %-4s   ratio range across seeds: [%.4f, %.4f]" % ("", "", a["ratio_min"], a["ratio_max"]))
    print("\nwrote reports/artifacts/sigma_trajectories.json")

if __name__ == "__main__":
    main()
