"""Build ONE neutral Walker state bank and evaluate all 16 final checkpoints on it.

The bank is balanced across arms by construction: every one of the 16 corrected
policies (PW and WML, seeds 301-308) contributes the same number of states at the
same set of episode depths. The SAME bank is used for all 16 checkpoints; no
policy is ever evaluated only on its own states.

Read-only with respect to checkpoints. Rollouts collect states only; nothing is
trained and no parameter is updated.
"""
from __future__ import annotations
import hashlib, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
import jax, jax.numpy as jnp                                          # noqa: E402
from scripts.load_ckpt import load                                    # noqa: E402
from scripts.critic_fidelity.common import Harness, ACTION_CLIP       # noqa: E402

SEEDS = list(range(301, 309))
ARMS = {"PW": "exports/WalkerRun_pathwise_fa_s%d_final",
        "WML": "exports/WalkerRun_weighted_mle_s%d_final"}
DEPTHS = (50, 150, 300, 500, 700, 900)
NENV, ROOT = 32, 20260905
BANK = "reports/artifacts/walker_fixed_state_bank.npz"
MIN_STD = 0.1

def build_bank():
    obs_all, src_all, depth_all = [], [], []
    for arm, pat in ARMS.items():
        for sd in SEEDS:
            ck = pat % sd
            h = Harness(ck, NENV)
            key = jax.random.fold_in(jax.random.PRNGKey(ROOT),
                                     hash((arm, sd)) % (2**31))
            k1, key = jax.random.split(key)
            o, _, st = h.reset(k1)
            for t in range(1, max(DEPTHS) + 1):
                ka, kb, key = jax.random.split(key, 3)
                a = jnp.clip(h.pi(o).sample(seed=ka), -ACTION_CLIP, ACTION_CLIP)
                o, _, st, _, _, _ = h.env.step(jax.random.split(kb, NENV), st, a)
                if t in DEPTHS:
                    obs_all.append(np.asarray(o)); src_all += ["%s-s%d" % (arm, sd)] * NENV
                    depth_all += [t] * NENV
            print("  collected %s seed %d" % (arm, sd), flush=True)
    obs = np.concatenate(obs_all, 0).astype(np.float32)
    np.savez(BANK, obs=obs, source=np.array(src_all), depth=np.array(depth_all),
             depths=np.array(DEPTHS), n_env=np.array(NENV), root=np.array(ROOT),
             burn_in_note=np.array("states are episode-depth stratified, not burn-in-50"))
    return obs, np.array(src_all), np.array(depth_all)

def main():
    if os.path.exists(BANK):
        z = np.load(BANK, allow_pickle=True)
        obs, src, dep = np.asarray(z["obs"], np.float32), np.array(z["source"]), np.array(z["depth"])
        print("reusing existing bank")
    else:
        obs, src, dep = build_bank()
    sha = hashlib.sha256(open(BANK, "rb").read()).hexdigest()
    print("\nSTATE_BANK_PATH = %s\nNUM_STATES = %d\nSHA256 = %s" % (BANK, obs.shape[0], sha))
    print("PROVENANCE = 16 corrected Walker policies (PW+WML, seeds 301-308), %d envs each,"
          " %d depths, RNG root %d" % (NENV, len(DEPTHS), ROOT))
    print("ARM BALANCE = PW %d states, WML %d states"
          % (int((np.char.startswith(src, "PW")).sum()), int((np.char.startswith(src, "WML")).sum())))
    print("EPISODE_DEPTH_DISTRIBUTION = " + ", ".join(
        "%d:%d" % (d, int((dep == d).sum())) for d in sorted(set(dep.tolist()))))

    # ---- evaluate all 16 checkpoints on the IDENTICAL bank --------------------
    res = {}
    ob = jnp.asarray(obs)
    for arm, pat in ARMS.items():
        for sd in SEEDS:
            c = load(pat % sd)
            _, sg = c.policy_dist(ob)
            sg = np.asarray(sg, np.float64)
            res[(arm, sd)] = sg
    d = res[("PW", 301)].shape[-1]

    def stats(sg):
        return dict(median=float(np.median(sg)), mean=float(sg.mean()),
                    p95=float(np.percentile(sg, 95)), maximum=float(sg.max()))
    out = dict(bank=BANK, bank_sha256=sha, n_states=int(obs.shape[0]),
               depths=list(DEPTHS), n_env=NENV, rng_root=ROOT, d=d,
               whole_policy={}, per_coordinate={}, paired={})
    print("\n=== whole-policy sigma on the fixed bank ===")
    print("  %-4s %-5s %10s %12s %12s %14s" % ("arm","seed","median","mean","p95","max"))
    for arm in ARMS:
        for sd in SEEDS:
            s = stats(res[(arm, sd)]); out["whole_policy"]["%s_s%d" % (arm, sd)] = s
            print("  %-4s %-5d %10.4f %12.4f %12.4f %14.1f"
                  % (arm, sd, s["median"], s["mean"], s["p95"], s["maximum"]))
    for arm in ARMS:
        for sd in SEEDS:
            sg = res[(arm, sd)]
            out["per_coordinate"]["%s_s%d" % (arm, sd)] = [
                dict(coord=j, **stats(sg[:, j])) for j in range(d)]

    # ---- paired differences ---------------------------------------------------
    print("\n=== paired WML - PW, whole-policy MEDIAN sigma ===")
    print("  %-6s %12s %12s %12s" % ("seed", "PW median", "WML median", "delta"))
    deltas = []
    for sd in SEEDS:
        a = float(np.median(res[("PW", sd)])); b = float(np.median(res[("WML", sd)]))
        deltas.append(b - a); out["paired"]["s%d" % sd] = dict(pw=a, wml=b, delta=b - a)
        print("  %-6d %12.4f %12.4f %+12.4f" % (sd, a, b, b - a))
    n_pos = int(sum(x > 0 for x in deltas))
    print("  WML median > PW median in %d of 8 seed pairs" % n_pos)

    print("\n=== paired WML - PW, per action coordinate (median sigma) ===")
    print("  %-6s" % "coord" + "".join("%10s" % ("s%d" % s) for s in SEEDS) + "%10s" % "n_pos")
    coord_pos = []
    percoord = {}
    for j in range(d):
        row = [float(np.median(res[("WML", s)][:, j]) - np.median(res[("PW", s)][:, j])) for s in SEEDS]
        npj = int(sum(x > 0 for x in row)); coord_pos.append(npj); percoord[j] = row
        print("  %-6d" % j + "".join("%+10.4f" % v for v in row) + "%10s" % ("%d/8" % npj))
    out["paired_percoord"] = {str(j): percoord[j] for j in range(d)}
    n_coord = int(sum(p >= 5 for p in coord_pos))
    out["rule"] = dict(seed_pairs_wml_gt_pw=n_pos, coords_positive_in_majority=n_coord,
                       criterion_1_met=bool(n_pos >= 7), criterion_2_met=bool(n_coord >= 5))
    verdict = ("BROAD ARM-LEVEL WML WIDENING SUPPORTED" if (n_pos >= 7 and n_coord >= 5)
               else ("MIXED / COORDINATE-SPECIFIC" if (n_pos >= 5 or n_coord >= 3) else "NOT SUPPORTED"))
    out["verdict"] = verdict
    print("\n  criterion 1: whole-policy median WML>PW in >=7/8 pairs -> %d/8  %s" % (n_pos, n_pos >= 7))
    print("  criterion 2: >=5/6 coords positive in a majority of pairs -> %d/6  %s" % (n_coord, n_coord >= 5))
    print("  VERDICT: %s" % verdict)

    # ---- figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for arm, col in (("PW", "#1f4e79"), ("WML", "#c0392b")):
        med = np.array([[np.median(res[(arm, s)][:, j]) for j in range(d)] for s in SEEDS])
        for r in med:
            axes[0].plot(range(d), r, color=col, alpha=0.35, lw=1.0, marker="o", ms=3)
        axes[0].plot(range(d), med.mean(0), color=col, lw=2.6, marker="o", ms=5, label=arm)
    axes[0].set_yscale("log"); axes[0].set_xlabel("action coordinate")
    axes[0].set_ylabel(r"median pre-tanh $\sigma$ on the fixed bank")
    axes[0].set_title("typical width (median), per coordinate, 8 seeds each", fontsize=9.5, loc="left")
    axes[0].axhline(MIN_STD, color="0.5", lw=1.0, ls="--")
    axes[0].legend(fontsize=8)
    for j in range(d):
        axes[1].scatter([j]*len(SEEDS), percoord[j], s=26, color="#444", alpha=0.8)
    axes[1].axhline(0.0, color="crimson", lw=1.2)
    axes[1].set_xlabel("action coordinate")
    axes[1].set_ylabel(r"paired  median$\sigma_{WML}$ - median$\sigma_{PW}$")
    axes[1].set_title("paired per-seed difference (8 points per coordinate)", fontsize=9.5, loc="left")
    for ax in axes:
        ax.grid(which="both", color="0.93", lw=0.7); ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = "reports/figures/fig_walker_fixed_bank_sigma.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    out["figure"] = p
    json.dump(out, open("reports/artifacts/walker_fixed_bank_sigma.json", "w"), indent=1)
    print("\nwrote %s and reports/artifacts/walker_fixed_bank_sigma.json" % p)

if __name__ == "__main__":
    main()
