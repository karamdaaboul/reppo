"""Experiment 1 -- ranking fidelity of Q_phi against the soft MC return.

For each state s_j: sample M actions from pi_old, score them with Q_phi, and score
them with a horizon-truncated soft MC return. Report per-state Spearman rho.

Guards implemented here:
  (a) noise ceiling  -- two independent MC estimates, rho(MC1, MC2)
  (b) median + IQR   -- the per-state distribution is heavy-tailed
  (c) random-critic control -- same architecture, fresh init, as the floor
  (d) top-1 agreement, reported separately from rho

States are cloned exactly via the MJX state pytree; nothing is reconstructed from
observations. Each chunk takes C decorrelated envs from a live rollout, clones each
M times, and evaluates all C*M perturbed actions in one batched step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import nnx  # noqa: E402

from scripts.critic_fidelity.common import (  # noqa: E402
    ACTION_CLIP,
    Harness,
    gather_states,
    spearman,
    summarize,
)
from src.networks.jax_models import CategoricalCriticNetwork, CriticNetwork  # noqa: E402


def random_critic(meta, seed=0):
    """Same architecture, fresh init -- the floor for guard (c)."""
    kwargs = dict(meta["critic_kwargs"])
    if meta["hl_gauss"]:
        return CategoricalCriticNetwork(**kwargs, rngs=nnx.Rngs(seed))
    for k in ("num_bins", "vmin", "vmax"):
        kwargs.pop(k, None)
    return CriticNetwork(**kwargs, rngs=nnx.Rngs(seed))


def run(ckpt, K, M, C, horizon, stride, burn_in, seed, R=1):
    B = C * M
    h = Harness(ckpt, B)
    rc = random_critic(h.meta, seed=123)
    key = jax.random.PRNGKey(seed)

    key, rk = jax.random.split(key)
    obs, _, st = h.reset(rk)

    # burn in so states are mid-episode rather than all at reset
    def advance(st, obs, n, key):
        for i in range(n):
            k1, k2, key = jax.random.split(key, 3)
            a = jnp.clip(h.pi(obs).sample(seed=k1), -ACTION_CLIP, ACTION_CLIP)
            obs, _, st, _, _, _ = h.env.step(jax.random.split(k2, B), st, a)
        return st, obs, key

    st, obs, key = advance(st, obs, burn_in, key)

    out = {k: [] for k in
           ["rho_q_mc1", "rho_mc1_mc2", "rho_rand_mc1", "top1_q", "top1_ceiling",
            "q_spread", "mc_spread"]}
    n_chunks = int(np.ceil(K / C))
    t0 = time.time()

    for ci in range(n_chunks):
        key, sk, k1, k2 = jax.random.split(key, 4)
        idx = jax.random.choice(sk, B, shape=(C,), replace=False)
        st_c = gather_states(st, idx)
        obs_c = obs[idx]

        # clone each selected state M times: (C,...) -> (C*M,...)
        st_rep = jax.tree.map(lambda x: jnp.repeat(x, M, axis=0), st_c)
        obs_rep = jnp.repeat(obs_c, M, axis=0)

        acts = h.pi(obs_c).sample(seed=k1, sample_shape=(M,))       # (M, C, D)
        acts = jnp.clip(jnp.transpose(acts, (1, 0, 2)), -ACTION_CLIP, ACTION_CLIP)
        acts_flat = acts.reshape(C * M, -1)

        q = np.asarray(h.q(obs_rep, acts_flat)).reshape(C, M)
        q_rand = np.asarray(h.q(obs_rep, acts_flat, critic=rc)).reshape(C, M)
        # R independent rollouts averaged per MC estimate; R=1 is the spec default
        def mc_estimate(tag):
            acc = 0.0
            for r in range(R):
                acc = acc + np.asarray(
                    h.soft_return(st_rep, obs_rep, acts_flat, horizon,
                                  jax.random.fold_in(jax.random.fold_in(k2, tag), r))
                )
            return (acc / R).reshape(C, M)

        mc1 = mc_estimate(1)
        mc2 = mc_estimate(2)

        out["rho_q_mc1"].append(spearman(q, mc1))
        out["rho_mc1_mc2"].append(spearman(mc1, mc2))
        out["rho_rand_mc1"].append(spearman(q_rand, mc1))
        out["top1_q"].append((q.argmax(1) == mc1.argmax(1)).astype(float))
        out["top1_ceiling"].append((mc1.argmax(1) == mc2.argmax(1)).astype(float))
        out["q_spread"].append(q.std(1))
        out["mc_spread"].append(mc1.std(1))

        st, obs, key = advance(st, obs, stride, key)
        if ci == 0:
            print(f"    chunk 0 done in {time.time() - t0:.1f}s (incl. compile)")

    res = {k: np.concatenate(v)[:K] for k, v in out.items()}
    res["_seconds"] = time.time() - t0
    res["_env_steps"] = int(K * M * horizon * 2 * R + n_chunks * stride * B)
    return res, h


def report(res, label):
    s = {k: summarize(res[k], k) for k in
         ["rho_q_mc1", "rho_mc1_mc2", "rho_rand_mc1", "top1_q", "top1_ceiling"]}
    print(f"\n--- {label} ---")
    print(f"  {'quantity':<16} {'median':>8} {'IQR':>16} {'mean':>8} {'sem':>7}")
    for k in ["rho_q_mc1", "rho_mc1_mc2", "rho_rand_mc1"]:
        v = s[k]
        print(f"  {k:<16} {v['median']:>8.3f} [{v['q1']:>6.3f},{v['q3']:>6.3f}] "
              f"{v['mean']:>8.3f} {v['sem']:>7.4f}")
    for k in ["top1_q", "top1_ceiling"]:
        v = s[k]
        print(f"  {k:<16} {'':>8} {'':>16} {v['mean']:>8.3f} {v['sem']:>7.4f}")

    ceil_med = s["rho_mc1_mc2"]["median"]
    ratio = s["rho_q_mc1"]["median"] / ceil_med if ceil_med > 0 else float("nan")
    # disattenuation: correlating against a noisy proxy attenuates rho by ~sqrt(reliability)
    disatt = s["rho_q_mc1"]["median"] / np.sqrt(ceil_med) if ceil_med > 0 else float("nan")
    print(f"  HEADLINE  rho(Q,MC1)/rho(MC1,MC2) = {ratio:.3f}   "
          f"[disattenuated rho/sqrt(ceiling) = {disatt:.3f}]")
    print(f"  floor     rho(rand,MC1)/ceiling  = "
          f"{s['rho_rand_mc1']['median'] / ceil_med if ceil_med > 0 else float('nan'):.3f}")
    if ceil_med < 0.2:
        print("  *** CEILING NEAR ZERO -- MC too noisy, this configuration is "
              "UNINFORMATIVE. Do not read the raw correlation. ***")
    return {"summary": s, "ratio": float(ratio), "disattenuated": float(disatt),
            "ceiling_median": float(ceil_med)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="exports/WalkerRun_s0")
    ap.add_argument("--K", type=int, default=2000)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--horizons", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--burn_in", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--R", type=int, default=1,
                    help="rollouts averaged per MC estimate (spec default 1)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    allres = {}
    for H in args.horizons:
        print(f"\n=== horizon H={H} ===")
        res, h = run(args.ckpt, args.K, args.M, args.C, H, args.stride,
                     args.burn_in, args.seed, args.R)
        print(f"  {res['_env_steps']:,} env steps in {res['_seconds']:.1f}s "
              f"({res['_env_steps'] / res['_seconds'] / 1e3:.0f}k steps/s)")
        allres[str(H)] = report(res, f"{os.path.basename(args.ckpt)} H={H} K={args.K}")
        allres[str(H)]["seconds"] = res["_seconds"]

    print("\n=== horizon sensitivity ===")
    print(f"  {'H':>5} {'ratio':>8} {'rho(Q,MC1)':>11} {'ceiling':>9} {'top1':>7}")
    for H in args.horizons:
        a = allres[str(H)]
        print(f"  {H:>5} {a['ratio']:>8.3f} {a['summary']['rho_q_mc1']['median']:>11.3f} "
              f"{a['ceiling_median']:>9.3f} {a['summary']['top1_q']['mean']:>7.3f}")
    spread = max(allres[str(H)]["ratio"] for H in args.horizons) - min(
        allres[str(H)]["ratio"] for H in args.horizons)
    print(f"  ratio spread across H: {spread:.3f} -> conclusions "
          f"{'MOVE with H' if spread > 0.05 else 'stable in H'}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(allres, f, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
