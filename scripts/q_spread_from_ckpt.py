"""Measure q_spread = sd_i(Q(s,a_i)/alpha) offline from an exported checkpoint.

This is the actual E-step signal: the spread of the softmax exponent across the M
actions drawn from pi_old at a state. It decides ESS.

Measuring it offline rather than during training matters for the pathwise arm: adding
the extra critic forward inside `actor_loss` changes XLA fusion and therefore float32
rounding, which breaks bit-identity with the pre-E-step implementation. Offline the
question is answered with no effect on the run at all.

Usage:
    JAX_PLATFORMS=cpu ./.venv/bin/python scripts/q_spread_from_ckpt.py exports/<dir> ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from scripts.critic_fidelity.common import ACTION_CLIP, Harness  # noqa: E402
from src.jaxrl.reppo import effective_sample_size, estep_weights  # noqa: E402


def measure(ckpt_dir, M=32, n_states=256, burn_in=50, seed=0):
    h = Harness(ckpt_dir, n_states)
    key = jax.random.PRNGKey(seed)
    key, rk = jax.random.split(key)
    obs, _, st = h.reset(rk)
    for _ in range(burn_in):
        k1, k2, key = jax.random.split(key, 3)
        a = jnp.clip(h.pi(obs).sample(seed=k1), -ACTION_CLIP, ACTION_CLIP)
        obs, _, st, _, _, _ = h.env.step(jax.random.split(k2, n_states), st, a)

    pi = h.ck.actor.actor(h.na(obs))
    a_i = jnp.clip(
        pi.sample(seed=jax.random.PRNGKey(5), sample_shape=(M,)), -1 + 1e-4, 1 - 1e-4
    )
    cobs = jnp.broadcast_to(h.nc(obs), (M, *h.nc(obs).shape))
    q_i = h.ck.critic.critic(cobs, a_i)
    alpha = jnp.float32(h.meta["alpha_entropy"])

    q_spread = (q_i / alpha).std(axis=0)
    w = estep_weights(q_i, alpha)
    ess = effective_sample_size(w, axis=0)
    sigma = pi.distribution.scale
    return {
        "alpha": float(alpha),
        "q_sd": float(q_i.std(axis=0).mean()),
        "q_spread": float(q_spread.mean()),
        "ess_mean": float(ess.mean()),
        "ess_median": float(jnp.median(ess)),
        "ess_min": float(ess.min()),
        "w_max": float(w.max(axis=0).mean()),
        "pi_sigma_mean": float(sigma.mean()),
        "entropy": float(-pi.log_prob(pi.sample(seed=jax.random.PRNGKey(7)))
                         .sum(-1).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--n_states", type=int, default=256)
    args = ap.parse_args()

    print(f"{'checkpoint':<38} {'steps':>11} {'ret':>7} {'alpha':>8} {'q_sd':>7} "
          f"{'q_spread':>9} {'ESS':>6} {'w_max':>6} {'sigma':>6} {'ent':>7}")
    for c in args.ckpts:
        meta = json.load(open(os.path.join(c, "meta.json")))
        r = measure(c, args.M, args.n_states)
        ret = meta.get("eval_return_at_snapshot", meta.get("final_eval_return"))
        print(f"{os.path.basename(c):<38} {meta['time_steps']:>11,} {ret:>7.1f} "
              f"{r['alpha']:>8.5f} {r['q_sd']:>7.4f} {r['q_spread']:>9.2f} "
              f"{r['ess_mean']:>6.2f} {r['w_max']:>6.3f} {r['pi_sigma_mean']:>6.3f} "
              f"{r['entropy']:>7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
