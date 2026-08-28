"""Validate the soft-return harness before running either experiment.

The decisive check is calibration: Q_phi is trained to predict exactly the quantity
``soft_return`` computes, and we already know this critic is well calibrated (its
reloaded eval return matched the logged one to 0.1%). So over on-policy states with
on-policy actions, ``mean MC soft return`` must track ``mean Q_phi``. A large gap
means the soft return is implemented wrong -- most likely a missing entropy term, a
wrong alpha, or a discounting error.

Also checks that state cloning gives the intended semantics: cloning one state B
times and executing B *identical* actions must give B identical next observations.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from scripts.critic_fidelity.common import ACTION_CLIP, Harness, gather_states, tile_state  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="exports/WalkerRun_s0")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    h = Harness(args.ckpt, args.batch)
    print(f"{h.meta['env_name']}: gamma={h.gamma} alpha={h.alpha:.6g} "
          f"act_dim={h.action_dim} obs_dim={h.obs_dim}")

    key = jax.random.PRNGKey(0)
    # burn in so states are on-policy mid-episode, not all at reset
    states, obss = h.collect_states(key, 60)
    st, obs = gather_states(states, -1), obss[-1]
    print(f"collected states: {jax.tree_util.tree_leaves(st)[0].shape[0]} envs")

    # --- clone determinism ---------------------------------------------------
    one = tile_state(st, 0, h.B)
    a_same = jnp.tile(h.det_action(obs[0:1]), (h.B, 1))
    o1, _, _, _, _, _ = h.env.step(jax.random.split(jax.random.PRNGKey(7), h.B), one, a_same)
    spread = float(jnp.abs(o1 - o1[0:1]).max())
    print(f"clone determinism: max spread across {h.B} clones = {spread:.3e} "
          f"-> {'OK' if spread == 0 else 'MISMATCH'}")

    # --- calibration: MC soft return vs Q -----------------------------------
    print("\ncalibration (on-policy actions):")
    print("  H     mean MC        mean Q       gap        corr(MC,Q)")
    a_on = jnp.clip(h.pi(obs).sample(seed=jax.random.PRNGKey(3)), -ACTION_CLIP, ACTION_CLIP)
    q_on = np.asarray(h.q(obs, a_on))
    for H in (50, 100, 200, 400):
        mc = np.asarray(h.soft_return(st, obs, a_on, H, jax.random.PRNGKey(11)))
        gap = mc.mean() - q_on.mean()
        corr = float(np.corrcoef(mc, q_on)[0, 1])
        print(f"  {H:<5d} {mc.mean():10.3f}   {q_on.mean():10.3f}   "
              f"{gap:+8.3f}   {corr:+.3f}")

    # --- how much of the return is the entropy term -------------------------
    mc_soft = np.asarray(h.soft_return(st, obs, a_on, 200, jax.random.PRNGKey(11), h.alpha))
    mc_plain = np.asarray(h.soft_return(st, obs, a_on, 200, jax.random.PRNGKey(11), 0.0))
    print(f"\nentropy contribution at H=200: soft={mc_soft.mean():.3f} "
          f"plain={mc_plain.mean():.3f} diff={mc_soft.mean() - mc_plain.mean():+.3f} "
          f"({100 * (mc_soft.mean() - mc_plain.mean()) / abs(mc_soft.mean()):+.1f}% of return)")

    # --- MC noise level ------------------------------------------------------
    m1 = np.asarray(h.soft_return(st, obs, a_on, 200, jax.random.PRNGKey(101)))
    m2 = np.asarray(h.soft_return(st, obs, a_on, 200, jax.random.PRNGKey(202)))
    print(f"two independent MC estimates: corr={np.corrcoef(m1, m2)[0, 1]:+.4f} "
          f"sd(diff)={np.std(m1 - m2):.3f} sd(MC)={np.std(m1):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
