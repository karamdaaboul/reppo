"""Assert the E-step routes asymmetric observations to the right networks.

On G1JoystickFlatTerrain the actor sees `state` (103) and the critic sees
`privileged_state` (216). The E-step must evaluate Q(s, a_i) on the CRITIC obs tiled
to the sample axis, and log pi on the ACTOR obs. Getting this backwards would still
run -- the shapes are only checked inside the networks -- so it is asserted here
rather than eyeballed.

The expressions below are copied from the weighted_mle branch of `actor_loss`
verbatim, so this exercises the same computation the trainer performs.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import nnx  # noqa: E402

from src.env_utils.jax_wrappers import MjxGymnaxWrapper  # noqa: E402
from src.networks.jax_models import (  # noqa: E402
    CategoricalCriticNetwork,
    SACActorNetworks,
)

TASK = "G1JoystickFlatTerrain"
M, B = 32, 8


def main() -> int:
    env = MjxGymnaxWrapper(TASK, episode_length=1000, reward_scale=1.0,
                           asymmetric_observation=True)
    actor_space, critic_space = env.observation_space(None)
    act_dim = env.action_space(None).shape[0]
    a_dim, c_dim = actor_space.shape[0], critic_space.shape[0]
    print(f"{TASK}: actor obs {a_dim}, critic obs {c_dim}, action dim {act_dim}")

    fails = []
    if a_dim == c_dim:
        fails.append(f"expected asymmetric obs, got actor==critic=={a_dim}")

    actor = SACActorNetworks(obs_dim=a_dim, action_dim=act_dim, hidden_dim=64,
                             ent_start=0.01, kl_start=0.01, layers=2,
                             with_eta=True, rngs=nnx.Rngs(0))
    critic = CategoricalCriticNetwork(obs_dim=c_dim, action_dim=act_dim, hidden_dim=64,
                                      num_bins=51, vmin=-10.0, vmax=10.0,
                                      encoder_layers=1, head_layers=1, pred_layers=1,
                                      rngs=nnx.Rngs(0))

    obs = jnp.zeros((B, a_dim))          # minibatch.obs
    critic_obs = jnp.zeros((B, c_dim))   # minibatch.critic_obs

    # --- exactly the lines from actor_loss -------------------------------------
    pi = actor.actor(obs)
    old_pi_action, old_logp = pi.sample_and_log_prob(sample_shape=(M,),
                                                     seed=jax.random.PRNGKey(0))
    old_pi_action = jnp.clip(old_pi_action, -1 + 1e-4, 1 - 1e-4)
    critic_obs_i = jnp.broadcast_to(critic_obs, (M, *critic_obs.shape))
    q_i = critic.critic(critic_obs_i, old_pi_action)
    logp_theta_i = pi.log_prob(old_pi_action).sum(-1)

    checks = [
        ("critic obs tiled to (M, B, critic_dim)", critic_obs_i.shape, (M, B, c_dim)),
        ("q_i is (M, B)", q_i.shape, (M, B)),
        ("actions are (M, B, act_dim)", old_pi_action.shape, (M, B, act_dim)),
        ("log_prob is (M, B)", logp_theta_i.shape, (M, B)),
        ("old log_prob is (M, B)", old_logp.sum(-1).shape, (M, B)),
    ]
    for name, got, want in checks:
        ok = got == want
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<42} got {got}, want {want}")
        if not ok:
            fails.append(f"{name}: got {got} want {want}")

    # the critic must genuinely consume 216 dims: feeding it the actor obs must fail
    try:
        critic.critic(jnp.broadcast_to(obs, (M, *obs.shape)), old_pi_action)
        fails.append("critic accepted the 103-dim ACTOR obs -- it is not "
                     "dimension-checking, so the routing is unverified")
        print("  FAIL  critic wrongly accepted actor obs")
    except Exception:
        print(f"  PASS  critic rejects the {a_dim}-dim actor obs (consumes {c_dim})")

    # and the actor must genuinely consume 103
    try:
        actor.actor(critic_obs)
        fails.append("actor accepted the 216-dim CRITIC obs")
        print("  FAIL  actor wrongly accepted critic obs")
    except Exception:
        print(f"  PASS  actor rejects the {c_dim}-dim critic obs (consumes {a_dim})")

    print()
    if fails:
        print("ASSERTION FAILED:")
        for f in fails:
            print("  - " + f)
        return 1
    print("ASSERTION PASSED: Q uses tiled critic obs (M,B,216); log_prob uses actor obs (103).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
