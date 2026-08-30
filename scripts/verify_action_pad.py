"""Verify redundant-action padding on WalkerRun before any training run.

Checks: actor output dim == critic action-input dim == E-step sample dim == 6+k,
and the simulator receives exactly 6 coordinates (asserted by a recording shim
inserted BELOW the pad, i.e. where the env boundary is).
"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import jax, jax.numpy as jnp
from flax import nnx
from src.env_utils.action_pad import ActionPad
from src.env_utils.jax_wrappers import ClipAction, LogWrapper, MjxGymnaxWrapper
from src.networks.jax_models import CategoricalCriticNetwork, SACActorNetworks

K = int(sys.argv[1]) if len(sys.argv) > 1 else 16
M, B = 32, 4
seen = {}

class Recorder(MjxGymnaxWrapper):
    def step(self, key, state, action):
        seen["sim_action_dim"] = int(action.shape[-1])
        return super().step(key, state, action)

base = Recorder("WalkerRun", episode_length=1000, reward_scale=1.0)
env = ActionPad(base, K) if K > 0 else base
env = ClipAction(LogWrapper(env, B))
obs_space, _ = env.observation_space(None)
act_dim = env.action_space(None).shape[0]
obs_dim = obs_space.shape[0]

actor = SACActorNetworks(obs_dim=obs_dim, action_dim=act_dim, hidden_dim=64, ent_start=0.01,
                         kl_start=0.01, layers=2, with_eta=True, rngs=nnx.Rngs(0))
critic = CategoricalCriticNetwork(obs_dim=obs_dim, action_dim=act_dim, hidden_dim=64, num_bins=151,
                                  vmin=0.0, vmax=150.0, encoder_layers=1, head_layers=1, pred_layers=1,
                                  rngs=nnx.Rngs(0))
obs, _, st = env.reset(jax.random.split(jax.random.PRNGKey(0), B))
pi = actor.actor(obs)
a_i, _ = pi.sample_and_log_prob(sample_shape=(M,), seed=jax.random.PRNGKey(1))   # E-step samples
q_i = critic.critic(jnp.broadcast_to(obs, (M, *obs.shape)), a_i)
det = actor.det_action(obs)
env.step(jax.random.split(jax.random.PRNGKey(2), B), st, det)

checks = [
    ("env action_space dim", act_dim, 6 + K),
    ("actor det_action dim", det.shape[-1], 6 + K),
    ("actor pre-squash output = 2*(6+k)", actor.actor_module(obs).shape[-1], 2 * (6 + K)),
    ("E-step sample dim", a_i.shape[-1], 6 + K),
    ("critic accepts (M,B,6+k) action -> (M,B)", q_i.shape, (M, B)),
    ("simulator received dim", seen["sim_action_dim"], 6),
]
ok = True
for name, got, want in checks:
    good = got == want; ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {name:<42} got {got}, want {want}")
try:
    critic.critic(jnp.broadcast_to(obs, (M, *obs.shape)), a_i[..., :6]); print("  FAIL  critic accepted a 6-dim action (would allow stripping)"); ok = False
except Exception:
    print(f"  PASS  critic rejects a 6-dim action (consumes {6+K})")
print("VERIFY", "PASSED" if ok else "FAILED", f"k={K}")
sys.exit(0 if ok else 1)
