"""A-audit: three across-state definitions on ONE sampled (state, action) tensor.
Sampler identical to probe_ckpt.py: B=256 envs, burn-in 200 steps under pi_old, 8 chunks
25 steps apart -> 2048 states, M=32 actions/state from pi_old, jax PRNGKey(0) chain,
action-sample key = fold_in(key,7) at each chunk. Critic = single CategoricalCriticNetwork
head, scalar Q = sum_b softmax(logits)_b * linspace(vmin,vmax,151)_b (no ensemble)."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax, jax.numpy as jnp, numpy as np
from scripts.critic_fidelity.common import ACTION_CLIP, Harness
ck = sys.argv[1]; B, burn, M, chunks, gap = 256, 200, 32, 8, 25
h = Harness(ck, B); key = jax.random.PRNGKey(0); key, rk = jax.random.split(key); obs, _, st = h.reset(rk)
Q = []
for step in range(burn + chunks * gap):
    k1, k2, key = jax.random.split(key, 3); dist = h.pi(obs)
    a = jnp.clip(dist.sample(seed=k1), -ACTION_CLIP, ACTION_CLIP)
    if step >= burn and (step - burn) % gap == 0:
        a_i = jnp.clip(dist.sample(seed=jax.random.fold_in(key, 7), sample_shape=(M,)), -1 + 1e-4, 1 - 1e-4)
        Q.append(np.asarray(h.ck.critic.critic(jnp.broadcast_to(h.nc(obs), (M, *h.nc(obs).shape)), a_i)).T)  # (B,M)
    obs, _, st, _, _, _ = h.env.step(jax.random.split(k2, B), st, a)
Q = np.concatenate(Q, 0)  # (S=2048, M=32)
S_, M_ = Q.shape
rng = np.random.default_rng(12345); j = rng.integers(0, M_, size=S_)
out = dict(ckpt=ck, step=h.meta["time_steps"], seed=h.meta["seed"], n_states=S_, actions_per_state=M_, pick_rng="np.random.default_rng(12345)",
  a_pop=float(Q.mean(1).std(ddof=0)), a_samp=float(Q.mean(1).std(ddof=1)),
  b_pop=float(Q.std(ddof=0)), b_samp=float(Q.std(ddof=1)),
  c_pop=float(Q[np.arange(S_), j].std(ddof=0)), c_samp=float(Q[np.arange(S_), j].std(ddof=1)),
  within_mean_pop=float(Q.std(1, ddof=0).mean()), within_median_pop=float(np.median(Q.std(1, ddof=0))),
  within_mean_samp=float(Q.std(1, ddof=1).mean()),
  q_p1=float(np.percentile(Q, 1)), q_p50=float(np.percentile(Q, 50)), q_p99=float(np.percentile(Q, 99)),
  range_p99_p1=float(np.percentile(Q, 99) - np.percentile(Q, 1)),
  state_mean_p1=float(np.percentile(Q.mean(1), 1)), state_mean_p99=float(np.percentile(Q.mean(1), 99)))
print(json.dumps(out, indent=1)); np.save(sys.argv[2], Q) if len(sys.argv) > 2 else None
