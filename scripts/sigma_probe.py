"""Per-coordinate sigma on visited states (same sampler as probe_ckpt.py: B=256, burn-in 200,
8 chunks 25 steps apart = 2048 states, PRNGKey(0)). Reports median/mean pre-squash sigma
over the real coords and over the padded coords, and their ratio."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax, jax.numpy as jnp, numpy as np
from scripts.critic_fidelity.common import ACTION_CLIP, Harness
ck = sys.argv[1]; B, burn, chunks, gap = 256, 200, 8, 25
h = Harness(ck, B); m = h.meta; pad = int(m.get("action_pad", 0)); d = int(m["action_dim"]); real = d - pad
key = jax.random.PRNGKey(0); key, rk = jax.random.split(key); obs, _, st = h.reset(rk); SG = []
for step in range(burn + chunks * gap):
    k1, k2, key = jax.random.split(key, 3); dist = h.pi(obs)
    if step >= burn and (step - burn) % gap == 0: SG.append(np.asarray(dist.distribution.scale))
    obs, _, st, _, _, _ = h.env.step(jax.random.split(k2, B), st, jnp.clip(dist.sample(seed=k1), -ACTION_CLIP, ACTION_CLIP))
sg = np.concatenate(SG, 0)  # (2048, d)
r = dict(ckpt=ck, step=int(m["time_steps"]), frac=m.get("checkpoint_frac"), seed=m["seed"], d=d, pad=pad, n_states=int(sg.shape[0]),
    sigma_real_median=float(np.median(sg[:, :real])), sigma_real_mean=float(sg[:, :real].mean()),
    sigma_pad_median=float(np.median(sg[:, real:])) if pad else None, sigma_pad_mean=float(sg[:, real:].mean()) if pad else None,
    sigma_per_coord_median=[float(v) for v in np.median(sg, 0)])
r["ratio_pad_over_real_median"] = (r["sigma_pad_median"] / r["sigma_real_median"]) if pad else None
print(json.dumps(r))
if len(sys.argv) > 2: json.dump(r, open(sys.argv[2], "w"))
