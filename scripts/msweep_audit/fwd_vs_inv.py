"""How big is the forward-vs-inverse log-prob gap the intervention branch introduces?

Control (sqrt_rho == 1.0) evaluates  pi.log_prob(clip(tanh(y)))   -- distrax inverts
the bijector, i.e. arctanh, internally. The intervention branch evaluates
  pi.distribution.log_prob(y) - fldj(y)                            -- forward only.
Same quantity mathematically; different op path, and the inverse path is the one that
clamps. Measured here at rho = 1 so the ONLY difference is the evaluation path.
"""
import os, sys
os.environ["JAX_PLATFORMS"]="cpu"; sys.path.insert(0,"/home/human/workspaces/reppo_original")
import jax, jax.numpy as jnp, numpy as np
from scripts.critic_fidelity.common import Harness, ACTION_CLIP
for ck in ["exports/HumanoidRun_weighted_mle_s2_final","exports/HumanoidRun_weighted_mle_s211_final"]:
    h=Harness(ck,128); k=jax.random.PRNGKey(20260902); k,rk=jax.random.split(k)
    obs,_,st=h.reset(rk)
    for _ in range(160):
        ak,sk,k=jax.random.split(k,3)
        a=jnp.clip(h.pi(obs).sample(seed=ak),-ACTION_CLIP,ACTION_CLIP)
        obs,_,st,_,_,_=h.env.step(h._step_keys(sk),st,a)
    pi=h.ck.actor.actor(h.na(obs))
    y,_lp=pi.distribution.sample_and_log_prob(seed=k,sample_shape=(32,))
    a_raw,fldj=jax.vmap(pi.bijector.forward_and_log_det)(y)
    a_clip=jnp.clip(a_raw,-1+1e-4,1-1e-4)
    lp_inv=pi.log_prob(a_clip).sum(-1)                       # control path
    lp_fwd=(pi.distribution.log_prob(y)-fldj).sum(-1)        # intervention path
    d=np.asarray(jnp.abs(lp_fwd-lp_inv)); rel=d/np.maximum(np.abs(np.asarray(lp_inv)),1e-9)
    clamped=np.asarray(jnp.abs(a_raw)>=1-1e-4).any(-1)
    print(f"{ck.split('/')[-1]}")
    print(f"  |logp| scale        : mean {float(jnp.abs(lp_inv).mean()):.3f}")
    print(f"  |fwd - inv|         : mean {d.mean():.3e}  p99 {np.percentile(d,99):.3e}  max {d.max():.3e}")
    print(f"  relative            : mean {rel.mean():.3e}  max {rel.max():.3e}")
    print(f"  on CLAMPED samples  : n={int(clamped.sum())}/{clamped.size}  "
          f"mean {d[clamped].mean() if clamped.any() else 0:.3e}  max {d[clamped].max() if clamped.any() else 0:.3e}")
    print(f"  on unclamped        : mean {d[~clamped].mean():.3e}  max {d[~clamped].max():.3e}")
    print()
