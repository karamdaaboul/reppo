"""STEP 1 HARD GATE: does an explicit pre-tanh draw reproduce
`pi.sample_and_log_prob(seed=key, sample_shape=(32,))` BIT-FOR-BIT?

Two candidate paths:
  (a) reuse distrax's own base distribution:
         x, lp_x = pi.distribution.sample_and_log_prob(seed=key, sample_shape=(32,))
         y, fldj = vmap(pi.bijector.forward_and_log_det)(x)
         lp_y    = vmap(subtract)(lp_x, fldj)
  (b) hand-rolled Gaussian draw:
         x = scale * normal(key, (32,B,d)) + loc

Exact equality only -- no allclose.
"""
import os, sys
os.environ["JAX_PLATFORMS"]="cpu"; sys.path.insert(0,"/home/human/workspaces/reppo_original")
import jax, jax.numpy as jnp, numpy as np
from scripts.critic_fidelity.common import Harness, ACTION_CLIP

N_ESTEP = 32
h = Harness("exports/HumanoidRun_weighted_mle_s2_final", 64)
key = jax.random.PRNGKey(20260902)
k, rk = jax.random.split(key)
obs, _, st = h.reset(rk)
for _ in range(40):                                  # get off the reset manifold
    ak, sk, k = jax.random.split(k, 3)
    a = jnp.clip(h.pi(obs).sample(seed=ak), -ACTION_CLIP, ACTION_CLIP)
    obs, _, st, _, _, _ = h.env.step(h._step_keys(sk), st, a)

nobs = h.na(obs)
pi = h.ck.actor.actor(nobs)
draw_key = jax.random.fold_in(jax.random.PRNGKey(11), 7)   # the "same key" for all paths

# ---------- reference: exactly what reppo.py:724-726 calls ----------
ref_a, ref_lp = pi.sample_and_log_prob(sample_shape=(N_ESTEP,), seed=draw_key)

# ---------- (a) distrax's own base + bijector ----------
x_a, lp_x = pi.distribution.sample_and_log_prob(seed=draw_key, sample_shape=(N_ESTEP,))
y_a, fldj = jax.vmap(pi.bijector.forward_and_log_det)(x_a)
lp_a = jax.vmap(jnp.subtract)(lp_x, fldj)

# ---------- (b) hand-rolled Gaussian ----------
loc, scale = h.ck.actor.gaussian(nobs)
rnd = jax.random.normal(draw_key, (N_ESTEP, *loc.shape), dtype=jnp.result_type(loc, scale))
x_b = scale[None] * rnd + loc[None]
y_b, fldj_b = jax.vmap(pi.bijector.forward_and_log_det)(x_b)
lp_b = jax.vmap(jnp.subtract)(pi.distribution.log_prob(x_b), fldj_b)

def rep(tag, a, lp, xref=None, x=None):
    ea = bool(jnp.array_equal(a, ref_a)); el = bool(jnp.array_equal(lp, ref_lp))
    da = float(jnp.abs(a-ref_a).max()); dl = float(jnp.abs(lp-ref_lp).max())
    nbits_a = int((a.view(jnp.int32) != ref_a.view(jnp.int32)).sum())
    nbits_l = int((lp.view(jnp.int32) != ref_lp.view(jnp.int32)).sum())
    print(f"  {tag}")
    print(f"    actions : exact={ea}  max|d|={da:.3e}  differing floats={nbits_a}/{a.size}")
    print(f"    log_prob: exact={el}  max|d|={dl:.3e}  differing floats={nbits_l}/{lp.size}")
    if x is not None and xref is not None:
        print(f"    pre-tanh: exact={bool(jnp.array_equal(x,xref))}  max|d|={float(jnp.abs(x-xref).max()):.3e}")
    return ea and el

print(f"states={nobs.shape[0]}  action_dim={loc.shape[-1]}  M={N_ESTEP}  "
      f"shapes ref_a={ref_a.shape} ref_lp={ref_lp.shape}\n")
print("BIT-IDENTITY vs pi.sample_and_log_prob(sample_shape=(32,), seed=key):")
ok_a = rep("(a) distrax base + bijector", y_a, lp_a)
ok_b = rep("(b) hand-rolled scale*normal+loc", y_b, lp_b, x_a, x_b)
print()
# the quantity the intervention needs: is x_a really the pre-tanh of ref_a?
print(f"  tanh(x_a) == ref_a exactly: {bool(jnp.array_equal(jnp.tanh(x_a), ref_a))}")
print(f"  x_a finite: {bool(jnp.isfinite(x_a).all())}   |x_a| max = {float(jnp.abs(x_a).max()):.4f}")
print()
print(f"GATE (a): {'PASS' if ok_a else 'FAIL'}")
print(f"GATE (b): {'PASS' if ok_b else 'FAIL'}")
