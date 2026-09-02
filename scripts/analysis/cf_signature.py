"""Deterministic signature of the actor/critic construction and gradients.

Run inside a given worktree so that its own src/ is imported. Comparing the
signatures of two source versions gives component-level evidence that adding the
covariance-freeze mechanism changes nothing when the flag is off. It is NOT the
end-to-end training parity, which needs a GPU; it is a cheap CPU-side check that can
be run while that job is queued.

Usage: cf_signature.py <worktree_root> <out.npz>
"""
from __future__ import annotations

import sys, os
root, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)

import jax                                   # noqa: E402
import jax.numpy as jnp                      # noqa: E402
import numpy as np                           # noqa: E402
from flax import nnx                         # noqa: E402

from src.networks.jax_models import CategoricalCriticNetwork, SACActorNetworks  # noqa: E402

AKW = dict(obs_dim=24, action_dim=6, hidden_dim=512, ent_start=0.014509912580251694,
           kl_start=0.01, use_norm=True, layers=3, use_skip=False)
CKW = dict(obs_dim=24, action_dim=6, hidden_dim=512, num_bins=151, vmin=0.0,
           vmax=150.0, use_norm=True, encoder_layers=2, head_layers=2,
           pred_layers=2, use_simplical_embedding=False, use_skip=False)


def flat(state):
    items = [(jax.tree_util.keystr(p), np.asarray(v))
             for p, v in jax.tree_util.tree_leaves_with_path(state)]
    return sorted(items, key=lambda kv: kv[0])


payload = {}
obs = np.asarray(jax.random.normal(jax.random.PRNGKey(11), (32, 24)), np.float32)
act = np.asarray(jnp.tanh(jax.random.normal(jax.random.PRNGKey(12), (32, 6))), np.float32)
payload["__obs__"] = obs
payload["__act__"] = act

for mode, with_eta in (("pathwise", False), ("weighted_mle", True)):
    a = SACActorNetworks(**AKW, with_eta=with_eta, with_betas=False, rngs=nnx.Rngs(499))
    for k, v in flat(nnx.state(a)):
        payload["actor:%s:%s" % (mode, k)] = v
    mu, sg = a.gaussian(jnp.asarray(obs))
    d = a.actor(jnp.asarray(obs)).distribution
    payload["mu:%s" % mode] = np.asarray(mu)
    payload["sg:%s" % mode] = np.asarray(sg)
    payload["dloc:%s" % mode] = np.asarray(d.loc)
    payload["dscale:%s" % mode] = np.asarray(d.scale)
    payload["det:%s" % mode] = np.asarray(a.det_action(jnp.asarray(obs)))
    payload["temp:%s" % mode] = np.asarray(a.temperature())
    payload["lag:%s" % mode] = np.asarray(a.lagrangian())
    if with_eta:
        payload["eta:%s" % mode] = np.asarray(a.eta())

    def loss(params):
        # Deliberately avoids distrax log_prob at the tanh boundary: a sample can
        # land at +-1 in float32, giving inf and then NaN gradients, which would
        # make every comparison below vacuous rather than informative.
        m = nnx.merge(nnx.graphdef(a), params)
        mu_, sg_ = m.gaussian(jnp.asarray(obs))
        u = jax.random.normal(jax.random.PRNGKey(13), mu_.shape)
        y = mu_ + sg_ * u
        logp = -0.5 * ((y - mu_) / sg_) ** 2 - jnp.log(sg_)
        return (jnp.sum(mu_ ** 2) + jnp.sum(sg_ ** 3) + jnp.sum(logp)
                + jnp.sum(m.temperature()))

    g = jax.grad(loss)(nnx.state(a))
    for k, v in flat(g):
        payload["grad:%s:%s" % (mode, k)] = v

c = CategoricalCriticNetwork(**CKW, rngs=nnx.Rngs(499))
for k, v in flat(nnx.state(c)):
    payload["critic:%s" % k] = v
payload["q"] = np.asarray(c.critic(jnp.asarray(obs), jnp.asarray(act)))
assert all(np.isfinite(v).all() for v in payload.values()), "non-finite in signature"

np.savez(out, **payload)
print("wrote %s  (%d arrays) from src at %s" % (out, len(payload), root))
