"""Mandatory frozen same-critic diagnostic (prereg section 9) + reload verification.

At each final corrected checkpoint, on IDENTICAL states with common random numbers:
    PW-1        one-sample pathwise gradient  (the training operator)
    PW-32       32-sample pathwise gradient   (query-budget control)
    ZO-32       centred value-only estimator  (the manuscript's g_ZO)
    c           exact nonlinear centred WML component
    v           full standardized WML mean score  (= ubar + c)
All in the whitened pre-tanh metric. Separates the algorithmic operator difference
from the action-query budget, the nonlinear softmax weighting and the uniform term.
PW-1 is never treated as PW-32.

Usage: fr_samecritic.py <ckpt_dir> <out.npz>
"""
from __future__ import annotations
import hashlib, json, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np                       # noqa: E402
from scripts.critic_fidelity.common import ACTION_CLIP, Harness   # noqa: E402

M, BURN, N_ENVS, ROOT = 32, 50, 512, 20260902

def key_for(tag, ck):
    d = hashlib.blake2b(("%s|%s" % (tag, ck)).encode(), digest_size=4).digest()
    return jax.random.fold_in(jax.random.PRNGKey(ROOT), int.from_bytes(d, "big") % (2**31))

def unit(v):
    return v / jnp.maximum(jnp.linalg.norm(v, axis=-1, keepdims=True), 1e-300)

def main(ckpt, out):
    tag = os.path.basename(ckpt)
    meta = json.load(open("%s/meta.json" % ckpt))
    h = Harness(ckpt, N_ENVS)
    d = h.action_dim
    key = key_for("states", tag); key, rk = jax.random.split(key)
    obs, _, st = h.reset(rk)
    for _ in range(BURN):
        k1, k2, key = jax.random.split(key, 3)
        a = jnp.clip(h.pi(obs).sample(seed=k1), -ACTION_CLIP, ACTION_CLIP)
        obs, _, st, _, _, _ = h.env.step(jax.random.split(k2, N_ENVS), st, a)

    # ---- reload verification: does the loaded policy reproduce the logged return? --
    tot = jnp.zeros(N_ENVS); key, ek = jax.random.split(key)
    o2, _, s2 = h.reset(ek)
    for _ in range(1000):
        k1, k2, key = jax.random.split(key, 3)
        aa = jnp.clip(h.pi(o2).sample(seed=k1), -ACTION_CLIP, ACTION_CLIP)
        o2, _, s2, rw, _, _ = h.env.step(jax.random.split(k2, N_ENVS), s2, aa)
        tot = tot + rw
    reload_ret = float(np.asarray(tot).mean())

    mu, sg = h.ck.actor.gaussian(h.na(jnp.asarray(obs)))
    mu = np.asarray(mu, np.float64); sg = np.asarray(sg, np.float64)
    cobs = np.asarray(h.nc(jnp.asarray(obs)), np.float32)
    # common random numbers: one (M,N,d) draw shared by every estimator
    u = np.asarray(jax.random.normal(key_for("u", tag), (M, len(mu), d), dtype=jnp.float64))
    y = mu[None] + sg[None] * u
    a_i = np.tanh(y)

    def q_of(acts):
        cb = jnp.broadcast_to(jnp.asarray(cobs), (acts.shape[0], *cobs.shape))
        return np.asarray(h.ck.critic.critic(cb, jnp.asarray(acts, jnp.float32)), np.float64)

    # whitened pathwise gradient at a sample: sigma * d/dy Q(s, tanh(y))
    def hgrad(yy):
        def f(y1, c):
            return h.ck.critic.critic(c, jnp.tanh(y1)).squeeze()
        gg = jax.vmap(jax.vmap(jax.grad(f), in_axes=(0, 0)), in_axes=(0, None))
        return np.asarray(gg(jnp.asarray(yy, jnp.float32), jnp.asarray(cobs)), np.float64)

    G = hgrad(y) * sg[None]                       # (M, N, d) whitened per-sample grads
    pw1, pw32 = G[0], G.mean(0)
    Q = q_of(a_i)
    zo32 = np.einsum("ib,ibd->bd", Q - Q.mean(0, keepdims=True), u) / M
    try:
        eta = float(np.asarray(h.ck.actor.eta()).ravel()[0]); eta_src = "measured"
    except AttributeError:                        # pathwise ckpt has no eta_param
        eta, eta_src = float(np.std(Q)), "recomputed_placeholder"
    z = Q / eta; z -= z.max(0, keepdims=True)
    w = np.exp(z); w /= w.sum(0, keepdims=True)
    v = np.einsum("ib,ibd->bd", w, u)
    ubar = u.mean(0)
    c = v - ubar
    cos = lambda p, q: float(np.mean(np.sum(unit(jnp.asarray(p)) * unit(jnp.asarray(q)), -1)))
    nrm = lambda p: float(np.mean(np.linalg.norm(p, axis=-1)))
    res = dict(tag=tag, env=meta["env_name"], mode=meta["actor_update_mode"],
               seed=meta["seed"], d=d, eta=eta, eta_src=eta_src,
               logged_return=meta["final_eval_return"], reload_return=reload_ret,
               n_pw1=nrm(pw1), n_pw32=nrm(pw32), n_zo32=nrm(zo32),
               n_c=nrm(c), n_v=nrm(v), n_ubar=nrm(ubar),
               cos_pw1_pw32=cos(pw1, pw32), cos_pw32_zo32=cos(pw32, zo32),
               cos_zo32_c=cos(zo32, c), cos_c_v=cos(c, v), cos_pw32_v=cos(pw32, v),
               cos_pw1_v=cos(pw1, v), R2_ubar_c=float(
                   np.sqrt(np.sum(np.linalg.norm(ubar, axis=-1)**2) /
                           max(np.sum(np.linalg.norm(c, axis=-1)**2), 1e-30))))
    np.savez_compressed(out, **res)
    print("%-48s ret logged %8.2f reload %8.2f | |PW1| %.4g |PW32| %.4g |ZO32| %.4g "
          "|c| %.4g |v| %.4g | cos(PW1,PW32) %.3f cos(PW32,ZO32) %.3f cos(ZO32,c) %.3f"
          % (tag, res["logged_return"], reload_ret, res["n_pw1"], res["n_pw32"],
             res["n_zo32"], res["n_c"], res["n_v"], res["cos_pw1_pw32"],
             res["cos_pw32_zo32"], res["cos_zo32_c"]), flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
