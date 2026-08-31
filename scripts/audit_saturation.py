"""B-audit: saturation definitions + log-prob gradient inflation, on a given checkpoint.
Two observation sources: (1) visited states (probe_ckpt sampler, 2048 states, PRNGKey(0));
(2) the ORIGINAL recipe of 2026-08-27 (transcript line 1791): B=64 synthetic obs ~ N(0,1)
via np.random.default_rng(0), u = mu + sigma*eps with jax PRNGKey(1), M=32, Dirichlet(1) weights."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax, jax.numpy as jnp, numpy as np, distrax
from scripts.critic_fidelity.common import ACTION_CLIP, Harness
from src.jaxrl.reppo import gaussian_logp
ck = sys.argv[1]; B, burn, M, chunks, gap = 256, 200, 32, 8, 25
h = Harness(ck, B); actor = h.ck.actor
def sat_stats(u):
    t = np.asarray(jnp.tanh(u)); r = {}
    for tau in (1 - 1e-4, 0.99):
        r[f"per_coord_tau{tau:g}"] = float((np.abs(t) > tau).mean())
        r[f"per_sample_any_tau{tau:g}"] = float((np.abs(t) > tau).any(-1).mean())
        r[f"per_sample_all_tau{tau:g}"] = float((np.abs(t) > tau).all(-1).mean())
    r["n_samples"] = int(t.shape[0] * t.shape[1]); r["n_coords"] = int(t.size); return r
def grad_compare(nobs, key_u, rng):
    mu_old, sg_old = [jax.lax.stop_gradient(x) for x in actor.gaussian(nobs)]
    u = mu_old + sg_old * jax.random.normal(key_u, (M, *mu_old.shape))
    w = jnp.asarray(rng.dirichlet(np.ones(M), size=nobs.shape[0]).T, dtype=jnp.float32)
    a_clip = jnp.clip(jnp.tanh(u), -1 + 1e-4, 1 - 1e-4)
    def via_tanh(p):
        mu, sg = p; pi = distrax.Transformed(distrax.Normal(mu, sg), distrax.Tanh())
        return jnp.sum(-jnp.sum(w * pi.log_prob(a_clip).sum(-1), 0))
    def on_u(p):
        mu, sg = p; return jnp.sum(-jnp.sum(w * gaussian_logp(u, mu[None], sg[None]), 0))
    gt, gb = jax.grad(via_tanh)((mu_old, sg_old)), jax.grad(on_u)((mu_old, sg_old))
    sat_any = np.asarray((jnp.abs(jnp.tanh(u)) > 1 - 1e-4).any(-1))  # (M,B)
    # per-sample sigma-gradient magnitude of the (unweighted) log-prob, conditional on saturation
    def lp_tanh(mu, sg, a): return distrax.Transformed(distrax.Normal(mu, sg), distrax.Tanh()).log_prob(a).sum()
    def lp_u(mu, sg, uu): return gaussian_logp(uu[None], mu[None], sg[None]).sum()
    g_t = jax.vmap(jax.vmap(jax.grad(lp_tanh, argnums=(0, 1)), in_axes=(0, 0, 0)), in_axes=(None, None, 0))(mu_old, sg_old, a_clip)
    g_u = jax.vmap(jax.vmap(jax.grad(lp_u, argnums=(0, 1)), in_axes=(0, 0, 0)), in_axes=(None, None, 0))(mu_old, sg_old, u)
    nt = np.asarray(jnp.linalg.norm(jnp.concatenate(g_t, -1), axis=-1)); nu = np.asarray(jnp.linalg.norm(jnp.concatenate(g_u, -1), axis=-1))
    ratio = nt / np.maximum(nu, 1e-12)
    q = lambda x: [float(v) for v in np.percentile(x, [50, 90, 99, 100])]
    return dict(sat=sat_stats(u), max_abs_ratio_mu=float(jnp.abs(gt[0]).max() / jnp.abs(gb[0]).max()),
        max_abs_ratio_sigma=float(jnp.abs(gt[1]).max() / jnp.abs(gb[1]).max()),
        max_abs_g_sigma_tanh=float(jnp.abs(gt[1]).max()), max_abs_g_sigma_u=float(jnp.abs(gb[1]).max()),
        max_abs_g_mu_tanh=float(jnp.abs(gt[0]).max()), max_abs_g_mu_u=float(jnp.abs(gb[0]).max()),
        norm_ratio_total=float(np.linalg.norm(np.concatenate([np.asarray(gt[0]).ravel(), np.asarray(gt[1]).ravel()])) / np.linalg.norm(np.concatenate([np.asarray(gb[0]).ravel(), np.asarray(gb[1]).ravel()]))),
        per_sample_grad_norm_ratio_saturated_p50_90_99_max=q(ratio[sat_any]) if sat_any.any() else None,
        per_sample_grad_norm_ratio_unsaturated_p50_90_99_max=q(ratio[~sat_any]),
        frac_samples_any_saturated=float(sat_any.mean()), max_atanh_err=float(jnp.abs(jnp.arctanh(a_clip) - u).max()))
res = dict(ckpt=ck, step=h.meta["time_steps"], seed=h.meta["seed"], d=h.action_dim)
# (2) original synthetic recipe
rng = np.random.default_rng(0); nobs = jnp.asarray(rng.normal(size=(64, h.obs_dim)), dtype=jnp.float32)
res["synthetic_obs_recipe"] = grad_compare(nobs, jax.random.PRNGKey(1), rng)
# (1) visited states
key = jax.random.PRNGKey(0); key, rk = jax.random.split(key); obs, _, st = h.reset(rk); NOBS = []
for step in range(burn + chunks * gap):
    k1, k2, key = jax.random.split(key, 3); a = jnp.clip(h.pi(obs).sample(seed=k1), -ACTION_CLIP, ACTION_CLIP)
    if step >= burn and (step - burn) % gap == 0: NOBS.append(h.na(obs))
    obs, _, st, _, _, _ = h.env.step(jax.random.split(k2, B), st, a)
nobs = jnp.concatenate(NOBS, 0)
res["visited_states"] = grad_compare(nobs, jax.random.PRNGKey(1), np.random.default_rng(0)); res["visited_states"]["n_states"] = int(nobs.shape[0])
print(json.dumps(res, indent=1))
