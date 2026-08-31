"""Offline checkpoint probe: critic-side return scale, within/across-state Q spread,
per-coordinate sigma (real vs padded), action saturation, success metric if exposed.
M=32 actions per state, 2048 states after >=200 burn-in steps under pi_old."""
import json, os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import jax, jax.numpy as jnp, numpy as np
from scripts.critic_fidelity.common import ACTION_CLIP, Harness

ck = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else None
B, burn, M, H, chunks, gap = 256, 200, 32, 400, 8, 25
h = Harness(ck, B); meta = h.meta
vmin, vmax = meta["critic_kwargs"]["vmin"], meta["critic_kwargs"]["vmax"]
pad = int(meta.get("action_pad", 0)); d = int(meta["action_dim"]); real = d - pad
key = jax.random.PRNGKey(0); key, rk = jax.random.split(key)
obs, _, st = h.reset(rk)
metric_keys = list(getattr(st.env_state, "metrics", {}).keys()) if hasattr(st, "env_state") else []
qs, qmean, sds, sds_real, sds_pad, sats, softs, e0, e1, sig_real, sig_pad, succ = ([] for _ in range(12))
for step in range(burn + chunks * gap):
    k1, k2, key = jax.random.split(key, 3)
    dist = h.pi(obs)
    a = jnp.clip(dist.sample(seed=k1), -ACTION_CLIP, ACTION_CLIP)
    if step >= burn and (step - burn) % gap == 0:
        kk = jax.random.fold_in(key, 7)
        a_i = jnp.clip(dist.sample(seed=kk, sample_shape=(M,)), -1 + 1e-4, 1 - 1e-4)
        cobs = jnp.broadcast_to(h.nc(obs), (M, *h.nc(obs).shape))
        q_i = h.ck.critic.critic(cobs, a_i)
        qs.append(np.asarray(q_i).ravel()); qmean.append(np.asarray(q_i.mean(0))); sds.append(np.asarray(q_i.std(0)))
        sats.append(float((jnp.abs(a_i) > 0.99).mean()))
        det = h.det_action(obs)
        if pad > 0:  # perturb only the real coords (padded held at tanh(mu)), and only the padded coords
            a_r = a_i.at[..., real:].set(jnp.broadcast_to(det[:, real:], (M, B, pad)))
            a_p = a_i.at[..., :real].set(jnp.broadcast_to(det[:, :real], (M, B, real)))
            sds_real.append(np.asarray(h.ck.critic.critic(cobs, a_r).std(0)))
            sds_pad.append(np.asarray(h.ck.critic.critic(cobs, a_p).std(0)))
        sc = np.asarray(dist.distribution.scale)  # pre-squash sigma per coord (B, d)
        sig_real.append(sc[:, :real].ravel()); sig_pad.append(sc[:, real:].ravel())
        p = jax.nn.softmax(h.ck.critic.critic_cat(h.nc(obs), a), axis=-1)
        e0.append(np.asarray(p[..., 0])); e1.append(np.asarray(p[..., -1]))
        softs.append(np.asarray(h.soft_return(st, obs, a, H, jax.random.fold_in(key, 11))))
        sk = [k for k in metric_keys if "success" in k]
        if sk: succ.append(np.stack([np.asarray(st.env_state.metrics[k]) for k in sk], -1))
    obs, _, st, _, _, _ = h.env.step(jax.random.split(k2, B), st, a)
q = np.concatenate(qs); qm = np.concatenate(qmean); sd = np.concatenate(sds); soft = np.concatenate(softs)
pc = lambda x, ps=(1, 50, 99): [float(v) for v in np.percentile(x, ps)]
res = dict(ckpt=ck, env=meta["env_name"], d=d, pad=pad, obs_dim=int(meta["obs_dim"]), gamma=float(meta["gamma"]),
    vmin=vmin, vmax=vmax, alpha=meta.get("alpha_entropy"), final_eval_return=meta.get("final_eval_return"),
    eval_curve=meta.get("eval_return_curve"), n_states=int(sd.size),
    q_p1_p50_p99=pc(q), q_min=float(q.min()), q_max=float(q.max()), mc_soft_p50=float(np.median(soft)),
    edge0_mean=float(np.concatenate(e0).mean()), edge0_max=float(np.concatenate(e0).max()),
    edgeTop_mean=float(np.concatenate(e1).mean()), edgeTop_max=float(np.concatenate(e1).max()),
    within_sd_mean=float(sd.mean()), within_sd_median=float(np.median(sd)),
    across_sd=float(qm.std()), rel_sd=float(sd.mean() / max(abs(np.median(q)), 1e-9)),
    within_over_across=float(sd.mean() / max(qm.std(), 1e-12)),
    action_sat=float(np.mean(sats)),
    sigma_real_median=float(np.median(np.concatenate(sig_real))), sigma_real_mean=float(np.mean(np.concatenate(sig_real))),
    sigma_pad_median=float(np.median(np.concatenate(sig_pad))) if pad else None,
    sigma_pad_mean=float(np.mean(np.concatenate(sig_pad))) if pad else None,
    within_sd_real_only=float(np.concatenate(sds_real).mean()) if pad else None,
    within_sd_pad_only=float(np.concatenate(sds_pad).mean()) if pad else None,
    metric_keys=metric_keys, success_keys=[k for k in metric_keys if "success" in k], success_mean=[float(v) for v in np.concatenate(succ).mean(0)] if succ else None)
print(json.dumps({k: v for k, v in res.items() if k != "eval_curve"}, indent=1))
if out: json.dump(res, open(out, "w"))
