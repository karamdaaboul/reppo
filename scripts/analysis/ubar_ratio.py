"""Step 2: frozen-checkpoint measurement of the uniform empirical-mean term.

Protocol: docs/prereg_ubar_ratio.md (committed at 5912170, before any number below).
Read-only: no training, no checkpoint written, no log touched.

State source matched to scripts/q_spread_from_ckpt.py:34-49 (Harness, reset,
burn_in=50 stochastic steps clipped to +-0.999). The temperature is NOT matched to
that script: it uses alpha_entropy, the trainer uses eta (reppo.py:741).

Primary decomposition is implementation-space (u_fit); raw-Gaussian (u_raw) is the
manuscript-level diagnostic. They differ only through the +-(1-1e-4) clip.

Usage: ubar_ratio.py <ckpt_dir> <out.npz>
"""
from __future__ import annotations
import hashlib, json, os, sys, time
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np                                   # noqa: E402
from scripts.critic_fidelity.common import ACTION_CLIP, Harness        # noqa: E402

M = 32
CLIP = 1.0 - 1e-4
BURN = 50                    # q_spread_from_ckpt.py:39
N_ENVS = 1024                # trainer's env count; two blocks give N=2048 states
N_BLOCKS = 2
N_CLOUD_STATES = 256
N_CLOUDS = 16
EPS = 1e-12
PROBE_ROOT = 20260902        # docs/prereg_ubar_ratio.md Sec. 3


def key_for(tag, ckpt):
    dig = hashlib.blake2b(f"{tag}|{ckpt}".encode(), digest_size=4).digest()
    return jax.random.fold_in(jax.random.PRNGKey(PROBE_ROOT),
                              int.from_bytes(dig, "big") % (2 ** 31))


def solve_eta_scalar(q, eps_e=0.5, lo=1e-4, hi=10.0, iters=100):
    """Single batch-shared eta from the registered MPO dual (reppo.py:168-181).

    Bounds are the network's own [eta_min, eta_max] (jax_models.py:340-341)."""
    q = np.asarray(q, np.float64)
    q = q.reshape(q.shape[0], -1).T                      # (states, M)

    def dg(eta):
        z = q / eta
        zm = z.max(1, keepdims=True)
        lse = zm[:, 0] + np.log(np.mean(np.exp(z - zm), 1))
        w = np.exp(z - zm); w /= w.sum(1, keepdims=True)
        return float(np.mean(eps_e + lse - (w * q).sum(1) / eta))

    a, b = lo, hi
    if dg(a) >= 0: return a
    if dg(b) <= 0: return b
    for _ in range(iters):
        m = np.sqrt(a * b)
        if dg(m) < 0: a = m
        else: b = m
    return float(np.sqrt(a * b))


def collect_states(h, key, n_envs):
    key, rk = jax.random.split(key)
    obs, _, st = h.reset(rk)
    for _ in range(BURN):
        k1, k2, key = jax.random.split(key, 3)
        a = jnp.clip(h.pi(obs).sample(seed=k1), -ACTION_CLIP, ACTION_CLIP)
        obs, _, st, _, _, _ = h.env.step(jax.random.split(k2, n_envs), st, a)
    return np.asarray(obs)


def decompose(u, w, q):
    """v, ubar, c, m_hat for a standardized residual collection u: (M, N, d)."""
    v = np.einsum("ib,ibd->bd", w, u)
    ubar = u.mean(0)
    c = v - ubar
    qc = q - q.mean(0, keepdims=True)
    m_hat = np.einsum("ib,ibd->bd", qc, u) / u.shape[0]
    return v, ubar, c, m_hat


def norm(x):
    return np.linalg.norm(x, axis=-1)


def cos(a, b):
    return np.sum(a * b, -1) / np.maximum(norm(a) * norm(b), EPS)


def metrics(u, w, q, eta):
    v, ubar, c, m_hat = decompose(u, w, q)
    lin = m_hat / eta
    nv, nu, nc, nl = norm(v), norm(ubar), norm(c), norm(lin)
    return dict(
        v=v, ubar=ubar, c=c, m_hat=m_hat,
        n_v=nv, n_ubar=nu, n_c=nc, n_lin=nl,
        norm_ratio=nc / (nl + EPS),
        cosine_linear=cos(c, lin),
        residual_linear=norm(c - lin) / (nc + EPS),
        R_exact=nu / (nc + EPS),
        R_linear=nu / (nl + EPS),
        cos_ubar_c=cos(ubar, c),
        cos_v_c=cos(v, c),
        cross_fraction=2 * np.sum(ubar * c, -1) / (nu ** 2 + nc ** 2 + EPS),
        direction_change=1 - cos(v, c),
    )


def sample_cloud(mu, sg, key, n, d):
    u = np.asarray(jax.random.normal(key, (M, n, d), dtype=jnp.float64), np.float64)
    y = mu[None] + sg[None] * u
    a_raw = np.tanh(y)
    a_fit = np.clip(a_raw, -CLIP, CLIP)
    u_fit = (np.arctanh(a_fit) - mu[None]) / sg[None]
    return u, u_fit, a_fit, float(np.mean(np.abs(a_raw) > CLIP))


def main(ckpt, out):
    t0 = time.time()
    meta = json.load(open(f"{ckpt}/meta.json"))
    tag = os.path.basename(ckpt)
    h = Harness(ckpt, N_ENVS)
    d = h.action_dim
    eta_measured = None
    try:
        eta_measured = float(np.asarray(h.ck.actor.eta()).ravel()[0])
    except AttributeError:
        pass

    obs = np.concatenate([collect_states(h, key_for(f"states{b}", tag), N_ENVS)
                          for b in range(N_BLOCKS)])
    N = obs.shape[0]
    pi = h.ck.actor.actor(jnp.asarray(obs))
    mu = np.asarray(pi.distribution.loc, np.float64)
    sg = np.asarray(pi.distribution.scale, np.float64)
    cobs = np.asarray(h.nc(jnp.asarray(obs)), np.float32)

    u_raw, u_fit, a_fit, clip_rate = sample_cloud(mu, sg, key_for("cloud0", tag), N, d)
    q = np.asarray(h.ck.critic.critic(
        jnp.broadcast_to(jnp.asarray(cobs), (M, *cobs.shape)),
        jnp.asarray(a_fit, jnp.float32)), np.float64)

    if eta_measured is not None:
        eta, eta_src = eta_measured, "measured"
    else:
        eta, eta_src = solve_eta_scalar(q), "recomputed_counterfactual"
    w = np.asarray(jax.nn.softmax(jnp.asarray(q / eta), axis=0), np.float64)

    mf = metrics(u_fit, w, q, eta)
    mr = metrics(u_raw, w, q, eta)
    logit_spread = (q / eta).std(0)
    ess = 1.0 / np.sum(w ** 2, 0)
    w_max = w.max(0)

    # --- 16 independent action clouds on a fixed 256-state subset --------------
    sub = slice(0, N_CLOUD_STATES)
    cl = {k: [] for k in ("R_exact", "n_ubar", "n_c", "cos_v_c", "residual_linear")}
    for r in range(N_CLOUDS):
        ur, uf, af, _ = sample_cloud(mu[sub], sg[sub], key_for(f"cloud{r+1}", tag),
                                     N_CLOUD_STATES, d)
        qq = np.asarray(h.ck.critic.critic(
            jnp.broadcast_to(jnp.asarray(cobs[sub]), (M, N_CLOUD_STATES, cobs.shape[-1])),
            jnp.asarray(af, jnp.float32)), np.float64)
        ww = np.asarray(jax.nn.softmax(jnp.asarray(qq / eta), axis=0), np.float64)
        mm = metrics(uf, ww, qq, eta)
        for k in cl: cl[k].append(mm[k])
    cl = {k: np.stack(v) for k, v in cl.items()}          # (N_CLOUDS, 256)

    np.savez_compressed(
        out, tag=tag, env=meta["env_name"], mode=meta["actor_update_mode"],
        seed=meta["seed"], d=d, action_pad=meta.get("action_pad", 0) or 0, M=M,
        eta=eta, eta_src=eta_src,
        eta_measured=(eta_measured if eta_measured is not None else np.nan),
        clip_rate=clip_rate, N=N,
        mu_absmean=np.abs(mu).mean(1), sigma_mean=sg.mean(1),
        sigma_min=sg.min(1), sigma_max=sg.max(1),
        q_sd=q.std(0), logit_spread=logit_spread, ess=ess, w_max=w_max,
        **{f"fit_{k}": v for k, v in mf.items() if v.ndim == 1},
        **{f"raw_{k}": v for k, v in mr.items() if v.ndim == 1},
        ubar_raw_vec=mr["ubar"], ubar_fit_vec=mf["ubar"],
        **{f"cloud_{k}": v for k, v in cl.items()},
    )
    print("%-46s d=%2d eta=%.6g (%s) clip=%.3f%% | R2_exact=%.4g medR=%.4g "
          "med cos(v,c)=%.4f med resid=%.4g med spread=%.4g ESS=%.3g  [%.0fs]"
          % (tag, d, eta, eta_src, 100 * clip_rate,
             np.sqrt(np.sum(mf["n_ubar"] ** 2) / np.sum(mf["n_c"] ** 2)),
             np.median(mf["R_exact"]), np.median(mf["cos_v_c"]),
             np.median(mf["residual_linear"]), np.median(logit_spread),
             np.median(ess), time.time() - t0), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
