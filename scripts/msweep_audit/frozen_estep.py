"""STEPS 2 + 5: frozen nested-prefix E-step diagnostic. No optimizer step, no training.

One common action cloud of N_MAX draws per state (common random numbers); nested
prefixes 32/128/512/2048 reuse its leading entries, so every M sees the same actions.
For each prefix the MPO eta dual is solved to eps_E on that prefix, exactly as
`eta_dual_loss` defines it, and the resulting E-step geometry is measured.

Conventions matched to src/jaxrl/reppo.py:
  * actions drawn from the actor's Tanh(Normal), then clipped to +-(1-1e-4)  (reppo.py:727)
  * Q from the TARGET critic; polyak=1.0 so target == live critic at export
  * softmax over the sample axis 0; ESS = 1/sum w^2
"""
import os, sys, time, json
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/home/human/workspaces/reppo_original")
import jax, jax.numpy as jnp, numpy as np
from scripts.critic_fidelity.common import Harness, ACTION_CLIP

CKPT   = sys.argv[1]
OUT    = sys.argv[2]
B      = int(sys.argv[3]) if len(sys.argv) > 3 else 128
BURN   = int(sys.argv[4]) if len(sys.argv) > 4 else 150
NCH, GAP = 8, 20
N_MAX  = 2048
PREFIX = [32, 128, 512, 2048]
EPS_E  = 0.5
ESTEP_CLIP = 1.0 - 1e-4
SCHUNK = 64            # sample-axis chunk for the critic
ROOT   = 20260902

t0 = time.time()
h = Harness(CKPT, B)
key = jax.random.PRNGKey(ROOT)

# ---- state bank: obs only (env states not needed for a frozen E-step) --------
def rollout(key, n):   # n is a Python int; lax.scan traces once
    k, rk = jax.random.split(key)
    obs, _, st = h.reset(rk)
    def body(carry, kk):
        st, obs = carry
        ak, sk = jax.random.split(kk)
        a = jnp.clip(h.pi(obs).sample(seed=ak), -ACTION_CLIP, ACTION_CLIP)
        nobs, _, nst, _, _, _ = h.env.step(h._step_keys(sk), st, a)
        return (nst, nobs), obs
    _, out = jax.lax.scan(body, (st, obs), jax.random.split(k, n))
    return out

key, rk = jax.random.split(key)
n_steps = BURN + NCH * GAP
obs_all = rollout(rk, n_steps)                      # (n_steps, B, obs_dim)
idx = [BURN + i * GAP for i in range(NCH)]
S = np.asarray(obs_all[np.array(idx)]).reshape(-1, h.obs_dim)  # (NCH*B, obs_dim)
print(f"[{time.time()-t0:6.1f}s] state bank {S.shape}", flush=True)

# ---- frozen policy + one common action cloud --------------------------------
S_j = jnp.asarray(S)
dist = h.pi(S_j)
mu, sg = h.ck.actor.gaussian(h.na(S_j))              # pre-squash Gaussian
mu, sg = np.asarray(mu, np.float64), np.asarray(sg, np.float64)
eta_ckpt = float(np.asarray(h.ck.actor.eta()).squeeze())
print(f"[{time.time()-t0:6.1f}s] eta_ckpt={eta_ckpt:.6g}  sigma mean={sg.mean():.4f} "
      f"p95={np.percentile(sg,95):.4f} max={sg.max():.4f}", flush=True)

key, ck_ = jax.random.split(key)
u = np.asarray(jax.random.normal(ck_, (N_MAX, S.shape[0], h.action_dim)), np.float64)
y = mu[None] + sg[None] * u
a_cloud = np.clip(np.tanh(y), -ESTEP_CLIP, ESTEP_CLIP)

# log pi_old(a_i) for the objective identity check (joint, summed over dims)
lp = np.asarray(dist.log_prob(jnp.asarray(a_cloud, jnp.float32)).sum(-1), np.float64)

# ---- Q on the whole cloud, chunked over the sample axis ----------------------
@jax.jit
def qfun(a_chunk, s):
    cobs = jnp.broadcast_to(s, (a_chunk.shape[0], *s.shape))
    return h.q(cobs, a_chunk)

qs = []
for i in range(0, N_MAX, SCHUNK):
    qs.append(np.asarray(qfun(jnp.asarray(a_cloud[i:i+SCHUNK], jnp.float32), S_j), np.float64))
    if (i // SCHUNK) % 8 == 0:
        print(f"[{time.time()-t0:6.1f}s]   Q {i+SCHUNK}/{N_MAX}", flush=True)
q = np.concatenate(qs, 0)                            # (N_MAX, nstates)
print(f"[{time.time()-t0:6.1f}s] Q cloud {q.shape}", flush=True)

# ---- MPO eta dual, solved on each prefix (float64, host) --------------------
def dual(eta, qm):
    qmax = qm.max(0)
    lse = np.log(np.mean(np.exp((qm - qmax) / eta), 0))
    return eta * EPS_E + np.mean(eta * lse + qmax)

def solve_eta(qm, lo=1e-4, hi=10.0):
    """Golden-section on the convex dual, over the network's own [eta_min, eta_max]."""
    gr = (np.sqrt(5) - 1) / 2
    a, b = np.log(lo), np.log(hi)
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = dual(np.exp(c), qm), dual(np.exp(d), qm)
    for _ in range(200):
        if fc < fd: b, d, fd = d, c, fc; c = b - gr * (b - a); fc = dual(np.exp(c), qm)
        else:       a, c, fc = c, d, fd; d = a + gr * (b - a); fd = dual(np.exp(d), qm)
        if b - a < 1e-10: break
    return float(np.exp((a + b) / 2))

def geometry(qm, um, eta, lpm):
    M = qm.shape[0]
    z = qm / eta
    z = z - z.max(0, keepdims=True)
    w = np.exp(z); w /= w.sum(0, keepdims=True)
    ess = 1.0 / np.sum(w**2, 0)
    Hw = -np.sum(np.where(w > 0, w * np.log(np.maximum(w, 1e-300)), 0.0), 0)
    kl_wu = np.log(M) - Hw                       # KL(w || uniform_M)
    d_vec = np.einsum('ij,ijk->jk', w, um)       # weighted mean whitened displacement
    resid = um - d_vec[None]
    sec = np.einsum('ij,ijk->jk', w, resid**2)   # weighted per-dim second moment
    return dict(
        M=M, eta=eta,
        ess=ess.mean(), ess_med=np.median(ess), ess_frac=(ess / M).mean(),
        ess_frac_med=float(np.median(ess / M)),
        w_max=w.max(0).mean(), w_max_med=float(np.median(w.max(0))),
        kl_wu=kl_wu.mean(), kl_wu_med=float(np.median(kl_wu)), ent_w=Hw.mean(),
        logit_spread=z.std(0).mean(), logit_spread_med=float(np.median(z.std(0))),
        q_max=qm.max(0).mean(), q_mean=qm.mean(0).mean(), q_std=qm.std(0).mean(),
        q_p95=np.percentile(qm, 95, axis=0).mean(),
        q_p99=np.percentile(qm, 99, axis=0).mean(),
        q_max_minus_med=(qm.max(0) - np.median(qm, 0)).mean(),
        q_weighted=np.sum(w * qm, 0).mean(),
        q_gain=(np.sum(w * qm, 0) - qm.mean(0)).mean(),      # E-step Q improvement
        disp_norm=np.linalg.norm(d_vec, axis=-1).mean(),     # ||d|| in whitened units
        disp_norm_med=float(np.median(np.linalg.norm(d_vec, axis=-1))),
        sec_trace_per_dim=sec.mean(),                        # 1.0 == pi_old width
        sum_w_err=float(np.abs(w.sum(0) - 1.0).max()),
        objective=float(np.mean(-np.sum(w * lpm, 0))),       # -sum_i w_i logp(a_i)
        uniform_objective=float(np.mean(-lpm.mean(0))),
        eta_at_clip=bool(eta <= 1.0001e-4 or eta >= 9.999),
    )

rows = []
for M in PREFIX:
    qm, um, lpm = q[:M], u[:M], lp[:M]
    e_star = solve_eta(qm)
    r = geometry(qm, um, e_star, lpm); r["dual_mode"] = "solved_on_prefix"
    rows.append(r)
    r2 = geometry(qm, um, eta_ckpt, lpm); r2["dual_mode"] = "eta_ckpt_fixed"
    rows.append(r2)
    print(f"[{time.time()-t0:6.1f}s] M={M:5d} eta*={e_star:.6g} "
          f"ESS/M={r['ess_frac']:.3f} KL(w||u)={r['kl_wu']:.3f} "
          f"qmax={r['q_max']:.2f} qw={r['q_weighted']:.2f} |d|={r['disp_norm']:.4f}",
          flush=True)

out = dict(ckpt=CKPT, meta=dict(final=h.meta.get("final_eval_return"),
           M_train=h.meta.get("estep_num_samples"), seed=h.meta.get("seed"),
           eta_ckpt=eta_ckpt, alpha=h.alpha),
           n_states=int(S.shape[0]), n_max=N_MAX, eps_e=EPS_E,
           sigma=dict(mean=float(sg.mean()), med=float(np.median(sg)),
                      p95=float(np.percentile(sg,95)), max=float(sg.max())),
           rows=rows)
json.dump(out, open(OUT, "w"), indent=1, default=float)
print(f"[{time.time()-t0:6.1f}s] wrote {OUT}", flush=True)
