#!/usr/bin/env python
"""Offline same-critic estimator probe.

WHAT THIS MEASURES, AND WHAT IT DOES NOT
========================================
Measures: the self-consistency (across independent action draws) of two
operator directions, computed FROM THE SAME FROZEN CRITIC, at the same visited
states, from the same samples. It isolates estimator variance from critic
quality, which the 64 confirmatory runs cannot do because each arm trained its
own critic.

Does NOT measure:
  * omega. The preregistered estimator-error quantity was never computed
    (reppo.py:901 gates the whole block on log_estimator_diag, which was false
    for all 64 runs). Nothing here recovers it.
  * any connection to return. This probe has no outcome variable. It cannot
    show that estimator noise caused the g1 gap, only whether the two operators
    differ in self-consistency at a fixed critic.
  * bias. Self-disagreement across draws is a variance-like quantity. A biased
    but stable estimator looks good here.

Do not report a result from this script as evidence that one operator is better.
It is evidence about estimator variance at a fixed critic, and nothing more.

ESTIMATORS (both in the whitened pre-tanh metric, matching reppo.py:912
`_h = _sg * grad`, so the two are directly comparable)
  z_i ~ N(0, I),  u_i = mu + sigma * z_i,  a_i = tanh(u_i),  Q_i = Q_phi(s, a_i)

  g_PW = mean_i [ sigma * d/dy Q_phi(s, tanh(y)) |_{y = u_i} ]
  g_ZO = (M/(M-1)) * mean_i [ (Q_i - Qbar) * z_i ]

  The (M/(M-1)) factor de-attenuates the known shrinkage of the centred
  zeroth-order estimator, as reppo.py:936 does.

REFERENCE (not ground truth, a fixed deterministic anchor)
  h_ref = sigma * d/dy Q_phi(s, tanh(y)) |_{y = mu}    (reppo.py:906-912)

SELF-DISAGREEMENT, per state, over R independent replicate draws
  rel_disp = || std_r(g) ||_2 / || mean_r(g) ||_2      (scale-free)
  pair_cos = mean over replicate pairs of cosine(g^(r), g^(r'))

Actions are NOT clamped to +/-(1-1e-4) here: both estimators must see identical
inputs, and the clamp is an arm-asymmetric artifact of the training loop, not a
property of the operators. Saturation at these checkpoints is <= 0.4% of
coordinates, so the choice is immaterial; --clamp restores it.
"""
import argparse, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import jax, jax.numpy as jnp, numpy as np
from scripts.critic_fidelity.common import ACTION_CLIP, Harness

BATCH, BURN, GAP, CHUNKS = 256, 200, 25, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--R", type=int, default=8, help="independent replicate draws")
    ap.add_argument("--nstates", type=int, default=512)
    ap.add_argument("--clamp", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    name = os.path.basename(a.ckpt.rstrip("/"))

    h = Harness(a.ckpt, BATCH)
    key = jax.random.PRNGKey(0)
    key, rk = jax.random.split(key)
    obs, _, st = h.reset(rk)
    OBS = []
    for step in range(BURN + CHUNKS * GAP):
        k1, k2, key = jax.random.split(key, 3)
        act = jnp.clip(h.pi(obs).sample(seed=k1), -ACTION_CLIP, ACTION_CLIP)
        if step >= BURN and (step - BURN) % GAP == 0:
            OBS.append(obs)
        obs, _, st, _, _, _ = h.env.step(jax.random.split(k2, BATCH), st, act)
    S = jnp.concatenate(OBS, 0)[: a.nstates]                      # raw obs (N, obs_dim)
    N = S.shape[0]

    mu, sg = h.ck.actor.gaussian(h.na(S))                          # pre-tanh Gaussian
    mu = jax.lax.stop_gradient(mu); sg = jax.lax.stop_gradient(sg)

    def q_of_y(y, obs_rep):                                        # y: (K, d)
        aa = jnp.tanh(y)
        if a.clamp:
            aa = jnp.clip(aa, -1 + 1e-4, 1 - 1e-4)
        return h.q(obs_rep, aa).reshape(-1).sum()

    grad_q = jax.jit(jax.grad(q_of_y))

    # deterministic anchor at y = mu  (reppo.py:906-912)
    h_ref = np.asarray(sg * grad_q(mu, S))

    M = a.M
    obs_rep = jnp.reshape(jnp.broadcast_to(S, (M, *S.shape)), (M * N, -1))
    pw, zo = [], []
    for r in range(a.R):
        z = jax.random.normal(jax.random.PRNGKey(1000 + r), (M, N, mu.shape[-1]))
        u = mu[None] + sg[None] * z
        g = grad_q(u.reshape(M * N, -1), obs_rep).reshape(M, N, -1)
        pw.append(np.asarray((sg[None] * g).mean(0)))
        aa = jnp.tanh(u)
        if a.clamp:
            aa = jnp.clip(aa, -1 + 1e-4, 1 - 1e-4)
        Q = h.q(obs_rep, aa.reshape(M * N, -1)).reshape(M, N)
        Qc = Q - Q.mean(0, keepdims=True)
        zo.append(np.asarray((M / (M - 1.0)) * (Qc[..., None] * z).mean(0)))
    PW, ZO = np.stack(pw), np.stack(zo)                            # (R, N, d)

    def stats(G):
        mean_r = G.mean(0)
        num = np.linalg.norm(G.std(0), axis=-1)
        den = np.maximum(np.linalg.norm(mean_r, axis=-1), 1e-12)
        rel = num / den
        cs = []
        for i, j in itertools.combinations(range(G.shape[0]), 2):
            n1 = np.linalg.norm(G[i], axis=-1); n2 = np.linalg.norm(G[j], axis=-1)
            cs.append((G[i] * G[j]).sum(-1) / np.maximum(n1 * n2, 1e-12))
        cs = np.stack(cs)
        ref_n = np.maximum(np.linalg.norm(h_ref, axis=-1), 1e-12)
        cref = (mean_r * h_ref).sum(-1) / np.maximum(np.linalg.norm(mean_r, axis=-1) * ref_n, 1e-12)
        return dict(
            rel_disp_median=float(np.median(rel)), rel_disp_p90=float(np.percentile(rel, 90)),
            pair_cos_median=float(np.median(cs)), pair_cos_mean=float(cs.mean()),
            cos_to_href_median=float(np.median(cref)),
            norm_median=float(np.median(np.linalg.norm(mean_r, axis=-1))),
        )

    res = dict(ckpt=name, arm="B" if "weighted_mle" in name else "A",
               M=M, R=a.R, n_states=int(N), action_dim=int(h.action_dim),
               seed=int(h.meta["seed"]), step=int(h.meta["time_steps"]),
               clamped=bool(a.clamp),
               href_norm_median=float(np.median(np.linalg.norm(h_ref, axis=-1))),
               pathwise=stats(PW), zeroth_order=stats(ZO))
    with open(f"{a.outdir}/{name}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
