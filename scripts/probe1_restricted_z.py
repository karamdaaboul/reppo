"""Probe 1 -- restricted-z oracle (prospective_padding_error_field_analysis.md, Sec. 2).

Holds (s, x) fixed on visited states and varies only the padded block z, under two
reference laws (checkpoint law and the common standardized N(0, I_k)). Measures the
centered padded error field, both estimators' padded-block error energies, and the
finite-M moments of the canonical ZO estimator against eqs (13)-(14) of
`wasted_step_fraction_proposition.md`.

Conventions are fixed by Amendment A (audited commit 3b96deb):

* pre-tanh coordinates y = (x, z); F(u_z) = Q_phi(s, tanh([x, mu_z + sigma_z u_z]))
  composed with tanh, and for arm B (weighted_mle) with the +-(1-1e-4) clip that
  `src/jaxrl/reppo.py:669` applies to the E-step actions BEFORE the critic call.
* Q is the LIVE critic's HL-Gauss categorical mean (`critic.critic`); there is no
  target critic.
* Gaussian direction law, diagonal state-dependent Sigma, M = 32.

Because the probe differentiates w.r.t. the whitened u_z, autodiff returns
H_z = Sigma_z^{1/2} grad_z Q directly -- the quantity eq. (8) needs.

L is NOT computed: a z-only step has L = 1 by construction (plan, Probe 1 row).

All Q-moments are accumulated around a per-state reference mean Q_ref estimated from a
small independent pre-pass. The padded error field here is O(1e-2) on a critic whose
output is O(50), so the textbook one-pass form E[Q^2] - E[Q]^2 cancels away roughly
seven significant digits and returns negative variances in float32; centering first
keeps every moment well conditioned.

Two independent sample streams per state, so that the eq-(13)/(14) comparison is not
self-fulfilling:
  * ORACLE stream (N_ORACLE draws) -> population moments a, nu, D, h, C.
  * BATCH stream  (R x M draws)    -> repeated-batch realizations of a_hat_M, R_PW.

Usage:  probe1_restricted_z.py <ckpt_dir> <law:ckpt|std> <out.npz>
"""

from __future__ import annotations

import hashlib
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from scripts.critic_fidelity.common import ACTION_CLIP, Harness  # noqa: E402

# --- probe constants ---------------------------------------------------------
B_ENVS, BURN, N_CHUNKS, CHUNK_GAP = 256, 200, 8, 25  # -> 2048 visited states
M = 32  # estep_num_samples (Amendment A recordables)
R_BATCHES = 2048  # ">= 2000 repeated batches per state"
N_ORACLE = 32768  # independent stream for the population moments
ORACLE_BLOCK = 2048  # scan block for the oracle stream
N_REF = 2048  # pre-pass draws used only to centre Q before accumulating moments
BATCH_CHUNK = 64  # batches pulled back to host per device call
STATE_CHUNK = 64  # states processed per device call
ARM_B_CLIP = 1.0 - 1e-4  # src/jaxrl/reppo.py:669
KEEP_FULL = 64  # states whose per-batch realizations are kept verbatim

# Fresh probe PRNG domain. Never a training key: training keys are folded off
# `cfg.seed` in {0..4} inside `make_train`, and nothing below ever touches those.
PROBE_ROOT = 20260831  # probe-only root, matching the plan's analysis-seed convention


def probe_key(tag: str, ckpt: str) -> jax.Array:
    """Deterministic fresh key per (purpose, checkpoint), disjoint from training.

    Uses blake2b rather than `hash()`, which is salted per interpreter run and would
    make the probe irreproducible.
    """
    dig = hashlib.blake2b(f"{tag}|{ckpt}".encode(), digest_size=4).digest()
    return jax.random.fold_in(
        jax.random.PRNGKey(PROBE_ROOT), int.from_bytes(dig, "big") % (2**31)
    )


def collect_states(h: Harness, key):
    """2048 visited states under pi_old, matching sigma_probe.py's sampler.

    Returns raw obs (N,obs), pre-tanh mu/sigma (N,d), env lane id, temporal chunk id.
    """
    obs_l, mu_l, sg_l, env_l, chunk_l = [], [], [], [], []
    key, rk = jax.random.split(key)
    obs, _, st = h.reset(rk)
    for step in range(BURN + N_CHUNKS * CHUNK_GAP):
        k1, k2, key = jax.random.split(key, 3)
        dist = h.pi(obs)
        if step >= BURN and (step - BURN) % CHUNK_GAP == 0:
            c = (step - BURN) // CHUNK_GAP
            obs_l.append(np.asarray(obs))
            mu_l.append(np.asarray(dist.distribution.loc))
            sg_l.append(np.asarray(dist.distribution.scale))
            env_l.append(np.arange(B_ENVS))
            chunk_l.append(np.full(B_ENVS, c))
        a = jnp.clip(dist.sample(seed=k1), -ACTION_CLIP, ACTION_CLIP)
        obs, _, st, _, _, _ = h.env.step(jax.random.split(k2, B_ENVS), st, a)
    return (
        np.concatenate(obs_l),
        np.concatenate(mu_l),
        np.concatenate(sg_l),
        np.concatenate(env_l),
        np.concatenate(chunk_l),
    )


def build_fns(h: Harness, r: int, k: int, clip: bool):
    """F(u_z) and its whitened gradient, batched over (state, sample)."""

    def q_of_u(u, cobs, x, mu_z, sg_z):
        z = mu_z + sg_z * u
        a = jnp.tanh(jnp.concatenate([x, z], axis=-1))
        if clip:
            a = jnp.clip(a, -ARM_B_CLIP, ARM_B_CLIP)
        return h.ck.critic.critic(cobs, a)

    def q_and_h(u, cobs, x, mu_z, sg_z):
        """u:(n,k) flat rows. Returns Q:(n,) and H_z = Sigma_z^{1/2} grad_z Q :(n,k)."""
        f = lambda uu: q_of_u(uu, cobs, x, mu_z, sg_z)
        q, pull = jax.vjp(f, u)
        (g,) = pull(jnp.ones_like(q))
        return q, g

    def sat_of_u(u, mu_z, sg_z):
        t = jnp.abs(jnp.tanh(mu_z + sg_z * u))
        return (t >= ARM_B_CLIP).astype(jnp.float32), (t > 0.99).astype(jnp.float32)

    # ---- reference mean, so every later moment is accumulated around zero -------
    def ref_block(bkey, cobs, x, mu_z, sg_z, S):
        u = jax.random.normal(bkey, (S, N_REF, k))
        rep = lambda a: jnp.repeat(a, N_REF, axis=0)
        q = q_of_u(u.reshape(-1, k), rep(cobs), rep(x), rep(mu_z), rep(sg_z))
        return q.reshape(S, N_REF).mean(1)

    # ---- oracle stream: moment sums of r = Q - Q_ref over N_ORACLE draws --------
    def oracle_block(carry, bkey, cobs, x, mu_z, sg_z, qref, S):
        u = jax.random.normal(bkey, (S, ORACLE_BLOCK, k))
        uf = u.reshape(-1, k)
        rep = lambda a: jnp.repeat(a, ORACLE_BLOCK, axis=0)
        q, H = q_and_h(uf, rep(cobs), rep(x), rep(mu_z), rep(sg_z))
        q = q.reshape(S, ORACLE_BLOCK) - qref[:, None]
        H = H.reshape(S, ORACLE_BLOCK, k)
        e = jnp.einsum
        add = (
            q.sum(1),  # S_Q
            (q**2).sum(1),  # S_Q2
            e("sn,snj->sj", q, u),  # S_Qu
            u.sum(1),  # S_u
            e("sn,snj,snl->sjl", q**2, u, u),  # S_Q2uu
            e("sn,snj,snl->sjl", q, u, u),  # S_Quu
            e("snj,snl->sjl", u, u),  # S_uu
            H.sum(1),  # S_H
            e("snj,snl->sjl", H, H),  # S_HH
            (H**2).sum((1, 2)),  # S_H2
        )
        return tuple(c + a for c, a in zip(carry, add)), None

    # ---- batch stream: per-batch estimator realizations ------------------------
    def batch_block(bkey, cobs, x, mu_z, sg_z, qref, S):
        """Returns per-batch a_hat (S,C,k), R_PW (S,C,k), sat fracs (S,C)."""
        C = BATCH_CHUNK
        u = jax.random.normal(bkey, (S, C, M, k))
        uf = u.reshape(-1, k)
        rep = lambda a: jnp.repeat(a, C * M, axis=0)
        q, H = q_and_h(uf, rep(cobs), rep(x), rep(mu_z), rep(sg_z))
        q = q.reshape(S, C, M) - qref[:, None, None]
        H = H.reshape(S, C, M, k)
        qc = q - q.mean(-1, keepdims=True)  # sample-mean baseline, eq. (11)
        a_hat = jnp.einsum("scm,scmj->scj", qc, u) / M
        r_pw = H.mean(2)
        sat_clip, sat99 = sat_of_u(u, mu_z[:, None, None, :], sg_z[:, None, None, :])
        return a_hat, r_pw, sat_clip.mean((2, 3)), sat99.mean((2, 3))

    return (
        jax.jit(ref_block, static_argnums=(5,)),
        jax.jit(oracle_block, static_argnums=(7,)),
        jax.jit(batch_block, static_argnums=(6,)),
    )


def run(ckpt: str, law: str, out: str):
    t0 = time.time()
    h = Harness(ckpt, B_ENVS)
    meta = h.meta
    d, k = int(meta["action_dim"]), int(meta["action_pad"])
    r = d - k
    arm = meta["actor_update_mode"]
    clip = arm == "weighted_mle"  # Amendment A answer 1

    obs, mu, sg, env_id, chunk_id = collect_states(h, probe_key("states", ckpt))
    N = obs.shape[0]
    cobs = np.asarray(h.nc(jnp.asarray(obs)))

    # x fixed per state: one fresh pre-tanh draw from the checkpoint law's real block
    kx = probe_key("xfix", ckpt)
    x = mu[:, :r] + sg[:, :r] * np.asarray(jax.random.normal(kx, (N, r)))

    if law == "ckpt":
        mu_z, sg_z = mu[:, r:].copy(), sg[:, r:].copy()
    elif law == "std":
        mu_z, sg_z = np.zeros((N, k), np.float32), np.ones((N, k), np.float32)
    else:
        raise ValueError(law)

    ref_block, oracle_block, batch_block = build_fns(h, r, k, clip)

    # accumulators (float64, on host). The oracle stream is accumulated in two
    # independent halves as well as in full, so that the eq-(14) comparison has an
    # oracle-side MC noise floor and not only a batch-side one.
    z64 = lambda *s: np.zeros(s, np.float64)
    ORC = lambda: [z64(N), z64(N), z64(N, k), z64(N, k), z64(N, k, k),
                   z64(N, k, k), z64(N, k, k), z64(N, k), z64(N, k, k), z64(N)]
    orc_half = [ORC(), ORC()]
    q_ref_all = z64(N)

    n_or_blocks = N_ORACLE // ORACLE_BLOCK
    n_b_blocks = R_BATCHES // BATCH_CHUNK
    ok = probe_key("oracle:" + law, ckpt)
    bk = probe_key("batch:" + law, ckpt)

    # per-state batch summaries
    mean_a, mean_p = z64(N, k), z64(N, k)
    cov_a, cov_p = z64(N, k, k), z64(N, k, k)
    m2_a, m2_p = z64(N, k), z64(N, k)  # per-component E[.^2]
    half_a = [z64(N, k), z64(N, k)]
    half_cov_a = [z64(N, k, k), z64(N, k, k)]
    sat_m, sat99_m = z64(N), z64(N)
    sat_cross_a, sat_cross_p, sat_var = z64(N), z64(N), z64(N)
    ea_energy, ep_energy = z64(N), z64(N)  # per-batch ||.||^2 means (for covariate)
    ea_e2, ep_e2 = z64(N), z64(N)
    zero_a, zero_p = z64(N), z64(N)
    keep = {"a_hat": [], "r_pw": [], "sat": []}

    for s0 in range(0, N, STATE_CHUNK):
        s1 = min(s0 + STATE_CHUNK, N)
        S = s1 - s0
        args = (
            jnp.asarray(cobs[s0:s1]),
            jnp.asarray(x[s0:s1]),
            jnp.asarray(mu_z[s0:s1]),
            jnp.asarray(sg_z[s0:s1]),
        )
        qref = ref_block(
            jax.random.fold_in(probe_key("qref:" + law, ckpt), s0), *args, S
        )
        q_ref_all[s0:s1] = np.asarray(qref, np.float64)

        # --- oracle stream ---
        carry = (
            jnp.zeros((S,)), jnp.zeros((S,)), jnp.zeros((S, k)), jnp.zeros((S, k)),
            jnp.zeros((S, k, k)), jnp.zeros((S, k, k)), jnp.zeros((S, k, k)),
            jnp.zeros((S, k)), jnp.zeros((S, k, k)), jnp.zeros((S,)),
        )
        for hf in (0, 1):
            cc = carry
            for b in range(hf * (n_or_blocks // 2), (hf + 1) * (n_or_blocks // 2)):
                cc, _ = oracle_block(
                    cc, jax.random.fold_in(jax.random.fold_in(ok, s0), b), *args,
                    qref, S
                )
            for i, v in enumerate(cc):
                orc_half[hf][i][s0:s1] += np.asarray(v, np.float64)

        # --- batch stream ---
        for b in range(n_b_blocks):
            ah, rp, sc, s99 = batch_block(
                jax.random.fold_in(jax.random.fold_in(bk, s0), b), *args, qref, S
            )
            ah = np.asarray(ah, np.float64)   # (S,C,k)
            rp = np.asarray(rp, np.float64)
            sc = np.asarray(sc, np.float64)   # (S,C)
            s99 = np.asarray(s99, np.float64)
            mean_a[s0:s1] += ah.sum(1); mean_p[s0:s1] += rp.sum(1)
            cov_a[s0:s1] += np.einsum("scj,scl->sjl", ah, ah)
            cov_p[s0:s1] += np.einsum("scj,scl->sjl", rp, rp)
            m2_a[s0:s1] += (ah**2).sum(1); m2_p[s0:s1] += (rp**2).sum(1)
            hi = 0 if b < n_b_blocks // 2 else 1
            half_a[hi][s0:s1] += ah.sum(1)
            half_cov_a[hi][s0:s1] += np.einsum("scj,scl->sjl", ah, ah)
            ea = (ah**2).sum(-1); ep = (rp**2).sum(-1)
            ea_energy[s0:s1] += ea.sum(1); ep_energy[s0:s1] += ep.sum(1)
            ea_e2[s0:s1] += (ea**2).sum(1); ep_e2[s0:s1] += (ep**2).sum(1)
            sat_m[s0:s1] += sc.sum(1); sat99_m[s0:s1] += s99.sum(1)
            sat_var[s0:s1] += (sc**2).sum(1)
            sat_cross_a[s0:s1] += (sc * ea).sum(1)
            sat_cross_p[s0:s1] += (sc * ep).sum(1)
            zero_a[s0:s1] += (ea == 0).sum(1); zero_p[s0:s1] += (ep == 0).sum(1)
            if s0 == 0 and len(keep["a_hat"]) < n_b_blocks:
                keep["a_hat"].append(ah[:KEEP_FULL].astype(np.float32))
                keep["r_pw"].append(rp[:KEEP_FULL].astype(np.float32))
                keep["sat"].append(sc[:KEEP_FULL].astype(np.float32))
        print(f"  states {s1}/{N}  {time.time()-t0:.0f}s", flush=True)

    Rr = float(R_BATCHES)

    def moments(acc, n):
        # sums are of r = Q - Q_ref, so q = Q - E[Q] = r - E[r] and every moment
        # below is a difference of like-magnitude terms.
        S_r, S_r2, S_ru, S_u, S_r2uu, S_ruu, S_uu, S_H, S_HH, S_H2 = acc
        mr = S_r / n
        mQ = q_ref_all + mr                       # absolute E[Q], for reporting only
        nu = S_r2 / n - mr**2                     # V_e, eq (8)/(18)
        a = S_ru / n - mr[:, None] * (S_u / n)    # a_z, eq (18)
        D = (S_r2uu - 2 * mr[:, None, None] * S_ruu
             + (mr**2)[:, None, None] * S_uu) / n
        hvec = S_H / n
        G2 = S_H2 / n                             # G_z^2, eq (8)
        C = S_HH / n - np.einsum("sj,sl->sjl", hvec, hvec)
        return mQ, nu, a, D, hvec, G2, C

    full = [x + y for x, y in zip(orc_half[0], orc_half[1])]
    mQ, nu, a, D, hvec, G2, C = moments(full, float(N_ORACLE))
    oh = [moments(orc_half[i], float(N_ORACLE) / 2) for i in (0, 1)]

    mean_a /= Rr; mean_p /= Rr
    cov_a = cov_a / Rr - np.einsum("sj,sl->sjl", mean_a, mean_a)
    cov_p = cov_p / Rr - np.einsum("sj,sl->sjl", mean_p, mean_p)
    m2_a /= Rr; m2_p /= Rr
    for i in (0, 1):
        n_h = Rr / 2
        half_a[i] /= n_h
        half_cov_a[i] = half_cov_a[i] / n_h - np.einsum(
            "sj,sl->sjl", half_a[i], half_a[i]
        )
    sat_m /= Rr; sat99_m /= Rr; sat_var = sat_var / Rr - sat_m**2
    ea_energy /= Rr; ep_energy /= Rr
    ea_e2 = ea_e2 / Rr - ea_energy**2
    ep_e2 = ep_e2 / Rr - ep_energy**2
    cov_sat_a = sat_cross_a / Rr - sat_m * ea_energy
    cov_sat_p = sat_cross_p / Rr - sat_m * ep_energy
    den_a = np.sqrt(np.maximum(sat_var, 0) * np.maximum(ea_e2, 0))
    den_p = np.sqrt(np.maximum(sat_var, 0) * np.maximum(ep_e2, 0))

    np.savez_compressed(
        out,
        ckpt=ckpt, law=law, arm=arm, seed=meta["seed"], d=d, k=k, r=r, M=M,
        R=R_BATCHES, n_oracle=N_ORACLE, clip=clip,
        env_id=env_id, chunk_id=chunk_id,
        mQ=mQ, nu=nu, a=a, D=D, h=hvec, G2=G2, C=C,
        nu_h0=oh[0][1], nu_h1=oh[1][1], a_h0=oh[0][2], a_h1=oh[1][2],
        D_h0=oh[0][3], D_h1=oh[1][3], G2_h0=oh[0][5], G2_h1=oh[1][5],
        mean_a=mean_a, cov_a=cov_a, m2_a=m2_a,
        mean_p=mean_p, cov_p=cov_p, m2_p=m2_p,
        half_a0=half_a[0], half_a1=half_a[1],
        half_cov_a0=half_cov_a[0], half_cov_a1=half_cov_a[1],
        sat_mean=sat_m, sat99_mean=sat99_m, sat_var=sat_var,
        corr_sat_zo=np.where(den_a > 0, cov_sat_a / np.maximum(den_a, 1e-300), np.nan),
        corr_sat_pw=np.where(den_p > 0, cov_sat_p / np.maximum(den_p, 1e-300), np.nan),
        e_zo=ea_energy, e_pw=ep_energy,
        zero_frac_zo=zero_a / Rr, zero_frac_pw=zero_p / Rr,
        sigma_z=sg_z, sigma_x=sg[:, :r], mu_z=mu_z,
        keep_a_hat=np.concatenate(keep["a_hat"], 1) if keep["a_hat"] else np.zeros(0),
        keep_r_pw=np.concatenate(keep["r_pw"], 1) if keep["r_pw"] else np.zeros(0),
        keep_sat=np.concatenate(keep["sat"], 1) if keep["sat"] else np.zeros(0),
    )
    print(f"wrote {out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])
