"""Probe 2 -- full-estimator oracle (prospective_padding_error_field_analysis.md, Sec. 2).

Probe 1 held (s, x) fixed and varied only the padded block, so its step was padded by
construction and L was identically 1. Probe 2 varies the WHOLE pre-tanh action
y = mu + sigma .* u over d = r + k coordinates, which is the estimator the algorithm
actually forms, and therefore measures

    L(g_hat) = ||R_z||^2 / (||R_x||^2 + ||R_z||^2),   R = Sigma^{1/2} g_hat,

eq. (1) of `wasted_step_fraction_proposition.md`, together with per-block error
energies over the real and padded blocks.

Per the committed probe table this row is DESCRIPTIVE: the difference from Probe 1
includes real signal and constant-in-z bias leakage, not the padded error field alone.
No prediction is attached and none is made here.

Numerical conventions are Probe 1's, and are mandatory (Amendment A.1 item 2):

* every Q-moment is accumulated around a per-state reference mean from an independent
  pre-pass, then reduced in float64 on the host;
* a V_e <= 0 tripwire column is reported and must read zero;
* two independent sample streams per state, so the eq-(13)/(14) gate is not
  self-fulfilling: an ORACLE stream for the population moments, and a BATCH stream for
  repeated finite-M realizations.

Conventions fixed by Amendment A (audited commit 3b96deb) and A.1:

* F(u) = Q_phi(s, tanh(mu + sigma .* u)), composed with tanh, and for arm B
  (weighted_mle) with the +-(1-1e-4) clip that `src/jaxrl/reppo.py:669` applies to the
  E-step actions BEFORE the critic call;
* Q is the LIVE critic's HL-Gauss categorical mean; there is no target critic;
* Gaussian direction law, diagonal state-dependent Sigma, M = 32;
* sigma uses the EFFECTIVE min_std = 0.1 (Amendment A.1 item 1), which is what the
  checkpoints trained with.

Differentiating w.r.t. the whitened u returns H = Sigma^{1/2} grad_y Q directly.

Usage:  probe2_full_estimator.py <ckpt_dir> <law:ckpt|std> <out.npz>
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
from src.jaxrl.estimators import (  # noqa: E402
    centred_zo,
    pathwise_mean,
    whitened_pathwise,
)

# --- probe constants (Probe 1's, so the two are directly comparable) ---------
B_ENVS, BURN, N_CHUNKS, CHUNK_GAP = 256, 200, 8, 25  # -> 2048 visited states
M = 32  # estep_num_samples
R_BATCHES = 2048  # ">= 2000 repeated batches per state"
N_ORACLE = 32768  # independent stream for the population moments
ORACLE_BLOCK = 1024  # smaller than Probe 1's: d = 22 here, not k = 16
N_REF = 2048  # pre-pass draws used only to centre Q
BATCH_CHUNK = 32
STATE_CHUNK = 32
ARM_B_CLIP = 1.0 - 1e-4  # src/jaxrl/reppo.py:669

PROBE_ROOT = 20260901  # Probe 2 root; distinct from Probe 1's 20260831


def probe_key(tag: str, ckpt: str) -> jax.Array:
    """Fresh PRNG, never the training key. blake2b, not hash(): the latter is salted."""
    dig = hashlib.blake2b(f"{tag}|{ckpt}".encode(), digest_size=4).digest()
    return jax.random.fold_in(
        jax.random.PRNGKey(PROBE_ROOT), int.from_bytes(dig, "big") % (2**31)
    )


def collect_states(h: Harness, key):
    """Visited on-policy states, thinned across chunks so they are not one trajectory."""
    key, rk = jax.random.split(key)
    obs, _, st = h.reset(rk)
    for _ in range(BURN):
        key, ak, sk = jax.random.split(key, 3)
        a = jnp.clip(h.pi(obs).sample(seed=ak), -ACTION_CLIP, ACTION_CLIP)
        obs, _, st, _, _, _ = h.env.step(jax.random.split(sk, h.B), st, a)

    O, MU, SG, EID, CID = [], [], [], [], []
    per = B_ENVS // N_CHUNKS
    for c in range(N_CHUNKS):
        for _ in range(CHUNK_GAP):
            key, ak, sk = jax.random.split(key, 3)
            a = jnp.clip(h.pi(obs).sample(seed=ak), -ACTION_CLIP, ACTION_CLIP)
            obs, _, st, _, _, _ = h.env.step(jax.random.split(sk, h.B), st, a)
        sl = slice(c * per, (c + 1) * per)
        mu, sg = h.ck.actor.gaussian(h.na(obs))
        O.append(np.asarray(obs)[sl])
        MU.append(np.asarray(mu)[sl])
        SG.append(np.asarray(sg)[sl])
        EID.append(np.arange(c * per, (c + 1) * per))
        CID.append(np.full(per, c))
    return (np.concatenate(O), np.concatenate(MU), np.concatenate(SG),
            np.concatenate(EID), np.concatenate(CID))


def build_fns(h: Harness, d: int, clip: bool):
    """F(u) over the FULL action and its whitened gradient, batched over (state, sample)."""

    def q_of_u(u, cobs, mu, sg):
        a = jnp.tanh(mu + sg * u)
        if clip:
            a = jnp.clip(a, -ARM_B_CLIP, ARM_B_CLIP)
        return h.ck.critic.critic(cobs, a)

    def q_and_h(u, cobs, mu, sg):
        """u:(n,d). Returns Q:(n,) and H = Sigma^{1/2} grad_y Q :(n,d)."""
        return whitened_pathwise(lambda uu: q_of_u(uu, cobs, mu, sg), u)

    def ref_block(bkey, cobs, mu, sg, S):
        u = jax.random.normal(bkey, (S, N_REF, d))
        rep = lambda a: jnp.repeat(a, N_REF, axis=0)
        q = q_of_u(u.reshape(-1, d), rep(cobs), rep(mu), rep(sg))
        return q.reshape(S, N_REF).mean(1)

    def oracle_block(carry, bkey, cobs, mu, sg, qref, S):
        """Population sums of r = Q - Q_ref, of u, and of H, over the full d."""
        u = jax.random.normal(bkey, (S, ORACLE_BLOCK, d))
        uf = u.reshape(-1, d)
        rep = lambda a: jnp.repeat(a, ORACLE_BLOCK, axis=0)
        q, H = q_and_h(uf, rep(cobs), rep(mu), rep(sg))
        q = q.reshape(S, ORACLE_BLOCK) - qref[:, None]
        H = H.reshape(S, ORACLE_BLOCK, d)
        e = jnp.einsum
        add = (
            q.sum(1),                       # S_r      -> E[Q] - Q_ref
            (q**2).sum(1),                  # S_r2     -> nu = V_e
            e("sn,snj->sj", q, u),          # S_ru     -> a  (eq. 18)
            u.sum(1),                       # S_u
            H.sum(1),                       # S_H      -> h  (eq. 7)
            e("snj,snl->sjl", H, H),        # S_HH     -> C = Cov(H)
            (H**2).sum((1, 2)),             # S_H2
        )
        return tuple(c + a for c, a in zip(carry, add)), None

    def batch_block(bkey, cobs, mu, sg, qref, S):
        """Per-batch whitened estimator realizations over the full d.

        Returns a_hat (S,C,d) and R_PW (S,C,d), plus the saturation covariate.
        """
        C = BATCH_CHUNK
        u = jax.random.normal(bkey, (S, C, M, d))
        uf = u.reshape(-1, d)
        rep = lambda a: jnp.repeat(a, C * M, axis=0)
        q, H = q_and_h(uf, rep(cobs), rep(mu), rep(sg))
        q = q.reshape(S, C, M) - qref[:, None, None]
        H = H.reshape(S, C, M, d)
        # sample-mean baseline, eq. (11). deattenuate=False is REQUIRED: this probe
        # stores the uncorrected a_hat and scripts/probe1_report.py's eq-(13) gate
        # predicts c = 1 - 1/M on exactly that convention.  reduce="einsum" pins the
        # op order the archived scripts/probe2_out/*.npz were baselined with.
        a_hat = centred_zo(q, u, axis=-1, deattenuate=False, reduce="einsum")
        r_pw = pathwise_mean(H, axis=2)
        t = jnp.abs(jnp.tanh(mu[:, None, None, :] + sg[:, None, None, :] * u))
        sat = (t >= ARM_B_CLIP).astype(jnp.float32).mean((2, 3))
        return a_hat, r_pw, sat

    return (
        jax.jit(ref_block, static_argnums=(4,)),
        jax.jit(oracle_block, static_argnums=(6,)),
        jax.jit(batch_block, static_argnums=(5,)),
    )


def run(ckpt: str, law: str, out: str):
    t0 = time.time()
    h = Harness(ckpt, B_ENVS)
    meta = h.meta
    d, k = int(meta["action_dim"]), int(meta["action_pad"])
    r = d - k
    arm = meta["actor_update_mode"]
    clip = arm == "weighted_mle"  # Amendment A answer 1

    obs, mu_ck, sg_ck, env_id, chunk_id = collect_states(h, probe_key("states", ckpt))
    N = obs.shape[0]
    cobs = np.asarray(h.nc(jnp.asarray(obs)))

    if law == "ckpt":
        mu, sg = mu_ck.copy(), sg_ck.copy()
    elif law == "std":
        # Common standardized law: the whitened coordinate is N(0, I_d) about the
        # SAME mean, so the padded/real split stays the checkpoint's own geometry.
        mu, sg = mu_ck.copy(), np.ones((N, d), np.float32)
    else:
        raise ValueError(law)

    ref_block, oracle_block, batch_block = build_fns(h, d, clip)

    z64 = lambda *s: np.zeros(s, np.float64)
    S_r, S_r2 = z64(N), z64(N)
    S_ru, S_u = z64(N, d), z64(N, d)
    S_H, S_HH, S_H2 = z64(N, d), z64(N, d, d), z64(N)
    q_ref_all = z64(N)

    # per-state batch summaries
    mean_a, mean_p = z64(N, d), z64(N, d)
    m2_a, m2_p = z64(N, d), z64(N, d)
    # L moments, per estimator
    L_a_sum, L_a_sq, L_p_sum, L_p_sq = z64(N), z64(N), z64(N), z64(N)
    # block energies
    Xa, Za, Xp, Zp = z64(N), z64(N), z64(N), z64(N)
    zero_a, zero_p = z64(N), z64(N)
    sat_m = z64(N)

    n_or = N_ORACLE // ORACLE_BLOCK
    n_bb = R_BATCHES // BATCH_CHUNK
    ok = probe_key("oracle:" + law, ckpt)
    bk = probe_key("batch:" + law, ckpt)

    for s0 in range(0, N, STATE_CHUNK):
        s1 = min(s0 + STATE_CHUNK, N)
        S = s1 - s0
        args = (jnp.asarray(cobs[s0:s1]), jnp.asarray(mu[s0:s1]), jnp.asarray(sg[s0:s1]))

        qref = ref_block(jax.random.fold_in(probe_key("qref:" + law, ckpt), s0), *args, S)
        q_ref_all[s0:s1] = np.asarray(qref, np.float64)

        carry = (jnp.zeros((S,)), jnp.zeros((S,)), jnp.zeros((S, d)), jnp.zeros((S, d)),
                 jnp.zeros((S, d)), jnp.zeros((S, d, d)), jnp.zeros((S,)))
        for b in range(n_or):
            carry, _ = oracle_block(
                carry, jax.random.fold_in(jax.random.fold_in(ok, s0), b), *args, qref, S
            )
        for dst, v in zip((S_r, S_r2, S_ru, S_u, S_H, S_HH, S_H2), carry):
            dst[s0:s1] += np.asarray(v, np.float64)

        for b in range(n_bb):
            a_hat, r_pw, sat = batch_block(
                jax.random.fold_in(jax.random.fold_in(bk, s0), b), *args, qref, S
            )
            A = np.asarray(a_hat, np.float64)   # (S,C,d)
            P = np.asarray(r_pw, np.float64)
            mean_a[s0:s1] += A.sum(1); mean_p[s0:s1] += P.sum(1)
            m2_a[s0:s1] += (A**2).sum(1); m2_p[s0:s1] += (P**2).sum(1)
            xa = (A[..., :r] ** 2).sum(-1); za = (A[..., r:] ** 2).sum(-1)
            xp = (P[..., :r] ** 2).sum(-1); zp = (P[..., r:] ** 2).sum(-1)
            Xa[s0:s1] += xa.sum(1); Za[s0:s1] += za.sum(1)
            Xp[s0:s1] += xp.sum(1); Zp[s0:s1] += zp.sum(1)
            ta, tp = xa + za, xp + zp
            zero_a[s0:s1] += (ta == 0).sum(1); zero_p[s0:s1] += (tp == 0).sum(1)
            la = np.where(ta > 0, za / np.maximum(ta, 1e-300), np.nan)
            lp = np.where(tp > 0, zp / np.maximum(tp, 1e-300), np.nan)
            L_a_sum[s0:s1] += np.nansum(la, 1); L_a_sq[s0:s1] += np.nansum(la**2, 1)
            L_p_sum[s0:s1] += np.nansum(lp, 1); L_p_sq[s0:s1] += np.nansum(lp**2, 1)
            sat_m[s0:s1] += np.asarray(sat, np.float64).sum(1)

        print(f"  states {s1}/{N}  {time.time()-t0:7.1f}s", flush=True)

    n_o = float(N_ORACLE)
    n_b = float(R_BATCHES)
    mr = S_r / n_o
    nu = S_r2 / n_o - mr**2                                 # V_e, eq. (8)/(18)
    a_pop = S_ru / n_o - mr[:, None] * (S_u / n_o)          # a, eq. (18)
    h_pop = S_H / n_o                                       # h, eq. (7)
    C_pop = S_HH / n_o - h_pop[:, :, None] * h_pop[:, None, :]

    np.savez_compressed(
        out,
        ckpt=ckpt, law=law, arm=arm, d=d, r=r, k=k, M=M,
        R_BATCHES=R_BATCHES, N_ORACLE=N_ORACLE, N=N,
        env_id=env_id, chunk_id=chunk_id,
        q_ref=q_ref_all, mean_Q=q_ref_all + mr, V_e=nu,
        a_pop=a_pop, h_pop=h_pop, C_pop=C_pop, S_H2=S_H2 / n_o,
        mean_a=mean_a / n_b, mean_p=mean_p / n_b,
        m2_a=m2_a / n_b, m2_p=m2_p / n_b,
        L_a_mean=L_a_sum / n_b, L_a_m2=L_a_sq / n_b,
        L_p_mean=L_p_sum / n_b, L_p_m2=L_p_sq / n_b,
        X_a=Xa / n_b, Z_a=Za / n_b, X_p=Xp / n_b, Z_p=Zp / n_b,
        zero_a=zero_a, zero_p=zero_p, sat=sat_m / n_b,
        seconds=time.time() - t0,
    )
    bad = int((nu <= 0).sum())
    print(f"[{os.path.basename(ckpt)} {law}] arm={arm} d={d} r={r} k={k} "
          f"V_e<=0: {bad}/{N}  {time.time()-t0:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])
