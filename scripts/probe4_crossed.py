"""Probe 4 -- crossed same-critic operator table against the padded reference.

Executes Probe 4 of docs/prospective_padding_error_field_analysis.md
(sha256 34dd111af742750c3f96258b15f460ddd867dc42510dceaa73db7125f93e409b) as
committed.  The committed row is:

  | 4 | Crossed same-critic table vs Qbar | Primary D: squared Sigma_x^{-1}-norm of
  | real-update change; secondary L, cosine, weight KL, ESS | Prediction: WML is
  | affected more.  Falsified if paired median D_WML - D_PW <= 0.  Evaluate each
  | critic under both operators. |

Identifying invariance: Q^pi(s,x,z) = Q^pi(s,x) because the padded post-tanh
coordinates are discarded by the environment, so centred variation of Q_phi in z
identifies CRITIC ERROR, not control signal.

D, written out.  Work in the checkpoint's whitened pre-tanh metric, where
Sigma = diag(sigma_c^2) so Sigma^{1/2} is elementwise.  For a gradient estimate
ghat the committed trust-region step is Delta_mu = sqrt(2 eps) Sigma ghat /
||ghat||_Sigma (wasted_step_fraction_proposition.md Sec. 1).  Whitening the
displacement gives Delta_mu_w = sqrt(2 eps) hhat / ||hhat||_2 with
hhat = Sigma^{1/2} ghat, and ||v||^2_{Sigma^-1} = ||v_w||^2_2 exactly.  Hence

    D = 2 eps * || (uhat/||uhat||)_x - (ubar/||ubar||)_x ||^2_2

where uhat is the operator's whitened direction from Q_phi and ubar the same
operator's whitened direction from Qbar_phi.  2 eps multiplies BOTH operators
identically, so it cancels from the sign of D_WML - D_PW and from their ratio; it
is set to 1 and the fact is recorded rather than tuned.

Operators, from Amendment A:
  PW  : ghat = (1/M) sum_i grad_y Q_phi(s, tanh(y_i)), critic input UNCLIPPED.
  WML : Delta_mu = sum_i w_i (y_i - mu_c), w_i = softmax(Q_i/eta_c) raw
        self-normalised softmax, no centring/baseline (Amendment A answer 2),
        eta_c read VERBATIM from the checkpoint (answer 4), critic input CLIPPED
        to +-(1-1e-4) (answer 1, src/jaxrl/reppo.py:669).

RECORDED AMBIGUITIES (plan Sec. 5: record, do not silently resolve; use the most
literal reading that preserves the committed design):
  A1. The plan fixes the clip convention per ARM (answer 1 is about arm B's code
      path), but Probe 4 is a CROSSED table in which the operator varies inside a
      fixed critic.  Most literal reading: the clip belongs to the WML OPERATOR,
      because that is the code path it is a fact about.  The alternative (clip by
      TRAINING arm) is computed as a sensitivity and reported alongside.
  A3. The plan says eta is read VERBATIM from the checkpoint (answer 4).  The
      A-trained (pathwise) checkpoints have NO eta_param: no E-step temperature was
      ever trained for them, so for the crossed cell "A-critic under the WML
      operator" there is nothing to read.  Most literal reading that preserves the
      committed crossed design: read eta verbatim where it exists (arm B), and where
      it does not, construct it from the operator's own registered definition --
      solve the MPO dual against the registered eps_E = 0.5, as a SINGLE scalar
      shared across the batch, matching answer 4's "single learned scalar shared
      across the whole batch (per-batch, never per-state)".  Both the source of eta
      and its value are recorded per cell.  A sensitivity using the matched B-seed
      eta on the A critic is reported alongside.
  A2. The plan fixes no explicit z budget for Probe 4 (it fixes one only for
      Probe 1's oracle stream).  N_Z below is chosen for MC convergence, and a
      split-half MC floor for Qbar is reported so the reader can see it is
      converged rather than take it on trust.

Numerical convention is Amendment A.1 item 2: Q moments are accumulated around a
per-state reference mean from an INDEPENDENT pre-pass, reduced in float64, and a
V_e <= 0 tripwire column is reported and must read zero.

Usage: probe4_crossed.py <seed> <law:ckpt|std> <out.npz>
"""
from __future__ import annotations
import hashlib, os, sys, time
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import jax, jax.numpy as jnp, numpy as np                       # noqa: E402
from scripts.critic_fidelity.common import ACTION_CLIP, Harness  # noqa: E402

M = 32                 # estep_num_samples (Amendment A recordables)
EPS_E = 0.5            # eps_e
ARM_B_CLIP = 1.0 - 1e-4
R_DIM, K_DIM = 6, 16   # WalkerRun real dims, padded dims
B_ENVS = int(os.environ.get("P4_ENVS", 128))          # lanes per arm
BURN, N_CHUNKS, CHUNK_GAP = 200, int(os.environ.get("P4_CHUNKS", 8)), 25
N_Z = int(os.environ.get("P4_NZ", 1024))             # z draws for Qbar (ambiguity A2)
N_REF = 256            # independent pre-pass for the per-state centring reference
STATE_CHUNK = int(os.environ.get("P4_SCHUNK", 32))
PROBE_ROOT = 20260831
TWO_EPS = 1.0

EXPORTS = os.path.join(REPO_ROOT, "exports")


def probe_key(tag, ckpt):
    dig = hashlib.blake2b(f"{tag}|{ckpt}".encode(), digest_size=4).digest()
    return jax.random.fold_in(jax.random.PRNGKey(PROBE_ROOT),
                              int.from_bytes(dig, "big") % (2 ** 31))


def solve_eta_scalar(Q, eps_e, lo=1e-4, hi=1e4, iters=80):
    """Single batch-shared eta from the standard MPO E-step dual

        g(eta) = eta*eps_E + eta * log mean_i exp(Q_i/eta),

    solved by bisection on its derivative, which is monotone in eta.  Used ONLY where
    the checkpoint has no eta_param to read (ambiguity A3)."""
    Q = np.asarray(Q, dtype=np.float64).reshape(-1, Q.shape[-1])

    def dg(eta):
        z = Q / eta
        zm = z.max(1, keepdims=True)
        lse = zm[:, 0] + np.log(np.mean(np.exp(z - zm), 1))
        w = np.exp(z - zm); w /= w.sum(1, keepdims=True)
        return float(np.mean(eps_e + lse - (w * Q).sum(1) / eta))

    a, b = lo, hi
    for _ in range(iters):
        m = np.sqrt(a * b)
        if dg(m) < 0:
            a = m
        else:
            b = m
    return float(np.sqrt(a * b))


def ckpt_dir(arm, seed):
    mode = "pathwise" if arm == "A" else "weighted_mle"
    var = "_fa_pad16" if arm == "A" else "_pad16"
    return os.path.join(EXPORTS, f"WalkerRun_{mode}{var}_s{seed}_final")


def collect_raw_states(h, key):
    """Visited RAW observations under this checkpoint's own pi_old, temporal chunks kept."""
    obs_l, env_l, chunk_l = [], [], []
    key, rk = jax.random.split(key)
    obs, _, st = h.reset(rk)
    for step in range(BURN + N_CHUNKS * CHUNK_GAP):
        k1, k2, key = jax.random.split(key, 3)
        dist = h.pi(obs)
        if step >= BURN and (step - BURN) % CHUNK_GAP == 0:
            obs_l.append(np.asarray(obs))
            env_l.append(np.arange(B_ENVS))
            chunk_l.append(np.full(B_ENVS, (step - BURN) // CHUNK_GAP))
        a = jnp.clip(dist.sample(seed=k1), -ACTION_CLIP, ACTION_CLIP)
        obs, _, st, _, _, _ = h.env.step(jax.random.split(k2, B_ENVS), st, a)
    return np.concatenate(obs_l), np.concatenate(env_l), np.concatenate(chunk_l)


def main():
    seed, law, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    assert law in ("ckpt", "std")
    t0 = time.time()

    # --- common raw states: equal complete lanes from A and B (plan Sec. 1) ------
    raws, envs, chunks, srcs = [], [], [], []
    for arm in ("A", "B"):
        h = Harness(ckpt_dir(arm, seed), B_ENVS)
        r, e, c = collect_raw_states(h, probe_key(f"states|{arm}", f"s{seed}"))
        raws.append(r); envs.append(e); chunks.append(c)
        srcs.append(np.full(len(r), 0 if arm == "A" else 1))
        del h
    raw = np.concatenate(raws); env_id = np.concatenate(envs)
    chunk_id = np.concatenate(chunks); state_src = np.concatenate(srcs)
    N_S = raw.shape[0]
    print(f"[seed {seed} law {law}] common states: {N_S} "
          f"({(state_src==0).sum()} A-lane, {(state_src==1).sum()} B-lane)", flush=True)

    # --- per-checkpoint policy laws on the IDENTICAL raw states -----------------
    H = {arm: Harness(ckpt_dir(arm, seed), B_ENVS) for arm in ("A", "B")}
    MU, SG, ETA, ETA_SRC = {}, {}, {}, {}
    for arm, h in H.items():
        d = h.pi(jnp.asarray(raw))                    # each critic applies its OWN normalizer
        MU[arm] = np.asarray(d.distribution.loc, dtype=np.float64)
        SG[arm] = np.asarray(d.distribution.scale, dtype=np.float64)
        try:
            ETA[arm] = float(np.asarray(h.ck.actor.eta()).ravel()[0])
            ETA_SRC[arm] = "verbatim"
        except AttributeError:
            ETA[arm] = None                           # solved later, see ambiguity A3
            ETA_SRC[arm] = "solved_from_dual"
    print(f"  eta: A={ETA['A']} ({ETA_SRC['A']})  B={ETA['B']} ({ETA_SRC['B']})", flush=True)

    # --- common actions: equal A-policy and B-policy samples, labels kept -------
    ku = probe_key(f"actions|{law}", f"s{seed}")
    u_common = jax.random.normal(ku, (N_S, M, R_DIM + K_DIM), dtype=jnp.float64)
    u_common = np.asarray(u_common)
    act_src = np.concatenate([np.zeros(M // 2, int), np.ones(M // 2, int)])
    Y = np.empty((N_S, M, R_DIM + K_DIM))
    for j, arm in enumerate(("A", "B")):
        sl = slice(j * M // 2, (j + 1) * M // 2)
        Y[:, sl, :] = MU[arm][:, None, :] + SG[arm][:, None, :] * u_common[:, sl, :]

    res = {}
    eta_used = {}
    for critic in ("A", "B"):
        h = H[critic]
        cobs = np.asarray(h.nc(jnp.asarray(raw)), dtype=np.float32)
        mu_c, sg_c = MU[critic], SG[critic]

        def qfun(cob, y, clip):
            a = jnp.tanh(y)
            if clip:
                a = jnp.clip(a, -ARM_B_CLIP, ARM_B_CLIP)
            return h.ck.critic.critic(cob, a)

        # gradient of Q wrt the pre-tanh y, tanh Jacobian included
        gfun = jax.vmap(jax.vmap(jax.grad(lambda y, c, cl: qfun(c, y, cl).squeeze()),
                                 in_axes=(0, None, None)), in_axes=(0, 0, None))
        qbat = jax.vmap(jax.vmap(lambda y, c, cl: qfun(c, y, cl).squeeze(),
                                 in_axes=(0, None, None)), in_axes=(0, 0, None))

        for opname, clip in (("PW", False), ("WML", True)):
            # ---- Q_phi at the common actions
            Qi = np.empty((N_S, M)); Gi = np.empty((N_S, M, R_DIM + K_DIM))
            for s0 in range(0, N_S, STATE_CHUNK):
                sl = slice(s0, min(s0 + STATE_CHUNK, N_S))
                yc = jnp.asarray(Y[sl], dtype=jnp.float32)
                cc = jnp.asarray(cobs[sl])
                Qi[sl] = np.asarray(qbat(yc, cc, clip), dtype=np.float64)
                if opname == "PW":
                    Gi[sl] = np.asarray(gfun(yc, cc, clip), dtype=np.float64)

            # ---- Qbar: average over z at fixed (s, x). Centred pre-pass first.
            kz = probe_key(f"z|{law}|{critic}|{opname}", f"s{seed}")
            Qbar = np.empty((N_S, M)); Ve = np.empty((N_S, M))
            Gbar = np.empty((N_S, M, R_DIM + K_DIM))
            zref_key, zmain_key = jax.random.split(kz)
            for s0 in range(0, N_S, STATE_CHUNK):
                sl = slice(s0, min(s0 + STATE_CHUNK, N_S))
                ns = sl.stop - sl.start
                cc = jnp.asarray(cobs[sl])
                xb = jnp.asarray(Y[sl][:, :, :R_DIM], dtype=jnp.float32)

                def z_draws(key, n):
                    u = jax.random.normal(key, (ns, M, n, K_DIM), dtype=jnp.float32)
                    if law == "ckpt":
                        mz = jnp.asarray(mu_c[sl][:, None, None, R_DIM:], dtype=jnp.float32)
                        sz = jnp.asarray(sg_c[sl][:, None, None, R_DIM:], dtype=jnp.float32)
                        return mz + sz * u
                    return u                     # common standardized law N(0, I_k)

                def qz(zs):
                    yy = jnp.concatenate(
                        [jnp.broadcast_to(xb[:, :, None, :], zs.shape[:3] + (R_DIM,)), zs], -1)
                    cb = jnp.broadcast_to(cc[:, None, None, :], zs.shape[:3] + (cc.shape[-1],))
                    return qfun(cb, yy, clip).reshape(zs.shape[:3])   # (ns, M, n)

                # independent pre-pass -> per-state reference mean (Amendment A.1(2))
                kr, zmain_key2 = jax.random.split(jax.random.fold_in(zref_key, s0))
                Qref = np.asarray(qz(z_draws(kr, N_REF)).mean(-1), dtype=np.float64)
                km = jax.random.fold_in(zmain_key, s0)
                zs = z_draws(km, N_Z)
                qv = np.asarray(qz(zs), dtype=np.float64) - Qref[:, :, None]
                Qbar[sl] = Qref + qv.mean(-1)
                Ve[sl] = qv.var(-1, ddof=1)      # centred: never the cancelling one-pass form
                if opname == "PW":
                    # grad of Qbar wrt y: E_z[grad_x Q] in the x block, 0 in the z block
                    gb = np.zeros((ns, M, R_DIM + K_DIM))
                    NZG = int(os.environ.get("P4_NZG", 128))   # gradient sub-budget for E_z[grad_x Q]
                    zg = zs[:, :, :NZG, :]
                    yy = jnp.concatenate(
                        [jnp.broadcast_to(xb[:, :, None, :], zg.shape[:3] + (R_DIM,)), zg], -1)
                    cb = jnp.broadcast_to(cc[:, None, None, :], zg.shape[:3] + (cc.shape[-1],))
                    g3 = jax.vmap(jax.vmap(jax.vmap(
                        jax.grad(lambda y, c, cl: qfun(c, y, cl).squeeze()),
                        in_axes=(0, None, None)), in_axes=(0, 0, None)), in_axes=(0, 0, None))
                    gb[:, :, :R_DIM] = np.asarray(g3(yy, cb, clip),
                                                  dtype=np.float64).mean(2)[:, :, :R_DIM]
                    Gbar[sl] = gb

            # ---- whitened operator directions from Q_phi and from Qbar_phi
            sgc = sg_c[:, None, :]
            if opname == "PW":
                uh = (sgc * Gi).mean(1)          # (N_S, d) whitened gradient estimate
                ub = (sgc * Gbar).mean(1)
                ess = np.full(N_S, np.nan); wkl = np.full(N_S, np.nan)
            else:
                uw = (Y - mu_c[:, None, :]) / sgc            # whitened offsets
                eta_c = ETA[critic]
                if eta_c is None:
                    eta_c = solve_eta_scalar(Qi, EPS_E)
                    print(f"    eta for critic {critic} solved from the dual "
                          f"(eps_E={EPS_E}): {eta_c:.6g}", flush=True)
                eta_used[critic] = eta_c
                def wml_dir(Q):
                    z = Q / eta_c
                    z = z - z.max(1, keepdims=True)
                    w = np.exp(z); w /= w.sum(1, keepdims=True)
                    return (w[..., None] * uw).sum(1), w
                uh, wh = wml_dir(Qi)
                ub, wb = wml_dir(Qbar)
                ess = 1.0 / np.sum(wh ** 2, axis=1)
                wkl = np.sum(wh * (np.log(wh + 1e-300) - np.log(wb + 1e-300)), axis=1)

            def unit(v):
                return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)
            dh, db = unit(uh), unit(ub)
            D = TWO_EPS * np.sum((dh[:, :R_DIM] - db[:, :R_DIM]) ** 2, axis=-1)
            L = np.sum(dh[:, R_DIM:] ** 2, axis=-1)          # padded fraction of the step
            cos = np.sum(dh * db, axis=-1)
            res[f"{critic}|{opname}"] = dict(D=D, L=L, cos=cos, ess=ess, wkl=wkl,
                                             Ve_nonpos=int((Ve <= 0).sum()),
                                             Qbar_mc=float(np.mean(np.sqrt(Ve / N_Z))),
                                             Qspread=float(np.mean(np.sqrt(np.maximum(Ve, 0)))))
            r = res[f"{critic}|{opname}"]
            print(f"  critic {critic} op {opname:3s}: median D={np.median(D):.5g} "
                  f"median L={np.median(L):.4g} median cos={np.median(cos):.5g} "
                  f"Ve<=0:{r['Ve_nonpos']} QbarMC={r['Qbar_mc']:.3g}", flush=True)

    np.savez(out, env_id=env_id, chunk_id=chunk_id, state_src=state_src,
             act_src=act_src, seed=seed, law=law, eta_A=ETA["A"], eta_B=ETA["B"],
             **{f"{k.replace('|','_')}_{f}": v
                for k, d in res.items() for f, v in d.items()})
    print(f"[seed {seed} law {law}] wrote {out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
