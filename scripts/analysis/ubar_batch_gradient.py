"""Step 3: trainer-faithful frozen batch reconstruction and actor-gradient decomposition.

Protocol: docs/prereg_ubar_ratio.md Sec. 8. Frozen-checkpoint, read-only. This is NOT
a training run: no learner state is updated or saved. Optimizer state is absent from
every export (Step 0.5), so the optional copied-state replay is not performed and the
gradient decomposition is the exact primary analysis.

Reproduces the trainer's layout exactly:
  * pool of num_envs*num_steps = 1024*128 = 131072 states, rolled under the frozen policy
  * shuffled into num_mini_batches minibatches (B=2048 DMC, B=8192 g1)
  * ONE (M, B, d) standard-normal draw per epoch, REUSED across every minibatch of that
    epoch -- the shared-PRNG pattern established at reppo.py:1093-1098 / 613 / 1085
  * theta == theta_old exactly, since actor_target.params = actor.params at reppo.py:603-607

Loss decomposition (weights stop-gradiented, reppo.py:751-754):
    L_full     = -mean_s sum_i w_si       log pi(a_si_fit|s)
    L_uniform  = -mean_s (1/M) sum_i      log pi(a_si_fit|s)
    L_centered = -mean_s sum_i (w_si-1/M) log pi(a_si_fit|s)

Metrics are reported in three spaces:
  * Euclidean parameter norms, split by mean head / scale head / shared trunk / full
    (parameterization dependent -- flagged as such);
  * action-space: the induced first-order change in the actor mean, whitened by sigma;
  * an invariant KL/Fisher norm: for a diagonal Gaussian,
        KL(new||old) ~ sum_j [ dmu_j^2/(2 sigma_j^2) + dlogsigma_j^2 ],
    so ||g||_KL := sqrt( mean_s sum_j [ dmu_j^2/sigma_j^2 + 2 dlogsigma_j^2 ] ),
    with (dmu, dlogsigma) obtained by JVP along the gradient direction. Cross-checked
    against the repo's own decoupled_kls at finite step size.

Usage: ubar_batch_gradient.py <ckpt_dir> <out.npz> [n_shuffles] [n_grad_minibatches]
"""
from __future__ import annotations
import hashlib, json, os, sys, time
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np                                   # noqa: E402
from flax import nnx                                                   # noqa: E402
import distrax                                                         # noqa: E402
from scripts.critic_fidelity.common import ACTION_CLIP, Harness        # noqa: E402
from src.jaxrl.reppo import decoupled_kls                              # noqa: E402

M, CLIP, EPS = 32, 1.0 - 1e-4, 1e-12
N_ENVS, N_STEPS = 1024, 128          # trainer pool: config/experiment_overrides/*
PROBE_ROOT = 20260902
MB_OF = {"G1JoystickFlatTerrain": 16}   # mjx_humanoid_large_data; DMC tasks use 64


def key_for(tag, ck):
    dig = hashlib.blake2b(f"{tag}|{ck}".encode(), digest_size=4).digest()
    return jax.random.fold_in(jax.random.PRNGKey(PROBE_ROOT),
                              int.from_bytes(dig, "big") % (2 ** 31))


def solve_eta_scalar(q, eps_e=0.5, lo=1e-4, hi=10.0, iters=100):
    q = np.asarray(q, np.float64).reshape(q.shape[0], -1).T

    def dg(eta):
        z = q / eta; zm = z.max(1, keepdims=True)
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


def leaf_group(path):
    """mean head / scale head / shared trunk. The output layer emits [loc | log_std]
    as one tensor (jax_models.py:522-523), so its last axis is split in half."""
    s = "/".join(str(p.key) if hasattr(p, "key") else str(p) for p in path)
    if "output_layer" in s:
        return "head"
    return "trunk"


def main(ckpt, out, n_shuffles=1, n_grad_mb=8):
    t0 = time.time()
    meta = json.load(open(f"{ckpt}/meta.json"))
    tag, env = os.path.basename(ckpt), meta["env_name"]
    n_mb = MB_OF.get(env, 64)
    B = (N_ENVS * N_STEPS) // n_mb
    h = Harness(ckpt, N_ENVS)
    d = h.action_dim
    try:
        eta_m = float(np.asarray(h.ck.actor.eta()).ravel()[0]); eta_src = "measured"
    except AttributeError:
        eta_m, eta_src = None, "recomputed_counterfactual"

    # ---- trainer-faithful pool -------------------------------------------------
    key = key_for("pool", tag)
    key, rk = jax.random.split(key)
    obs, _, st = h.reset(rk)
    pool = []
    for _ in range(N_STEPS):
        k1, k2, key = jax.random.split(key, 3)
        pool.append(np.asarray(obs))
        a = jnp.clip(h.pi(obs).sample(seed=k1), -ACTION_CLIP, ACTION_CLIP)
        obs, _, st, _, _, _ = h.env.step(jax.random.split(k2, N_ENVS), st, a)
    pool = np.concatenate(pool)                                  # (131072, obs)
    P = pool.shape[0]

    gdef, params = nnx.split(h.ck.actor)
    rows, grows = [], []

    for ep in range(n_shuffles):
        perm = np.asarray(jax.random.permutation(key_for(f"shuf{ep}", tag), P))
        # ONE draw per epoch, reused across every minibatch of that epoch
        u_ep = np.asarray(jax.random.normal(key_for(f"u{ep}", tag), (M, B, d),
                                            dtype=jnp.float64), np.float64)
        for mb in range(n_mb):
            idx = perm[mb * B:(mb + 1) * B]
            o = pool[idx]
            pi = h.ck.actor.actor(jnp.asarray(o))
            mu = np.asarray(pi.distribution.loc, np.float64)
            sg = np.asarray(pi.distribution.scale, np.float64)
            a_raw = np.tanh(mu[None] + sg[None] * u_ep)
            a_fit = np.clip(a_raw, -CLIP, CLIP)
            u_fit = (np.arctanh(a_fit) - mu[None]) / sg[None]
            cobs = np.asarray(h.nc(jnp.asarray(o)), np.float32)
            q = np.asarray(h.ck.critic.critic(
                jnp.broadcast_to(jnp.asarray(cobs), (M, *cobs.shape)),
                jnp.asarray(a_fit, jnp.float32)), np.float64)
            eta = eta_m if eta_m is not None else solve_eta_scalar(q)
            w = np.asarray(jax.nn.softmax(jnp.asarray(q / eta), axis=0), np.float64)

            v = np.einsum("ib,ibd->bd", w, u_fit)
            ub = u_fit.mean(0); c = v - ub
            ub_raw = u_ep.mean(0)
            mv, mu_b, mc = v.mean(0), ub.mean(0), c.mean(0)
            rows.append(dict(
                tag=tag, env=env, mode=meta["actor_update_mode"], seed=meta["seed"],
                d=d, B=B, epoch=ep, minibatch=mb, eta=eta, eta_src=eta_src,
                n_mean_v=float(np.linalg.norm(mv)), n_mean_ubar=float(np.linalg.norm(mu_b)),
                n_mean_c=float(np.linalg.norm(mc)),
                n_mean_ubar_raw=float(np.linalg.norm(ub_raw.mean(0))),
                R_batch_action=float(np.linalg.norm(mu_b) / (np.linalg.norm(mc) + EPS)),
                cos_meanv_meanc=float(mv @ mc / max(np.linalg.norm(mv) * np.linalg.norm(mc), EPS)),
                clip_rate=float(np.mean(np.abs(a_raw) > CLIP))))

            if mb < n_grad_mb:
                grows.append(grad_decomposition(gdef, params, o, a_fit, w, mu, sg, d,
                                                dict(tag=tag, env=env,
                                                     mode=meta["actor_update_mode"],
                                                     seed=meta["seed"], d=d, B=B,
                                                     epoch=ep, minibatch=mb)))
        print("  shuffle %d done (%d minibatches, %d gradient triples) [%.0fs]"
              % (ep, n_mb, min(n_grad_mb, n_mb), time.time() - t0), flush=True)

    np.savez_compressed(out, action=np.array(rows, dtype=object),
                        grad=np.array(grows, dtype=object), allow_pickle=True)
    import csv
    for name, rr in (("action", rows), ("grad", grows)):
        if rr:
            with open(out.replace(".npz", f"_{name}.csv"), "w", newline="") as f:
                wcsv = csv.DictWriter(f, fieldnames=list(rr[0].keys())); wcsv.writeheader()
                for r in rr: wcsv.writerow(r)
    g0 = grows[0] if grows else {}
    print("%-46s d=%2d B=%d eta=%.5g (%s) | med R_batch_action=%.4g | "
          "R_theta full=%.4g mean-out=%.4g KL=%.4g  [%.0fs]"
          % (tag, d, B, rows[0]["eta"], eta_src,
             float(np.median([r["R_batch_action"] for r in rows])),
             g0.get("R_theta_full", np.nan), g0.get("R_meanout", np.nan),
             g0.get("R_kl", np.nan), time.time() - t0), flush=True)


def grad_decomposition(gdef, params, o, a_fit, w, mu, sg, d, info):
    """g_full = g_uniform + g_centered, in parameter, action-mean and KL metrics."""
    ob = jnp.asarray(o)

    def L(p, wts):
        m = nnx.merge(gdef, p)
        return -jnp.mean(jnp.sum(jnp.asarray(wts) * m.actor(ob).log_prob(
            jnp.asarray(a_fit)).sum(-1), axis=0))

    wu = np.full_like(w, 1.0 / M)
    gf, gu, gc = (jax.grad(L)(params, x) for x in (w, wu, w - 1.0 / M))

    def flat(g, sel=None):
        vs = []
        leaves, _ = jax.tree_util.tree_flatten_with_path(g)
        for path, leaf in leaves:
            grp = leaf_group(path)
            arr = np.asarray(leaf, np.float64)
            if "output_layer" in "/".join(str(getattr(p, "key", p)) for p in path):
                half = arr.shape[-1] // 2
                if sel == "mean_head": arr = arr[..., :half]
                elif sel == "scale_head": arr = arr[..., half:]
                elif sel == "trunk": continue
            elif sel in ("mean_head", "scale_head"):
                continue
            vs.append(arr.ravel())
        return np.concatenate(vs) if vs else np.zeros(1)

    res = dict(info)
    for sel, nm in ((None, "full"), ("mean_head", "meanhead"),
                    ("scale_head", "scalehead"), ("trunk", "trunk")):
        F, U, C = flat(gf, sel), flat(gu, sel), flat(gc, sel)
        res[f"resid_{nm}"] = float(np.linalg.norm(F - U - C) / max(np.linalg.norm(F), EPS))
        res[f"R_theta_{nm}"] = float(np.linalg.norm(U) / (np.linalg.norm(C) + EPS))
        res[f"cos_full_centered_{nm}"] = float(F @ C / max(np.linalg.norm(F) * np.linalg.norm(C), EPS))
        res[f"cos_uniform_centered_{nm}"] = float(U @ C / max(np.linalg.norm(U) * np.linalg.norm(C), EPS))

    # ---- policy-output effect by JVP: induced (dmu, dlogsigma) along -g ---------
    def outputs(p):
        m = nnx.merge(gdef, p)
        loc, scale = m.gaussian(ob)
        return loc, jnp.log(scale)

    def induced(g):
        _, (dmu, dls) = jax.jvp(outputs, (params,), (jax.tree.map(lambda x: -x, g),))
        dmu = np.asarray(dmu, np.float64); dls = np.asarray(dls, np.float64)
        meanout = float(np.sqrt(np.mean(np.sum((dmu / sg) ** 2, -1))))
        kl = float(np.sqrt(np.mean(np.sum((dmu / sg) ** 2 + 2 * dls ** 2, -1))))
        return meanout, kl, dmu, dls

    mo_f, kl_f, dmu_f, dls_f = induced(gf)
    mo_u, kl_u, dmu_u, dls_u = induced(gu)
    mo_c, kl_c, dmu_c, dls_c = induced(gc)
    res.update(meanout_full=mo_f, meanout_uniform=mo_u, meanout_centered=mo_c,
               kl_full=kl_f, kl_uniform=kl_u, kl_centered=kl_c,
               R_meanout=mo_u / (mo_c + EPS), R_kl=kl_u / (kl_c + EPS),
               resid_meanout=float(np.linalg.norm(dmu_f - dmu_u - dmu_c)
                                   / max(np.linalg.norm(dmu_f), EPS)),
               cos_meanout_full_centered=float(
                   np.sum(dmu_f * dmu_c) / max(np.linalg.norm(dmu_f) * np.linalg.norm(dmu_c), EPS)))
    # cross-check the KL metric against the repo's own decoupled_kls at finite step
    t = 1e-4
    pert = jax.tree.map(lambda p, g: p - t * g, params, gf)
    m0, m1 = nnx.merge(gdef, params), nnx.merge(gdef, pert)
    mu0, sg0 = m0.gaussian(ob); mu1, sg1 = m1.gaussian(ob)
    kmu, ksg = decoupled_kls(mu1, sg1, mu0, sg0)
    res["kl_full_fd_check"] = float(np.sqrt(2 * float(jnp.mean(kmu + ksg))) / t)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 1,
         int(sys.argv[4]) if len(sys.argv) > 4 else 8)
