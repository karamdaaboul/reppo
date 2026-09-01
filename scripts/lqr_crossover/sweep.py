"""E1a / E1b sweep driver. One pass over shared u; everything else is reconstruction.

    python scripts/lqr_crossover/sweep.py --d 8 [--coarse] [--smoke]

WHY THE PRODUCT GRID IS NEVER ENUMERATED. The spec asks for one row per
(d, sigma, omega, M, state, replicate, estimator): 7 x 20 x 20 x 10^4 x 64 x 2 = 3.6e9
rows. That is infeasible in any format, and pyarrow is not installed. The resolution is
not a smaller grid -- it is that both operators are EXACTLY linear in the critic
(gate G6d asserts this), so with g = g_smooth + g_error,

    ||g_full - g*||^2 = ||D_sm(sigma)||^2
                      + 2 (eps/sigma) <D_sm(sigma), De_unit(c)>      <- spec Sec. 5.2
                      + (eps/sigma)^2 ||De_unit(c)||^2               <- spec Sec. 5.1

and, crucially, D_sm depends on (state, sigma) but NOT on omega, while De_unit depends
on (state, c = sigma*omega) with the amplitude eps factored out entirely. So the three
statistics S1(sigma), S2(sigma, c), S3(c) determine every cell of the grid, they all come
from the SAME draws, and the Sec. 5.1/5.2 requirement that the error-only and smooth
columns be written from one pass is satisfied structurally rather than by discipline.

The sigma and omega grids share a log ratio, so c takes 53 distinct values rather than
400, and every sigma column brackets the crossover at every d (see the prereg Sec. 2 for
why the spec's omega <= 300 does not).

Output: one .npz per (d, arm) under out/, batch-resolved for the hierarchical bootstrap,
plus a tidy row appended to out/index.jsonl. Aggregation happens only in analyze.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

# three levels: this package sits at scripts/lqr_crossover/, not scripts/
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import scripts.lqr_crossover  # noqa: F401,E402

import jax  # noqa: E402
import numpy as np  # noqa: E402
from jax import numpy as jnp  # noqa: E402

from scripts.lqr_crossover import OUT, SEED_ROOT, error_field as EF, lqr  # noqa: E402

RATIO = 300.0 ** (1.0 / 19.0)
N_SIGMA, N_OMEGA = 20, 34
N_C = N_SIGMA + N_OMEGA - 1                       # 53 distinct sigma*omega values
SIGMAS = 0.01 * RATIO ** np.arange(N_SIGMA)
OMEGAS = 0.1 * RATIO ** np.arange(N_OMEGA)
CS = 1e-3 * RATIO ** np.arange(N_C)
M_DEFAULT = 32
DS = (1, 2, 4, 8, 16, 32, 64)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def make_kernel(d, M, r):
    """Jitted per-state kernel: scan over replicate batches, emit batch-resolved stats.

    All statistics come from ONE draw of u per batch. Returned per batch:
        s1_{pw,zo}  (n_sigma,)                  smooth-part squared error
        s3_{pw,zo}  (n_sigma, n_omega)          error-only squared error, in units of
                                                (eps/sigma)^2
        s2_{pw,zo}  (n_sigma, n_sigma, n_omega) cross-term inner product, first axis
                                                indexing the smooth part's sigma

    GENERAL RANK. With e = (eps/sqrt r) sum_j sin(omega v_j^T a + phi_j) and orthonormal
    V, the pathwise error stays inside span(V) and its squared norm is
    (eps/sigma)^2 (c^2/r) sum_j a_j^2, while the zeroth-order one is driven by the scalar
    f_i = (1/sqrt r) sum_j sin(theta_j + c t_ji). r = 1 is the spec's rank-one field and
    reduces to the closed forms of reference.py; r > 1 has no closed form, which is
    exactly why Rule B of the prereg is falsifiable and Rule A is not.

    THETA IS NOT CONSTANT ACROSS THE GRID. theta_j = omega v_j^T mu + phi_j genuinely
    depends on omega, so two cells with the same product c = sigma*omega have DIFFERENT
    phases. Folding theta into a per-state constant would make the collapse check of the
    prereg trivially true instead of testing anything, so the error-only part is
    evaluated on the full (sigma, omega) grid. What makes the collapse hold is that
    phi ~ U[0, 2pi) makes theta uniform for every omega, so the STATE-AVERAGED crossover
    depends only on c -- which is what gate G10 checks and the collapse residual measures.

    The loop is scan-over-omega with an inner vmap-over-sigma, so the largest live array
    is (n_sigma, R, M, r) rather than (n_sigma, n_omega, R, M, r).
    """
    rf = float(r)

    def one_batch(carry, key):
        u = jax.random.normal(key, (R_BATCH, M, d))

        # ---- smooth part. grad_a Q^pi is affine in a, so g_PW - g* = -2 sigma H ubar
        # exactly, and the ZO estimate follows from q_i = const - sigma^2 qq + sigma ql.
        Hu = u @ H_j                                        # (R, M, d)
        qq = jnp.sum(Hu * u, -1)                            # (R, M)
        ql = u @ g_j                                        # (R, M)
        ubar = u.mean(1)                                    # (R, d)

        def smooth(sig):
            q = -(sig ** 2) * qq + sig * ql                 # (R, M)
            qc = q - q.mean(1, keepdims=True)
            zo = (qc[..., None] * u).mean(1) * (M / (M - 1.0)) / sig
            return -2.0 * sig * (ubar @ H_j), zo - g_j      # (R,d), (R,d)

        D_pw, D_zo = jax.vmap(smooth)(sig_j)                # (S, R, d) each
        T = jnp.einsum("rmd,jd->rmj", u, V_j)               # (R, M, r)

        # ---- error-only part, amplitude eps factored out entirely (which is why the
        # E1a contour is exactly eps-independent, prereg Sec. 3).
        def per_omega(_, om):
            th = om * vtmu_j + phi_j                        # (r,)

            def per_sigma(sig):
                c = sig * om
                arg = th[None, None, :] + c * T             # (R, M, r)
                damp = jnp.exp(-0.5 * c * c)
                a_pw = jnp.cos(arg).mean(1) - damp * jnp.cos(th)[None, :]     # (R, r)
                de_pw = (c / jnp.sqrt(rf)) * (a_pw @ V_j)                     # (R, d)
                f = jnp.sin(arg).sum(-1) / jnp.sqrt(rf)                       # (R, M)
                fc = f - f.mean(1, keepdims=True)
                b = (fc[..., None] * u).mean(1) * (M / (M - 1.0))             # (R, d)
                tgt = (c * damp / jnp.sqrt(rf)) * (jnp.cos(th) @ V_j)         # (d,)
                de_zo = b - tgt[None, :]
                return (
                    (de_pw ** 2).sum(-1).mean(),                              # s3_pw
                    (de_zo ** 2).sum(-1).mean(),                              # s3_zo
                    jnp.einsum("srj,rj->s", D_pw, de_pw) / R_BATCH,           # s2_pw (S,)
                    jnp.einsum("srj,rj->s", D_zo, de_zo) / R_BATCH,           # s2_zo (S,)
                )

            return None, jax.vmap(per_sigma)(sig_j)

        _, (s3p, s3z, s2p, s2z) = jax.lax.scan(per_omega, None, om_j)
        # scan stacks on axis 0 = omega; move to (sigma, omega) / (S, sigma, omega)
        out = dict(
            s1_pw=(D_pw ** 2).sum(-1).mean(-1),
            s1_zo=(D_zo ** 2).sum(-1).mean(-1),
            s3_pw=s3p.T, s3_zo=s3z.T,
            s2_pw=jnp.transpose(s2p, (2, 1, 0)),
            s2_zo=jnp.transpose(s2z, (2, 1, 0)),
        )
        return carry, out

    def kernel(key, H, g, V, vtmu, phi, sigmas, omegas, R, n_batch):
        nonlocal H_j, g_j, V_j, vtmu_j, phi_j, sig_j, om_j, R_BATCH
        H_j, g_j, V_j, vtmu_j, phi_j = H, g, V, vtmu, phi
        sig_j, om_j, R_BATCH = sigmas, omegas, R
        keys = jax.random.split(key, n_batch)
        _, out = jax.lax.scan(one_batch, None, keys)
        return out

    H_j = g_j = V_j = vtmu_j = phi_j = sig_j = om_j = R_BATCH = None
    return kernel


def run_d(d, *, M=M_DEFAULT, n_states=64, n_batch=40, r_batch=250, kind="rank1",
          rank=None, eps_frac=0.05, normalize="unit_H", cost="identity", tag=""):
    t0 = time.time()
    s = lqr.build_system(d, seed=SEED_ROOT + d, normalize=normalize, cost=cost)
    rng = np.random.default_rng(SEED_ROOT + 2000 + d)
    states = lqr.sample_states(s, rng, n_states)
    H, g, mu = lqr.q_coeffs(s, states)
    pe = EF.draw_error(rng, n_states, d, kind=kind, rank=rank, omega=1.0)

    # eps is set per (state, sigma) from the closed-form Q spread, never as an absolute
    # number, so the amplitude is scale-matched across d (guard R2).
    q_spread = np.stack([lqr.q_spread_closed_form(s, states, sg) for sg in SIGMAS], 1)
    eps = eps_frac * q_spread                                   # (n_states, n_sigma)

    kernel = jax.jit(make_kernel(d, M, pe.rank), static_argnames=("R", "n_batch"))
    acc = {k: [] for k in ("s1_pw", "s1_zo", "s3_pw", "s3_zo", "s2_pw", "s2_zo")}
    vtmu = np.einsum("srd,sd->sr", pe.V, mu)                 # v_j^T mu, per state
    for i in range(n_states):
        key = jax.random.fold_in(jax.random.PRNGKey(SEED_ROOT + 3000 + d), i)
        out = kernel(key, jnp.asarray(H), jnp.asarray(g[i]),
                     jnp.asarray(pe.V[i]), jnp.asarray(vtmu[i]),
                     jnp.asarray(pe.phi[i]),
                     jnp.asarray(SIGMAS), jnp.asarray(OMEGAS),
                     R=r_batch, n_batch=n_batch)
        for k in acc:
            acc[k].append(np.asarray(out[k]))
    res = {k: np.stack(v) for k, v in acc.items()}              # (n_states, n_batch, ...)

    name = f"d{d}_{kind}{'' if rank is None else rank}_M{M}_{normalize}_{cost}{tag}"
    path = os.path.join(OUT, name + ".npz")
    np.savez_compressed(
        path, sigmas=SIGMAS, omegas=OMEGAS, cs=CS, eps=eps, q_spread=q_spread,
        phi=pe.phi, vtmu=vtmu, d=d, M=M, n_states=n_states, n_batch=n_batch,
        r_batch=r_batch, kind=kind, rank=pe.rank, eps_frac=eps_frac,
        normalize=normalize, cost=cost,
        rho_closed=s.rho_closed, cond_H=s.cond_H, tr_H2=s.tr_H2, P_fro=s.P_fro,
        lyap_resid_rel=s.lyap_resid_rel, retries=s.retries,
        alpha_Q=s.scale["alpha_Q"], alpha_s=s.scale["alpha_s"],
        omega_inf_conventions=np.array(
            [f"{a}/{b}" for a, b in EF.ALL_CONVENTIONS], dtype=object),
        # realised frequency per unit nominal omega, all four conventions. The
        # primary (registered) one is grad_norm=2 / val_norm=inf.
        omega_inf_factor=np.array(
            [EF.omega_inf(pe, grad_norm=a, val_norm=b) if b == "inf" else np.nan
             for a, b in EF.ALL_CONVENTIONS]),
        git_sha=git_sha(), seconds=time.time() - t0, **res)

    row = dict(name=name, d=d, M=M, kind=kind, rank=int(pe.rank), eps_frac=eps_frac,
               normalize=normalize, cost=cost, n_states=n_states,
               n_rep=n_batch * r_batch, rho_closed=s.rho_closed, cond_H=s.cond_H,
               tr_H2=s.tr_H2, lyap_resid_rel=s.lyap_resid_rel, retries=s.retries,
               eps_over_qspread=float(eps_frac), git_sha=git_sha(),
               seconds=round(time.time() - t0, 2), npz=os.path.basename(path))
    with open(os.path.join(OUT, "index.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"  d={d:3d} {kind}{'' if rank is None else rank} -> {os.path.basename(path)} "
          f"({time.time() - t0:.1f}s, rho_cl={s.rho_closed:.3f})")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=None, help="single d; default all seven")
    ap.add_argument("--M", type=int, default=M_DEFAULT)
    ap.add_argument("--kind", default="rank1", choices=("rank1", "rank_r", "full"))
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--eps-frac", type=float, default=0.05)
    ap.add_argument("--normalize", default="unit_H", choices=("unit_H", "none"))
    ap.add_argument("--cost", default="identity", choices=("identity", "random_psd"))
    ap.add_argument("--coarse", action="store_true",
                    help="M6 gate: N=200, is the contour interior at every d?")
    ap.add_argument("--smoke", action="store_true", help="1/100 scale")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    kw = dict(M=a.M, kind=a.kind, rank=a.rank, eps_frac=a.eps_frac,
              normalize=a.normalize, cost=a.cost, tag=a.tag)
    if a.coarse:
        kw.update(n_states=16, n_batch=8, r_batch=25, tag=a.tag + "_coarse")
    elif a.smoke:
        kw.update(n_states=4, n_batch=4, r_batch=25, tag=a.tag + "_smoke")

    os.makedirs(OUT, exist_ok=True)
    ds = [a.d] if a.d else list(DS)
    print(f"sweep: d={ds} M={a.M} kind={a.kind} eps_frac={a.eps_frac} "
          f"normalize={a.normalize} sha={git_sha()[:10]}")
    t0 = time.time()
    for d in ds:
        run_d(d, **kw)
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
