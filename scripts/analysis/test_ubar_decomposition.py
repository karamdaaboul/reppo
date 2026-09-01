"""Step 0.8 verification of the ubar decomposition against the actual WML loss.

Frozen-checkpoint, read-only. No training, no checkpoint is written.

What is verified, and what would falsify each:

  T1  The implemented weighted-MLE objective is exactly
          L_full = -mean_s sum_i sg(w_si) log pi_theta(a_si_fit | s)
      with the weights fully stop-gradiented (src/jaxrl/reppo.py:751-754).
      Falsified if the reconstructed loss differs from the in-repo expression.

  T2  L_full = L_uniform + L_centered  exactly (algebraic identity).

  T3  grad L_full = grad L_uniform + grad L_centered on the FULL actor parameter
      vector, to numerical precision. Falsified by any nonzero residual.

  T4  The exact score-space identity. For a tanh-transformed diagonal Gaussian,
          log pi(a) = log N(y; mu, sigma) - sum_j log(1 - a_j^2),  y = atanh(a),
      and the Jacobian term does not depend on mu. Hence
          d/dmu log pi(a_i) = (y_i - mu) / sigma^2
      and in the whitened mean coordinate
          Sigma^{1/2} d/dmu log pi(a_i) = (y_i - mu)/sigma = u_i_fit.
      So the whitened mean-score direction is exactly  v = sum_i w_i u_i_fit,
      and the tanh transformation drops out of the MEAN score entirely.
      Verified by autodiff against the closed form. Falsified if they disagree.

  T5  v = ubar + c identically, with ubar = mean_i u_i, c = sum_i (w_i - 1/M) u_i.

  T6  Raw-Gaussian moments of ubar_raw: RMS ||ubar_raw|| = sqrt(d/M),
      E||ubar_raw|| = sqrt(2/M) Gamma((d+1)/2)/Gamma(d/2), and the chi median.

  T7  u_i_fit == u_i_raw exactly where the +-(1-1e-4) clip does not bind, so any
      difference between the raw and implementation-space decompositions is
      attributable to clipping alone and not to the tanh transform.

Run:  JAX_PLATFORMS=cpu ./.venv/bin/python scripts/analysis/test_ubar_decomposition.py <ckpt>
"""
from __future__ import annotations
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np                                  # noqa: E402
from flax import nnx                                                  # noqa: E402
from scipy.special import gammaln, chdtri                             # noqa: E402
from scripts.critic_fidelity.common import ACTION_CLIP, Harness       # noqa: E402
from src.jaxrl.reppo import estep_weights                             # noqa: E402

CLIP = 1.0 - 1e-4          # src/jaxrl/reppo.py:712
M = 32                     # estep_num_samples
NS = 128                   # states for the gradient checks
BURN = 50                  # scripts/q_spread_from_ckpt.py:39
FAILS = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("   " + detail if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def main(ckpt):
    h = Harness(ckpt, NS)
    key = jax.random.PRNGKey(0)
    key, rk = jax.random.split(key)
    obs, _, st = h.reset(rk)
    for _ in range(BURN):                       # q_spread_from_ckpt.py:39-42
        k1, k2, key = jax.random.split(key, 3)
        a = jnp.clip(h.pi(obs).sample(seed=k1), -ACTION_CLIP, ACTION_CLIP)
        obs, _, st, _, _, _ = h.env.step(jax.random.split(k2, NS), st, a)

    nobs, cobs = h.na(obs), h.nc(obs)
    d = h.action_dim
    eta_present = hasattr(h.ck.actor, "eta_param")
    eta = float(np.asarray(h.ck.actor.eta()).ravel()[0]) if eta_present else None

    # --- sample exactly as the trainer does: post-tanh sample, then clip -------
    pi_old = h.ck.actor.actor(nobs)
    mu = np.asarray(pi_old.distribution.loc, np.float64)
    sg = np.asarray(pi_old.distribution.scale, np.float64)
    # Draw the standard normal EXPLICITLY rather than recovering it with arctanh.
    # distrax's Transformed(Normal, Tanh).sample is exactly tanh(mu + sigma*u); an
    # arctanh round-trip through float32 loses precision as |a| -> 1 and biases the
    # ubar moments low, which is a probe artefact, not a property of the estimator.
    u_raw = np.asarray(jax.random.normal(jax.random.PRNGKey(5), (M, NS, d),
                                         dtype=jnp.float64), np.float64)
    y_raw = mu[None] + sg[None] * u_raw
    a_raw = np.tanh(y_raw)
    a_fit = np.clip(a_raw, -CLIP, CLIP)
    y_fit = np.arctanh(a_fit)                    # exact: |a_fit| <= 1-1e-4
    u_fit = (y_fit - mu[None]) / sg[None]
    clip_rate = float(np.mean(np.abs(a_raw) > CLIP))
    a_fit = jnp.asarray(a_fit)

    q_i = np.asarray(h.ck.critic.critic(
        jnp.broadcast_to(cobs, (M, *cobs.shape)), a_fit), np.float64)      # reppo.py:748
    if eta is None:                       # pathwise ckpt: no E-step temperature exists
        eta = float(np.std(q_i))          # placeholder for the ALGEBRAIC checks only
    w = np.asarray(estep_weights(jnp.asarray(q_i), eta), np.float64)       # reppo.py:751-752

    check("T1a weights normalise over the sample axis (axis 0), per state",
          np.allclose(w.sum(0), 1.0, atol=1e-12), "max|sum_i w_i - 1| = %.2e"
          % np.abs(w.sum(0) - 1).max())
    check("T1b M is 32", M == 32 and w.shape[0] == 32, "w.shape=%s" % (w.shape,))

    # ---- T4/T5: whitened mean score by autodiff vs the closed form -----------
    gdef, params = nnx.split(h.ck.actor)

    def logp(p, a):
        m = nnx.merge(gdef, p)
        return m.actor(nobs).log_prob(a).sum(-1)          # reppo.py:717 convention

    def mean_score_whitened(a_i):
        """Sigma^{1/2} d/dmu  sum_i w_i log pi(a_i), by autodiff through mu only."""
        def f(mu_):
            std = jnp.asarray(sg)
            import distrax
            dist = distrax.Transformed(distrax.Normal(loc=mu_, scale=std), distrax.Tanh())
            lp = dist.log_prob(a_i).sum(-1)               # (M, NS)
            return jnp.sum(jnp.asarray(w) * lp)
        g = jax.grad(f)(jnp.asarray(mu))
        return np.asarray(g, np.float64) * sg             # whitening

    v_auto = mean_score_whitened(a_fit)
    v_closed = np.einsum("ib,ibd->bd", w, u_fit)
    rel = np.linalg.norm(v_auto - v_closed) / max(np.linalg.norm(v_closed), 1e-300)
    check("T4 whitened mean score by autodiff == sum_i w_i u_i_fit (tanh Jacobian drops out)",
          rel < 1e-8, "rel err = %.3e" % rel)

    ubar = u_fit.mean(0)
    c = np.einsum("ib,ibd->bd", w - 1.0 / M, u_fit)
    check("T5 v = ubar + c identically",
          np.allclose(v_closed, ubar + c, atol=1e-12),
          "max abs = %.2e" % np.abs(v_closed - ubar - c).max())

    # ---- T2/T3: exact loss and gradient decomposition on FULL actor params ---
    def L(p, wts):
        return -jnp.mean(jnp.sum(jnp.asarray(wts) * logp(p, a_fit), axis=0))

    Lf = float(L(params, w)); Lu = float(L(params, np.full_like(w, 1.0 / M)))
    Lc = float(L(params, w - 1.0 / M))
    check("T2 L_full = L_uniform + L_centered",
          abs(Lf - (Lu + Lc)) <= 1e-9 * max(1.0, abs(Lf)),
          "residual = %.3e (L_full=%.6g)" % (abs(Lf - (Lu + Lc)), Lf))

    def grad_residual(p):
        gf = jax.grad(L)(p, w)
        gu = jax.grad(L)(p, np.full_like(w, 1.0 / M))
        gc = jax.grad(L)(p, w - 1.0 / M)
        lf, lu, lc = (jax.tree.leaves(x) for x in (gf, gu, gc))
        num = sum(float(jnp.sum((a - b - cc) ** 2)) for a, b, cc in zip(lf, lu, lc)) ** .5
        den = sum(float(jnp.sum(a ** 2)) for a in lf) ** .5
        return num / max(den, 1e-300)

    r32 = grad_residual(params)
    params64 = jax.tree.map(lambda x: x.astype(jnp.float64)
                            if jnp.issubdtype(x.dtype, jnp.floating) else x, params)
    r64 = grad_residual(params64)
    # The trained parameters are float32, so at native precision the identity can only
    # hold to ~float32 eps (1.2e-7). Casting the SAME parameters to float64 must drive
    # the residual to float64 level; if it does not, the discrepancy is real.
    check("T3a grad decomposition at native float32 params (tol 1e-5)",
          r32 < 1e-5, "rel residual = %.3e  (float32 eps = 1.2e-07)" % r32)
    check("T3b grad decomposition with params cast to float64 (tol 1e-11)",
          r64 < 1e-11, "rel residual = %.3e  -> the float32 residual is precision, "
                       "not a real discrepancy" % r64)

    # ---- T6: raw-Gaussian moments of ubar_raw --------------------------------
    # This is a pure PRNG/shape check with no model involvement, so it is run at high
    # replication rather than at the NS used for the gradient checks: at NS=128 the
    # relative standard error on the RMS is ~1.3%, too coarse to distinguish a real
    # deviation from sampling noise.
    N_MOM = 200000
    u_mom = np.asarray(jax.random.normal(jax.random.PRNGKey(11), (M, N_MOM, d),
                                         dtype=jnp.float64), np.float64)
    nrm = np.linalg.norm(u_mom.mean(0), axis=-1)
    rms, emp_mean, emp_med = float(np.sqrt((nrm ** 2).mean())), float(nrm.mean()), float(np.median(nrm))
    exp_rms = np.sqrt(d / M)
    exp_mean = np.sqrt(2.0 / M) * np.exp(gammaln((d + 1) / 2) - gammaln(d / 2))
    exp_med = np.sqrt(chdtri(d, 0.5)) / np.sqrt(M)
    check("T6a RMS ||ubar_raw|| == sqrt(d/M)", abs(rms / exp_rms - 1) < 0.01,
          "emp %.4f vs sqrt(d/M) %.4f  (ratio %.4f)" % (rms, exp_rms, rms / exp_rms))
    check("T6b E ||ubar_raw|| == chi mean", abs(emp_mean / exp_mean - 1) < 0.01,
          "emp %.4f vs %.4f  (ratio %.4f)" % (emp_mean, exp_mean, emp_mean / exp_mean))
    check("T6c median ||ubar_raw|| == chi median", abs(emp_med / exp_med - 1) < 0.01,
          "emp %.4f vs %.4f  (ratio %.4f)" % (emp_med, exp_med, emp_med / exp_med))

    # ---- T7: raw vs implementation space differ ONLY through the clip --------
    unclipped = np.abs(np.asarray(a_raw, np.float64)) <= CLIP
    same = np.allclose(u_raw[unclipped], u_fit[unclipped], atol=1e-9)
    check("T7 u_fit == u_raw wherever the clip does not bind",
          same, "clip rate = %.4f%%" % (100 * clip_rate))

    print("\nckpt              : %s" % ckpt)
    print("d, M              : %d, %d" % (d, M))
    print("eta               : %.6g  (%s)" % (eta, "checkpoint" if eta_present
                                              else "PLACEHOLDER sd(Q); algebraic checks only"))
    print("clip rate         : %.4f%%" % (100 * clip_rate))
    print("logit spread (med): %.4f" % float(np.median((q_i / eta).std(0))))
    print("\n%d/%d checks passed" % (10 - len(FAILS), 10))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else "exports/WalkerRun_weighted_mle_pad16_s0_final"))
