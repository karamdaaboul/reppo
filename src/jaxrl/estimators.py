"""Pure gradient-estimator core, shared by the training diagnostics, the offline
probes, and the LQR crossover harness.

Everything here is a plain function of arrays and one critic *callable*. There is no
dependence on `nnx`, on parameter trees, on `ReppoConfig`, or on the optimizer, and
nothing in this module calls `jax.config.update` -- precision is whatever the caller
established. That is what lets `scripts/lqr_crossover/` drive the very same code paths
the DMC arms run, in float64, against a closed-form ground truth.

METRIC. Everything lives in the WHITENED pre-tanh coordinate u, where

    y = mu + Sigma^{1/2} u,     u ~ N(0, I),     Sigma diagonal,

so Sigma^{1/2} is elementwise multiplication by sigma. This is the metric of
`docs/wasted_step_fraction_proposition.md`; see its Sec. 1 for why the dual norm
||g||_Sigma is the right one for gradients and ||v||_{Sigma^-1} for displacements.
Callers wanting an action-space gradient divide by sigma exactly once, outside.

SQUASHING. The core never sees an action, a critic, a `critic_obs`, or a tanh. Its
only critic-facing argument is a callable on the whitened coordinate:

    production:  q_of_u = lambda u: critic.critic(cobs, jnp.tanh(mu + sg * u))
    LQR toy:     q_of_u = lambda u: q_pi(sys, s, mu + sigma * u) + e(s, mu + sigma * u)

Differentiating that closure with `jax.vjp` returns Sigma^{1/2} grad_y Q with the
tanh Jacobian folded in automatically in production and the identity Jacobian in the
toy. There is deliberately no `if squash:` anywhere.

BIT-IDENTITY. Two call sites are pinned by regression gates that compare *bits*, not
values: `scripts/verify_estep.py` (`check_a`, a full training run against the frozen
upstream snapshot) and the archived `scripts/probe2_out/*.npz`. Those two sites reduce
the centred estimator differently -- `reppo.py` broadcasts then takes a mean over the
sample axis, `probe2` contracts with `einsum` then divides by M. The two are
mathematically equal but need not round identically in float32, so `centred_zo` takes
a `reduce` argument and each call site keeps the op order it was baselined with. Do
not "unify" that away without re-running both gates.
"""

from __future__ import annotations

from typing import Callable

import jax
from jax import numpy as jnp

_EINSUM_LETTERS = "abcdefghijklmnopqrstuvw"


def _sample_axes(q, axis: int) -> tuple[int, int]:
    """Sample axis in `q` and the corresponding axis in `u`.

    `u` carries one extra trailing coordinate axis, so a non-negative axis index means
    the same position in both, while a negative one shifts by one in `u`. Callers use
    both conventions: reppo.py passes axis=0 with q:(M,B), u:(M,B,d); probe2 passes
    axis=-1 with q:(S,C,M), u:(S,C,M,d).
    """
    return axis, (axis if axis >= 0 else axis - 1)


def whiten_from_squashed(action, mu, sigma):
    """Recover the whitened draw behind an already-squashed action.

        u = (arctanh(a) - mu) / sigma

    Production-only: the LQR harness draws `u` directly and never calls this. Note the
    caller is responsible for any clipping applied before the critic saw the action --
    the E-step path clips to +-(1 - 1e-4) (`reppo.py`), the pathwise path does not, and
    `arctanh` of an unclipped +-1 is not finite.
    """
    return (jnp.arctanh(action) - mu) / sigma


def whitened_grad_at_mean(q_of_y, mu, sigma):
    """h = Sigma^{1/2} grad_y Q(s, .) evaluated at y = mu.  eq (7)/(19).

    `q_of_y` maps the pre-tanh action to per-state critic values; it is summed here so
    a single `jax.grad` yields the per-state gradients.

    Computed as `sigma * grad_y(...)` rather than as a vjp w.r.t. u. The two are
    mathematically identical and numerically are not: the op order is baselined into
    `scripts/verify_estep.py`.
    """

    def _summed(y):
        return q_of_y(y).sum()

    return sigma * jax.lax.stop_gradient(jax.grad(_summed)(mu))


def whitened_pathwise(q_of_u: Callable, u):
    """Critic values and the whitened pathwise gradient at every sampled point.

    Returns `(q, H)` with `H = Sigma^{1/2} grad_y Q`, obtained by differentiating the
    whitened closure directly, so the chain rule through any squash is handled by JAX
    rather than by hand.
    """
    q, pull = jax.vjp(q_of_u, u)
    (g,) = pull(jnp.ones_like(q))
    return q, g


def pathwise_mean(H, axis: int = -2):
    """r_PW = mean_i H_i -- the pathwise operator's estimate of the blurred gradient."""
    return H.mean(axis)


def deattenuation_factor(M, dtype):
    """M / (M - 1), the correction for the known (1 - 1/M) shrinkage of eq (13).

    Emitted in `dtype` rather than hardcoded float32. In production `dtype` is float32
    and the op is unchanged; under `jax_enable_x64` the LQR harness gets float64
    instead of a silent downcast that would sit right on top of the quantity the
    crossover is a ratio of.
    """
    m = jnp.asarray(M, dtype)
    return m / (m - 1.0)


def centred_zo(q, u, *, axis: int = 0, deattenuate: bool = False,
               return_terms: bool = False, reduce: str = "broadcast"):
    """Centred zeroth-order estimator in the whitened metric.  eq (13)/(16).

        a_hat_M = (1/M) sum_i (Q_i - Qbar) u_i,        E[a_hat_M] = (1 - 1/M) h

    The sample-mean baseline Qbar is what makes this the score-function estimator for
    the Gaussian mean with a variance-reducing baseline; it is also the source of the
    (1 - 1/M) attenuation, which `deattenuate=True` removes.

    `deattenuate` defaults to False because `scripts/probe2_full_estimator.py` stores
    the UNcorrected estimate and the shipped eq-(13) gate in `scripts/probe1_report.py`
    predicts c = 1 - 1/M on exactly that convention. Correcting it there would silently
    invalidate numbers already in `reports/`. An MSE comparison, by contrast, must
    correct -- see spec Sec. 4.

    `reduce` selects the contraction op order; see the module docstring.
    """
    qaxis, uaxis = _sample_axes(q, axis)
    qc = q - q.mean(axis=qaxis, keepdims=True)

    terms = None
    if reduce == "broadcast" or return_terms:
        terms = qc[..., None] * u

    if reduce == "broadcast":
        a_hat = terms.mean(axis=uaxis)
    elif reduce == "einsum":
        nq = q.ndim
        qs = _EINSUM_LETTERS[:nq]
        ax = qaxis % nq
        out = qs[:ax] + qs[ax + 1:] + "z"
        a_hat = jnp.einsum(f"{qs},{qs}z->{out}", qc, u) / q.shape[qaxis]
    else:
        raise ValueError(f"unknown reduce={reduce!r}")

    if deattenuate:
        a_hat = a_hat * deattenuation_factor(q.shape[qaxis], a_hat.dtype)

    return (a_hat, terms) if return_terms else a_hat


def softmax_displacement(w, u, axis: int = 0):
    """What the MPO E-step ACTUALLY moves the mean by, in the whitened metric.

        d = sum_i w_i u_i = argmax_mu sum_i w_i log N(u_i; mu, I)

    This is NOT the centred estimator above. The shipped `actor_update_mode`
    ="weighted_mle" arm optimises a softmax-weighted MLE, whose mean displacement is
    this convex combination; `centred_zo` is the estimator Claim 4 and Proposition 1
    are stated about. Keeping both here is what lets the two be compared on the same
    samples instead of assumed equivalent.
    """
    return jnp.sum(w[..., None] * u, axis=axis)


def zo_diagnostics(h, a_hat, terms, M, *, axis: int = 0):
    """Bias/variance decomposition of the centred estimator against the pathwise h.

    `a_hat` is the UNcorrected estimate and `terms` its per-sample summands, both as
    returned by `centred_zo(..., return_terms=True)`.

    Self-check, state by state:  rel_l2_sq == bias2_proxy + var_proxy * (M/(M-1))^2.
    `est_rel_l2` is a mean OF A RATIO and by Jensen sits below sqrt(est_rel_l2_sq); do
    not mix the two.

    Op order is baselined by `scripts/verify_estep.py`; `ddof=0` on the variance and
    the 1e-12 floors are load-bearing, not incidental.
    """
    dtype = h.dtype
    _M = jnp.asarray(M, dtype)

    h_norm = jnp.linalg.norm(h, axis=-1)
    a_norm = jnp.linalg.norm(a_hat, axis=-1)

    # de-attenuate the known (1 - 1/M) shrinkage before comparing
    a_deatt = a_hat * (_M / (_M - 1.0))
    den = jnp.maximum(h_norm, 1e-12)
    cos = jnp.sum(a_hat * h, axis=-1) / jnp.maximum(a_norm * h_norm, 1e-12)
    err2 = jnp.square(jnp.linalg.norm(a_deatt - h, axis=-1))
    rel_l2 = jnp.linalg.norm(a_deatt - h, axis=-1) / den
    # sampling-noise energy of the mean over M: tr(Cov)/M
    var = terms.var(axis=axis).sum(axis=-1) / _M
    # squared bias with the sampling-noise energy removed
    bias2 = err2 - var * jnp.square(_M / (_M - 1.0))

    return dict(
        est_M=_M,
        est_h_norm=h_norm.mean(),
        est_a_norm=a_norm.mean(),
        est_cos=cos.mean(),
        est_rel_l2=rel_l2.mean(),
        est_rel_l2_sq=(err2 / jnp.square(den)).mean(),
        est_var_proxy=(var / jnp.square(den)).mean(),
        est_bias2_proxy=(bias2 / jnp.square(den)).mean(),
        est_nonfinite=(1.0 - jnp.isfinite(cos).astype(dtype)).mean(),
    )
