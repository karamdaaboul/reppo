"""Thin wrapper over the shared estimator core. DO NOT reimplement the estimators here.

The point of Sec. 6.1 of the spec is that a fresh g_PW / g_ZO written for the toy
proves the toy is correct and proves nothing about the operators running in the fork.
So the arithmetic lives in `src/jaxrl/estimators.py`, which the training diagnostics
and `scripts/probe2_full_estimator.py` also call, and this module does exactly one
thing on top of it: convert the whitened gradient to an action-space one.

    core returns  Sigma^{1/2} g          (the metric of the proposition document)
    the toy wants g                      (so it can be compared to grad_a Q^pi)

so everything here divides by sigma exactly once. That single conversion is the only
place a scale bug could hide, which is why gate G6c targets it specifically.
"""

from __future__ import annotations

from jax import numpy as jnp

from src.jaxrl.estimators import (
    centred_zo,
    deattenuation_factor,
    pathwise_mean,
    softmax_displacement,
    whitened_pathwise,
)

__all__ = [
    "zo_gradient", "pw_gradient", "estep_displacement", "both_from_shared_u",
]


def pw_gradient(q_of_u, u, sigma, *, axis=-2):
    """g_PW = (1/M) sum_i grad_a Q(s, mu + sigma u_i), in action space."""
    _, H = whitened_pathwise(q_of_u, u)
    return pathwise_mean(H, axis=axis) / sigma


def zo_gradient(q_of_u, u, sigma, *, axis=-2, deattenuate=True):
    """g_ZO = (1/(sigma M)) sum_i (Q_i - Qbar) u_i, in action space.

    `deattenuate=True` applies the M/(M-1) correction of Appendix A.2. It is MANDATORY
    for any MSE comparison: E[g_ZO] = (1 - 1/M) g*, and while the trust region absorbs
    that scalar in training, an MSE does not -- skipping it hands the pathwise arm a
    free ~3% advantage at M = 32.
    """
    q = q_of_u(u)
    qaxis = axis + 1 if axis < 0 else axis
    a_hat = centred_zo(q, u, axis=qaxis, deattenuate=deattenuate, reduce="broadcast")
    return a_hat / sigma


def estep_displacement(q_of_u, u, sigma, eta, *, axis=-2):
    """The shipped weighted_mle arm's actual mean move, mapped to action space.

    NOT the centred estimator: `actor_update_mode="weighted_mle"` optimises a
    softmax-weighted MLE, whose mean displacement is sum_i w_i u_i with
    w = softmax(Q/eta). Claim 4 and Proposition 1 are stated about `zo_gradient`; this
    is what actually ships, and the two are compared here on identical samples rather
    than assumed equivalent.

    Returned as a displacement per unit sigma so it lives in the same units as the
    gradients above; it is a direction, not a gradient, and its magnitude is set by eta.
    """
    q = q_of_u(u)
    qaxis = axis + 1 if axis < 0 else axis
    w = jnp.exp((q - q.max(axis=qaxis, keepdims=True)) / eta)
    w = w / w.sum(axis=qaxis, keepdims=True)
    return softmax_displacement(w, u, axis=qaxis) / sigma


def both_from_shared_u(q_of_u, u, sigma, *, axis=-2, deattenuate=True):
    """Both operators on the SAME draws, in one pass.

    Sharing u is what makes the error-only decomposition of spec Sec. 5.1 exact rather
    than approximate, and what makes the Sec. 5.2 cross term computable at all.
    """
    q, H = whitened_pathwise(q_of_u, u)
    qaxis = axis + 1 if axis < 0 else axis
    a_hat = centred_zo(q, u, axis=qaxis, reduce="broadcast")
    if deattenuate:
        a_hat = a_hat * deattenuation_factor(q.shape[qaxis], a_hat.dtype)
    return {
        "q": q,
        "g_pw": pathwise_mean(H, axis=axis) / sigma,
        "g_zo": a_hat / sigma,
    }
