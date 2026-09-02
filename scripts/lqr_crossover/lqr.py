"""Discounted LQR with a closed-form Q^pi, and everything the sweep needs from it.

The point of using an LQR rather than a maze is that Q^pi, its action-gradient, and
the Gaussian-blurred estimand are all available in closed form, so estimator error is
measured against an EXACT vector and never against a Monte Carlo reference. Nothing in
this module rolls anything out except `value_mc`, which exists only to check the
closed forms (gate G2).

Setting.  s' = A s + B a + w,  w ~ N(0, W);  r = -(s^T Qc s + a^T Rc a);
policy pi(a|s) = N(-K s, sigma^2 I).  Cost matrices are Qc/Rc so they cannot be
confused with the Q function.

Closed forms (derived in the spec Sec. 2, re-verified here against `value_mc`):

    P  = Qc + K^T Rc K + gamma A_K^T P A_K,            A_K = A - B K
    v  = [sigma^2 tr(Rc) + gamma sigma^2 tr(B^T P B) + gamma tr(P W)] / (1 - gamma)
    V^pi(s) = -(s^T P s + v)

    H  = Rc + gamma B^T P B
    b(s) = -2 gamma B^T P A s
    Q^pi(s, a)      = -a^T H a + b^T a + const(s)
    grad_a Q^pi     = -2 H a + b

Because Q^pi is quadratic, Gaussian blurring shifts it by a constant, so the smooth
part of the estimand is exactly grad_a Q^pi(s, mu). Substituting a = mu + sigma u,

    Q^pi(s, mu + sigma u) = const(s, sigma) - sigma^2 (u^T H u) + sigma (u^T g*)

with g* = -2 H mu + b. Both estimators are invariant to the additive constant -- the
zeroth-order one centres it away, the pathwise one differentiates it away -- so the
sweep needs only `qq = u^T H u` and `ql = u^T g*`, and one (M,d)x(d,d) matmul serves
every sigma on the grid. That identity is why the sweep is minutes rather than hours.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

from jax import numpy as jnp


@dataclasses.dataclass(frozen=True)
class LQRSystem:
    d: int
    n: int
    gamma: float
    A: np.ndarray
    B: np.ndarray
    Qc: np.ndarray
    Rc: np.ndarray
    W: np.ndarray
    K: np.ndarray
    K_opt: np.ndarray
    P: np.ndarray
    H: np.ndarray
    rho_open: float
    rho_closed: float
    cond_H: float
    tr_H2: float
    P_fro: float
    lyap_resid_rel: float
    ctrb_smin: float
    retries: int
    sigma_ref: float
    scale: dict
    seed: int

    @property
    def A_K(self) -> np.ndarray:
        return self.A - self.B @ self.K


def solve_P(A, B, Qc, Rc, K, gamma):
    """Discounted Lyapunov solve for the policy's value matrix.

    scipy's solve_discrete_lyapunov(a, q) solves X = a X a^H + q, so passing
    a = sqrt(gamma) A_K^T gives X = gamma A_K^T X A_K + q, which is the equation above.
    """
    A_K = A - B @ K
    q = Qc + K.T @ Rc @ K
    return solve_discrete_lyapunov(np.sqrt(gamma) * A_K.T, q)


def lyap_residual_rel(sys_or_args) -> float:
    """Relative Frobenius residual of the P equation. Gate G1 (< 1e-12).

    Relative, not the spec's absolute 1e-10: ||P||_F grows with d, so an absolute
    bound is not scale-free.
    """
    s = sys_or_args
    R = s.P - s.Qc - s.K.T @ s.Rc @ s.K - s.gamma * s.A_K.T @ s.P @ s.A_K
    return float(np.linalg.norm(R, "fro") / np.linalg.norm(s.P, "fro"))


def dare_gain(A, B, Qc, Rc, gamma):
    """Optimal discounted gain.

    The discount is absorbed by the change of variable s~ = gamma^{t/2} s, which turns
    the discounted problem into a standard one for (sqrt(gamma) A, sqrt(gamma) B).
    """
    Ad, Bd = np.sqrt(gamma) * A, np.sqrt(gamma) * B
    P = solve_discrete_are(Ad, Bd, Qc, Rc)
    K = np.linalg.solve(Rc + gamma * B.T @ P @ B, gamma * B.T @ P @ A)
    return K, P


def _ctrb_smin(A, B) -> float:
    """Smallest singular value of the controllability matrix, normalised by ||B||."""
    n = A.shape[0]
    blocks, M = [B], B
    for _ in range(n - 1):
        M = A @ M
        blocks.append(M)
    C = np.concatenate(blocks, axis=1)
    return float(np.linalg.svd(C, compute_uv=False).min() / max(np.linalg.norm(B), 1e-300))


def build_system(
    d: int,
    *,
    gamma: float = 0.99,
    rho_open: float = 1.05,
    k_perturb: float = 1.15,
    cost: str = "identity",
    n_ratio: int = 2,
    normalize: str = "unit_H",
    sigma_ref: float = 0.1,
    seed: int = 0,
    max_retries: int = 64,
) -> LQRSystem:
    """Draw a stabilisable LQR with a suboptimal stabilising policy.

    Rejection sampling guards R1: a near-marginal closed loop inflates the stationary
    covariance, so Q^pi acquires an enormous scale and the EFFECTIVE error amplitude
    moves without the nominal one moving. Every rejection is counted and the accepted
    system's diagnostics are logged with the results.

    `k_perturb` scales the DARE gain to give a deliberately suboptimal but stabilising
    policy -- the operators are being compared away from the optimum, where the
    action-gradient is nonzero.

    `sigma_ref` fixes the state law once for the whole sweep. States must be SHARED
    across the sigma grid for the paired comparison to mean anything, so the
    stationary covariance is evaluated at this one reference width and recorded.
    """
    n = n_ratio * d
    rng = np.random.default_rng(seed)
    retries = 0
    for attempt in range(max_retries):
        A0 = rng.normal(size=(n, n)) / np.sqrt(n)
        A = rho_open * A0 / max(np.abs(np.linalg.eigvals(A0)).max(), 1e-300)
        B = rng.normal(size=(n, d)) / np.sqrt(n)
        if cost == "identity":
            Qc, Rc = np.eye(n), np.eye(d)
        elif cost == "random_psd":
            Gq, Gr = rng.normal(size=(n, n)), rng.normal(size=(d, d))
            Qc = np.eye(n) + Gq @ Gq.T / n
            Rc = np.eye(d) + Gr @ Gr.T / d
        else:
            raise ValueError(f"unknown cost={cost!r}")
        W = np.eye(n)

        smin = _ctrb_smin(A, B)
        if smin < 1e-8:
            retries += 1
            continue
        try:
            K_opt, _ = dare_gain(A, B, Qc, Rc, gamma)
        except Exception:
            retries += 1
            continue
        K = k_perturb * K_opt
        rho_cl = float(np.abs(np.linalg.eigvals(A - B @ K)).max())
        if not np.isfinite(rho_cl) or rho_cl > 0.99:
            retries += 1
            continue

        P = solve_P(A, B, Qc, Rc, K, gamma)
        P = 0.5 * (P + P.T)
        H = Rc + gamma * B.T @ P @ B
        H = 0.5 * (H + H.T)
        cond_H = float(np.linalg.cond(H))
        if cond_H > 50.0:
            retries += 1
            continue

        sys_ = LQRSystem(
            d=d, n=n, gamma=gamma, A=A, B=B, Qc=Qc, Rc=Rc, W=W, K=K, K_opt=K_opt,
            P=P, H=H,
            rho_open=float(np.abs(np.linalg.eigvals(A)).max()),
            rho_closed=rho_cl,
            cond_H=cond_H,
            tr_H2=float(np.trace(H @ H)),
            P_fro=float(np.linalg.norm(P, "fro")),
            lyap_resid_rel=0.0,
            ctrb_smin=smin,
            retries=retries,
            sigma_ref=sigma_ref,
            scale={"alpha_Q": 1.0, "alpha_s": 1.0, "normalize": normalize},
            seed=seed,
        )
        sys_ = dataclasses.replace(sys_, lyap_resid_rel=lyap_residual_rel(sys_))
        return dataclasses.replace(sys_, scale=_scale_factors(sys_, normalize))
    raise RuntimeError(f"build_system(d={d}) exhausted {max_retries} retries")


def _scale_factors(sys_: LQRSystem, normalize: str) -> dict:
    """Two scalars that hold the sweep's d-axis comparable. Guard R2.

    A GLOBAL rescale of Q cancels out of everything here -- both estimators are
    homogeneous and eps is set as a fraction of the within-state Q spread -- so it is
    not enough. What actually drifts with d is the SHAPE ratio tr(H^2)/||g*||^2, and
    that needs two knobs:

        alpha_Q  scales the critic, hence H and g* together;
        alpha_s  scales the state law, hence mu and b, hence g* alone.

    Setting ||alpha_Q H||_2 = 1 and E_s||g*|| = 1 pins both. Without this, the ZO
    smooth-part variance carries a sigma^4 curvature term that grows like d^3 (for
    H = h I, E[(u^T H u)^2 ||u||^2] = h^2 d(d+2)(d+4)), so E1b's crossover at d = 64
    would be driven by an artefact of a globally quadratic Q rather than by the error
    field. E1a is untouched either way: by D1 its contour is exactly sigma*omega =
    c*(d), independent of both scalars.

    NOTE alpha_s moves the evaluation states off the closed-loop stationary law. That
    is a declared distortion, not an oversight: Q^pi is exact everywhere, the states
    are only evaluation points, and the unnormalised arm is run alongside.
    """
    if normalize == "none":
        return {"alpha_Q": 1.0, "alpha_s": 1.0, "normalize": normalize}
    if normalize != "unit_H":
        raise ValueError(f"unknown normalize={normalize!r}")
    hnorm = float(np.linalg.norm(sys_.H, 2))
    alpha_Q = 1.0 / hnorm
    # E_s ||g*_raw|| under the reference stationary law, before any scaling.
    S = state_cov(sys_, sigma=sys_.sigma_ref)
    rng = np.random.default_rng(sys_.seed + 7919)
    L = np.linalg.cholesky(S + 1e-12 * np.eye(sys_.n))
    s = rng.normal(size=(4096, sys_.n)) @ L.T
    g = _grad_raw(sys_, s)
    alpha_s = hnorm / float(np.linalg.norm(g, axis=-1).mean())
    return {"alpha_Q": alpha_Q, "alpha_s": alpha_s, "normalize": normalize}


def _grad_raw(sys_: LQRSystem, s: np.ndarray) -> np.ndarray:
    """grad_a Q^pi(s, mu(s)) before any normalisation. s:(...,n) -> (...,d)."""
    mu = -s @ sys_.K.T
    b = -2.0 * sys_.gamma * s @ (sys_.B.T @ sys_.P @ sys_.A).T
    return -2.0 * mu @ sys_.H.T + b


def q_coeffs(sys_: LQRSystem, s: np.ndarray):
    """(H, g_star, mu) for the NORMALISED critic, at states s:(...,n).

    These three arrays are the entire interface between the LQR and the sweep: with
    them, Q(s, mu + sigma u) = const - sigma^2 u^T H u + sigma u^T g*.
    """
    aQ, aS = sys_.scale["alpha_Q"], sys_.scale["alpha_s"]
    ss = aS * s
    mu = -ss @ sys_.K.T
    g = aQ * _grad_raw(sys_, ss)
    return aQ * sys_.H, g, mu


def value_offset(sys_: LQRSystem, sigma: float) -> float:
    """The scalar v in V^pi(s) = -(s^T P s + v)."""
    g = sys_.gamma
    return float(
        (sigma**2 * np.trace(sys_.Rc)
         + g * sigma**2 * np.trace(sys_.B.T @ sys_.P @ sys_.B)
         + g * np.trace(sys_.P @ sys_.W))
        / (1.0 - g)
    )


def v_pi(sys_: LQRSystem, s: np.ndarray, sigma: float) -> np.ndarray:
    return -(np.einsum("...i,ij,...j->...", s, sys_.P, s) + value_offset(sys_, sigma))


def q_pi(sys_: LQRSystem, s: np.ndarray, a: np.ndarray, sigma: float) -> np.ndarray:
    """Exact Q^pi(s, a), unnormalised. Used by the gates, not by the hot path."""
    g = sys_.gamma
    nxt = s @ sys_.A.T + a @ sys_.B.T
    return -(
        np.einsum("...i,ij,...j->...", s, sys_.Qc, s)
        + np.einsum("...i,ij,...j->...", a, sys_.Rc, a)
        + g * np.einsum("...i,ij,...j->...", nxt, sys_.P, nxt)
        + g * np.trace(sys_.P @ sys_.W)
        + g * value_offset(sys_, sigma)
    )


def grad_a_q_pi(sys_: LQRSystem, s: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Exact grad_a Q^pi(s, a), unnormalised."""
    b = -2.0 * sys_.gamma * s @ (sys_.B.T @ sys_.P @ sys_.A).T
    return -2.0 * a @ sys_.H.T + b


def q_of_u_factory(sys_: LQRSystem, s, sigma: float) -> Callable:
    """Whitened critic closure for the shared estimator core.

    This is the toy's counterpart of production's
    `lambda u: critic.critic(cobs, tanh(mu + sg*u))`; the core cannot tell them apart,
    which is the whole point of Sec. 6.1 of the spec.
    """
    H, g, mu = q_coeffs(sys_, s)
    H_j, g_j, mu_j = jnp.asarray(H), jnp.asarray(g), jnp.asarray(mu)

    def q_of_u(u):
        # g* is per state, (S, d); u carries arbitrary replicate/sample axes between
        # the state axis and the coordinate axis, so it must be reshaped rather than
        # left to trailing-dim broadcasting (which silently misaligns, cf. G5/G6c).
        gg = g_j.reshape(g_j.shape[0], *([1] * (u.ndim - 2)), g_j.shape[1])
        # Reduced form: Q(s, mu + sigma u) = const - sigma^2 (u^T H u) + sigma (u^T g*).
        # The additive constant is dropped because both operators are blind to it (the
        # zeroth-order one centres it away, the pathwise one differentiates it away),
        # and this is EXACTLY the form the sweep kernel evaluates, so the gates test
        # the arithmetic the sweep actually runs.
        return (
            -(sigma**2) * jnp.einsum("...i,ij,...j->...", u, H_j, u)
            + sigma * jnp.sum(gg * u, axis=-1)
        )

    return q_of_u


def q_spread_closed_form(sys_: LQRSystem, s: np.ndarray, sigma: float) -> np.ndarray:
    """sd over a ~ pi(.|s) of Q(s, a), exactly.

        Var = sigma^2 ||g*||^2 + 2 sigma^4 tr(H^2)

    The cross term vanishes because E[(u^T H u)(u^T g)] is an odd moment. This is what
    eps is set as a fraction of, so the error amplitude is scale-matched across d
    rather than fixed as an absolute number (guard R2).
    """
    H, g, _ = q_coeffs(sys_, s)
    return np.sqrt(
        sigma**2 * np.sum(g * g, axis=-1) + 2.0 * sigma**4 * np.trace(H @ H)
    )


def state_cov(sys_: LQRSystem, *, sigma: float, discounted: bool = False) -> np.ndarray:
    """Closed-loop state second moment.

    Undiscounted stationary:  S = A_K S A_K^T + W + sigma^2 B B^T.
    Discounted occupancy:     S_g = gamma (A_K S_g A_K^T + W + sigma^2 B B^T) + ...
    The undiscounted form is the registered choice for E2 (see the prereg): it is what
    a batch average over visited states actually computes.
    """
    A_K = sys_.A_K
    Wn = sys_.W + sigma**2 * sys_.B @ sys_.B.T
    if not discounted:
        return solve_discrete_lyapunov(A_K, Wn)
    g = sys_.gamma
    return solve_discrete_lyapunov(np.sqrt(g) * A_K, g * Wn)


def sample_states(sys_: LQRSystem, rng, n_states: int, *, sigma=None) -> np.ndarray:
    """Draw states from the closed-loop stationary law at the reference width."""
    S = state_cov(sys_, sigma=sys_.sigma_ref if sigma is None else sigma)
    L = np.linalg.cholesky(S + 1e-12 * np.eye(sys_.n))
    return rng.normal(size=(n_states, sys_.n)) @ L.T


def value_mc(sys_: LQRSystem, s0: np.ndarray, rng, *, sigma: float,
             n_traj: int, horizon: int):
    """Monte Carlo V^pi(s0) by rollout. Gate G2 only -- the sweep never calls this.

    The one end-to-end check that the environment, the discount, and the value formula
    agree. Returns (mean, standard error).
    """
    n, d, g = sys_.n, sys_.d, sys_.gamma
    s = np.broadcast_to(s0, (n_traj, n)).copy()
    tot = np.zeros(n_traj)
    disc = 1.0
    for _ in range(horizon):
        a = -s @ sys_.K.T + sigma * rng.normal(size=(n_traj, d))
        r = -(np.einsum("ti,ij,tj->t", s, sys_.Qc, s)
              + np.einsum("ti,ij,tj->t", a, sys_.Rc, a))
        tot += disc * r
        s = s @ sys_.A.T + a @ sys_.B.T + rng.normal(size=(n_traj, n))
        disc *= g
    return float(tot.mean()), float(tot.std(ddof=1) / np.sqrt(n_traj))


def dare_cost(sys_: LQRSystem, sigma: float) -> float:
    """Optimal discounted cost from the DARE solution, for E2's reference point."""
    _, P_opt = dare_gain(sys_.A, sys_.B, sys_.Qc, sys_.Rc, sys_.gamma)
    g = sys_.gamma
    return float(
        (sigma**2 * np.trace(sys_.Rc)
         + g * sigma**2 * np.trace(sys_.B.T @ P_opt @ sys_.B)
         + g * np.trace(P_opt @ sys_.W)) / (1.0 - g)
    )
