"""The planted critic error field, and the frequency it actually realises.

    e(s, a) = (eps / sqrt(r)) * sum_{j=1..r} sin( omega * v_j^T a + phi_j )

with V = (v_1..v_r) orthonormal rows in R^d, phi_j ~ U[0, 2pi) drawn per state. r = 1
is the spec's rank-one field; r = d with V = I is its full-rank robustness arm; the
rank ladder in between is what makes rank an axis rather than two points (guard R3).

WHY PLANT RATHER THAN LEARN. The paper's central quantity omega is currently an
unmeasured proxy. Here every norm of e and of grad e is available in closed form, so
omega stops being estimated and becomes an input. That is the entire reason this
experiment exists; it is also its main limitation, and the prereg says so.

THE FREQUENCY IS NOT ONE NUMBER. Claim 4 writes omega = ||grad e||_inf / ||e||_inf
without saying which norm ||grad e||_inf is. For r = 1 every reading coincides, which
is exactly why the rank-one arm cannot detect the ambiguity. For r = d they do not:

    ||e||_inf            = eps * sqrt(r)          (orthonormal rows, so all r sines
                                                   can peak at once)
    sup_a ||grad e||_2   = eps * omega
    max_j sup_a |d e/d a_j| = eps * omega / sqrt(r)     (coordinate basis)

so the realised frequency is omega/sqrt(r) under the L2 reading and omega/r under the
componentwise one. The fitted crossover exponent moves from +0.5 to 0 or to -0.5
accordingly. `omega_inf` therefore takes the convention as an explicit argument and
the sweep records all four; the registered primary is grad_norm="2", val_norm="inf",
the Lipschitz reading that matches Mohamed et al. (2020).

BLURRING IS EXACT. The Gaussian characteristic function gives

    E_u[ sin(omega v^T (mu + sigma u) + phi) ] = exp(-sigma^2 omega^2 / 2) sin(theta)

with theta = omega v^T mu + phi, since v is a unit vector so v^T u ~ N(0, 1). Hence the
full estimand grad (Q_phi)_Sigma (mu) is closed form and no Monte Carlo enters the
target anywhere.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from jax import numpy as jnp


@dataclasses.dataclass(frozen=True)
class PlantedError:
    kind: str          # "rank1" | "rank_r" | "full"
    rank: int
    omega: float
    V: np.ndarray      # (n_states, r, d) orthonormal rows
    phi: np.ndarray    # (n_states, r)


def draw_error(rng, n_states: int, d: int, *, kind: str = "rank1",
               rank: int | None = None, omega: float = 1.0) -> PlantedError:
    """Draw V and phi per state. eps is applied later, per (state, sigma)."""
    if kind == "rank1":
        r = 1
    elif kind == "full":
        r = d
    elif kind == "rank_r":
        r = int(rank)
    else:
        raise ValueError(f"unknown kind={kind!r}")
    r = max(1, min(r, d))

    if kind == "full":
        # Coordinate basis, matching the spec's e_full literally. By rotational
        # symmetry of u this is not a restriction, but it does fix what "componentwise
        # sup-norm" means, which is the whole point of the norm sweep.
        V = np.broadcast_to(np.eye(d), (n_states, d, d)).copy()
    else:
        G = rng.normal(size=(n_states, r, d))
        # Row-orthonormalise per state.
        V = np.empty_like(G)
        for i in range(n_states):
            Qm, _ = np.linalg.qr(G[i].T)          # (d, r)
            V[i] = Qm.T[:r]
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(n_states, r))
    return PlantedError(kind=kind, rank=r, omega=float(omega), V=V, phi=phi)


def _align(arr, a, xp=np):
    """Broadcast a per-state array over an action array's middle axes.

    V is (S, r, d) and phi is (S, r); an action array is (S, *mid, d) where `mid` is
    any number of replicate/sample axes. Aligning them needs explicit reshaping.

    NOT einsum with a named state index: "srd,...d->...r" drops `s` from the output and
    therefore SUMS over states, which is silently wrong and produces a plausible-looking
    array. Gate G5 caught exactly that.
    """
    nmid = a.ndim - 2
    return arr.reshape(arr.shape[0], *([1] * nmid), *arr.shape[1:])


def _project(pe: PlantedError, a, xp=np):
    """v_j^T a for every j, aligned per state.  (S, *mid, d) -> (S, *mid, r)"""
    V = xp.asarray(pe.V)
    return (_align(V, a, xp) * a[..., None, :]).sum(-1)


def theta(pe: PlantedError, mu: np.ndarray) -> np.ndarray:
    """theta_j(s) = omega v_j^T mu + phi_j.  (S, r) for mu of shape (S, d)."""
    return pe.omega * _project(pe, np.asarray(mu)) + pe.phi


def e_value(pe: PlantedError, eps, mu, sigma, u):
    """e(s, mu + sigma u). u:(S, *mid, d). Exact."""
    a = mu + sigma * u
    s = jnp.sin(pe.omega * _project(pe, a, jnp) + _align(jnp.asarray(pe.phi), a, jnp))
    return eps / jnp.sqrt(float(pe.rank)) * s.sum(-1)


def e_grad(pe: PlantedError, eps, a):
    """grad_a e(s, a). Exact.  (S, *mid, d) -> (S, *mid, d)"""
    V = jnp.asarray(pe.V)
    c = jnp.cos(pe.omega * _project(pe, a, jnp) + _align(jnp.asarray(pe.phi), a, jnp))
    amp = eps * pe.omega / jnp.sqrt(float(pe.rank))
    return amp * (c[..., None] * _align(V, a, jnp)).sum(-2)


def blurred_e_grad(pe: PlantedError, eps, mu, sigma):
    """grad (e)_Sigma (mu), in closed form. Spec Sec. 3.1.

        eps * omega / sqrt(r) * exp(-sigma^2 omega^2 / 2) * sum_j cos(theta_j) v_j
    """
    mu = np.asarray(mu)
    th = theta(pe, mu)
    c = pe.omega * sigma
    amp = eps * pe.omega / np.sqrt(float(pe.rank)) * np.exp(-0.5 * c * c)
    w = amp * np.cos(th) if np.ndim(amp) == 0 else np.asarray(amp)[..., None] * np.cos(th)
    return (w[..., None] * pe.V).sum(-2)


def eps_for_fraction(q_spread, frac: float):
    """Error amplitude as a fixed fraction of the within-state Q spread.

    Fixed relative to the MEASURED spread, never as an absolute number, or the sweep is
    not comparable across d (guard R2). The realised fraction is recorded per row.
    """
    return frac * np.asarray(q_spread)


def e_sup_norm(eps, rank: int):
    """||e||_inf = eps * sqrt(r): orthonormal rows let all r sines peak together."""
    return np.asarray(eps) * np.sqrt(float(rank))


def e_rms_norm(eps, rank: int, c, th):
    """sd over a ~ pi(.|s) of e, exactly: eps * sqrt(mean_j Vsin(c, theta_j))."""
    from scripts.lqr_crossover.reference import vsin
    return np.asarray(eps) * np.sqrt(np.mean(vsin(c, th), axis=-1))


def omega_inf(pe: PlantedError, *, grad_norm: str = "2", val_norm: str = "inf",
              c=None, th=None) -> float:
    """The frequency this field ACTUALLY realises, under a stated norm convention.

    grad_norm="2"   : sup_a ||grad e||_2      = eps * omega
    grad_norm="inf" : max_j sup_a |d_j e|     = eps * omega / sqrt(r)
    val_norm="inf"  : ||e||_inf               = eps * sqrt(r)
    val_norm="rms"  : sd_{a~pi} e             = eps * sqrt(mean_j Vsin)

    eps cancels in every combination, which is why the E1a contour is eps-independent.
    Returns a scalar for val_norm="inf" and a per-state array for "rms".
    """
    r = float(pe.rank)
    num = pe.omega if grad_norm == "2" else pe.omega / np.sqrt(r)
    if val_norm == "inf":
        return num / np.sqrt(r)
    if val_norm == "rms":
        from scripts.lqr_crossover.reference import vsin
        return num / np.sqrt(np.mean(vsin(c, th), axis=-1))
    raise ValueError(f"unknown val_norm={val_norm!r}")


ALL_CONVENTIONS = (("2", "inf"), ("inf", "inf"), ("2", "rms"), ("inf", "rms"))
PRIMARY_CONVENTION = ("2", "inf")
