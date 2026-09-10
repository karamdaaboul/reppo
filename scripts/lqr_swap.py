"""Where pathwise and zeroth order swap: against error size, against dimension, blind.

    python scripts/lqr_swap.py --stage {gates,components,measure,figures,report}

Registered in docs/prereg_lqr_swap.md. Read it before changing anything here. The grids,
the metrics, the gate tolerances and the three predictions are frozen there, and the
predictions are committed below its amendment line before any measurement runs.

WHAT THIS ADDS. The path study confirmed four panels and left three limits.
Graph A measures how the swap point moves with the size of the critic error. Graph B
measures the whole-path swap against the action dimension, the sqrt(d) question that
d = 2 could not ask. Graph C predicts a swap from separately measured parts, commits the
number, and then runs paths on both sides of it.

THE MODEL, AND WHY IT CAN FAIL. Both operators are exactly linear in the critic, so
with the error scaled by m the total error splits exactly into a wave-free part, an
error-channel part carrying m^2, and a cross term carrying m. The prediction drops the
cross term. The crossover study found that term is not negligible at d <= 4, so dropping
it is precisely what the blind test puts at risk.

WHAT IS NOT HERE. No E-step arm, so the conditioning that failed the path study's G0a
never enters. No difference of `lqr.q_pi`, so the cancellation that failed its G0b
cannot recur. No quadrature, so the comparator that failed its G0c is absent. Every
equivalence gate names its denominator, which amendment A0 explains at length.

The per-step primitives come from scripts/lqr_paths.py, which imports the harness. This
file composes them and adds no estimator arithmetic of its own.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import scripts.lqr_crossover  # noqa: F401,E402  (platform + x64, must be first)

import jax  # noqa: E402
import numpy as np  # noqa: E402
from jax import numpy as jnp  # noqa: E402

import scripts.lqr_paths as LP  # noqa: E402
from scripts.lqr_crossover import SEED_ROOT  # noqa: E402
from scripts.lqr_crossover import analyze as A  # noqa: E402
from scripts.lqr_crossover import error_field as EF  # noqa: E402
from scripts.lqr_crossover import estimators as E  # noqa: E402
from scripts.lqr_crossover import lqr  # noqa: E402
from scripts.lqr_crossover import reference as R  # noqa: E402
from scripts.lqr_crossover import sweep as SW  # noqa: E402

# ---------------------------------------------------------------- frozen constants
M = LP.M                                 # 32 samples per estimator call
DS = (2, 4, 8, 16, 32, 64)               # Graph B dimensions
MS_A = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
M_B = 30.0                               # Graph B error size
M_C = 3.0                                # Graph C error size
GRID = np.logspace(-1.0, 3.0, 41)        # sigma*omega, the one-step grid
RATIOS = 2.0 ** np.linspace(-2.0, 2.0, 9)   # 0.25 .. 4 in half octaves, 9 points
REPS = 10_000
CHUNK = 1000                             # replicate chunk, memory only
PERSIST = 3                              # a crossing must stay below for 3 more points
NBOOT = 10_000
BOOT_SEED = 20260902
ARMS = ("pw", "zo")                      # no E-step arm in this study
TOL_G0 = 1e-13
G1_Z = 4.0
G3_TOL = 0.02
PUBLISHED_FULL_D2_CSTAR = 1.522          # reports/lqr_crossover_corrected.md Sec. 7.1
PATH_STUDY_TIE_D2 = 1.5387               # docs/prereg_lqr_paths.md A1, gate G3a
PATH_STUDY_ANGULAR = {1.0: 39.985, 10.0: 4.1721, 30.0: 1.9345}   # same file, A1

FIG_DIR = os.path.join(REPO_ROOT, "reports", "figures")
NAVY, BRICK, GRAY = "#1F4E79", "#C0504D", "#9a9a9a"
FIG_SIZE = (3.4, 2.6)


# ------------------------------------------------------------------ small utilities
def traced_pe(su, omega):
    """The planted field at a frequency that may be a tracer.

    `lqr_paths.at_omega` casts with `float()`, which a traced omega cannot survive.
    Building the frozen dataclass directly lets one compiled step function serve a whole
    grid of frequencies, which is the difference between 12 compilations and 108.
    """
    return EF.PlantedError(su.pe.kind, su.pe.rank, omega, su.pe.V, su.pe.phi)


def swap_from(xs: np.ndarray, diff: np.ndarray, persist: int = PERSIST):
    """First x at which `diff` turns negative and stays negative for `persist` more.

    `diff` is ZO's metric minus PW's. The persistence requirement is what stops one
    noisy cell from being read as a crossing. Returns (x, status).
    """
    n = len(diff)
    for k in range(n):
        if diff[k] >= 0:
            continue
        if k == 0:
            return float("nan"), "below_grid"
        if not np.all(diff[k:min(k + 1 + persist, n)] < 0):
            continue
        if n - k <= persist:
            return float("nan"), "too_close_to_edge"
        lo, hi = diff[k - 1], diff[k]
        f = lo / (lo - hi)
        lx = np.log(xs)
        return float(np.exp(lx[k - 1] + f * (lx[k] - lx[k - 1]))), "ok"
    return float("nan"), "no_crossing"


def rho_swap(xs: np.ndarray, rho: np.ndarray):
    """The sigma*omega at which rho = 1, by log-log interpolation. (x, status)."""
    lr = np.log(np.maximum(rho, 1e-300))
    lx = np.log(xs)
    for k in range(1, len(lr)):
        if lr[k - 1] > 0 >= lr[k]:
            f = lr[k - 1] / (lr[k - 1] - lr[k])
            return float(np.exp(lx[k - 1] + f * (lx[k] - lx[k - 1]))), "ok"
    return float("nan"), ("all_above" if lr[-1] > 0 else "all_below")


def _sqerr(g, target):
    return np.asarray(jnp.sum((g - target) ** 2, axis=-1))


# ----------------------------------------------------------------- the setups, by d
_SETUPS: dict[int, LP.Setup] = {}


def setup_d(d: int) -> LP.Setup:
    if d not in _SETUPS:
        _SETUPS[d] = LP.setup(d)
    return _SETUPS[d]


def block(su: LP.Setup, offset: int, reps: int = REPS):
    """One replicate block of whitened draws. Never materialised beyond a chunk."""
    return jax.random.PRNGKey(SEED_ROOT + offset + su.d), reps


def _chunks(reps: int, chunk: int):
    i = 0
    while i < reps:
        n = min(chunk, reps - i)
        yield i, n
        i += n


# ------------------------------------------------------------------------ components
def components(su: LP.Setup, offset: int = 10000) -> dict:
    """V^s, V^e and the cross term C, from one pass over shared draws at eps_study.

    Both arms are unbiased for the exact blurred gradient, so every quantity here is a
    variance or a covariance about an exact mean and never about a sampled one. The
    error channel is taken as the paired difference g(Q_phi) - g(Q^pi) on identical
    draws, which by linearity is exactly g(e).
    """
    key, reps = block(su, offset)
    sg, d = su.sigma, su.d
    g_sm = LP.g_lin_np(su, su.mu0)                      # exact smooth target at mu_0
    n_x = len(GRID)
    acc_s = np.zeros(2)
    acc_e = np.zeros((n_x, 2))
    acc_c = np.zeros((n_x, 2))
    tgt_e = np.stack([EF.blurred_e_grad(LP.at_omega(su, float(x) / sg), su.eps_study,
                                        su.mu0[None, :], sg)[0] for x in GRID])

    for i0, n in _chunks(reps, CHUNK):
        u = jax.random.normal(jax.random.fold_in(key, i0), (n, M, d))
        q0 = LP.make_q_of_u(su, jnp.asarray(su.mu0), 1.0, 0.0)
        o0 = E.both_from_shared_u(q0, u, sg, axis=1)
        g0 = {a: o0["g_pw" if a == "pw" else "g_zo"] for a in ARMS}
        Ds = {a: np.asarray(g0[a]) - g_sm[None, :] for a in ARMS}
        for j, a in enumerate(ARMS):
            acc_s[j] += float(np.sum(Ds[a] ** 2))
        for k, x in enumerate(GRID):
            om = float(x) / sg
            q1 = LP.make_q_of_u(su, jnp.asarray(su.mu0), om, su.eps_study)
            o1 = E.both_from_shared_u(q1, u, sg, axis=1)
            for j, a in enumerate(ARMS):
                g1a = np.asarray(o1["g_pw" if a == "pw" else "g_zo"])
                ge = g1a - np.asarray(g0[a])
                De = ge - tgt_e[k][None, :]
                acc_e[k, j] += float(np.sum(De ** 2))
                acc_c[k, j] += float(np.sum(De * Ds[a]))
    return dict(V_s=(acc_s / reps).tolist(), V_e=(acc_e / reps).tolist(),
                C=(acc_c / reps).tolist(), grid=GRID.tolist(), reps=reps,
                d=d, sigma=sg, eps_study=su.eps_study,
                target_dist=su.target_dist, state_index=su.idx)


def predicted_swap(comp: dict, m: float):
    """The model's swap: V_ZO^s + m^2 V_ZO^e = V_PW^s + m^2 V_PW^e. C is dropped."""
    Vs = np.asarray(comp["V_s"])
    Ve = np.asarray(comp["V_e"])
    diff = (Vs[1] - Vs[0]) + m * m * (Ve[:, 1] - Ve[:, 0])
    return swap_from(np.asarray(comp["grid"]), diff)


def error_only_tie(comp: dict):
    Ve = np.asarray(comp["V_e"])
    return swap_from(np.asarray(comp["grid"]), Ve[:, 1] - Ve[:, 0])


# --------------------------------------------------------------------- one-step sweep
def measure_one_step(su: LP.Setup, ms, offset: int = 11000):
    """Total one-step MSE per replicate, per grid point, per arm, at each error size.

    An independent draw block from `components`, so that the comparison of Section 6 is
    not the same randomness the prediction of Section 5 was built from.
    """
    key, reps = block(su, offset)
    sg, d = su.sigma, su.d
    n_x, n_m = len(GRID), len(ms)
    se = np.zeros((reps, n_x, n_m, 2))          # squared error
    cs = np.zeros((reps, n_x, n_m, 2))          # cosine, for the angular bridge
    for k, x in enumerate(GRID):
        om = float(x) / sg
        pe_o = LP.at_omega(su, om)
        for jm, m in enumerate(ms):
            eps = m * su.eps_study
            tgt = LP.g_lin_np(su, su.mu0) + EF.blurred_e_grad(
                pe_o, eps, su.mu0[None, :], sg)[0]
            tj = jnp.asarray(tgt)
            nt = float(np.linalg.norm(tgt))
            for i0, n in _chunks(reps, CHUNK):
                u = jax.random.normal(jax.random.fold_in(key, i0), (n, M, d))
                o = E.both_from_shared_u(
                    LP.make_q_of_u(su, jnp.asarray(su.mu0), om, eps), u, sg, axis=1)
                for j, a in enumerate(ARMS):
                    g = o["g_pw" if a == "pw" else "g_zo"]
                    se[i0:i0 + n, k, jm, j] = _sqerr(g, tj)
                    cs[i0:i0 + n, k, jm, j] = np.asarray(
                        jnp.sum(g * tj, -1)
                        / jnp.maximum(jnp.linalg.norm(g, axis=-1) * nt, 1e-300))
    return se, cs


def bootstrap_one_step(se: np.ndarray, ms, nboot=NBOOT, seed=BOOT_SEED):
    """Percentile intervals for the measured swap, resampling replicates jointly.

    The resample is a multinomial reweighting of the replicates, applied to every grid
    point and every error size at once, so the whole swap-against-m curve moves as one.
    """
    reps, n_x, n_m, _ = se.shape
    dif = (se[..., 1] - se[..., 0]).reshape(reps, n_x * n_m)
    rng = np.random.default_rng(seed)
    out = np.full((nboot, n_m), np.nan)
    done = 0
    while done < nboot:
        nb = min(2000, nboot - done)
        W = rng.multinomial(reps, np.full(reps, 1.0 / reps), size=nb) / reps
        means = (W @ dif).reshape(nb, n_x, n_m)
        for b in range(nb):
            for jm in range(n_m):
                out[done + b, jm] = swap_from(GRID, means[b, :, jm])[0]
        done += nb
    return out


# ------------------------------------------------------------------------- path runs
def make_runner(su: LP.Setup, arm: str, n_steps: int):
    """jit(vmap(scan)) with the frequency and the amplitude as traced arguments."""
    base = jax.random.PRNGKey(SEED_ROOT + 8000 + su.d)
    mu0_j = jnp.asarray(su.mu0)
    sg, d = su.sigma, su.d

    def one_seed(seed, omega, eps):
        key0 = jax.random.fold_in(base, seed)
        pe_o = traced_pe(su, omega)

        def body(m, t):
            u = jax.random.normal(jax.random.fold_in(key0, t), (M, d))
            q = LP.make_q_of_u(su, m, omega, eps, pe=pe_o)
            gh = LP._unit(LP.arm_direction(arm, q, u, sg, axis=0))
            m2 = m + LP.STEP_FRAC * sg * gh
            return m2, m2

        _, mus = jax.lax.scan(body, mu0_j, jnp.arange(n_steps))
        return mus

    return jax.jit(jax.vmap(one_seed, in_axes=(0, None, None)))


def run_grid(su: LP.Setup, xs, m: float, n_steps_by_x) -> dict:
    """Both arms, 100 seeds, every grid point. Nothing is compared until all finish."""
    seeds = jnp.arange(LP.R_SEEDS)
    eps = m * su.eps_study
    J = np.zeros((len(xs), 2, LP.R_SEEDS))
    runners = {}
    for k, x in enumerate(xs):
        om = float(x) / su.sigma
        n = int(n_steps_by_x[k])
        ref = LP.reference_path(su, om, eps, n)
        for j, a in enumerate(ARMS):
            if (a, n) not in runners:
                runners[(a, n)] = make_runner(su, a, n)
            mus = np.asarray(runners[(a, n)](seeds, om, eps))
            J[k, j] = np.linalg.norm(mus - ref[None], axis=-1).mean(1) / su.sigma
    return dict(J=J, xs=np.asarray(xs), m=m, d=su.d)


def bootstrap_paths(Js: dict, nboot=NBOOT, seed=BOOT_SEED):
    """One resample of the seed identifiers, applied to every grid point and every d."""
    rng = np.random.default_rng(seed)
    ds = sorted(Js)
    out = {d: np.full(nboot, np.nan) for d in ds}
    done = 0
    while done < nboot:
        nb = min(2000, nboot - done)
        idx = rng.integers(0, LP.R_SEEDS, (nb, LP.R_SEEDS))
        for d in ds:
            J = Js[d]["J"]                       # (n_x, 2, R)
            med = np.median(J[:, :, idx], axis=-1)      # (n_x, 2, nb)
            rho = med[:, 1, :] / med[:, 0, :]           # (n_x, nb)
            for b in range(nb):
                out[d][done + b] = rho_swap(Js[d]["xs"], rho[:, b])[0]
        done += nb
    return out


# ---------------------------------------------------------------------------- gates
class _Dev:
    """Three denominators for one equivalence check, per prereg amendment A0.

    `scaled` is the scored one: max|x - y| / max|y|, both maxima over the whole test
    set, so no single near-zero item can inflate it. `vec` divides each item's deviation
    by that item's norm and `comp` divides component by component. Both collapse when
    the quantity under test approaches zero, which is what failed this gate as first
    registered. All three are printed and stored.
    """

    def __init__(self):
        self.dev = self.mag = 0.0
        self.vec = self.comp = 0.0
        self.minfrac = np.inf

    def add(self, x, y):
        x, y = np.atleast_2d(x), np.atleast_2d(y)
        d = np.abs(x - y)
        n = np.linalg.norm(y, axis=-1)
        self.dev = max(self.dev, float(d.max()))
        self.mag = max(self.mag, float(np.abs(y).max()))
        self.vec = max(self.vec, float(np.max(d.max(-1) / np.maximum(n, 1e-300))))
        self.comp = max(
            self.comp, float(np.max(d / np.maximum(np.abs(y), 1e-300))))
        self.minfrac = min(self.minfrac,
                           float(np.min(np.abs(y).min(-1) / np.maximum(n, 1e-300))))

    @property
    def scaled(self):
        return self.dev / max(self.mag, 1e-300)


def _add_equivalence(G, name, stats: dict):
    ok = all(v.scaled <= TOL_G0 for v in stats.values())
    reg = all(v.comp <= TOL_G0 for v in stats.values())
    detail = "; ".join(
        f"{k}: scaled {v.scaled:.2e}, per vector {v.vec:.2e}, per component "
        f"{v.comp:.2e}, min component/vector {v.minfrac:.1e}"
        for k, v in stats.items())
    G.add(name, "PASS" if ok else "FAIL",
          f"bound {TOL_G0:.0e} on max|x-y|/max|y| (A0); " + detail,
          registered="PASS" if reg else "FAIL",
          detail_registered="scored per component: "
                            + "; ".join(f"{k} {v.comp:.2e}" for k, v in stats.items()))


def gate_g0a(su: LP.Setup, G: LP.Gate):
    """PW and ZO, as this experiment calls them, against the harness composition."""
    keys = jax.random.split(jax.random.PRNGKey(SEED_ROOT + 12000 + su.d), 100)
    pe_o = LP.at_omega(su, 1.0)
    qf = lqr.q_of_u_factory(su.sys_, su.s[None, :], su.sigma)
    mu3 = jnp.asarray(su.mu0)[None, None, :]
    eps3 = jnp.asarray(np.array([su.eps_study]))[:, None]

    def q_h(uu):
        return qf(uu) + EF.e_value(pe_o, eps3, mu3, su.sigma, uu)

    q_mine = LP.make_q_of_u(su, jnp.asarray(su.mu0), 1.0, su.eps_study)
    st = {a: _Dev() for a in ARMS}
    for k in keys:
        u = jax.random.normal(k, (M, su.d))
        for a in ARMS:
            x = np.asarray(LP.arm_direction(a, q_mine, u, su.sigma, axis=0))
            y = np.asarray(
                LP.arm_direction(a, q_h, u[None], su.sigma, axis=1)).reshape(su.d)
            st[a].add(x, y)
    _add_equivalence(G, f"G0a arms reproduce the harness [d={su.d}]", st)


def gate_g0b(su: LP.Setup, G: LP.Gate):
    """The moving-mean closure's gradient against the harness's exact action gradient.

    Constant free by construction: no difference of `lqr.q_pi` is formed anywhere, which
    is the comparison that failed in the path study.
    """
    rng = np.random.default_rng(SEED_ROOT + 12100 + su.d)
    ms = su.mu0 + np.linspace(0.0, 1.0, 10)[:, None] * (su.a_star - su.mu0)
    u = jnp.asarray(rng.normal(size=(100, su.d)))
    st = {"grad": _Dev()}
    for m in ms:
        m_j = jnp.asarray(m)
        q = LP.make_q_of_u(su, m_j, 1.0, 0.0)
        _, pull = jax.vjp(q, u)
        (got,) = pull(jnp.ones(u.shape[0]))
        want = su.sigma * su.alpha_Q * lqr.grad_a_q_pi(
            su.sys_, su.ss, np.asarray(m + su.sigma * np.asarray(u)))
        st["grad"].add(np.asarray(got), want)
    _add_equivalence(G, f"G0b moving-mean closure gradient [d={su.d}]", st)


def gate_g0c(G: LP.Gate, out_dir: str):
    """The dimension refactor changes nothing at d = 2.

    The reference is the committed run itself, not a transcribed literal: every panel
    .npz carries the sigma, eps and mu_0 it was produced with, so the comparison is
    bitwise against the artifact rather than against a rounded number in this file.
    """
    su = setup_d(2)
    src = os.path.join(REPO_ROOT, "results", "lqr_paths")
    ok_setup, det, ok_arrays = True, [], True
    for name in ("study_slow", "big_slow", "study_fast", "big_fast"):
        p = os.path.join(src, f"paths_{name}.npz")
        if not os.path.exists(p):
            det.append(f"{name}: absent")
            ok_arrays = False
            continue
        z = np.load(p, allow_pickle=True)
        ok_setup &= (su.sigma == float(z["sigma"])
                     and su.eps_study == float(z["eps_study_ref"])
                     if "eps_study_ref" in z else su.sigma == float(z["sigma"]))
        ok_setup &= bool((su.mu0 == z["mu0"]).all())
        run = LP.make_runner(su, "pw", float(z["omega"]), float(z["eps"]),
                             int(z["n_steps"]))
        mus, _ = run(jnp.arange(LP.R_SEEDS))
        got = np.asarray(mus)
        same = bool((got == z["mu_pw"]).all())
        mx = float(np.max(np.abs(got - z["mu_pw"])))
        det.append(f"{name}: bitwise {same}, max abs {mx:.1e}")
        ok_arrays &= same
    G.add("G0c dimension refactor is a no-op at d = 2",
          "PASS" if (ok_setup and ok_arrays) else "FAIL",
          f"sigma and mu_0 match the artifacts: {ok_setup}; "
          "committed path arrays " + "; ".join(det))


def gate_g1(su: LP.Setup, G: LP.Gate, xs, label=""):
    """Unbiasedness at mu_0, at the frequencies this experiment actually uses."""
    key, _ = block(su, 10000)
    det, ok = [], True
    for x in xs:
        om = float(x) / su.sigma if x is not None else 1.0
        eps = 0.0 if x is None else su.eps_study * M_B
        tgt = LP.g_lin_np(su, su.mu0)
        if x is not None:
            tgt = tgt + EF.blurred_e_grad(LP.at_omega(su, om), eps,
                                          su.mu0[None, :], su.sigma)[0]
        acc = {a: [] for a in ARMS}
        for i0, n in _chunks(4000, CHUNK):
            u = jax.random.normal(jax.random.fold_in(key, 900000 + i0), (n, M, su.d))
            o = E.both_from_shared_u(
                LP.make_q_of_u(su, jnp.asarray(su.mu0), om, eps), u, su.sigma, axis=1)
            for a in ARMS:
                acc[a].append(np.asarray(o["g_pw" if a == "pw" else "g_zo"]))
        zz = []
        for a in ARMS:
            g = np.concatenate(acc[a])
            sem = g.std(0, ddof=1) / np.sqrt(len(g))
            z = (g.mean(0) - tgt) / np.maximum(sem, 1e-300)
            zz.append(float(np.max(np.abs(z))))
        ok &= max(zz) <= G1_Z
        det.append(f"{'eps=0' if x is None else f'x={x:.3g}'}: "
                   f"|z| PW {zz[0]:.2f} ZO {zz[1]:.2f}")
    G.add(f"G1 unbiased at mu_0 [d={su.d}{label}]", "PASS" if ok else "FAIL",
          f"bound {G1_Z:.0f} SE; " + "; ".join(det))


def gate_g2(su: LP.Setup, G: LP.Gate, xs, m: float, label: str):
    """The reference path reaches a stationary point, at every grid point."""
    eps = m * su.eps_study
    n_by_x, det, ok = [], [], True
    for x in xs:
        om = float(x) / su.sigma
        for n in (LP.N_STEPS, LP.N_STEPS_G2):
            ref = LP.reference_path(su, om, eps, n)
            tail = ref[-20:]
            rad = float(np.max(np.linalg.norm(tail - tail.mean(0), axis=-1)) / su.sigma)
            if rad <= 1.0:
                break
        n_by_x.append(n)
        ok &= rad <= 1.0
        det.append(f"{rad:.2f}@{n}")
    G.add(f"G2 reference paths settled [{label}]", "PASS" if ok else "FAIL",
          "tail radius in sigma at N: " + " ".join(det) + " (bound 1.0)")
    return n_by_x


def gate_g3(G: LP.Gate, comp2: dict):
    """The published full-rank d = 2 crossover, and this study's own tie beside it."""
    name = f"d2_full_M{M}_unit_H_identity_paths_g3b"
    path = os.path.join(scripts.lqr_crossover.OUT, name + ".npz")
    if not os.path.exists(path):
        SW.run_d(2, M=M, kind="full", n_states=LP.N_STATES, n_batch=20, r_batch=100,
                 tag="_paths_g3b")
    z = A.load(path)
    c, ok_root = A.crossover_by_c(z)
    rel = abs(c / PUBLISHED_FULL_D2_CSTAR - 1.0)
    tie, st = error_only_tie(comp2)
    G.add("G3 published full-rank d = 2 crossover",
          "PASS" if (ok_root and rel <= G3_TOL) else "FAIL",
          f"c* = {c:.4f} against {PUBLISHED_FULL_D2_CSTAR:.3f}, rel {rel:.4f} "
          f"(bound {G3_TOL}); this study's own tie {tie:.4f} ({st}) against the path "
          f"study's single-state {PATH_STUDY_TIE_D2:.4f}, "
          f"rel {abs(tie / PATH_STUDY_TIE_D2 - 1):.4f} (descriptive)")


# --------------------------------------------------------------------------- stages
def stage_gates(out_dir: str):
    G = LP.Gate()
    print("Gates (prereg Sec. 7).", jax.devices(), jnp.zeros(1).dtype)
    for d in DS:
        su = setup_d(d)
        print(f"  d={d:2d}  state {su.idx}  sigma {su.sigma:.6g}  "
              f"eps_study {su.eps_study:.6g}  ||a*-mu0|| {su.target_dist:.6g}")
        gate_g0a(su, G)
        gate_g0b(su, G)
    gate_g0c(G, out_dir)
    rec = dict(rows=G.rows,
               setups={d: dict(state_index=setup_d(d).idx, sigma=setup_d(d).sigma,
                               eps_study=setup_d(d).eps_study,
                               target_dist=setup_d(d).target_dist) for d in DS})
    with open(os.path.join(out_dir, "gates.json"), "w") as f:
        json.dump(rec, f, indent=2, default=float)
    print(("\nALL GATES PASS" if G.ok() else "\nGATE FAILURE") + " -> gates.json")
    return G.ok()


def stage_components(out_dir: str):
    G = LP.Gate()
    comps, preds = {}, {}
    for d in DS:
        su = setup_d(d)
        t0 = time.time()
        comps[d] = components(su)
        tie, st_tie = error_only_tie(comps[d])
        sw30, st30 = predicted_swap(comps[d], M_B)
        preds[d] = dict(tie=tie, tie_status=st_tie, swap30=sw30, swap30_status=st30)
        print(f"  d={d:2d}  tie {tie:.4f} ({st_tie})  predicted swap(m=30) "
              f"{sw30:.4f} ({st30})  [{time.time() - t0:.1f}s]", flush=True)

    su2 = setup_d(2)
    comp2 = comps[2]
    swaps_m = {}
    for m in MS_A:
        x, st = predicted_swap(comp2, m)
        swaps_m[m] = dict(swap=x, status=st)
        print(f"  d= 2  m={m:6.1f}  predicted swap {x:.4f} ({st})")
    swC, stC = predicted_swap(comp2, M_C)

    gridB = {d: (RATIOS * preds[d]["swap30"]).tolist() for d in DS}
    gridC = (RATIOS * swC).tolist()

    gate_g3(G, comp2)
    for d in DS:
        gate_g1(setup_d(d), G, [None, preds[d]["tie"], preds[d]["swap30"]])
    nB = {d: gate_g2(setup_d(d), G, gridB[d], M_B, f"B d={d}") for d in DS}
    nC = gate_g2(su2, G, gridC, M_C, "C d=2 m=3")

    rec = dict(rows=G.rows, components={str(d): comps[d] for d in DS},
               predicted_swap_m={str(m): swaps_m[m] for m in MS_A},
               predicted_swap_C=dict(swap=swC, status=stC, m=M_C),
               tie={str(d): preds[d]["tie"] for d in DS},
               predicted_swap30={str(d): preds[d]["swap30"] for d in DS},
               grid_B={str(d): gridB[d] for d in DS}, grid_C=gridC,
               n_steps_B={str(d): nB[d] for d in DS}, n_steps_C=nC,
               ratios=RATIOS.tolist())
    with open(os.path.join(out_dir, "components.json"), "w") as f:
        json.dump({k: v for k, v in rec.items() if k != "components"}, f,
                  indent=2, default=float)
    with open(os.path.join(out_dir, "components_full.json"), "w") as f:
        json.dump({str(d): {k: v for k, v in comps[d].items() if k != "C"}
                   for d in DS}, f, indent=2, default=float)
    # The cross term is written apart, and the report reads it only after the verdicts.
    with open(os.path.join(out_dir, "crossterm.json"), "w") as f:
        json.dump({str(d): comps[d]["C"] for d in DS}, f, indent=2, default=float)
    print(("\nALL GATES PASS" if G.ok() else "\nGATE FAILURE") + " -> components.json")
    return G.ok()


def stage_measure(out_dir: str):
    with open(os.path.join(out_dir, "components.json")) as f:
        comp = json.load(f)
    for name in ("gates.json", "components.json"):
        with open(os.path.join(out_dir, name)) as f:
            rows = json.load(f)["rows"]
        bad = [r["name"] for r in rows if r["blocking"] and r["status"] != "PASS"]
        if bad:
            sys.exit(f"refusing to measure: blocking gates failing in {name}: {bad}")

    su2 = setup_d(2)
    t0 = time.time()
    se, cs = measure_one_step(su2, MS_A)
    swaps = {}
    for jm, m in enumerate(MS_A):
        x, st = swap_from(GRID, (se[..., 1] - se[..., 0]).mean(0)[:, jm])
        swaps[m] = dict(swap=x, status=st)
    boots = bootstrap_one_step(se, MS_A)
    ang = {}
    for jm, m in enumerate(MS_A):
        if m in PATH_STUDY_ANGULAR:
            ang[m] = float(LP.crossing(GRID, (cs[..., 0] - cs[..., 1]).mean(0)[:, jm]))
    rec_a = dict(
        ms=list(MS_A), grid=GRID.tolist(),
        measured={str(m): swaps[m] for m in MS_A},
        ci={str(m): [float(np.nanpercentile(boots[:, jm], 2.5)),
                     float(np.nanpercentile(boots[:, jm], 97.5)),
                     int(np.isfinite(boots[:, jm]).sum())]
            for jm, m in enumerate(MS_A)},
        mse_pw={str(m): se[..., 0].mean(0)[:, jm].tolist()
                for jm, m in enumerate(MS_A)},
        mse_zo={str(m): se[..., 1].mean(0)[:, jm].tolist()
                for jm, m in enumerate(MS_A)},
        angular_bridge={str(m): ang[m] for m in ang},
        angular_path_study={str(m): v for m, v in PATH_STUDY_ANGULAR.items()},
        seconds=time.time() - t0)
    with open(os.path.join(out_dir, "graph_a.json"), "w") as f:
        json.dump(rec_a, f, indent=2, default=float)
    print(f"  Graph A done ({time.time() - t0:.1f}s): "
          + " ".join(f"m={m:g}:{swaps[m]['swap']:.3f}" for m in MS_A), flush=True)

    Js = {}
    for d in DS:
        t1 = time.time()
        xs = comp["grid_B"][str(d)]
        Js[d] = run_grid(setup_d(d), xs, M_B, comp["n_steps_B"][str(d)])
        print(f"  Graph B d={d:2d} done ({time.time() - t1:.1f}s)", flush=True)
    JC = run_grid(su2, comp["grid_C"], M_C, comp["n_steps_C"])
    np.savez_compressed(
        os.path.join(out_dir, "paths.npz"),
        **{f"J_d{d}": Js[d]["J"] for d in DS},
        **{f"xs_d{d}": Js[d]["xs"] for d in DS},
        J_C=JC["J"], xs_C=JC["xs"], git_sha=LP.git_sha(),
        script_sha256=LP.sha256_file(os.path.abspath(__file__)))
    print(f"  Graph C done; all seeds finished ({time.time() - t0:.1f}s)")
    return True


def _analyse(out_dir: str):
    z = np.load(os.path.join(out_dir, "paths.npz"), allow_pickle=True)
    Js = {d: dict(J=z[f"J_d{d}"], xs=z[f"xs_d{d}"]) for d in DS}
    JC = dict(J=z["J_C"], xs=z["xs_C"])
    res = {}
    for d in DS:
        med = np.median(Js[d]["J"], axis=-1)
        rho = med[:, 1] / med[:, 0]
        x, st = rho_swap(Js[d]["xs"], rho)
        res[d] = dict(rho=rho.tolist(), swap=x, status=st, xs=Js[d]["xs"].tolist())
    bs = bootstrap_paths(Js)
    for d in DS:
        b = bs[d][np.isfinite(bs[d])]
        res[d]["ci"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)),
                        int(len(b))]
    # slope of log(path swap) on log d, with the same joint resample
    ds = np.array([d for d in DS if np.isfinite(res[d]["swap"])], float)
    ys = np.array([res[d]["swap"] for d in DS if np.isfinite(res[d]["swap"])])
    slope = float(np.polyfit(np.log(ds), np.log(ys), 1)[0]) if len(ds) >= 2 else np.nan
    sl = []
    for b in range(len(bs[DS[0]])):
        yy = np.array([bs[d][b] for d in DS], float)
        msk = np.isfinite(yy)
        if msk.sum() >= 2:
            sl.append(np.polyfit(np.log(np.array(DS, float)[msk]),
                                 np.log(yy[msk]), 1)[0])
    sl = np.array(sl)
    med = np.median(JC["J"], axis=-1)
    rhoC = med[:, 1] / med[:, 0]
    rng = np.random.default_rng(BOOT_SEED)
    ciC = []
    for k in range(len(rhoC)):
        b = []
        for _ in range(NBOOT // 10):
            i = rng.integers(0, LP.R_SEEDS, LP.R_SEEDS)
            b.append(np.median(JC["J"][k, 1, i]) / np.median(JC["J"][k, 0, i]))
        ciC.append([float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
    return dict(B={str(d): res[d] for d in DS}, slope=slope,
                slope_ci=[float(np.percentile(sl, 2.5)),
                          float(np.percentile(sl, 97.5)), int(len(sl))],
                C=dict(xs=JC["xs"].tolist(), rho=rhoC.tolist(), ci=ciC))


def stage_diag(out_dir: str):
    """Two diagnostics asked for after the blind commit. Neither changes a criterion.

    First, the structure of G1's z-scores by dimension and arm. The gate is a maximum
    over many coordinates, so a value near its bound of four is what a maximum of that
    many standard normals does on its own. The mean of z^2 is the calibration statistic
    that a maximum cannot give: it is 1 when the arm is unbiased and the standard error
    is right, whatever the number of coordinates.

    Second, an attribution of the 4.4 per cent gap between this study's error-only tie
    at d = 2 and the path study's. The registered tie stays 1.4711; nothing here
    replaces it.
    """
    out = {"g1": [], "tie": {}}
    print("G1 z-scores by dimension and arm. mean z^2 is 1 under the null.")
    print(f"{'d':>3} {'reps':>6} {'arm':>4} {'max|z|':>7} {'at':>22} {'mean z^2':>9} "
          f"{'n':>5}")
    for d in DS:
        su = setup_d(d)
        key, _ = block(su, 10000)
        tie = json.load(open(os.path.join(out_dir, "components.json")))["tie"][str(d)]
        sw = json.load(open(os.path.join(out_dir, "components.json"))
                       )["predicted_swap30"][str(d)]
        cfgs = [("eps=0", None, 0.0), (f"tie={tie:.3f}", tie, M_B * su.eps_study),
                (f"swap30={sw:.3f}", sw, M_B * su.eps_study)]
        for reps, tag in ((4000, "4000"), (REPS, "10000")):
            for a in ARMS:
                zs, where = [], None
                best = -1.0
                for nm, x, eps in cfgs:
                    om = 1.0 if x is None else float(x) / su.sigma
                    tgt = LP.g_lin_np(su, su.mu0)
                    if x is not None:
                        tgt = tgt + EF.blurred_e_grad(
                            LP.at_omega(su, om), eps, su.mu0[None, :], su.sigma)[0]
                    acc = []
                    for i0, n in _chunks(reps, CHUNK):
                        u = jax.random.normal(
                            jax.random.fold_in(key, 900000 + i0), (n, M, su.d))
                        o = E.both_from_shared_u(
                            LP.make_q_of_u(su, jnp.asarray(su.mu0), om, eps), u,
                            su.sigma, axis=1)
                        acc.append(np.asarray(o["g_pw" if a == "pw" else "g_zo"]))
                    g = np.concatenate(acc)
                    sem = g.std(0, ddof=1) / np.sqrt(len(g))
                    z = (g.mean(0) - tgt) / np.maximum(sem, 1e-300)
                    zs.append(z)
                    if np.abs(z).max() > best:
                        best = float(np.abs(z).max())
                        where = f"{nm} coord {int(np.argmax(np.abs(z)))}"
                z = np.concatenate(zs)
                row = dict(d=d, reps=reps, arm=a, max_abs_z=float(np.abs(z).max()),
                           at=where, mean_z2=float((z ** 2).mean()), n=int(z.size))
                out["g1"].append(row)
                print(f"{d:>3} {tag:>6} {a:>4} {row['max_abs_z']:7.2f} {where:>22} "
                      f"{row['mean_z2']:9.3f} {row['n']:5d}", flush=True)

    # ---- the tie at d = 2, by four routes plus a Monte Carlo interval
    su = setup_d(2)
    comp = json.load(open(os.path.join(out_dir, "components_full.json")))["2"]
    Ve = np.asarray(comp["V_e"])
    r1 = swap_from(np.asarray(comp["grid"]), Ve[:, 1] - Ve[:, 0])[0]

    fine = np.logspace(-1.0, 3.0, 161)
    key, _ = block(su, 10000)
    acc = np.zeros((len(fine), 2))
    per = np.zeros((REPS, len(fine)))
    tgt_e = np.stack([EF.blurred_e_grad(LP.at_omega(su, float(x) / su.sigma),
                                        su.eps_study, su.mu0[None, :], su.sigma)[0]
                      for x in fine])
    for i0, n in _chunks(REPS, CHUNK):
        u = jax.random.normal(jax.random.fold_in(key, i0), (n, M, su.d))
        o0 = E.both_from_shared_u(
            LP.make_q_of_u(su, jnp.asarray(su.mu0), 1.0, 0.0), u, su.sigma, axis=1)
        for k, x in enumerate(fine):
            o1 = E.both_from_shared_u(
                LP.make_q_of_u(su, jnp.asarray(su.mu0), float(x) / su.sigma,
                               su.eps_study), u, su.sigma, axis=1)
            dd = []
            for j, a in enumerate(ARMS):
                ge = (np.asarray(o1["g_pw" if a == "pw" else "g_zo"])
                      - np.asarray(o0["g_pw" if a == "pw" else "g_zo"]))
                De = ge - tgt_e[k][None, :]
                se = (De ** 2).sum(-1)
                acc[k, j] += float(se.sum())
                dd.append(se)
            per[i0:i0 + n, k] = dd[1] - dd[0]
    acc /= REPS
    r2 = swap_from(fine, acc[:, 1] - acc[:, 0])[0]
    rng = np.random.default_rng(BOOT_SEED)
    bs = []
    for _ in range(1000):
        i = rng.integers(0, REPS, REPS)
        bs.append(swap_from(fine, per[i].mean(0))[0])
    bs = np.array(bs)
    bs = bs[np.isfinite(bs)]

    # the path study's route: the harness kernel at this state, collapsed onto c
    kernel = jax.jit(SW.make_kernel(2, M, su.pe.rank), static_argnames=("R", "n_batch"))
    vtmu = np.einsum("srd,sd->sr", su.pe.V, su.mu0[None, :])
    ko = kernel(jax.random.fold_in(jax.random.PRNGKey(SEED_ROOT + 3000 + 2), su.idx),
                jnp.asarray(su.Hn), jnp.asarray(su.g_mu), jnp.asarray(su.pe.V[0]),
                jnp.asarray(vtmu[0]), jnp.asarray(su.pe.phi[0]),
                jnp.asarray(SW.SIGMAS), jnp.asarray(SW.OMEGAS), R=250, n_batch=40)
    z = {"sigmas": SW.SIGMAS, "omegas": SW.OMEGAS,
         "s3_pw": np.asarray(ko["s3_pw"])[None], "s3_zo": np.asarray(ko["s3_zo"])[None]}
    r3 = A.crossover_by_c(z)[0]
    # the same kernel data, one sigma column, rooted in omega
    col = int(np.argmin(np.abs(np.log(SW.SIGMAS / su.sigma))))
    r4 = float(SW.SIGMAS[col]) * A.solve_crossover(
        np.log(SW.OMEGAS),
        (np.log(z["s3_zo"].mean(1)) - np.log(z["s3_pw"].mean(1)))[0, col])[0]
    out["tie"] = dict(
        registered_41=float(r1), fine_161=float(r2),
        boot_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        kernel_collapsed=float(r3), kernel_one_column=float(r4),
        kernel_column_sigma=float(SW.SIGMAS[col]), sigma=su.sigma,
        path_study_g3a=PATH_STUDY_TIE_D2)
    print("\nError-only tie at d = 2, by route:")
    for k, v in out["tie"].items():
        print(f"  {k:>20}: {v}")
    with open(os.path.join(out_dir, "diag.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out


def stage_report(out_dir: str):
    with open(os.path.join(out_dir, "components.json")) as f:
        comp = json.load(f)
    with open(os.path.join(out_dir, "graph_a.json")) as f:
        ga = json.load(f)
    res = _analyse(out_dir)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(dict(graph_a=ga, paths=res, predictions=comp), f, indent=2,
                  default=float)
    print(json.dumps({k: res[k] for k in ("slope", "slope_ci")}, indent=2))
    return res


def stage_figures(out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 8,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6})
    with open(os.path.join(out_dir, "components.json")) as f:
        comp = json.load(f)
    with open(os.path.join(out_dir, "graph_a.json")) as f:
        ga = json.load(f)
    res = _analyse(out_dir)
    os.makedirs(FIG_DIR, exist_ok=True)

    # A
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = np.array(ga["ms"], float)
    y = np.array([ga["measured"][str(m)]["swap"] for m in ga["ms"]], float)
    lo = np.array([ga["ci"][str(m)][0] for m in ga["ms"]], float)
    hi = np.array([ga["ci"][str(m)][1] for m in ga["ms"]], float)
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", ms=3, color=NAVY,
                ecolor=NAVY, elinewidth=0.7, capsize=1.5, lw=0)
    ax.axhline(comp["tie"]["2"], color=BRICK, ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$m=\epsilon/\epsilon_{\rm study}$")
    ax.set_ylabel(r"swap point $\sigma\omega$")
    ax.annotate("error only", (x[0], comp["tie"]["2"]), textcoords="offset points",
                xytext=(2, 3), color=BRICK, fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG_DIR, "lqr_swap_vs_error.pdf")); plt.close(fig)

    # B
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    dd = np.array(DS, float)
    y = np.array([res["B"][str(d)]["swap"] for d in DS], float)
    lo = np.array([res["B"][str(d)]["ci"][0] for d in DS], float)
    hi = np.array([res["B"][str(d)]["ci"][1] for d in DS], float)
    ax.errorbar(dd, y, yerr=[y - lo, hi - y], fmt="o", ms=3, color=NAVY,
                ecolor=NAVY, elinewidth=0.7, capsize=1.5, lw=0)
    ax.plot(dd, [R.c_star_asymptote(d, M) for d in DS], color=BRICK, ls="--", lw=0.8)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(dd); ax.set_xticklabels([str(int(d)) for d in dd])
    ax.set_xlabel(r"$d$"); ax.set_ylabel(r"path swap $\sigma\omega$")
    ax.annotate(r"$\sqrt{d}$", (dd[-1], R.c_star_asymptote(DS[-1], M)),
                textcoords="offset points", xytext=(-14, 4), color=BRICK, fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG_DIR, "lqr_swap_vs_dim.pdf")); plt.close(fig)

    # C
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    xs = np.array(res["C"]["xs"], float)
    rho = np.array(res["C"]["rho"], float)
    ci = np.array(res["C"]["ci"], float)
    ax.axhline(1.0, color=GRAY, lw=0.6)
    ax.axvline(comp["predicted_swap_C"]["swap"], color=BRICK, ls="--", lw=0.8)
    ax.errorbar(xs, rho, yerr=[rho - ci[:, 0], ci[:, 1] - rho], fmt="o-", ms=3,
                color=NAVY, ecolor=NAVY, elinewidth=0.7, capsize=1.5, lw=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma\omega$"); ax.set_ylabel(r"$\rho=J_{\rm ZO}/J_{\rm PW}$")
    ax.annotate("predicted", (comp["predicted_swap_C"]["swap"], rho.max()),
                textcoords="offset points", xytext=(3, -2), color=BRICK, fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG_DIR, "lqr_blind_test.pdf")); plt.close(fig)
    print("  wrote three figures to reports/figures/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("gates", "components", "measure", "diag", "figures",
                             "report"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "lqr_swap"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    ok = True
    if a.stage == "gates":
        ok = stage_gates(a.out)
    elif a.stage == "components":
        ok = stage_components(a.out)
    elif a.stage == "measure":
        stage_measure(a.out)
    elif a.stage == "diag":
        stage_diag(a.out)
    elif a.stage == "figures":
        stage_figures(a.out)
    elif a.stage == "report":
        stage_report(a.out)
    LP.record_env(a.out, a.stage, time.time() - t0)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
