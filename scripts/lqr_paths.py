"""Optimisation paths of a Gaussian policy's mean on a planted LQR critic, d = 2.

    python scripts/lqr_paths.py --stage {gates,calibrate,paths,figures,report}

Registered in docs/prereg_lqr_paths.md. Read that before changing anything here: the
panels, the seeds, the gates, the metrics and the figure spec are all frozen there, and
the calibration outcome is appended below its amendment line before any path run.

WHAT THIS DRAWS. One path per gradient estimator, over the contours of the critic the
estimator sees. Both estimators follow the same average route, which is the gradient of
the Gaussian-blurred critic; which of them jitters more depends on how fast the critic
error wiggles relative to the policy width sigma. That is the whole content of the
figure, and it is an illustration at d = 2, not evidence about dimension.

NOTHING HERE REIMPLEMENTS THE HARNESS. The system, the planted error field, its
closed-form blur, the estimator core and the E-step dual are all imported from
scripts/lqr_crossover/ and src/jaxrl/estimators.py, so a number produced here is a
number about the operators the fork runs. The one piece of arithmetic this file adds is
that the policy mean MOVES: `lqr.q_of_u_factory` pins the mean at -K s, which is exactly
what a path cannot do. The moving-mean closure uses the reduced form documented in the
header of scripts/lqr_crossover/lqr.py, with the action-gradient taken from
`lqr.grad_a_q_pi`, and gate G0b checks it against the harness's independent `lqr.q_pi`.

AXIS CONVENTION. Every estimator call passes a NON-NEGATIVE sample axis. That is the
convention of the production call site (src/jaxrl/reppo.py:1247) and of every E-step
call site in the study. `scripts/lqr_crossover/estimators.py:estep_displacement` is not
called: with a negative `axis` it forwards the q-axis to `softmax_displacement`, which
then contracts the coordinate axis instead of the sample axis. It has no call sites in
the repository, so the defect is latent; it is recorded here and not repaired, because
the study's files are a measurement instrument.

CPU. `scripts/lqr_crossover/__init__.py` sets JAX_PLATFORMS=cpu and asserts it at
import, so this script runs on CPU on every machine including a GPU workstation. It
records the device list rather than choosing one. The job is minutes.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform as _platform
import socket
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import scripts.lqr_crossover  # noqa: F401,E402  (platform + x64, must be first)

import jax  # noqa: E402
import numpy as np  # noqa: E402
from jax import numpy as jnp  # noqa: E402
from scipy.special import roots_hermitenorm  # noqa: E402

from scripts.lqr_crossover import SEED_ROOT  # noqa: E402
from scripts.lqr_crossover import analyze as A  # noqa: E402
from scripts.lqr_crossover import error_field as EF  # noqa: E402
from scripts.lqr_crossover import estimators as E  # noqa: E402
from scripts.lqr_crossover import lqr  # noqa: E402
from scripts.lqr_crossover import reference as R  # noqa: E402
from scripts.lqr_crossover import sweep as SW  # noqa: E402
from scripts.lqr_crossover.estep_arm import (  # noqa: E402,F401
    EPS_E, ETA_GRID, solve_eta,
)
from src.jaxrl.estimators import softmax_displacement  # noqa: E402

# ---------------------------------------------------------------- frozen constants
D = 2                                   # action dimension. n = 2d = 4 state dimension.
M = 32                                  # samples per estimator call, study primary
N_STATES = 32                           # the study's full-rank arm state count
EPS_FRAC = 0.05                         # eps_study = 5% of the within-state Q spread
N_STEPS = 80                            # path length; doubled once to 160 only by G2
N_STEPS_G2 = 160
STEP_FRAC = 0.2                         # step 0.2 sigma; only direction from the arm
R_SEEDS = 100                           # path seeds 0..99
SEED_DISPLAY = 0                        # the seed drawn in the figures
MIN_TARGET_DIST = 1e-3                  # state rejection threshold on ||a* - mu_0||

TIE = float(R.c_star_asymptote(D, M))   # sqrt(d M/(M-1)) = 1.4368 at d=2, M=32
SLOW_C = TIE / 4.0
FAST_MULT = 4.0
CAL_GRID = np.logspace(-1.0, 3.0, 41)   # sigma*omega values for the calibration
CAL_REPS = 10_000
CAL_BOOT = 1000
EPS_BIG_CANDIDATES = (10.0, 30.0, 100.0)
EPS_BIG_MAX_RTOT = 1.5 * TIE
BOOT_SEED = 20260902
NBOOT_PATHS = 10_000
PUBLISHED_FULL_D2_CSTAR = 1.522         # reports/lqr_crossover_corrected.md Sec. 7.1
G3B_TOL = 0.02
G0_TOL = 1e-14
G0A_ESTEP_TOL = 1e-11                   # prereg Sec. 13, amendment A2.1
G0B_TOL = 1e-12
G0C_TOL = 1e-7
G1_Z = 4.0
ARMS = ("pw", "zo", "estep")

# Figure spec, frozen in the prereg Sec. 9.
FIG_DIR = os.path.join(REPO_ROOT, "reports", "figures")
COL = {"pw": "#1F4E79", "zo": "#C0504D"}
FIG_SIZE = (3.4, 3.0)
FIG_LEVELS = 15
FIG_GRID_MIN, FIG_GRID_MAX = 400, 600
FIG_PTS_PER_PERIOD = 8


# ------------------------------------------------------------------------ provenance
def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
    except Exception:
        return True


def prereg_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", "docs/prereg_lqr_paths.md"],
            cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record_env(out_dir: str, stage: str, seconds: float, extra: dict | None = None):
    """Append one row to env.json. The script's own sha256 is recorded because the
    gates and the calibration are run BEFORE this file is committed, so `git_sha` alone
    would not identify the code that produced them (the provenance gap the crossover
    audit found in msweep_hi.py, reports/lqr_crossover_audit.md Sec. 2)."""
    path = os.path.join(out_dir, "env.json")
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = json.load(f)
    import importlib
    vers = {}
    for mod in ("jax", "numpy", "scipy", "matplotlib"):
        try:
            vers[mod] = importlib.import_module(mod).__version__
        except Exception:
            vers[mod] = "absent"
    rows.append(dict(
        stage=stage, hostname=socket.gethostname(), platform=_platform.platform(),
        python=sys.version.split()[0], devices=[str(x) for x in jax.devices()],
        x64=bool(jnp.zeros(1).dtype == jnp.float64), versions=vers,
        git_sha=git_sha(), git_dirty=git_dirty(), prereg_sha=prereg_sha(),
        script_sha256=sha256_file(os.path.abspath(__file__)),
        seconds=round(seconds, 3), timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **(extra or {})))
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


# ----------------------------------------------------------------------------- setup
@dataclasses.dataclass(frozen=True)
class Setup:
    sys_: object
    idx: int                 # state index actually used
    s: np.ndarray            # (n,) the state, unnormalised
    ss: np.ndarray           # (n,) alpha_s * s, the normalised state
    mu0: np.ndarray          # (d,) start point, -K s in normalised coordinates
    a_star: np.ndarray       # (d,) argmax_a Q^pi(s, a)
    sigma: float
    eps_study: float
    Hn: np.ndarray           # (d,d) alpha_Q * H, the normalised curvature
    g_mu: np.ndarray         # (d,) grad_a Q^pi at mu0, normalised
    bvec: np.ndarray         # (d,) the affine constant of grad_a Q^pi
    alpha_Q: float
    alpha_s: float
    pe: EF.PlantedError      # single-state error field, V (1,r,d), phi (1,r)
    target_dist: float


def setup() -> Setup:
    """The system, the state, the start point and the width, exactly as registered.

    ONE random stream, consumed in the study's order: states first, then the phases.
    That is what makes the state and the phases here the first state and phases of the
    study's published full-rank d = 2 arm rather than a fresh draw.
    """
    sys_ = lqr.build_system(D, seed=SEED_ROOT + D)
    rng = np.random.default_rng(SEED_ROOT + 2000 + D)
    states = lqr.sample_states(sys_, rng, N_STATES)
    pe_all = EF.draw_error(rng, N_STATES, D, kind="full", omega=1.0)

    Hn, g_all, mu_all = lqr.q_coeffs(sys_, states)
    idx, step_vec = None, None
    for i in range(N_STATES):
        v = 0.5 * np.linalg.solve(Hn, g_all[i])
        if np.linalg.norm(v) >= MIN_TARGET_DIST:
            idx, step_vec = i, v
            break
    if idx is None:
        raise RuntimeError("no state with ||a* - mu_0|| >= 1e-3 among the 32 drawn")

    s = states[idx]
    mu0 = mu_all[idx]
    a_star = mu0 + step_vec
    dist = float(np.linalg.norm(step_vec))
    sigma = dist / 10.0
    eps_study = float(EPS_FRAC * lqr.q_spread_closed_form(sys_, s, sigma))

    aQ = float(sys_.scale["alpha_Q"])
    aS = float(sys_.scale["alpha_s"])
    ss = aS * s
    # exactly the constant term of lqr.grad_a_q_pi, so g_lin(mu0) reproduces q_coeffs.
    bvec = -2.0 * sys_.gamma * ss @ (sys_.B.T @ sys_.P @ sys_.A).T

    pe = EF.PlantedError(kind=pe_all.kind, rank=int(pe_all.rank), omega=1.0,
                         V=pe_all.V[idx:idx + 1].copy(),
                         phi=pe_all.phi[idx:idx + 1].copy())
    return Setup(sys_=sys_, idx=idx, s=s, ss=ss, mu0=mu0, a_star=a_star, sigma=sigma,
                 eps_study=eps_study, Hn=Hn, g_mu=g_all[idx], bvec=bvec,
                 alpha_Q=aQ, alpha_s=aS, pe=pe, target_dist=dist)


def at_omega(su: Setup, omega: float) -> EF.PlantedError:
    return EF.PlantedError(su.pe.kind, su.pe.rank, float(omega), su.pe.V, su.pe.phi)


# ------------------------------------------------------------- critic and estimators
def g_lin_np(su: Setup, m: np.ndarray) -> np.ndarray:
    """grad_a Q^pi(s, m), normalised. Identical expression to lqr.grad_a_q_pi."""
    return su.alpha_Q * lqr.grad_a_q_pi(su.sys_, su.ss, m)


def make_q_of_u(su: Setup, m, omega: float, eps, pe=None):
    """Q_phi(s, m + sigma u) up to a constant in u, for a mean m that may be traced.

    Same reduced form as lqr.q_of_u_factory, with the linear coefficient evaluated at m
    instead of at -K s, plus the harness's own error field. Both operators are blind to
    the dropped constant: the zeroth-order one centres it away, the pathwise one
    differentiates it away.
    """
    pe_o = at_omega(su, omega) if pe is None else pe
    Hn_j = jnp.asarray(su.Hn)
    Hu_j = jnp.asarray(su.sys_.H)
    b_j = jnp.asarray(su.bvec)
    sg = su.sigma

    def q_of_u(u):
        gl = su.alpha_Q * (-2.0 * m @ Hu_j.T + b_j)
        quad = -(sg ** 2) * jnp.einsum("...i,ij,...j->...", u, Hn_j, u)
        lin = sg * jnp.sum(gl * u, axis=-1)
        return quad + lin + EF.e_value(pe_o, eps, m, sg, u)

    return q_of_u


def arm_direction(arm: str, q_of_u, u, sigma: float, axis: int = 0):
    """One estimator's direction, in action space. `axis` is the sample axis of u."""
    if arm in ("pw", "zo"):
        out = E.both_from_shared_u(q_of_u, u, sigma, axis=axis)
        return out["g_pw" if arm == "pw" else "g_zo"]
    if arm == "estep":
        # composed exactly as scripts/lqr_crossover/estep_arm.py composes it
        q = q_of_u(u)
        eta = solve_eta(q)
        w = jax.nn.softmax(q / eta, axis=-1)
        return softmax_displacement(w, u, axis=axis) / sigma
    raise ValueError(f"unknown arm={arm!r}")


def _unit(v):
    return v / jnp.maximum(jnp.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def make_runner(su: Setup, arm: str, omega: float, eps: float, n_steps: int):
    """jit(vmap(scan)): R seeds at once, one scan over steps, u drawn from (seed, t).

    Returns a function of the seed indices giving (mus, ghats) with the seed axis first.
    """
    base = jax.random.PRNGKey(SEED_ROOT + 8000 + D)
    mu0_j = jnp.asarray(su.mu0)
    sg = su.sigma

    def one_seed(seed):
        key0 = jax.random.fold_in(base, seed)

        def body(m, t):
            u = jax.random.normal(jax.random.fold_in(key0, t), (M, D))
            q_of_u = make_q_of_u(su, m, omega, eps)
            gh = _unit(arm_direction(arm, q_of_u, u, sg, axis=0))
            m_next = m + STEP_FRAC * sg * gh
            return m_next, (m_next, gh)

        _, (mus, ghats) = jax.lax.scan(body, mu0_j, jnp.arange(n_steps))
        return mus, ghats

    return jax.jit(jax.vmap(one_seed))


def g_star_np(su: Setup, m: np.ndarray, omega: float, eps: float) -> np.ndarray:
    """The exact estimand: the gradient of the Gaussian-blurred critic at mean m.

    The smooth part needs no blur correction because a Gaussian blur shifts a quadratic
    by a constant. The error part is the harness's closed form.
    """
    pe_o = at_omega(su, omega)
    m2 = np.atleast_2d(m)
    out = g_lin_np(su, m2) + EF.blurred_e_grad(pe_o, eps, m2, su.sigma)
    return out.reshape(np.shape(m))


def reference_path(su: Setup, omega: float, eps: float, n_steps: int) -> np.ndarray:
    """The noise-free path. numpy, one per panel, never inside a jit."""
    m = np.array(su.mu0, dtype=float)
    out = np.empty((n_steps, D))
    for t in range(n_steps):
        g = g_star_np(su, m, omega, eps)
        m = m + STEP_FRAC * su.sigma * g / max(float(np.linalg.norm(g)), 1e-300)
        out[t] = m
    return out


# ------------------------------------------------------------------------ one-step MC
def cal_block(su: Setup) -> jax.Array:
    """One shared block of replicate draws, used at every grid point and every eps.

    Sharing it is what makes the calibration paired across arms, across grid points and
    across eps, so a difference between two curves is not replicate noise.
    """
    return jax.random.normal(jax.random.PRNGKey(SEED_ROOT + 7000 + D),
                             (CAL_REPS, M, D))


def one_step_cos(su: Setup, u, omega: float, eps: float) -> dict:
    """cos(g_hat, g*) per replicate at mu_0, for PW and de-attenuated ZO."""
    q_of_u = make_q_of_u(su, jnp.asarray(su.mu0), omega, eps)
    out = E.both_from_shared_u(q_of_u, u, su.sigma, axis=1)
    gs = jnp.asarray(g_star_np(su, su.mu0, omega, eps))
    res = {}
    for arm in ("pw", "zo"):
        g = out["g_pw" if arm == "pw" else "g_zo"]
        res[arm] = np.asarray(
            jnp.sum(g * gs, -1)
            / jnp.maximum(jnp.linalg.norm(g, axis=-1) * jnp.linalg.norm(gs), 1e-300))
        res[arm + "_mean"] = np.asarray(g.mean(0))
        res[arm + "_sem"] = np.asarray(g.std(0, ddof=1) / np.sqrt(g.shape[0]))
    res["g_star"] = np.asarray(gs)
    return res


def omega_rms_ratio(su: Setup, omega: float) -> float:
    """omega_RMS / omega at this state, from the study's own closed form.

    `audit.rms_product` is the function that produced the RMS column of
    reports/lqr_crossover_audit.md Sec. 3, so it is called here rather than rewritten.
    It expects a swept file; it is handed a one-cell view of this state, and only its
    `omega_rms_over_omega` column is read. The `s3` arrays it also consumes are set to
    one, which makes its variance-ratio columns meaningless and unused.
    """
    from scripts.lqr_crossover import audit as AUD
    vtmu = np.einsum("srd,sd->sr", su.pe.V, su.mu0[None, :])
    z = {"sigmas": np.array([su.sigma]), "omegas": np.array([float(omega)]),
         "d": D, "rank": su.pe.rank, "phi": su.pe.phi, "vtmu": vtmu,
         "s3_pw": np.ones((1, 1, 1, 1)), "s3_zo": np.ones((1, 1, 1, 1))}
    return float(AUD.rms_product(z)["omega_rms_over_omega"].iloc[0])


def crossing(xs: np.ndarray, diff: np.ndarray) -> float:
    """First positive-to-negative crossing of `diff`, interpolated in log x.

    `diff` is ZO's mean angular error minus PW's. It starts positive (the zeroth-order
    operator pays the classical dimension factor on the smooth part) and, if the error
    field is fast enough, ends negative. Returns nan when no such crossing exists on the
    grid, which is a reportable outcome and not a failure to be papered over.
    """
    lx = np.log(xs)
    for k in range(len(diff) - 1):
        if diff[k] > 0.0 >= diff[k + 1]:
            f = diff[k] / (diff[k] - diff[k + 1])
            return float(np.exp(lx[k] + f * (lx[k + 1] - lx[k])))
    return float("nan")


# --------------------------------------------------------------------------- G stages
class Gate:
    """Records two verdicts per gate: as registered, and under amendment A2.

    Three registered comparators measure floating-point cancellation inside the
    comparison rather than the quantity they were written to test (prereg Sec. 13, A2).
    Their registered verdict stays in the record permanently; `status` carries the
    amended one, which is what the path stage refuses on.
    """

    def __init__(self):
        self.rows = []

    def add(self, name, status, detail, blocking=True,
            registered=None, detail_registered=None):
        registered = status if registered is None else registered
        self.rows.append(dict(name=name, status=status, detail=detail,
                              status_as_registered=registered,
                              detail_as_registered=detail_registered or detail,
                              blocking=bool(blocking)))
        tag = f"[{status}]" if status == registered else f"[{status} / {registered} " \
                                                         f"as registered]"
        print(f"  {tag} {name}: {detail}", flush=True)
        if detail_registered and detail_registered != detail:
            print(f"        as registered: {detail_registered}", flush=True)

    def ok(self):
        return all(r["status"] == "PASS" or not r["blocking"] for r in self.rows)


def gate_g0a(su: Setup, G: Gate):
    """The path code's per-step function against the harness composition, at mu_0."""
    keys = jax.random.split(jax.random.PRNGKey(SEED_ROOT + 9000 + D), 100)
    pe_o = at_omega(su, 1.0)
    eps = su.eps_study
    mine_f = {a: [] for a in ARMS}
    harn_f = {a: [] for a in ARMS}
    # harness route, exactly as estep_arm.py composes it
    qf = lqr.q_of_u_factory(su.sys_, su.s[None, :], su.sigma)
    mu_j = jnp.asarray(su.mu0)[None, None, :]
    eps_j = jnp.asarray(np.array([eps]))[:, None]

    def q_h(uu):
        return qf(uu) + EF.e_value(pe_o, eps_j, mu_j, su.sigma, uu)

    q_mine = make_q_of_u(su, jnp.asarray(su.mu0), 1.0, eps)
    for k in keys:
        u = jax.random.normal(k, (M, D))
        u3 = u[None]
        for a in ARMS:
            mine_f[a].append(np.asarray(arm_direction(a, q_mine, u, su.sigma, axis=0)))
            harn_f[a].append(np.asarray(
                arm_direction(a, q_h, u3, su.sigma, axis=1)).reshape(D))
    worst, bitwise = 0.0, 0
    per_arm = {}
    for a in ARMS:
        x, y = np.array(mine_f[a]), np.array(harn_f[a])
        rel = np.abs(x - y) / np.maximum(np.abs(y), 1e-300)
        per_arm[a] = float(rel.max())
        worst = max(worst, per_arm[a])
        bitwise += int((x == y).all(axis=1).sum())
    per = ", ".join(f"{a} {per_arm[a]:.2e}" for a in ARMS)
    amended = all(per_arm[a] <= (G0A_ESTEP_TOL if a == "estep" else G0_TOL)
                  for a in ARMS)
    G.add("G0a arms reproduce the harness composition",
          "PASS" if amended else "FAIL",
          f"per arm {per}; bounds PW/ZO {G0_TOL:.0e}, "
          f"ESTEP {G0A_ESTEP_TOL:.0e} (A2.1); "
          f"bitwise identical {bitwise}/300 outputs",
          registered="PASS" if worst <= G0_TOL else "FAIL",
          detail_registered=f"max rel {worst:.3e} against one bound {G0_TOL:.0e}")
    return worst


def gate_g0b(su: Setup, G: Gate):
    """The moving-mean closure against the harness's independent exact Q^pi."""
    rng = np.random.default_rng(SEED_ROOT + 9100 + D)
    ms = su.mu0 + np.linspace(0.0, 1.0, 10)[:, None] * (su.a_star - su.mu0)
    us = rng.normal(size=(100, 2, M, D))
    worst_reg, worst_smooth, worst_abs, qmax = 0.0, 0.0, 0.0, 0.0
    pe_o = at_omega(su, 1.0)
    for m in ms:
        m_j = jnp.asarray(m)
        for eps, tag in ((su.eps_study, "reg"), (0.0, "smooth")):
            q = make_q_of_u(su, m_j, 1.0, eps)
            for u, up in us:
                a = m + su.sigma * np.asarray(u)
                ap = m + su.sigma * np.asarray(up)
                qa = lqr.q_pi(su.sys_, su.ss, a, su.sigma)
                qb = lqr.q_pi(su.sys_, su.ss, ap, su.sigma)
                exact = su.alpha_Q * (qa - qb)
                if eps:
                    exact = exact + np.asarray(
                        EF.e_value(pe_o, eps, m_j, su.sigma, jnp.asarray(u))
                        - EF.e_value(pe_o, eps, m_j, su.sigma, jnp.asarray(up)))
                got = np.asarray(q(jnp.asarray(u)) - q(jnp.asarray(up)))
                rel = float(np.max(np.abs(got - exact)
                                   / np.maximum(np.abs(exact), 1e-300)))
                worst_abs = max(worst_abs, float(np.max(np.abs(got - exact))))
                qmax = max(qmax, float(np.abs(qa).max()), float(np.abs(qb).max()))
                if tag == "reg":
                    worst_reg = max(worst_reg, rel)
                else:
                    worst_smooth = max(worst_smooth, rel)
    # A2.2: lqr.q_pi carries the additive constant gamma v/(1-gamma), so differencing
    # two of its values loses about eight digits before the comparison starts. The
    # criterion is therefore the cancellation floor of the comparison itself.
    floor = 10.0 * float(np.finfo(np.float64).eps) * su.alpha_Q * qmax
    G.add("G0b moving-mean closure vs lqr.q_pi",
          "PASS" if worst_abs <= floor else "FAIL",
          f"max abs dev {worst_abs:.3e} against the cancellation floor {floor:.3e} "
          f"(10 eps_mach alpha_Q max|q_pi|, max|q_pi| = {qmax:.6g}) (A2.2)",
          registered="PASS" if max(worst_reg, worst_smooth) <= G0B_TOL else "FAIL",
          detail_registered=f"max rel {worst_reg:.3e} with e_value on both sides, "
                            f"{worst_smooth:.3e} at eps = 0; bound {G0B_TOL:.0e}")


def _gh_nodes(n):
    x, w = roots_hermitenorm(n)
    return x, w / w.sum()


def gate_g0c(su: Setup, G: Gate, panels=None):
    """The exact target against finite differences, and the blurred error gradient
    against Gauss-Hermite quadrature wherever the blur factor is representable."""
    rng = np.random.default_rng(SEED_ROOT + 9200 + D)
    ms = su.mu0 + rng.normal(size=(10, D)) * su.target_dist
    h = 1e-6 * max(su.target_dist, 1.0)
    worst = 0.0
    for m in ms:
        num = np.empty(D)
        for j in range(D):
            e = np.zeros(D)
            e[j] = h
            num[j] = su.alpha_Q * (lqr.q_pi(su.sys_, su.ss, m + e, su.sigma)
                                   - lqr.q_pi(su.sys_, su.ss, m - e, su.sigma))
            num[j] = num[j] / (2 * h)
        got = g_lin_np(su, m)
        worst = max(worst, float(np.max(np.abs(num - got)
                                        / np.maximum(np.abs(got), 1e-300))))
    G.add("G0c smooth target vs central differences of lqr.q_pi",
          "PASS" if worst <= G0C_TOL else "FAIL",
          f"max rel {worst:.3e} (bound {G0C_TOL:.0e}) over 10 means")

    if not panels:
        G.add("G0c blurred error gradient vs quadrature", "DEFER",
              "needs the calibrated panels", blocking=True)
        return
    det, det_reg = [], []
    ok, ok_reg = True, True
    for p in panels:
        c = p["c"]
        damp = float(np.exp(-0.5 * c * c))
        pe_o = at_omega(su, p["omega"])
        closed = EF.blurred_e_grad(pe_o, p["eps"], su.mu0[None, :], su.sigma)[0]
        if damp == 0.0:
            # The comparator is retired here: the true value is zero and the quadrature
            # returns the un-cancelled integrand scale (A2.3).
            good = bool(np.all(closed == 0.0))
            ok &= good
            ok_reg &= good
            det.append(f"{p['name']}: c={c:.3g}, blur underflows, closed form exactly "
                       f"zero: {good}")
            det_reg.append(det[-1])
            continue
        vals = []
        for n in (200, 400):
            x, w = _gh_nodes(n)
            U = np.stack(np.meshgrid(x, x, indexing="ij"), -1).reshape(-1, D)
            W = np.outer(w, w).reshape(-1)
            a = su.mu0[None, :] + su.sigma * U
            g = np.asarray(EF.e_grad(pe_o, p["eps"], jnp.asarray(a)))
            vals.append((W[:, None] * g).sum(0))
        dev = float(np.max(np.abs(vals[1] - closed)))
        res = float(np.max(np.abs(vals[1] - vals[0])))          # comparator's own limit
        rel = dev / max(float(np.max(np.abs(closed))), 1e-300)
        conv = res / max(float(np.max(np.abs(vals[1]))), 1e-300)
        # Two readings of "relative deviation", both reported: componentwise (the
        # literal first run) and against the vector magnitude. The registered verdict
        # is adjudicated on the componentwise one, which is the stricter of the two.
        rel_cw = float(np.max(np.abs(vals[1] - closed)
                              / np.maximum(np.abs(closed), 1e-300)))
        conv_cw = float(np.max(np.abs(vals[1] - vals[0])
                               / np.maximum(np.abs(vals[1]), 1e-300)))
        # A2.3: the closed form must agree to the quadrature's demonstrated resolution.
        good = dev <= max(G0C_TOL * float(np.max(np.abs(closed))), 2.0 * res)
        ok &= good
        ok_reg &= (rel_cw <= G0C_TOL and conv_cw <= G0C_TOL)
        det.append(f"{p['name']}: c={c:.3g}, abs dev {dev:.2e} against quadrature "
                   f"resolution {res:.2e}")
        det_reg.append(f"{p['name']}: c={c:.3g}, rel {rel_cw:.2e} componentwise / "
                       f"{rel:.2e} on the vector, self-convergence {conv_cw:.2e}")
    G.add("G0c blurred error gradient vs quadrature", "PASS" if ok else "FAIL",
          "; ".join(det) + f" (A2.3, bound max({G0C_TOL:.0e} rel, 2 x resolution))",
          registered="PASS" if ok_reg else "FAIL",
          detail_registered="; ".join(det_reg) + f" (bound {G0C_TOL:.0e})")


def gate_g1(su: Setup, G: Gate, panels=None):
    """Unbiasedness at the start: both arms' means equal g* within Monte Carlo error."""
    u = cal_block(su)
    configs = [dict(name="eps=0", omega=1.0, eps=0.0)]
    if panels:
        configs += [dict(name=p["name"], omega=p["omega"], eps=p["eps"])
                    for p in panels]
    else:
        G.add("G1 estimator means at mu_0 (panel frequencies)", "DEFER",
              "needs the calibrated panels", blocking=True)
    worst, det, ok = 0.0, [], True
    for cfg in configs:
        r = one_step_cos(su, u, cfg["omega"], cfg["eps"])
        zz = []
        for arm in ("pw", "zo"):
            z = (r[arm + "_mean"] - r["g_star"]) / np.maximum(r[arm + "_sem"], 1e-300)
            zz.append(float(np.max(np.abs(z))))
        w = max(zz)
        worst = max(worst, w)
        ok &= w <= G1_Z
        det.append(f"{cfg['name']}: max|z| PW {zz[0]:.2f} ZO {zz[1]:.2f}")
    name = ("G1 estimator means at mu_0" if panels
            else "G1 estimator means at mu_0 (eps = 0)")
    G.add(name, "PASS" if ok else "FAIL",
          f"bound {G1_Z:.0f} SE over {CAL_REPS} replicates; " + "; ".join(det))


def gate_g3(su: Setup, G: Gate):
    """The error-only tie, by the harness's paired-difference kernel and root find."""
    # G3a: this state, this phase draw.
    kernel = jax.jit(SW.make_kernel(D, M, su.pe.rank),
                     static_argnames=("R", "n_batch"))
    vtmu = np.einsum("srd,sd->sr", su.pe.V, su.mu0[None, :])
    out = kernel(jax.random.fold_in(jax.random.PRNGKey(SEED_ROOT + 3000 + D), su.idx),
                 jnp.asarray(su.Hn), jnp.asarray(su.g_mu),
                 jnp.asarray(su.pe.V[0]), jnp.asarray(vtmu[0]),
                 jnp.asarray(su.pe.phi[0]),
                 jnp.asarray(SW.SIGMAS), jnp.asarray(SW.OMEGAS),
                 R=250, n_batch=40)
    z = {"sigmas": SW.SIGMAS, "omegas": SW.OMEGAS,
         "s3_pw": np.asarray(out["s3_pw"])[None],
         "s3_zo": np.asarray(out["s3_zo"])[None]}
    c_state, okb = A.crossover_by_c(z)
    dev = abs(np.log(c_state / TIE)) if okb and np.isfinite(c_state) else np.inf
    G.add("G3a error-only tie at this state",
          "PASS" if dev <= np.log(1.5) else "FAIL",
          f"c* = {c_state:.4f} against the asymptote {TIE:.4f}; "
          f"|log ratio| {dev:.4f} (bound {np.log(1.5):.4f})")

    # G3b: reproduce the study's published full-rank d = 2 number.
    tag = "_paths_g3b"
    name = f"d{D}_full_M{M}_unit_H_identity{tag}"
    path = os.path.join(scripts.lqr_crossover.OUT, name + ".npz")
    if os.path.exists(path):
        print(f"  (G3b reusing existing {os.path.basename(path)})", flush=True)
    else:
        assert not os.path.exists(path), path
        SW.run_d(D, M=M, kind="full", n_states=N_STATES, n_batch=20, r_batch=100,
                 tag=tag)
    zz = A.load(path)
    c_pub, ok2 = A.crossover_by_c(zz)
    rel = abs(c_pub / PUBLISHED_FULL_D2_CSTAR - 1.0)
    G.add("G3b reproduces the published full-rank d = 2 c*",
          "PASS" if (ok2 and rel <= G3B_TOL) else "FAIL",
          f"c* = {c_pub:.4f} against the published {PUBLISHED_FULL_D2_CSTAR:.3f}; "
          f"rel {rel:.4f} (bound {G3B_TOL})")
    return float(c_state), float(c_pub)


def gate_g2(su: Setup, G: Gate, panels):
    """The reference path has reached a stationary point of the blurred critic."""
    ok_all = True
    for p in panels:
        for n in (N_STEPS, N_STEPS_G2):
            ref = reference_path(su, p["omega"], p["eps"], n)
            tail = ref[-20:]
            rad = float(np.max(np.linalg.norm(tail - tail.mean(0), axis=-1)))
            if rad <= su.sigma:
                break
        p["n_steps"] = n
        good = rad <= su.sigma
        ok_all &= good
        G.add(f"G2 reference path settled [{p['name']}]",
              "PASS" if good else "FAIL",
              f"tail radius {rad / su.sigma:.3f} sigma at N = {n} (bound 1.0)")
    return ok_all


# ---------------------------------------------------------------------------- stages
def stage_gates(su: Setup, out_dir: str, panels=None):
    G = Gate()
    print("Gates (prereg Sec. 7). float64:", jnp.zeros(1).dtype, jax.devices())
    gate_g0a(su, G)
    gate_g0b(su, G)
    gate_g0c(su, G, panels)
    gate_g1(su, G, panels)
    if panels:
        gate_g2(su, G, panels)
        c_state = c_pub = None
    else:
        c_state, c_pub = gate_g3(su, G)
    rec = dict(rows=G.rows, tie_asymptote=TIE, panels=panels,
               c_state=c_state, c_published=c_pub)
    tagfile = "gates_panels.json" if panels else "gates.json"
    with open(os.path.join(out_dir, tagfile), "w") as f:
        json.dump(rec, f, indent=2, default=float)
    print(("\nALL GATES PASS" if G.ok() else "\nGATE FAILURE") + f" -> {tagfile}")
    return G.ok()


def stage_calibrate(su: Setup, out_dir: str):
    """Place the panels: the one-step total-error crossover r_tot(eps), per eps."""
    u = cal_block(su)
    eps_set = {"study": su.eps_study}
    for k in EPS_BIG_CANDIDATES:
        eps_set[f"x{int(k)}"] = k * su.eps_study

    curves, rtot = {}, {}
    rng = np.random.default_rng(BOOT_SEED)
    for key, eps in eps_set.items():
        cos_pw = np.empty((CAL_REPS, len(CAL_GRID)))
        cos_zo = np.empty((CAL_REPS, len(CAL_GRID)))
        for k, c in enumerate(CAL_GRID):
            r = one_step_cos(su, u, float(c) / su.sigma, eps)
            cos_pw[:, k], cos_zo[:, k] = r["pw"], r["zo"]
        # mean angular error difference, ZO minus PW, per replicate then averaged
        dper = cos_pw - cos_zo
        diff = dper.mean(0)
        c_star = crossing(CAL_GRID, diff)
        boots = np.empty(CAL_BOOT)
        for b in range(CAL_BOOT):
            i = rng.integers(0, CAL_REPS, CAL_REPS)
            boots[b] = crossing(CAL_GRID, dper[i].mean(0))
        fin = boots[np.isfinite(boots)]
        ci = (float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))) \
            if len(fin) > 2 else (float("nan"), float("nan"))
        rtot[key] = dict(eps=float(eps), r_tot=c_star, ci=ci,
                         n_finite_boot=int(len(fin)),
                         ang_pw=(1.0 - cos_pw.mean(0)).tolist(),
                         ang_zo=(1.0 - cos_zo.mean(0)).tolist())
        curves[key] = diff
        print(f"  eps {key:>6}  = {eps:.6g}   r_tot = {c_star:.4f}  "
              f"95% [{ci[0]:.4f}, {ci[1]:.4f}]", flush=True)

    # eps_big by the registered rule
    big_key, big_note = None, ""
    for k in EPS_BIG_CANDIDATES:
        key = f"x{int(k)}"
        r = rtot[key]["r_tot"]
        if np.isfinite(r) and r <= EPS_BIG_MAX_RTOT:
            big_key = key
            break
    if big_key is None:
        big_key = f"x{int(EPS_BIG_CANDIDATES[-1])}"
        big_note = ("no candidate satisfied r_tot <= 1.5 * tie; "
                    f"eps_big = {EPS_BIG_CANDIDATES[-1]:g} x eps_study by the rule")

    panels = []
    for eps_key, label in (("study", "study"), (big_key, "big")):
        eps = rtot[eps_key]["eps"]
        for regime in ("slow", "fast"):
            c = SLOW_C if regime == "slow" else FAST_MULT * rtot[eps_key]["r_tot"]
            omega = float(c) / su.sigma
            conv = {f"{a}/{b}": float(np.mean(np.atleast_1d(
                EF.omega_inf(at_omega(su, omega), grad_norm=a, val_norm=b, c=c,
                             th=EF.theta(at_omega(su, omega), su.mu0[None, :])))))
                for a, b in EF.ALL_CONVENTIONS}
            panels.append(dict(name=f"{label}_{regime}", eps_key=eps_key, label=label,
                               regime=regime, eps=float(eps), c=float(c),
                               omega=omega,
                               omega_rms_over_omega=omega_rms_ratio(su, omega),
                               omega_inf_conventions=conv,
                               n_steps=N_STEPS))
    rec = dict(state_index=su.idx, target_dist=su.target_dist, sigma=su.sigma,
               sigma_above_min_std=bool(su.sigma >= 0.1),
               eps_study=su.eps_study, eps_big_key=big_key,
               eps_big_multiple=float(rtot[big_key]["eps"] / su.eps_study),
               eps_big_note=big_note, tie_asymptote=TIE, slow_c=SLOW_C,
               grid=CAL_GRID.tolist(), reps=CAL_REPS, boot=CAL_BOOT,
               rtot=rtot, panels=panels)
    with open(os.path.join(out_dir, "calibration.json"), "w") as f:
        json.dump(rec, f, indent=2, default=float)
    with open(os.path.join(out_dir, "calibration.csv"), "w") as f:
        f.write("eps_key,sigma_omega,ang_err_pw,ang_err_zo\n")
        for key in rtot:
            for k, c in enumerate(CAL_GRID):
                f.write(f"{key},{c:.10g},{rtot[key]['ang_pw'][k]:.10g},"
                        f"{rtot[key]['ang_zo'][k]:.10g}\n")
    print("\npanels:")
    for p in panels:
        print(f"  {p['name']:>12}  eps {p['eps']:.6g}  sigma*omega {p['c']:.4f}  "
              f"omega {p['omega']:.4f}  "
              f"omega_RMS/omega {p['omega_rms_over_omega']:.4f}")
    return rec


def stage_paths(su: Setup, out_dir: str, panels):
    """Every seed of a panel finishes before anything is compared."""
    seeds = jnp.arange(R_SEEDS)
    man = []
    for p in panels:
        t0 = time.time()
        n = int(p["n_steps"])
        ref = reference_path(su, p["omega"], p["eps"], n)
        data = dict(mu0=su.mu0, sigma=su.sigma, reference=ref, seeds=np.arange(R_SEEDS),
                    **{k: p[k] for k in ("name", "eps", "c", "omega", "n_steps")})
        for arm in ARMS:
            run = make_runner(su, arm, p["omega"], p["eps"], n)
            mus, ghats = run(seeds)
            data[f"mu_{arm}"] = np.asarray(mus)
            data[f"ghat_{arm}"] = np.asarray(ghats)
        path = os.path.join(out_dir, f"paths_{p['name']}.npz")
        np.savez_compressed(path, git_sha=git_sha(), prereg_sha=prereg_sha(),
                            script_sha256=sha256_file(os.path.abspath(__file__)),
                            seconds=time.time() - t0, **data)
        man.append((os.path.basename(path), sha256_file(path), os.path.getsize(path)))
        print(f"  {p['name']:>12} -> {os.path.basename(path)} "
              f"({time.time() - t0:.1f}s)", flush=True)
    with open(os.path.join(out_dir, "npz_manifest.csv"), "w") as f:
        f.write("name,sha256,bytes\n")
        for row in man:
            f.write(f"{row[0]},{row[1]},{row[2]}\n")


def _metrics(su: Setup, out_dir: str, panels):
    rng = np.random.default_rng(BOOT_SEED)
    res = {}
    for p in panels:
        z = np.load(os.path.join(out_dir, f"paths_{p['name']}.npz"), allow_pickle=True)
        ref = z["reference"]
        J, cosm, endd = {}, {}, {}
        for arm in ARMS:
            mu = z[f"mu_{arm}"]                       # (R, N, d)
            dev = np.linalg.norm(mu - ref[None], axis=-1) / su.sigma
            J[arm] = dev.mean(1)                      # (R,)
            endd[arm] = float(np.median(dev[:, -1]))
            gs = g_star_np(su, mu.reshape(-1, D), p["omega"],
                           p["eps"]).reshape(mu.shape)
            gh = z[f"ghat_{arm}"]
            c = np.sum(gh * gs, -1) / np.maximum(np.linalg.norm(gs, axis=-1), 1e-300)
            cosm[arm] = float(np.median(c))
        out = dict(median_J={a: float(np.median(J[a])) for a in ARMS},
                   median_cos=cosm, end_dist=endd)
        for num, den, key in (("zo", "pw", "rho"), ("estep", "pw", "rho_estep")):
            point = float(np.median(J[num]) / np.median(J[den]))
            b = np.empty(NBOOT_PATHS)
            for i in range(NBOOT_PATHS):
                idx = rng.integers(0, R_SEEDS, R_SEEDS)
                b[i] = np.median(J[num][idx]) / np.median(J[den][idx])
            out[key] = dict(point=point,
                            ci=[float(np.percentile(b, 2.5)),
                                float(np.percentile(b, 97.5))])
        pred = "gt1" if p["regime"] == "slow" else "lt1"
        lo, hi = out["rho"]["ci"]
        if (pred == "gt1" and lo > 1) or (pred == "lt1" and hi < 1):
            verdict = "confirmed"
        elif (pred == "gt1" and hi < 1) or (pred == "lt1" and lo > 1):
            verdict = "refuted"
        else:
            verdict = "inconclusive"
        out.update(verdict=verdict, prediction=pred, panel=p)
        res[p["name"]] = out
    return res


def stage_report(su: Setup, out_dir: str, cal):
    res = _metrics(su, out_dir, cal["panels"])
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    return res


# --------------------------------------------------------------------------- figures
def stage_figures(su: Setup, out_dir: str, panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    lims = {}
    for label in ("study", "big"):
        pts = []
        for p in panels:
            if p["label"] != label:
                continue
            z = np.load(os.path.join(out_dir, f"paths_{p['name']}.npz"))
            for arm in ("pw", "zo"):
                pts.append(z[f"mu_{arm}"][SEED_DISPLAY])
            pts.append(su.mu0[None] + su.sigma * np.array([[1, 0], [-1, 0],
                                                           [0, 1], [0, -1]]))
        P = np.concatenate(pts, 0)
        lo, hi = P.min(0), P.max(0)
        pad = 0.10 * (hi - lo).max()
        ctr, half = 0.5 * (lo + hi), 0.5 * (hi - lo).max() + pad
        lims[label] = (ctr - half, ctr + half)

    made = []
    for p in panels:
        z = np.load(os.path.join(out_dir, f"paths_{p['name']}.npz"))
        lo, hi = lims[p["label"]]
        span = float((hi - lo).max())
        period = 2.0 * np.pi / p["omega"]
        need = int(np.ceil(FIG_PTS_PER_PERIOD * span / period))
        use_qpi = need > FIG_GRID_MAX
        ng = int(np.clip(need, FIG_GRID_MIN, FIG_GRID_MAX))
        gx = np.linspace(lo[0], hi[0], ng)
        gy = np.linspace(lo[1], hi[1], ng)
        A2 = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, D)
        Z = su.alpha_Q * lqr.q_pi(su.sys_, su.ss, A2, su.sigma)
        if not use_qpi:
            pe_o = at_omega(su, p["omega"])
            u = (A2 - su.mu0) / su.sigma
            Z = Z + np.asarray(EF.e_value(pe_o, p["eps"], jnp.asarray(su.mu0),
                                          su.sigma, jnp.asarray(u)))
        Z = Z.reshape(ng, ng)

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.contour(gx, gy, Z.T, levels=FIG_LEVELS, colors="#bcbcbc", linewidths=0.4)
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(su.mu0[0] + su.sigma * np.cos(th), su.mu0[1] + su.sigma * np.sin(th),
                color="#9a9a9a", lw=0.5)
        for arm, lab in (("pw", "PW"), ("zo", "ZO")):
            mu = np.concatenate([su.mu0[None], z[f"mu_{arm}"][SEED_DISPLAY]], 0)
            ax.plot(mu[:, 0], mu[:, 1], color=COL[arm], lw=1.0, label=lab, zorder=3)
            ax.plot(mu[-1, 0], mu[-1, 1], "o", color=COL[arm], ms=3.2, zorder=4)
        ax.plot(su.mu0[0], su.mu0[1], "D", color="black", ms=3.4, zorder=5)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_xlabel(r"$a_1$")
        ax.set_ylabel(r"$a_2$")
        ax.legend(frameon=False, loc="best", handlelength=1.4)
        fig.tight_layout(pad=0.3)
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, f"lqr_paths_{p['name']}.pdf")
        fig.savefig(path)
        plt.close(fig)
        made.append(dict(name=p["name"], path=path, grid=ng, needed=need,
                         background="Q^pi" if use_qpi else "Q_phi",
                         periods_across=span / period))
        print(f"  {p['name']:>12} -> {os.path.basename(path)}  grid {ng} "
              f"(needed {need})  background "
              f"{'Q^pi (error period too fine)' if use_qpi else 'Q_phi'}", flush=True)
    with open(os.path.join(out_dir, "figures.json"), "w") as f:
        json.dump(made, f, indent=2, default=float)
    return made


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("gates", "calibrate", "paths", "figures", "report"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "lqr_paths"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    su = setup()
    print(f"state index {su.idx}  ||a* - mu_0|| = {su.target_dist:.6g}  "
          f"sigma = {su.sigma:.6g} ({'>=' if su.sigma >= 0.1 else '<'} min_std 0.1)  "
          f"eps_study = {su.eps_study:.6g}")

    calpath = os.path.join(a.out, "calibration.json")
    cal = None
    if os.path.exists(calpath):
        with open(calpath) as f:
            cal = json.load(f)

    if a.stage == "gates":
        ok = stage_gates(su, a.out, cal["panels"] if cal else None)
        record_env(a.out, "gates" + ("_panels" if cal else ""), time.time() - t0)
        sys.exit(0 if ok else 1)
    if a.stage == "calibrate":
        stage_calibrate(su, a.out)
    elif a.stage == "paths":
        gp = os.path.join(a.out, "gates_panels.json")
        if not os.path.exists(gp):
            sys.exit("refusing to run: panel gates have not been run "
                     "(gates_panels.json)")
        with open(gp) as f:
            g = json.load(f)
        bad = [r["name"] for r in g["rows"] if r["blocking"] and r["status"] != "PASS"]
        if bad:
            sys.exit(f"refusing to run: blocking gates not passing: {bad}")
        stage_paths(su, a.out, g["panels"])
    elif a.stage == "figures":
        stage_figures(su, a.out, cal["panels"])
    elif a.stage == "report":
        stage_report(su, a.out, cal)
    record_env(a.out, a.stage, time.time() - t0)


if __name__ == "__main__":
    main()
