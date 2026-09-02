"""Crossed frozen-critic dispersion analysis (docs/prereg_crossed_dispersion.md, d1de4e8).

Read-only with respect to every checkpoint. No training. The only environment
interaction is state-bank collection, which the preregistration defines as such.

  bank <task> <out.npz>              collect the preregistered 2048-state bank
  run  <task> <bank.npz> <out.csv>   the crossed design under BOTH reference laws

Preregistered constants live in PREREG and are never taken from the command line.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from functools import partial

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
import numpy as np                                            # noqa: E402
from jax.scipy.special import logsumexp                       # noqa: E402

from scripts.critic_fidelity.common import ACTION_CLIP, Harness    # noqa: E402
from scripts.load_ckpt import load                                 # noqa: E402

PREREG = dict(
    root=20260904,                                # sec 7 / 13
    n_bank=2048, per_policy=128, burn_in=50,      # sec 7
    s_eval=128, R=200, M=32,                      # sec 13
    eta_lo=1e-4, eta_hi=10.0,                     # sec 8; jax_models.py eta()
    seeds=tuple(range(301, 309)),                 # sec 10, corrected tier
    arms={"PW": "pathwise_fa", "WML": "weighted_mle"},
)
TASK_ENV = {"walker": "WalkerRun", "g1": "G1JoystickFlatTerrain"}
OPERATORS = ("PW-32", "PW-1", "ZO-32", "WML-32")


def fold(tag, *parts):
    s = "|".join([str(tag)] + [str(p) for p in parts])
    h = hashlib.blake2b(s.encode(), digest_size=4).digest()
    return jax.random.fold_in(jax.random.PRNGKey(PREREG["root"]),
                              int.from_bytes(h, "big") % (2 ** 31))


def ckpt_dir(task, arm, seed):
    return os.path.join(REPO, "exports",
                        "%s_%s_s%d_final" % (TASK_ENV[task], PREREG["arms"][arm], seed))


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ------------------------------------------------------------------ state bank
def collect_bank(task, out):
    n, burn = PREREG["per_policy"], PREREG["burn_in"]
    parts, src, prov = [], [], []
    for arm in ("PW", "WML"):
        for seed in PREREG["seeds"]:
            d = ckpt_dir(task, arm, seed)
            h = Harness(d, n)
            key = fold("bank", task, arm, seed)
            key, rk = jax.random.split(key)
            obs, _, st = h.reset(rk)
            for _ in range(burn):                      # q_spread_from_ckpt.py:38-42
                k1, k2, key = jax.random.split(key, 3)
                a = jnp.clip(h.pi(obs).sample(seed=k1), -ACTION_CLIP, ACTION_CLIP)
                obs, _, st, _, _, _ = h.env.step(jax.random.split(k2, n), st, a)
            parts.append(np.asarray(obs))
            src += ["%s-s%d" % (arm, seed)] * n
            prov.append(dict(arm=arm, seed=seed, ckpt=os.path.relpath(d, REPO),
                             actor_sha256=sha256_file(os.path.join(d, "actor.npz")),
                             n_states=n))
            print("  %-4s seed %d collected %d states" % (arm, seed, n), flush=True)
    raw = np.concatenate(parts, 0)
    assert raw.shape[0] == PREREG["n_bank"], raw.shape
    idx = np.sort(np.asarray(
        jax.random.permutation(fold("s_eval", task), raw.shape[0]), np.int64
    )[:PREREG["s_eval"]])
    np.savez(out, obs=raw, source=np.array(src), eval_idx=idx,
             task=np.array(task), burn_in=np.array(burn),
             root=np.array(PREREG["root"]), provenance=np.array(json.dumps(prov)))
    print("bank %s  states %d  eval %d  distinct %d\nsha256 %s"
          % (out, raw.shape[0], len(idx), len(np.unique(raw, axis=0)), sha256_file(out)))


# ------------------------------------------------------------------- estimators
def _dual_g(F, eta, eps_e):
    """g(eta) = eta*eps_e + eta*(logsumexp(F/eta) - log M).  F (...,M), eta (...)."""
    e = eta[..., None]
    return jnp.squeeze(
        e * eps_e + e * (logsumexp(F / e, axis=-1, keepdims=True)
                         - jnp.log(F.shape[-1])), -1)


def solve_eta_dual(F, eps_e, lo, hi, iters=80):
    """Golden-section minimisation of the convex E-step dual, in log(eta).

    No grid: a grid over (R,S,M) would allocate hundreds of MB. Golden section on a
    convex function is deterministic and needs no initial guess.
    """
    phi = (jnp.sqrt(5.0) - 1.0) / 2.0
    a = jnp.full(F.shape[:-1], jnp.log(lo))
    b = jnp.full(F.shape[:-1], jnp.log(hi))
    for _ in range(iters):
        c, d_ = b - phi * (b - a), a + phi * (b - a)
        gc, gd = _dual_g(F, jnp.exp(c), eps_e), _dual_g(F, jnp.exp(d_), eps_e)
        take = gc < gd
        a = jnp.where(take, a, c)
        b = jnp.where(take, d_, b)
    return jnp.clip(jnp.exp(0.5 * (a + b)), lo, hi)


def build_updates(F, grads, u, sigma, eps_e, lo, hi):
    """All preregistered operators from one action cloud.

    F (R,S,M) | grads (R,S,M,d) = grad_y Q(s,tanh(y)) | u (R,S,M,d) | sigma (S,d)
    sigma is the REFERENCE-LAW sigma (prereg sec 3).
    """
    sg = sigma[None, :, None, :]
    out = {}
    out["PW-32"] = grads.mean(2)
    out["PW-1"] = grads[:, :, 0, :]                        # nested: shares innovations
    dF = F - F.mean(2, keepdims=True)                      # centered; no softmax, no ubar
    out["ZO-32"] = (dF[..., None] * (u / sg)).mean(2)
    eta = solve_eta_dual(F, eps_e, lo, hi)                 # (R,S)
    w = jax.nn.softmax(F / eta[..., None], axis=-1)
    out["WML-32"] = (w[..., None] * (sg * u)).sum(2)       # sum_i w_i (y_i - mu)
    return out, eta, w


def dispersion(g, sigma):
    """Prereg sec 2: unit whitened norm, then D(s) = 1 - ||gbar||^2_{Sigma^-1}."""
    wn = jnp.sqrt(jnp.sum((g / sigma[None]) ** 2, axis=-1, keepdims=True))
    n_zero = jnp.sum(wn <= 0)
    gh = g / jnp.where(wn > 0, wn, 1.0)
    gbar = gh.mean(0)
    return 1.0 - jnp.sum((gbar / sigma) ** 2, axis=-1), jnp.squeeze(wn, -1), n_zero


# ------------------------------------------------------------------------- cell
def _critic_batch(ck, raw_S, y, chunk):
    """F and grad_y Q(s, tanh(y)), chunked over the cloud axis to bound memory."""
    R, S, M, d = y.shape
    ob = raw_S.shape[-1]

    @jax.jit
    def one(yc):
        rc = jnp.broadcast_to(raw_S[None, :, None, :], (yc.shape[0], S, M, ob))
        rf, yf = rc.reshape(-1, ob), yc.reshape(-1, d)
        F = ck.q_scalar(rf, jnp.tanh(yf))
        g = jax.vmap(jax.grad(lambda yy, ss: ck.q_scalar(ss, jnp.tanh(yy))))(yf, rf)
        return F.reshape(yc.shape[0], S, M), g.reshape(yc.shape[0], S, M, d)

    Fs, Gs = [], []
    for i in range(0, R, chunk):
        f, g = one(y[i:i + chunk])
        Fs.append(f); Gs.append(g)
    return jnp.concatenate(Fs, 0), jnp.concatenate(Gs, 0)


def run_task(task, bank_path, out_csv, chunk=25):
    z = np.load(bank_path, allow_pickle=True)
    raw_all = np.asarray(z["obs"], np.float32)
    raw = jnp.asarray(raw_all[np.asarray(z["eval_idx"], np.int64)])
    S, R, M = raw.shape[0], PREREG["R"], PREREG["M"]
    assert S == PREREG["s_eval"], S
    rows, diag, checks = [], [], []

    for seed in PREREG["seeds"]:
        cks = {a: load(ckpt_dir(task, a, seed)) for a in ("PW", "WML")}
        d = int(cks["PW"].meta["action_dim"])
        # prereg sec 6: innovations depend on (task, seed) ONLY -- not on the
        # reference law and not on the critic source
        u = jax.random.normal(fold("u", task, seed), (R, S, M, d))
        u_hash = hashlib.sha256(np.asarray(u, np.float32).tobytes()).hexdigest()

        per_law = {}
        for law, law_arm in (("A", "PW"), ("B", "WML")):
            mu, sigma = cks[law_arm].policy_dist(raw)
            mu, sigma = jnp.asarray(mu), jnp.asarray(sigma)
            y = mu[None, :, None, :] + sigma[None, :, None, :] * u
            a_act = jnp.tanh(y)                              # prereg sec 9: no clip
            sat = float(jnp.mean(jnp.abs(a_act) > ACTION_CLIP))
            per_law[law] = dict(sigma=sigma, y=y, sat=sat)

            for csrc in ("PW", "WML"):
                ck = cks[csrc]
                eps_e = float(ck.meta["eps_e"])
                F, G = _critic_batch(ck, raw, y, chunk)
                ops, eta, w = build_updates(F, G, u, sigma, eps_e,
                                            PREREG["eta_lo"], PREREG["eta_hi"])
                # the dual condition, checked not assumed
                gopt = _dual_g(F, eta, eps_e)
                gpert = jnp.minimum(_dual_g(F, eta * 1.01, eps_e),
                                    _dual_g(F, eta * 0.99, eps_e))
                for name, g in ops.items():
                    D_s, wn, nz = dispersion(g, sigma)
                    rows.append(dict(
                        task=task, seed=seed, law=law, law_arm=law_arm, critic=csrc,
                        operator=name, D=float(jnp.mean(D_s)),
                        D_se_states=float(jnp.std(D_s, ddof=1) / jnp.sqrt(S)),
                        raw_norm_median=float(jnp.median(wn)),
                        n_zero_norm=int(nz), saturation=sat, M=(1 if name == "PW-1" else M),
                        R=R, S_eval=S, u_sha256=u_hash))
                diag.append(dict(
                    task=task, seed=seed, law=law, critic=csrc,
                    eta_median=float(jnp.median(eta)), eta_min=float(jnp.min(eta)),
                    eta_max=float(jnp.max(eta)),
                    eta_at_lo=float(jnp.mean(eta <= PREREG["eta_lo"] * 1.001)),
                    eta_at_hi=float(jnp.mean(eta >= PREREG["eta_hi"] * 0.999)),
                    dual_gap_max=float(jnp.max(gopt - gpert)),
                    w_max_median=float(jnp.median(w.max(-1))),
                    sigma_median=float(jnp.median(sigma)),
                    sigma_p95=float(jnp.percentile(sigma, 95)),
                    sigma_min=float(jnp.min(sigma)), sigma_max=float(jnp.max(sigma)),
                    sigma_across_states=float(jnp.median(jnp.std(sigma, axis=0))),
                    sigma_across_coords=float(jnp.median(jnp.std(sigma, axis=1))),
                    saturation=sat,
                    eta_saved=(float(np.asarray(ck.actor.eta()).ravel()[0])
                               if csrc == "WML" else float("nan"))))
            print("  seed %d law %s done" % (seed, law), flush=True)

        # integrity: only the law changed
        checks.append(dict(seed=seed, u_sha256=u_hash,
                           law_changes_sigma=bool(not np.array_equal(
                               np.asarray(per_law["A"]["sigma"]),
                               np.asarray(per_law["B"]["sigma"]))),
                           law_changes_y=bool(not np.array_equal(
                               np.asarray(per_law["A"]["y"]),
                               np.asarray(per_law["B"]["y"])))))

    with open(out_csv, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)
    base = out_csv.rsplit(".", 1)[0]
    with open(base + "_diagnostics.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(diag[0].keys()))
        wtr.writeheader(); wtr.writerows(diag)
    with open(base + "_checks.json", "w") as f:
        json.dump(dict(bank=bank_path, bank_sha256=sha256_file(bank_path),
                       prereg=PREREG, checks=checks), f, indent=1, default=str)
    print("wrote %s (%d rows), %s, %s" % (out_csv, len(rows),
                                          base + "_diagnostics.csv",
                                          base + "_checks.json"))


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "bank":
        collect_bank(sys.argv[2], sys.argv[3])
    elif m == "run":
        run_task(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit("modes: bank | run")
