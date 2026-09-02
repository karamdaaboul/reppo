"""Integrity follow-up for the WalkerRun gate: are the two anomalies mechanistic?

A. The E-step dual gap. run mode reports max(g(eta) - min(g(1.01 eta), g(0.99 eta)))
   which is > 0 in some cells. If eta is the minimiser this is <= 0, so a positive
   value is either a solver failure or the preregistered clip [1e-4, 10] binding.
   Split the gap by whether eta is interior.

B. Zero-norm pathwise updates, present only under the B (WML) law. If they are
   tanh saturation the vanishing gradient must coincide with |tanh(y)| -> 1, and
   must be identical across critic sources because y is shared.

Read-only. Uses the frozen bank. R is reduced here; this diagnoses a mechanism,
it does not restate any gate number.
"""
from __future__ import annotations
import sys
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, "/rwthfs/rz/cluster/home/qzi10910/repos/reppo")
import scripts.analysis.crossed_dispersion as CD
from scripts.load_ckpt import load

R_CHK = 20
z = np.load("reports/artifacts/cd_bank_walker_corrected.npz", allow_pickle=True)
raw = jnp.asarray(np.asarray(z["obs"], np.float32)[np.asarray(z["eval_idx"], np.int64)])
S = raw.shape[0]
M = CD.PREREG["M"]
lo, hi = CD.PREREG["eta_lo"], CD.PREREG["eta_hi"]

print("cell = seed 301; R reduced to %d for this diagnostic; S=%d M=%d" % (R_CHK, S, M))
cks = {a: load(CD.ckpt_dir("walker", a, 301)) for a in ("PW", "WML")}
d = int(cks["PW"].meta["action_dim"])
u = jax.random.normal(CD.fold("u", "walker", 301), (CD.PREREG["R"], S, M, d))[:R_CHK]

for law, arm in (("A", "PW"), ("B", "WML")):
    mu, sigma = cks[arm].policy_dist(raw)
    mu, sigma = jnp.asarray(mu), jnp.asarray(sigma)
    y = mu[None, :, None, :] + sigma[None, :, None, :] * u
    sat_frac = float(jnp.mean(jnp.abs(jnp.tanh(y)) > 0.999))
    print("\n--- law %s (reference arm %s) | saturation |tanh y|>0.999 = %.4f"
          % (law, arm, sat_frac))

    zeros_by_critic = {}
    for csrc in ("PW", "WML"):
        ck = cks[csrc]
        eps_e = float(ck.meta["eps_e"])
        F, G = CD._critic_batch(ck, raw, y, 5)
        eta = CD.solve_eta_dual(F, eps_e, lo, hi)
        gopt = CD._dual_g(F, eta, eps_e)
        gp = jnp.minimum(CD._dual_g(F, eta * 1.01, eps_e),
                         CD._dual_g(F, eta * 0.99, eps_e))
        gap = gopt - gp
        interior = (eta > lo * 1.001) & (eta < hi * 0.999)
        fi, fc = float(jnp.mean(interior)), float(jnp.mean(~interior))
        gi = float(jnp.max(jnp.where(interior, gap, -jnp.inf))) if fi > 0 else float("nan")
        gc = float(jnp.max(jnp.where(~interior, gap, -jnp.inf))) if fc > 0 else float("nan")
        # A: gap split by clipping
        print("  critic %-4s eta interior %.3f clipped %.3f | max gap interior %+.3g"
              "  max gap clipped %+.3g" % (csrc, fi, fc, gi, gc))
        # B: zero pathwise norms vs saturation
        pw32 = G.mean(2)
        wn = jnp.sqrt(jnp.sum((pw32 / sigma[None]) ** 2, -1))
        zmask = wn <= 0
        zeros_by_critic[csrc] = np.asarray(zmask)
        if float(jnp.mean(zmask)) > 0:
            satpc = jnp.mean((jnp.abs(jnp.tanh(y)) > 0.999).all(-1).all(-1))
            zs = float(jnp.mean(jnp.where(zmask, jnp.mean(
                (jnp.abs(jnp.tanh(y)) > 0.999).astype(jnp.float32), axis=(2, 3)), 0.0)
            ) / max(float(jnp.mean(zmask)), 1e-12))
            nz = float(jnp.mean(jnp.where(~zmask, jnp.mean(
                (jnp.abs(jnp.tanh(y)) > 0.999).astype(jnp.float32), axis=(2, 3)), 0.0)
            ) / max(float(jnp.mean(~zmask)), 1e-12))
            print("      PW-32 zero-norm frac %.5f | mean saturation in zero-norm "
                  "clouds %.4f vs %.4f elsewhere" % (float(jnp.mean(zmask)), zs, nz))
        else:
            print("      PW-32 zero-norm frac 0.00000")
    same = np.array_equal(zeros_by_critic["PW"], zeros_by_critic["WML"])
    print("  zero-norm mask identical across critic sources: %s "
          "(y is shared, so saturation must give an identical mask)" % same)
