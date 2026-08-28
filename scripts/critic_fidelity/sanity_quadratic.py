"""Guard (g): validate the smoothed-gradient estimator on a closed-form problem.

f(a) = -||a - a*||^2, true gradient -2(a - a*).

For a quadratic the antithetic difference cancels every even-order term exactly:

    f(a + s u) - f(a - s u) = -4 s (a - a*) . u  =  2 s (grad f . u)

so ``(f+ - f-) / (2 s) = grad f . u`` with **no s dependence at all**. That gives two
tests of very different strength:

  1. EXACT: hold u fixed, sweep s -- the estimate must be numerically identical.
     Any bug in the sigma bookkeeping shows up here immediately, with no sampling
     noise to hide behind.
  2. STATISTICAL: the estimator is unbiased, so cosine similarity to the true
     gradient must approach 1 as N grows, at a rate set by D/N.

Run before pointing the harness at a critic.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from scripts.critic_fidelity.common import antithetic_grad, cosine  # noqa: E402


def quadratic(a, a_star):
    return -jnp.sum((a - a_star) ** 2, axis=-1)


def estimate(a, a_star, u, sigma):
    fp = quadratic(a + sigma * u, a_star)
    fm = quadratic(a - sigma * u, a_star)
    return np.asarray(antithetic_grad(np.asarray(fp), np.asarray(fm), np.asarray(u), sigma))


def main() -> int:
    failures = []
    rng = np.random.default_rng(0)

    print("=" * 72)
    print("Test 1 -- EXACT sigma-invariance (fixed u, quadratic)")
    print("=" * 72)
    for D in (6, 21):
        a_star = jnp.asarray(rng.normal(size=(D,)), dtype=jnp.float32)
        a = jnp.asarray(rng.normal(size=(D,)), dtype=jnp.float32)
        u = jnp.asarray(rng.normal(size=(64, D)), dtype=jnp.float32)
        g_true = np.asarray(-2.0 * (a - a_star))
        ref = None
        worst = 0.0
        print(f"  D={D}:")
        for sigma in [1e-3, 1e-2, 1e-1, 3e-1, 1.0]:
            g = estimate(a, a_star, u, sigma)
            if ref is None:
                ref = g
            dev = float(np.abs(g - ref).max() / max(np.abs(ref).max(), 1e-12))
            worst = max(worst, dev)
            print(
                f"    sigma={sigma:<6g} |g|={np.linalg.norm(g):8.4f} "
                f"cos(g,g_true)={cosine(g, g_true):.6f}  dev_vs_ref={dev:.2e}"
            )
        # float32 evaluation of f at large sigma loses digits; 1e-3 is generous headroom
        if worst > 1e-3:
            failures.append(f"D={D}: estimate varies with sigma (dev {worst:.2e})")
        print(f"    -> worst deviation across sigma: {worst:.2e}")

    print()
    print("=" * 72)
    print("Test 2 -- unbiasedness: cosine to true gradient vs N")
    print("=" * 72)
    for D in (6, 21):
        a_star = jnp.asarray(rng.normal(size=(D,)), dtype=jnp.float32)
        a = jnp.asarray(rng.normal(size=(D,)), dtype=jnp.float32)
        g_true = np.asarray(-2.0 * (a - a_star))
        print(f"  D={D}:  (theory: cos -> 1 as N/D grows)")
        prev = -1.0
        for N in (8, 16, 64, 256, 1024):
            cs = []
            for rep in range(32):
                u = jnp.asarray(
                    np.random.default_rng(1000 + rep).normal(size=(N, D)), dtype=jnp.float32
                )
                cs.append(cosine(estimate(a, a_star, u, 0.1), g_true))
            m = float(np.mean(cs))
            print(f"    N={N:<5d} mean cos = {m:.4f}")
            if N == 64:
                n64 = m
            prev = m
        # the headline configuration used by experiment 2
        thresh = 0.90 if D == 6 else 0.70
        if n64 < thresh:
            failures.append(f"D={D}: N=64 cosine {n64:.3f} below {thresh}")
        if prev < 0.97:
            failures.append(f"D={D}: does not converge (N=1024 cosine {prev:.3f})")

    print()
    print("=" * 72)
    print("Test 3 -- antithetic pairing actually reduces variance")
    print("=" * 72)
    for D in (6, 21):
        a_star = jnp.asarray(rng.normal(size=(D,)), dtype=jnp.float32)
        a = jnp.asarray(rng.normal(size=(D,)), dtype=jnp.float32)
        g_true = np.asarray(-2.0 * (a - a_star))
        anti, one = [], []
        for rep in range(64):
            u = jnp.asarray(
                np.random.default_rng(5000 + rep).normal(size=(64, D)), dtype=jnp.float32
            )
            anti.append(cosine(estimate(a, a_star, u, 0.1), g_true))
            # one-sided: (f(a+su) - f(a)) / s
            fp = quadratic(a + 0.1 * u, a_star)
            f0 = quadratic(a, a_star)
            g1 = np.asarray(
                ((np.asarray(fp) - float(f0))[:, None] * np.asarray(u)).sum(0) / (0.1 * 64)
            )
            one.append(cosine(g1, g_true))
        print(
            f"  D={D}: antithetic cos={np.mean(anti):.4f}+-{np.std(anti):.4f}   "
            f"one-sided cos={np.mean(one):.4f}+-{np.std(one):.4f}"
        )
        if np.mean(anti) < np.mean(one):
            failures.append(f"D={D}: antithetic no better than one-sided")

    print()
    if failures:
        print("GUARD (g) FAILED:")
        for f in failures:
            print("  - " + f)
        return 1
    print("GUARD (g) PASSED: estimator is sigma-invariant on a quadratic and unbiased.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
