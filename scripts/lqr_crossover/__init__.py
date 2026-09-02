"""LQR estimator-crossover harness (CPU-only, float64).

Measures the Claim 4 crossover between the pathwise and zeroth-order operators in a
setting where Q^pi is available in closed form and the critic error e = Q_phi - Q^pi is
PLANTED with a chosen frequency, so omega = ||grad e|| / ||e||_inf is an exact input
rather than the proxy of Section 8.

The estimator core is imported from `src.jaxrl.estimators`, the same module the
training diagnostics and the offline probes call, so a pass here is evidence about the
shipped operators and not merely about this harness. See docs/prereg_lqr_crossover.md.

IMPORT ORDER IS LOAD-BEARING. `src/jaxrl/__init__.py` calls `jax.config.update` at
import time, and `jax_enable_x64` must be set before any array is created, so this
module configures the platform and precision before anything under `src.jaxrl` is
reachable. Always `import scripts.lqr_crossover` first in an entry point.

float64 is not a nicety: the crossover is located where two variances are equal by
construction, and float32 does not resolve a ratio of near-equal quantities.

CPU is not a nicety either: two CUDA devices are visible on the development box and
the DMC ladder owns them.
"""

from __future__ import annotations

import hashlib
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

# G11, asserted at import and again inside the sweep kernel.
assert jax.devices()[0].platform == "cpu", (
    f"expected CPU, got {jax.devices()[0].platform}; the DMC ladder owns the GPUs"
)
assert jnp.zeros(1).dtype == jnp.float64, "x64 did not take effect"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Date-derived, house convention. A NEW namespace: these are CPU analysis runs and
# must never consume the confirmatory seeds 101-108 (see ledger/README.md).
SEED_ROOT = 20260902


def probe_key(tag: str) -> jax.Array:
    """Fresh PRNG, never the training key. blake2b, not hash(): the latter is salted."""
    dig = hashlib.blake2b(tag.encode(), digest_size=4).digest()
    return jax.random.fold_in(
        jax.random.PRNGKey(SEED_ROOT), int.from_bytes(dig, "big") % (2**31)
    )


__all__ = ["OUT", "REPO_ROOT", "SEED_ROOT", "probe_key"]
