"""Isotropic-width KL budget for the sec:entfail mechanism claim.

Exact, not the small-y approximation. Emits reports/artifacts/mech_kl_width.json.
Read-only: no checkpoints, no training.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from scipy.optimize import brentq

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
BOUND, S0, S1 = 0.1, 0.279, 0.850          # kl_bound; sigma before/after, from sec:entfail

def kl_sigma(c, d):
    """D_KL( N(mu, s^2 I) || N(mu, c^2 s^2 I) ) = (d/2)(c^-2 - 1 + 2 log c)."""
    return 0.5 * d * (c ** -2 - 1.0 + 2.0 * np.log(c))

def main():
    try:
        from mujoco_playground import registry
        d = int(registry.load("HumanoidRun").action_size)
        d_src = "mujoco_playground registry, HumanoidRun.action_size"
    except Exception as e:
        d, d_src = 21, "registry unavailable (%s); documented value" % type(e).__name__
    c_obs = S1 / S0
    c_max = brentq(lambda c: kl_sigma(c, d) - BOUND, 1.0 + 1e-12, 5.0)
    n_exact = float(np.log(c_obs) / np.log(c_max))
    y_approx = float(np.sqrt(BOUND / d))               # from KL ~ d y^2
    c_approx = 1.0 + y_approx
    n_approx = float(np.log(c_obs) / np.log(c_approx))
    out = dict(
        d=d, d_source=d_src, kl_bound=BOUND,
        sigma_before=S0, sigma_after=S1, c_observed=float(c_obs),
        kl_sigma_at_c_observed=float(kl_sigma(c_obs, d)),
        c_max_exact=float(c_max), y_max_exact=float(c_max - 1.0),
        applications_exact=n_exact, applications_exact_ceil=int(np.ceil(n_exact)),
        y_small_y_approximation=y_approx,
        applications_small_y=n_approx, applications_small_y_ceil=int(np.ceil(n_approx)),
        outer_iterations_for_this_experiment="NOT RECOVERABLE",
        outer_iterations_source="reports/corrected_replication_code_trace.md:129,133",
        conditional_if_shipped_mjx_dmc=dict(
            num_envs=1024, num_steps=128, total_time_steps=50_000_000, num_eval=20,
            outer_iterations=399, outer_iterations_per_eval=19,
            inner_updates_per_outer_iteration=512),
        budget_scope="per OUTER iteration; actor_target is a hard copy held fixed for "
                     "all inner updates (src/jaxrl/reppo.py:693-697)",
        caveat="best-case width-only bound; mean movement and anisotropy also consume budget",
    )
    for k, v in out.items():
        if not isinstance(v, dict):
            print("  %-34s %s" % (k, v))
    json.dump(out, open("reports/artifacts/mech_kl_width.json", "w"), indent=1)
    print("\nwrote reports/artifacts/mech_kl_width.json")

if __name__ == "__main__":
    main()
