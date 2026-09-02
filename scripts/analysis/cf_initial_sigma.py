"""Measure the ACTUAL initial policy width (phase 1.6).

Candidate freeze value A is nominally exp(0) + min_std. That name is only honest if
the initial log_std output is provably zero for every state. It is not: the actor
trunk ends in an `nnx.Linear` whose kernel uses the default lecun-normal
initialisation (src/networks/jax_models.py:35-49), so the pre-squash log_std at
initialisation is a random, state-dependent, seed-dependent quantity.

This measures how far from 1.1 the real initial width actually is, on the committed
256-state WalkerRun bank, across several model seeds.

Usage: cf_initial_sigma.py <bank.npz> <ckpt_for_normalizer> <out.json>
"""

from __future__ import annotations

import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import nnx  # noqa: E402

from src.networks.jax_models import SACActorNetworks  # noqa: E402

# exactly the kwargs make_init uses for WalkerRun (reppo.py:364-377); note that
# min_std is NOT passed, so the constructor default 0.1 applies -- the config field
# `actor_min_std: 0.0` is never plumbed through.
KW = dict(obs_dim=24, action_dim=6, hidden_dim=512, ent_start=0.014509912580251694,
          kl_start=0.01, use_norm=True, layers=3, use_skip=False)
SEEDS = [0, 1, 2, 301, 302, 303, 304, 305]


def main(bank, ckpt, out):
    z = np.load(bank, allow_pickle=True)
    obs = np.asarray(z["__obs__"], np.float32)
    nz = np.load(os.path.join(ckpt, "normalizer.npz"))
    eps = 0.01
    nobs = jnp.asarray((obs - nz["mean"]) / np.sqrt(nz["var"] + eps))

    min_std = 0.1
    nominal = math.exp(0.0) + min_std
    res = {"nominal_zero_logstd_reference": nominal, "min_std": min_std,
           "n_states": int(obs.shape[0]), "bank": os.path.basename(bank),
           "normalizer_from": os.path.basename(ckpt), "per_seed": {}}

    allsig = []
    for s in SEEDS:
        a = SACActorNetworks(**KW, with_eta=False, with_betas=False, rngs=nnx.Rngs(s))
        assert float(a.min_std) == min_std, a.min_std
        _, sg = a.gaussian(nobs)
        sg = np.asarray(sg, np.float64)
        allsig.append(sg)
        res["per_seed"][str(s)] = {
            "min": float(sg.min()), "median": float(np.median(sg)),
            "mean": float(sg.mean()), "p95": float(np.percentile(sg, 95)),
            "max": float(sg.max()),
            "per_coord_median": np.median(sg, 0).round(4).tolist(),
            "frac_within_1pct_of_nominal":
                float((np.abs(sg - nominal) < 0.01 * nominal).mean()),
        }
    A = np.concatenate(allsig, 0)
    res["pooled"] = {
        "min": float(A.min()), "median": float(np.median(A)), "mean": float(A.mean()),
        "p95": float(np.percentile(A, 95)), "max": float(A.max()),
        "std_across_states_within_seed":
            float(np.mean([s.std() for s in allsig])),
        "spread_of_seed_medians":
            float(np.std([np.median(s) for s in allsig], ddof=1)),
        "frac_within_1pct_of_nominal":
            float((np.abs(A - nominal) < 0.01 * nominal).mean()),
        "exactly_nominal": bool(np.all(A == nominal)),
    }
    with open(out, "w") as f:
        json.dump(res, f, indent=1)

    p = res["pooled"]
    print("NOMINAL exp(0)+min_std                = %.17g" % nominal)
    print("measured initial sigma, %d seeds x %d states x 6 coords:"
          % (len(SEEDS), obs.shape[0]))
    print("  min %.4f   median %.4f   mean %.4f   p95 %.4f   max %.4f"
          % (p["min"], p["median"], p["mean"], p["p95"], p["max"]))
    print("  sd across states within a seed : %.4f" % p["std_across_states_within_seed"])
    print("  sd of the per-seed medians     : %.4f" % p["spread_of_seed_medians"])
    print("  fraction within 1%% of nominal  : %.4f" % p["frac_within_1pct_of_nominal"])
    print("  exactly equal to nominal       : %s" % p["exactly_nominal"])
    print()
    print("VERDICT: initial log_std is %s zero"
          % ("EXACTLY" if p["exactly_nominal"] else "NOT"))
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
