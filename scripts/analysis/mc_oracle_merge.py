"""Merge replicate shards of a pilot run into one archive.

A shard is a run of scripts/analysis/mc_oracle_walker.py with MCO_REPS=lo:hi. Because
every rollout key folds on the ABSOLUTE replicate index, the shards together are
bit-identical to what a single unsharded run would have produced.

This refuses to merge unless the deterministic parts of every shard agree exactly --
Q_phi, the action grid, mu, sigma, u, the branch layout and the state selection -- so
a mismatched checkpoint, bank or pilot tag is caught rather than silently averaged.

Usage: mc_oracle_merge.py <out.npz> <shard0.npz> <shard1.npz> ...
"""

from __future__ import annotations

import sys

import numpy as np

STACK = ("q_oracle", "q_oracle_prefix", "n_done")
MUST_MATCH = ("q_phi", "action", "mu", "sigma", "u", "y", "offsets", "branch_c",
              "branch_j", "branch_sign", "state_index", "source", "ckpt", "tag",
              "horizon", "subset", "alpha", "gamma", "n_rep", "n_state", "k", "pilot")


def main(out, shards):
    zs = [np.load(p, allow_pickle=True) for p in shards]
    order = np.argsort([int(z["rep_lo"]) for z in zs])
    zs = [zs[i] for i in order]
    shards = [shards[i] for i in order]

    for k in MUST_MATCH:
        if k not in zs[0].files:
            continue
        for z, p in zip(zs[1:], shards[1:]):
            if not np.array_equal(zs[0][k], z[k]):
                raise SystemExit("shard %s disagrees with %s on %r" % (p, shards[0], k))

    lo = [int(z["rep_lo"]) for z in zs]
    hi = [int(z["rep_hi"]) for z in zs]
    n_rep = int(zs[0]["n_rep"])
    if lo[0] != 0 or hi[-1] != n_rep or any(hi[i] != lo[i + 1] for i in range(len(zs) - 1)):
        raise SystemExit("shards do not tile 0:%d exactly: %s"
                         % (n_rep, list(zip(lo, hi))))

    payload = {k: zs[0][k] for k in zs[0].files if k not in STACK + ("rep_lo", "rep_hi")}
    for k in STACK:
        payload[k] = np.concatenate([z[k] for z in zs], axis=1)
    assert payload["q_oracle"].shape[1] == n_rep, payload["q_oracle"].shape
    np.savez(out, **payload)
    print("merged %d shards -> %s  q_oracle %s  finite %s"
          % (len(zs), out, payload["q_oracle"].shape,
             bool(np.isfinite(payload["q_oracle"]).all())))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
