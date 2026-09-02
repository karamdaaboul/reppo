"""Consolidate the four raw pilot archives into reports/artifacts/mc_oracle_pilot_raw.npz.

Rollout arrays are stored as float32 (the oracle returns are O(50) in value units, so
float32 carries ~6 significant digits, far below the MC standard error); the
deterministic Q_phi values, the action grid and the design metadata stay float64.

Usage: mc_oracle_package.py <indir> <out.npz>
"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

RUNS = (("PW", "pilot_PW.npz"), ("WML", "pilot_WML.npz"),
        ("PW_H1000", "horizon_PW.npz"), ("WML_H1000", "horizon_WML.npz"))
F32 = ("q_oracle", "q_oracle_prefix", "n_done")


def main(indir, out):
    payload = {}
    for tag, fn in RUNS:
        z = np.load(os.path.join(indir, fn), allow_pickle=True)
        for k in z.files:
            v = z[k]
            if k in F32:
                v = v.astype(np.float32)
            payload["%s/%s" % (tag, k)] = v
        print("%-10s %s  q_phi %s  q_oracle %s" %
              (tag, fn, z["q_phi"].shape, z["q_oracle"].shape))
    np.savez_compressed(out, **payload)
    h = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print("wrote %s  %.1f MB  sha256 %s" %
          (out, os.path.getsize(out) / 1e6, h))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
