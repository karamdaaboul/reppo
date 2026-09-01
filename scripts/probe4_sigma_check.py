"""Committed contamination check from docs/prereg_action_padding.md.

  CONTAMINATED (report separately, claim neither) if: at the final checkpoint, median
  sigma over the padded coordinates exceeds 1.5x median sigma over the real 6
  coordinates.

Evaluated on visited states under each checkpoint's own policy.  This is a check on
the REGENERATED checkpoints; the original k=16 runs were already described as
contaminated in the padding plan's own language, and Probe 4 does not attempt to
rehabilitate the return-level experiment either way.
"""
import glob, json, os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.critic_fidelity.common import ACTION_CLIP, Harness

R_DIM, B, BURN = 6, 256, 200
rows = []
for d in sorted(glob.glob("exports/WalkerRun_*pad16_s*_final")):
    h = Harness(d, B)
    key = jax.random.PRNGKey(999)
    key, rk = jax.random.split(key)
    obs, _, st = h.reset(rk)
    for t in range(BURN):
        k1, k2, key = jax.random.split(key, 3)
        a = jnp.clip(h.pi(obs).sample(seed=k1), -ACTION_CLIP, ACTION_CLIP)
        obs, _, st, _, _, _ = h.env.step(jax.random.split(k2, B), st, a)
    sg = np.asarray(h.pi(obs).distribution.scale, dtype=np.float64)
    real, pad = np.median(sg[:, :R_DIM]), np.median(sg[:, R_DIM:])
    arm = "A" if "pathwise" in d else "B"
    rows.append(dict(ckpt=os.path.basename(d), arm=arm,
                     sigma_real_med=real, sigma_pad_med=pad, ratio=pad / real,
                     contaminated=bool(pad > 1.5 * real)))
    print("%-44s arm %s  sigma_real=%.4f sigma_pad=%.4f  ratio=%.3f  %s"
          % (rows[-1]["ckpt"], arm, real, pad, pad / real,
             "CONTAMINATED" if pad > 1.5 * real else "ok"), flush=True)
import pandas as pd
df = pd.DataFrame(rows)
df.to_csv("reports/artifacts/probe4_sigma_contamination.csv", index=False)
print("\ncommitted rule: contaminated if median sigma_pad > 1.5 * median sigma_real")
print("contaminated checkpoints: %d/%d   (arm A %d/5, arm B %d/5)"
      % (df.contaminated.sum(), len(df),
         df[df.arm == "A"].contaminated.sum(), df[df.arm == "B"].contaminated.sum()))
