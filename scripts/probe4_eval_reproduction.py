"""Integrity check: does loading a regenerated checkpoint reproduce its logged eval?

Rolls each exported k=16 checkpoint's own policy in the padded environment with the
frozen normalizer and compares the mean episode return against `final_eval_return`
stored in meta.json.  A large mismatch would mean the export or the loader is wrong
and every Probe 4 number computed on it would be suspect.

Uses the STOCHASTIC policy, which is what the trainer evaluates, so agreement is
expected only up to evaluation noise; the check reports the gap in units of the
across-episode standard error.
"""
import glob, json, os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.critic_fidelity.common import ACTION_CLIP, Harness

B, HORIZON = 256, 1000
rows = []
for d in sorted(glob.glob("exports/WalkerRun_*pad16_s*_final")):
    m = json.load(open(f"{d}/meta.json"))
    h = Harness(d, B)
    key = jax.random.PRNGKey(12345)
    key, rk = jax.random.split(key)
    obs, _, st = h.reset(rk)
    tot = jnp.zeros(B)
    for t in range(HORIZON):
        k1, k2, key = jax.random.split(key, 3)
        a = jnp.clip(h.pi(obs).sample(seed=k1), -ACTION_CLIP, ACTION_CLIP)
        obs, _, st, rew, done, _ = h.env.step(jax.random.split(k2, B), st, a)
        tot = tot + rew
    tot = np.asarray(tot)
    logged = m["final_eval_return"]
    se = tot.std(ddof=1) / np.sqrt(B)
    rows.append(dict(ckpt=os.path.basename(d), logged=logged, rolled=float(tot.mean()),
                     se=float(se), gap_in_se=float((tot.mean() - logged) / se),
                     rel_gap=float((tot.mean() - logged) / abs(logged))))
    print("%-44s logged=%8.2f rolled=%8.2f +-%5.2f  gap=%+6.2f SE  rel=%+.4f"
          % (rows[-1]["ckpt"], logged, tot.mean(), se, rows[-1]["gap_in_se"],
             rows[-1]["rel_gap"]), flush=True)

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv("reports/artifacts/probe4_eval_reproduction.csv", index=False)
worst = df.rel_gap.abs().max()
print("\nmax |relative gap| = %.4f over %d checkpoints -> %s"
      % (worst, len(df), "PASS (<5%)" if worst < 0.05 else "CHECK"))
