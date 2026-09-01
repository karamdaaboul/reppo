#!/usr/bin/env python
"""Saturation + two-way KL reconstruction on an exported final checkpoint.

Read-only w.r.t. the run tree: loads a checkpoint, writes one JSON to --outdir.

Visited-state recipe mirrors scripts/audit_saturation.py: B=256 envs, 200 burn-in
steps, then 8 snapshots every 25 steps, PRNGKey(0).

Saturation uses the POLICY log-prob clamp tau = 1 - 1e-4 (reppo.py:712), which is
distinct from the env ClipAction bound ACTION_CLIP = 0.999.

KL reconstruction, on the same batch and the same samples:
  (a) code path  : logp_old at the UNCLIPPED sample (sample_and_log_prob returns it
                   before reppo.py:712 clips), logp_theta at the CLIPPED action.
  (b) fixed path : logp_old also evaluated at the CLIPPED action.
Only the current actor is exported (no actor_target), so pi_old == pi_theta here.
(b) is therefore identically 0 and (a)-(b) isolates the clamp-induced term alone.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import jax, jax.numpy as jnp, numpy as np
from scripts.critic_fidelity.common import ACTION_CLIP, Harness

TAU = 1.0 - 1e-4
B, BURN, CHUNKS, GAP = 256, 200, 8, 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--outdir", required=True)
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    name = os.path.basename(a.ckpt.rstrip("/"))
    arm = "B" if "weighted_mle" in name else "A"
    M = 32 if arm == "B" else 16                      # reppo.py:703-707

    h = Harness(a.ckpt, B)
    key = jax.random.PRNGKey(0); key, rk = jax.random.split(key)
    obs, _, st = h.reset(rk); NOBS = []
    for step in range(BURN + CHUNKS * GAP):
        k1, k2, key = jax.random.split(key, 3)
        act = jnp.clip(h.pi(obs).sample(seed=k1), -ACTION_CLIP, ACTION_CLIP)
        if step >= BURN and (step - BURN) % GAP == 0:
            NOBS.append(h.na(obs))
        obs, _, st, _, _, _ = h.env.step(jax.random.split(k2, B), st, act)
    nobs = jnp.concatenate(NOBS, 0)

    pi = h.ck.actor.actor(nobs)
    u, logp_unclipped = pi.sample_and_log_prob(sample_shape=(M,), seed=jax.random.PRNGKey(1))
    u_clipped = jnp.clip(u, -1 + 1e-4, 1 - 1e-4)      # reppo.py:712

    sat = np.abs(np.asarray(u)) > TAU
    logp_old_unclipped = logp_unclipped.sum(-1)       # code path
    logp_old_clipped = pi.log_prob(u_clipped).sum(-1)  # fixed path
    logp_theta = pi.log_prob(u_clipped).sum(-1)        # pi_old == pi_theta

    kl_a = float(jnp.mean(logp_old_unclipped.mean(0) - logp_theta.mean(0)))
    kl_b = float(jnp.mean(logp_old_clipped.mean(0) - logp_theta.mean(0)))
    per_state_a = np.asarray(logp_old_unclipped.mean(0) - logp_theta.mean(0))

    res = dict(
        ckpt=name, arm=arm, M=M, tau=TAU, n_states=int(nobs.shape[0]),
        action_dim=int(h.action_dim), step=int(h.meta["time_steps"]), seed=int(h.meta["seed"]),
        frac_coords_saturated=float(sat.mean()),
        frac_vectors_any_saturated=float(sat.any(-1).mean()),
        frac_vectors_all_saturated=float(sat.all(-1).mean()),
        kl_code_path=kl_a, kl_fixed_path=kl_b, kl_difference=kl_a - kl_b,
        kl_diff_p50=float(np.percentile(per_state_a, 50)),
        kl_diff_p90=float(np.percentile(per_state_a, 90)),
        kl_diff_max=float(per_state_a.max()),
        max_atanh_err=float(jnp.abs(jnp.arctanh(u_clipped) - jnp.arctanh(jnp.clip(u, -1 + 1e-7, 1 - 1e-7))).max()),
    )
    with open(f"{a.outdir}/{name}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
