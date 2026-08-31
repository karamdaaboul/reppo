"""C5 recheck: within-state Q-spread under the numerically safe path.

Replicates `scripts/probe_ckpt.py`'s C5 protocol EXACTLY -- B=256 envs, 200-step
burn-in, 8 temporal chunks 25 steps apart (2048 states), M=32 actions per state,
`PRNGKey(0)`, the same key-split order and the same fold_in(key, 7) action draw -- and
computes the three within-state spreads two ways on the *identical* draws:

  old : jnp.std over the float32 Q tensor, as probe_ckpt.py does.
  new : Q centred on a per-state reference mean from an independent pre-pass, then
        std accumulated in float64.

Because both paths see the same Q values, any difference between them is pure float32
error, with no Monte-Carlo component at all -- the sharpest available test of whether
the published C5 amplitudes were affected by the cancellation that corrupted the first
Probe 1 pass.

sd_all : all d coordinates perturbed.  sd_real : padded coords pinned at tanh(mu).
sd_pad : real coords pinned at tanh(mu).
"""

from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from scripts.critic_fidelity.common import ACTION_CLIP, Harness  # noqa: E402

B, BURN, M, CHUNKS, GAP = 256, 200, 32, 8, 25
M_REF = 32  # independent pre-pass draws used only to centre Q


def run(ck):
    h = Harness(ck, B)
    meta = h.meta
    d = int(meta["action_dim"])
    pad = int(meta.get("action_pad", 0))
    real = d - pad
    key = jax.random.PRNGKey(0)  # as originally
    key, rk = jax.random.split(key)
    obs, _, st = h.reset(rk)
    acc = {f"{w}_{v}": [] for w in ("old", "new") for v in ("all", "real", "pad")}

    def variants(a_i, det):
        a_r = a_i.at[..., real:].set(jnp.broadcast_to(det[:, real:], (M, B, pad)))
        a_p = a_i.at[..., :real].set(jnp.broadcast_to(det[:, :real], (M, B, real)))
        return {"all": a_i, "real": a_r, "pad": a_p}

    for step in range(BURN + CHUNKS * GAP):
        k1, k2, key = jax.random.split(key, 3)
        dist = h.pi(obs)
        a = jnp.clip(dist.sample(seed=k1), -ACTION_CLIP, ACTION_CLIP)
        if step >= BURN and (step - BURN) % GAP == 0:
            kk = jax.random.fold_in(key, 7)  # the original C5 action key
            a_i = jnp.clip(dist.sample(seed=kk, sample_shape=(M,)), -1 + 1e-4, 1 - 1e-4)
            cobs = jnp.broadcast_to(h.nc(obs), (M, *h.nc(obs).shape))
            det = h.det_action(obs)
            # independent pre-pass, used only as a centring constant per state
            kr = jax.random.fold_in(key, 20260831)
            a_ref = jnp.clip(dist.sample(seed=kr, sample_shape=(M_REF,)), -1 + 1e-4, 1 - 1e-4)
            cref = jnp.broadcast_to(h.nc(obs), (M_REF, *h.nc(obs).shape))
            ref = {
                v: np.asarray(h.ck.critic.critic(cref, av), np.float64).mean(0)
                for v, av in variants(a_ref, det).items()
            }
            for v, av in variants(a_i, det).items():
                q = h.ck.critic.critic(cobs, av)
                acc[f"old_{v}"].append(np.asarray(q.std(0)))                    # fp32
                r = np.asarray(q, np.float64) - ref[v][None, :]                 # centred
                acc[f"new_{v}"].append(r.std(0))                                # float64
        obs, _, st, _, _, _ = h.env.step(jax.random.split(k2, B), st, a)

    out = {k: np.concatenate(v) for k, v in acc.items()}
    res = {"ckpt": ck, "arm": meta["actor_update_mode"], "seed": int(meta["seed"]),
           "pad": pad, "n_states": int(out["old_all"].size)}
    for v in ("all", "real", "pad"):
        o, n = out[f"old_{v}"], out[f"new_{v}"]
        res[f"old_sd_{v}"] = float(o.mean())
        res[f"new_sd_{v}"] = float(n.mean())
        res[f"reldiff_{v}"] = float((n.mean() - o.mean()) / n.mean())
        res[f"max_abs_per_state_{v}"] = float(np.max(np.abs(n - o)))
        res[f"max_rel_per_state_{v}"] = float(np.max(np.abs(n - o) / np.maximum(n, 1e-30)))
        res[f"n_nonfinite_old_{v}"] = int((~np.isfinite(o)).sum())
        res[f"n_negvar_new_{v}"] = int((n < 0).sum())
    return res


if __name__ == "__main__":
    allres = [run(c) for c in sys.argv[1:]]
    print(json.dumps(allres, indent=1))
