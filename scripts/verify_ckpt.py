"""Verify an exported REPPO checkpoint in a fresh process.

Checks, in order:

1. Reload actor + critic from the export directory (no training state in scope).
2. Replay the trainer's evaluation loop with the restored deterministic policy and
   the saved normalizer statistics, then compare the return to the value the
   training run logged in ``metrics.npz`` and to the paper's reference CSV.
3. Finite-difference check on ``q_grad_a``.
4. Assert ``q_scalar`` is finite and inside ``[vmin, vmax]``.

Usage::

    CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python scripts/verify_ckpt.py exports/WalkerRun_s0
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from scripts.load_ckpt import load  # noqa: E402
from src.env_utils.jax_wrappers import (  # noqa: E402
    ClipAction,
    LogWrapper,
    MjxGymnaxWrapper,
    NormalizeVec,
)


class _NormParams:
    """Minimal stand-in for NormalizeVecObsEnvState.

    ``NormalizeVec.reset`` reads only these five fields off the object it is handed,
    so the saved statistics are enough to seed evaluation exactly as training did
    (src/jaxrl/reppo.py passes ``train_state.last_env_state`` for the same purpose).
    """

    def __init__(self, nz):
        self.mean = jnp.asarray(nz["mean"])
        self.var = jnp.asarray(nz["var"])
        self.critic_mean = jnp.asarray(nz["critic_mean"])
        self.critic_var = jnp.asarray(nz["critic_var"])
        self.count = jnp.asarray(nz["count"])


def replay_eval(ck, ckpt_dir: str, num_envs: int, seed: int) -> dict:
    """Mirror of make_eval_fn / evaluation_fn from src/jaxrl/reppo.py."""
    meta = ck.meta
    env = MjxGymnaxWrapper(
        meta["env_name"],
        episode_length=meta["max_episode_steps"],
        reward_scale=meta.get("reward_scaling", 1.0),
        push_distractions=False,
        asymmetric_observation=False,
    )
    env = LogWrapper(env, num_envs)
    env = ClipAction(env)
    if meta.get("normalize_env", True):
        env = NormalizeVec(env)

    nz = np.load(os.path.join(ckpt_dir, "normalizer.npz"))
    norm_state = _NormParams(nz) if meta.get("normalize_env", True) else None
    reward_scale = 1.0 / meta.get("reward_scaling", 1.0)

    def step_env(carry, _):
        key, env_state, obs = carry
        key, env_key = jax.random.split(key)
        # Obs arrives already normalized by the wrapper, exactly as in training,
        # so the actor is called directly rather than through policy_dist().
        action = ck.actor.det_action(obs)
        step_key = jax.random.split(env_key, num_envs)
        obs, _, env_state, _, _, info = env.step(step_key, env_state, action)
        return (key, env_state, obs), info

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    obs, _, env_state = env.reset(jax.random.split(init_key, num_envs), norm_state)
    _, infos = jax.lax.scan(
        f=step_env,
        init=(key, env_state, obs),
        xs=None,
        length=meta["max_episode_steps"],
    )
    done = infos["returned_episode"]
    return {
        "episode_return": float(
            infos["returned_episode_returns"].mean(where=done) * reward_scale
        ),
        "episode_return_std": float(infos["returned_episode_returns"].std(where=done)),
        "episode_length": float(infos["returned_episode_lengths"].mean(where=done)),
        "num_episodes": int(done.sum()),
    }


EPS_F32 = 1.2e-7


def check_grad(
    ck, obs_dim: int, action_dim: int, n: int = 6, h: float = 1e-1
) -> tuple[float, float]:
    """Central-difference check of q_grad_a. Returns (worst rel err, tolerance).

    Two design points, both forced by float32:

    * Directional derivatives along a random unit vector, normalized by ``||g||_2``,
      rather than per-component differences -- individual components can be near zero,
      which makes a per-component relative error meaningless.
    * The tolerance is *derived*, not fixed. Central differencing cancels leading
      digits, so achievable relative accuracy is about ``|Q| * eps_f32 / (h * ||g||)``.
      For a converged critic (Q ~ 50, ||g|| ~ 0.34) that floor is 1.2e-3 at h=1e-2 --
      above any sane fixed threshold, and it rises as the critic's scale rises.
      Measured error tracks the 1/h prediction closely (1.05e-3 measured vs 1.25e-3
      predicted at h=1e-2), which is what confirms the residual is round-off in the
      probe rather than error in the gradient.

    h defaults to 1e-1, well clear of the cancellation regime; error is still
    decreasing monotonically with h there, so truncation is not yet the binding term.
    """
    rng = np.random.default_rng(0)
    worst = 0.0
    tol = 0.0
    for _ in range(n):
        s = jnp.asarray(rng.normal(size=(obs_dim,)), dtype=jnp.float32)
        a = jnp.asarray(rng.uniform(-0.9, 0.9, size=(action_dim,)), dtype=jnp.float32)
        g = np.asarray(ck.q_grad_a(s, a))
        gnorm = max(float(np.linalg.norm(g)), 1e-6)
        u = rng.normal(size=action_dim)
        u = jnp.asarray(u / np.linalg.norm(u), dtype=jnp.float32)
        qp, qm = ck.q_scalar(s, a + h * u), ck.q_scalar(s, a - h * u)
        fd = float((qp - qm) / (2 * h))
        exact = float(np.dot(g, np.asarray(u)))
        worst = max(worst, abs(fd - exact) / gnorm)
        tol = max(tol, 20.0 * abs(float(qp)) * EPS_F32 / (h * gnorm))
    return worst, max(tol, 1e-3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--num_envs", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    ckpt_dir = os.path.abspath(args.ckpt_dir)
    ck = load(ckpt_dir)
    meta = ck.meta
    print(f"Loaded {meta['env_name']} seed={meta['seed']} steps={meta['time_steps']}")
    print(f"  alpha_entropy={meta['alpha_entropy']:.6f}  alpha_kl={meta['alpha_kl']:.6f}")

    # --- 2. eval replay -----------------------------------------------------
    res = replay_eval(ck, ckpt_dir, args.num_envs, args.seed)
    logged = meta.get("final_eval_return")
    print("\n[eval replay]")
    print(f"  reloaded return : {res['episode_return']:.3f} "
          f"(+-{res['episode_return_std']:.1f} over {res['num_episodes']} episodes)")
    print(f"  logged  return  : {logged:.3f}")
    gap = res["episode_return"] - logged
    print(f"  gap             : {gap:+.3f} ({100 * gap / max(abs(logged), 1e-9):+.2f}%)")

    ref = os.path.join(REPO_ROOT, "results", "mujoco_playground", f"{meta['env_name']}.csv")
    if os.path.exists(ref):
        rows = np.genfromtxt(ref, delimiter=",", names=True)
        steps = rows["steps"]
        idx = int(np.argmin(np.abs(steps - meta["time_steps"])))
        trials = np.array([rows[n][idx] for n in rows.dtype.names if n.startswith("trial_")])
        print(f"  paper @ {int(steps[idx]):,} steps: {trials.mean():.1f} +- {trials.std():.1f} "
              f"(n={len(trials)})")

    # --- 3 & 4. critic sanity ----------------------------------------------
    print("\n[critic]")
    worst, gtol = check_grad(ck, meta["obs_dim"], meta["action_dim"])
    print(f"  q_grad_a vs directional finite differences: worst rel err {worst:.2e} "
          f"(float32 probe tolerance {gtol:.2e})")

    rng = np.random.default_rng(1)
    s = jnp.asarray(rng.normal(size=(256, meta["obs_dim"])), dtype=jnp.float32)
    a = jnp.asarray(rng.uniform(-1, 1, size=(256, meta["action_dim"])), dtype=jnp.float32)
    q = np.asarray(ck.q_scalar(s, a))
    vmin, vmax = meta["critic_kwargs"]["vmin"], meta["critic_kwargs"]["vmax"]
    print(f"  q_scalar over 256 random (s,a): min={q.min():.3f} max={q.max():.3f} "
          f"mean={q.mean():.3f}")
    ok_finite = bool(np.isfinite(q).all())
    ok_range = bool((q >= vmin).all() and (q <= vmax).all())
    print(f"  all finite: {ok_finite}   within [{vmin}, {vmax}]: {ok_range}")

    mu, sigma = ck.policy_dist(s[:4])
    print(f"\n[actor]  mu {tuple(mu.shape)} sigma {tuple(sigma.shape)} "
          f"sigma range [{float(sigma.min()):.3f}, {float(sigma.max()):.3f}]")
    acts, logps = ck.policy_sample(s[0], jax.random.PRNGKey(0), 8)
    print(f"  policy_sample -> actions {tuple(acts.shape)} log_probs {tuple(logps.shape)} "
          f"| |a|max={float(jnp.abs(acts).max()):.3f}")

    failures = []
    if not ok_finite:
        failures.append("q_scalar produced non-finite values")
    if not ok_range:
        failures.append(f"q_scalar outside [{vmin}, {vmax}]")
    if worst > gtol:
        failures.append(
            f"q_grad_a disagrees with finite differences ({worst:.2e} > {gtol:.2e})"
        )
    if failures:
        print("\nFAIL: " + "; ".join(failures))
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
