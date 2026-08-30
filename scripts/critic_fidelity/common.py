"""Shared machinery for the critic-fidelity measurements.

Ground truth is the SOFT return the critic was actually trained on. Verified against
``src/jaxrl/reppo.py``:

* ``collect_rollout`` (reppo.py:376-379) builds
  ``soft_reward_t = r_t - gamma * log pi(a'_{t+1}|s_{t+1}) * alpha``
  where ``a'_{t+1}`` is an *independent* sample from ``pi(.|s_{t+1})`` and
  ``alpha = actor.temperature()``.
* ``learn_step`` (reppo.py:421-450) accumulates those into a lambda-return that
  bootstraps on ``value = Q(s_{t+1}, a'_{t+1})``, again at a sampled action.

So the horizon-H truncated estimate used here is

    R(s, a) = sum_{t<H} gamma^t * (r_t - gamma * alpha * log pi(a'_{t+1}|s_{t+1}))
              + gamma^H * Q_phi(s_H, a_H),     a_H ~ pi(.|s_H)

with the bootstrap moved earlier if the episode ends first (DMC truncates at 1000
steps and never terminates, and training bootstraps on truncation).

Two deliberate deviations from the training loop, both to keep measurement clean:

* The observation normalizer is FROZEN at the checkpoint's saved statistics rather
  than continuing to update. A drifting normalizer would make states collected early
  and late incomparable.
* ``NormalizeVec`` is dropped from the wrapper stack and normalization applied by
  hand, because its state mixes unbatched running statistics with batched env state,
  which makes per-state cloning error-prone.
"""

from __future__ import annotations

import os
import sys
from functools import partial

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from scripts.load_ckpt import load  # noqa: E402
from src.env_utils.action_pad import ActionPad  # noqa: E402
from src.env_utils.jax_wrappers import ClipAction, LogWrapper, MjxGymnaxWrapper  # noqa: E402

ACTION_CLIP = 0.999  # ClipAction bounds (src/env_utils/jax_wrappers.py:255)


class Harness:
    """Checkpoint + environment at a fixed batch size, with frozen normalization."""

    def __init__(self, ckpt_dir: str, batch_size: int):
        self.ck = load(ckpt_dir)
        self.meta = self.ck.meta
        self.B = batch_size
        self.gamma = float(self.meta["gamma"])
        self.alpha = float(self.meta["alpha_entropy"])
        self.action_dim = int(self.meta["action_dim"])
        self.obs_dim = int(self.meta["obs_dim"])

        nz = np.load(os.path.join(ckpt_dir, "normalizer.npz"))
        eps = float(self.meta["normalizer_eps"])
        self._mean = jnp.asarray(nz["mean"])
        self._istd = 1.0 / jnp.sqrt(jnp.asarray(nz["var"]) + eps)
        self._cmean = jnp.asarray(nz["critic_mean"])
        self._cistd = 1.0 / jnp.sqrt(jnp.asarray(nz["critic_var"]) + eps)

        env = MjxGymnaxWrapper(
            self.meta["env_name"],
            episode_length=int(self.meta["max_episode_steps"]),
            reward_scale=float(self.meta.get("reward_scaling", 1.0)),
            push_distractions=False,
            asymmetric_observation=False,
        )
        k = int(self.meta.get("action_pad", 0))
        if k > 0:
            env = ActionPad(env, k)
        env = LogWrapper(env, batch_size)
        self.env = ClipAction(env)

    # --- normalization -----------------------------------------------------
    def na(self, obs):
        return (obs - self._mean) * self._istd

    def nc(self, obs):
        return (obs - self._cmean) * self._cistd

    # --- policy / critic ---------------------------------------------------
    def pi(self, obs):
        """distrax Tanh(Normal) at raw obs."""
        return self.ck.actor.actor(self.na(obs))

    def det_action(self, obs):
        return self.ck.actor.det_action(self.na(obs))

    def q(self, obs, action, critic=None):
        c = self.ck.critic if critic is None else critic
        return c.critic(self.nc(obs), action)

    # --- rollout -----------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        return self.env.reset(jax.random.split(key, self.B))

    def _step_keys(self, key):
        return jax.random.split(key, self.B)

    @partial(jax.jit, static_argnums=(0, 2))
    def collect_states(self, key, n_steps):
        """Roll pi_old for n_steps, returning the visited (state, obs) at each step.

        Returned pytrees have leading axes (n_steps, B, ...).
        """
        key, rk = jax.random.split(key)
        obs, _, st = self.reset(rk)

        def body(carry, k):
            st, obs = carry
            ak, sk = jax.random.split(k)
            a = self.pi(obs).sample(seed=ak)
            a = jnp.clip(a, -ACTION_CLIP, ACTION_CLIP)
            nobs, _, nst, _, _, _ = self.env.step(self._step_keys(sk), st, a)
            return (nst, nobs), (st, obs)

        _, out = jax.lax.scan(body, (st, obs), jax.random.split(key, n_steps))
        return out

    def soft_return(self, state, obs, first_action, horizon, key, alpha=None):
        """Total soft return. See `soft_return_parts` for the decomposition."""
        acc, boot = self.soft_return_parts(state, obs, first_action, horizon, key, alpha)
        return acc + boot

    @partial(jax.jit, static_argnums=(0, 4))
    def soft_return_parts(self, state, obs, first_action, horizon, key, alpha=None):
        """Horizon-truncated soft return of executing `first_action` then following pi.

        `state`/`obs`/`first_action` carry a leading axis of size self.B.
        Returns (B,) soft returns.

        Returns (realized, bootstrap) where `realized` is the discounted sum of
        soft rewards actually collected and `bootstrap` is the discounted
        gamma^H * Q(s_H, a_H) tail (moved earlier if the episode ends first). The
        split lets the bootstrap's contribution to across-action variance be measured
        directly instead of argued about.

        `alpha` is a TRACED argument, not read off self: `self` is a static jit
        argument, so mutating an attribute between calls would silently reuse the
        cached trace and the entropy term would appear to have no effect.
        """
        gamma = self.gamma
        alpha = jnp.asarray(self.alpha if alpha is None else alpha, dtype=jnp.float32)

        def one_step(carry, k):
            st, obs, acc, disc, alive, boot = carry
            ak, ek, sk, forced = k["a"], k["e"], k["s"], k["forced"]

            # step 0 executes the supplied action; later steps sample from pi
            dist = self.pi(obs)
            sampled = jnp.clip(dist.sample(seed=ak), -ACTION_CLIP, ACTION_CLIP)
            act = jnp.where(forced[:, None] > 0, first_action, sampled)

            nobs, _, nst, rew, done, _ = self.env.step(self._step_keys(sk), st, act)

            # entropy term uses an independent draw at s_{t+1}, as collect_rollout does
            ndist = self.pi(nobs)
            a_next, logp = ndist.sample_and_log_prob(seed=ek)
            logp = logp.sum(-1)
            soft_r = rew - gamma * alpha * logp

            acc = acc + alive * disc * soft_r
            # value at s_{t+1} under a freshly sampled action, matching training
            v_next = self.q(nobs, jnp.clip(a_next, -ACTION_CLIP, ACTION_CLIP))
            newly_done = alive * done.astype(jnp.float32)
            boot = boot + newly_done * disc * gamma * v_next
            alive = alive * (1.0 - done.astype(jnp.float32))
            disc = disc * gamma
            return (nst, nobs, acc, disc, alive, boot), None

        B = self.B
        keys = {
            "a": jax.random.split(jax.random.fold_in(key, 1), horizon),
            "e": jax.random.split(jax.random.fold_in(key, 2), horizon),
            "s": jax.random.split(jax.random.fold_in(key, 3), horizon),
            "forced": jnp.concatenate(
                [jnp.ones((1, B)), jnp.zeros((horizon - 1, B))], axis=0
            ),
        }
        init = (
            state,
            obs,
            jnp.zeros((B,)),
            jnp.ones((B,)),
            jnp.ones((B,)),
            jnp.zeros((B,)),
        )
        (st, obs, acc, disc, alive, boot), _ = jax.lax.scan(one_step, init, keys)

        # states that never ended bootstrap at the horizon
        a_H = jnp.clip(
            self.pi(obs).sample(seed=jax.random.fold_in(key, 4)),
            -ACTION_CLIP,
            ACTION_CLIP,
        )
        tail = alive * disc * self.q(obs, a_H)
        return acc, boot + tail


def tile_state(state, idx, reps):
    """Clone entry `idx` of a batched LogEnvState `reps` times.

    Every leaf of LogEnvState carries a leading num_envs axis
    (src/env_utils/jax_wrappers.py:118-135), so a uniform tree map is safe here --
    unlike NormalizeVecObsEnvState, which mixes in unbatched running statistics.
    """
    return jax.tree.map(lambda x: jnp.repeat(x[idx : idx + 1], reps, axis=0), state)


def gather_states(state, idx):
    """Select a set of indices out of a batched LogEnvState."""
    return jax.tree.map(lambda x: x[idx], state)


def spearman(a, b, axis=-1):
    """Spearman rho along `axis`, computed from average ranks."""
    ra = _rankdata(a, axis)
    rb = _rankdata(b, axis)
    ra = ra - ra.mean(axis=axis, keepdims=True)
    rb = rb - rb.mean(axis=axis, keepdims=True)
    num = (ra * rb).sum(axis=axis)
    den = np.sqrt((ra**2).sum(axis=axis) * (rb**2).sum(axis=axis))
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def _rankdata(x, axis=-1):
    """Average ranks, ties shared (matches scipy.stats.rankdata 'average')."""
    x = np.asarray(x)
    order = np.argsort(x, axis=axis, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    n = x.shape[axis]
    idx = np.arange(1, n + 1, dtype=np.float64)
    shape = [1] * x.ndim
    shape[axis] = n
    np.put_along_axis(ranks, order, np.broadcast_to(idx.reshape(shape), x.shape), axis)
    # average tied ranks
    xs = np.take_along_axis(x, order, axis=axis)
    same = np.zeros_like(xs, dtype=bool)
    same[..., 1:] = np.diff(xs, axis=axis) == 0 if axis in (-1, x.ndim - 1) else False
    if same.any():
        rs = np.take_along_axis(ranks, order, axis=axis)
        out = rs.copy()
        flat_x = xs.reshape(-1, n)
        flat_r = rs.reshape(-1, n)
        flat_o = out.reshape(-1, n)
        for row in range(flat_x.shape[0]):
            i = 0
            while i < n:
                j = i
                while j + 1 < n and flat_x[row, j + 1] == flat_x[row, i]:
                    j += 1
                if j > i:
                    flat_o[row, i : j + 1] = flat_r[row, i : j + 1].mean()
                i = j + 1
        np.put_along_axis(ranks, order, flat_o.reshape(xs.shape), axis=axis)
    return ranks


def cosine(a, b, axis=-1):
    num = (a * b).sum(axis=axis)
    den = np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis)
    return np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)


def summarize(x, name=""):
    """Median / IQR / mean, since these distributions are heavy-tailed."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"name": name, "n": 0}
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {
        "name": name,
        "n": int(x.size),
        "median": float(med),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "mean": float(x.mean()),
        "sem": float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("nan"),
    }


def antithetic_grad(values_plus, values_minus, u, sigma):
    """Antithetic smoothed-gradient estimator.

        g(sigma) = 1/(2 sigma N) * sum_n [ f(a + sigma u_n) - f(a - sigma u_n) ] u_n

    Shapes: values_plus/minus are (..., N), u is (..., N, D). Returns (..., D).

    For a quadratic the antithetic difference cancels the even-order terms exactly,
    so the estimator is unbiased at every sigma -- which is what makes the quadratic
    a usable correctness check on the harness (guard g).
    """
    n = u.shape[-2]
    diff = (values_plus - values_minus)[..., :, None]
    return (diff * u).sum(axis=-2) / (2.0 * sigma * n)
