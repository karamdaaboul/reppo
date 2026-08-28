"""Load a REPPO checkpoint standalone, with no training state in scope.

``load(ckpt_dir)`` returns a handle exposing exactly four callables::

    q_scalar(s, a)            -> float      scalar Q via the HL-Gauss bin expectation
    q_grad_a(s, a)            -> array      jax.grad of q_scalar w.r.t. the action
    policy_dist(s)            -> (mu, sigma)
    policy_sample(s, key, n)  -> (actions, log_probs)

All four take **raw, unnormalized** observations; the saved running statistics are
applied internally, with the actor and critic using their own separate estimators
exactly as ``NormalizeVec`` does during training.

``mu`` and ``sigma`` are the parameters of the *pre-squash* Gaussian. The behaviour
policy is ``Tanh(Normal(mu, sigma))``, so the deterministic action the trainer's
evaluator uses is ``tanh(mu)`` (matching ``SACActorNetworks.det_action``).

All four accept a single ``(obs_dim,)`` observation or a batch ``(N, obs_dim)``;
``q_scalar`` returns a scalar or ``(N,)``, ``q_grad_a`` returns ``(action_dim,)`` or
``(N, action_dim)``. ``policy_sample`` returns ``(n, action_dim)`` actions with
``(n,)`` *joint* log-probabilities.

Example::

    from scripts.load_ckpt import load
    ck = load("exports/WalkerRun_s0")
    q  = ck.q_scalar(obs, act)
    g  = ck.q_grad_a(obs, act)

This venv's JAX requires its CUDA backend by default; for CPU-only analysis set
``JAX_PLATFORMS=cpu`` (``CUDA_VISIBLE_DEVICES=""`` alone raises at import).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Callable

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import nnx  # noqa: E402

from scripts.export_ckpt import flatten_with_paths  # noqa: E402
from src.networks.jax_models import (  # noqa: E402
    CategoricalCriticNetwork,
    CriticNetwork,
    SACActorNetworks,
)


@dataclass
class ReppoCheckpoint:
    """Handle over a reloaded checkpoint. See module docstring."""

    q_scalar: Callable
    q_grad_a: Callable
    policy_dist: Callable
    policy_sample: Callable
    meta: dict
    actor: nnx.Module
    critic: nnx.Module


def _load_flat(path: str) -> list[np.ndarray]:
    """Read leaves back in their saved order (keys are ``NNNN|path``)."""
    z = np.load(path)
    return [z[k] for k in sorted(z.files, key=lambda k: int(k.split("|", 1)[0]))]


def _restore(module: nnx.Module, leaves: list[np.ndarray], expected_paths: list[str]):
    """Unflatten saved leaves into a freshly constructed module."""
    fresh_state = nnx.state(module)
    paths, _ = flatten_with_paths(fresh_state)
    if paths != expected_paths:
        only_fresh = [p for p in paths if p not in set(expected_paths)]
        only_saved = [p for p in expected_paths if p not in set(paths)]
        raise ValueError(
            "Checkpoint structure does not match the rebuilt module.\n"
            f"  in module, not in checkpoint: {only_fresh}\n"
            f"  in checkpoint, not in module: {only_saved}\n"
            "The constructor args in meta.json no longer describe this network."
        )
    _, treedef = jax.tree_util.tree_flatten(fresh_state)
    restored = jax.tree_util.tree_unflatten(treedef, [jnp.asarray(x) for x in leaves])
    nnx.update(module, restored)
    return module


def load(ckpt_dir: str) -> ReppoCheckpoint:
    """Rebuild actor and critic from an export directory."""
    with open(os.path.join(ckpt_dir, "meta.json")) as f:
        meta = json.load(f)

    rngs = nnx.Rngs(0)  # values are overwritten by the restore; the seed is irrelevant
    actor_kwargs = dict(meta["actor_kwargs"])
    # `with_eta` / `with_betas` decide whether those leaves exist in the param tree.
    # Older checkpoints predate one or both flags, so infer from what was actually
    # saved rather than trusting the recorded kwargs.
    paths = meta["actor_leaf_paths"]
    actor_kwargs["with_eta"] = any("eta_param" in p for p in paths)
    actor_kwargs["with_betas"] = any("beta_mu_param" in p for p in paths)
    actor = SACActorNetworks(**actor_kwargs, rngs=rngs)
    critic_cls = CategoricalCriticNetwork if meta["hl_gauss"] else CriticNetwork
    critic_kwargs = dict(meta["critic_kwargs"])
    if not meta["hl_gauss"]:
        for k in ("num_bins", "vmin", "vmax"):
            critic_kwargs.pop(k, None)
    critic = critic_cls(**critic_kwargs, rngs=nnx.Rngs(0))

    _restore(actor, _load_flat(os.path.join(ckpt_dir, "actor.npz")), meta["actor_leaf_paths"])
    _restore(critic, _load_flat(os.path.join(ckpt_dir, "critic.npz")), meta["critic_leaf_paths"])

    nz = np.load(os.path.join(ckpt_dir, "normalizer.npz"))
    eps = meta["normalizer_eps"]
    a_mean, a_var = jnp.asarray(nz["mean"]), jnp.asarray(nz["var"])
    c_mean, c_var = jnp.asarray(nz["critic_mean"]), jnp.asarray(nz["critic_var"])
    normalize_env = meta.get("normalize_env", True)

    def norm_actor(s):
        s = jnp.asarray(s)
        return (s - a_mean) / jnp.sqrt(a_var + eps) if normalize_env else s

    def norm_critic(s):
        s = jnp.asarray(s)
        return (s - c_mean) / jnp.sqrt(c_var + eps) if normalize_env else s

    def _q_one(s, a):
        """Scalar Q for a single (s, a). Bin expectation lives in critic.critic()."""
        return critic.critic(norm_critic(s), jnp.asarray(a))

    def q_scalar(s, a):
        s, a = jnp.asarray(s), jnp.asarray(a)
        if s.ndim == 1:
            return _q_one(s, a)
        return jax.vmap(_q_one)(s, a)

    def q_grad_a(s, a):
        s, a = jnp.asarray(s), jnp.asarray(a)
        grad_one = jax.grad(lambda aa, ss: _q_one(ss, aa))
        if s.ndim == 1:
            return grad_one(a, s)
        return jax.vmap(grad_one)(a, s)

    def policy_dist(s):
        """Pre-squash Gaussian params. The policy is Tanh(Normal(mu, sigma))."""
        loc = actor.actor_module(norm_actor(s))
        mu, log_std = jnp.split(loc, 2, axis=-1)
        sigma = jnp.exp(log_std) + actor.min_std
        return mu, sigma

    def policy_sample(s, key, n):
        """Sample n actions with their JOINT log-probabilities.

        distrax's Tanh(Normal) here has event_shape () and batch_shape (action_dim,),
        so its log_prob is per-dimension. The trainer always sums over the action axis
        (src/jaxrl/reppo.py:359, 373, 539, 551-563); this matches that convention, so
        log_probs has one scalar per sampled action vector rather than one per element.
        """
        pi = actor.actor(norm_actor(s))
        actions, log_probs = pi.sample_and_log_prob(seed=key, sample_shape=(n,))
        return actions, log_probs.sum(-1)

    return ReppoCheckpoint(
        q_scalar=q_scalar,
        q_grad_a=q_grad_a,
        policy_dist=policy_dist,
        policy_sample=policy_sample,
        meta=meta,
        actor=actor,
        critic=critic,
    )


if __name__ == "__main__":
    ck = load(sys.argv[1])
    m = ck.meta
    print(f"{m['env_name']} seed={m['seed']} steps={m['time_steps']}")
    print(f"  obs_dim={m['obs_dim']} action_dim={m['action_dim']} bins={m['critic_kwargs']['num_bins']}")
    print(f"  final_eval_return={m.get('final_eval_return')}")
    print(f"  alpha_entropy={m.get('alpha_entropy')} alpha_kl={m.get('alpha_kl')}")
