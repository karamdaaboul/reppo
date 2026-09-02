"""Export a trained REPPO train state into a standalone, reloadable checkpoint.

The upstream JAX path persists *no* parameters: ``run()`` in ``src/jaxrl/reppo.py``
discards the final ``SACTrainState`` and only writes ``metrics.npz``. This module
supplies the missing capture step. It is imported by ``scripts/train_and_export.py``,
which keeps the state that ``train_fn`` returns and hands it here.

Layout written per run::

    <out_dir>/
        actor.npz        flattened actor params (incl. both learned duals)
        critic.npz       flattened critic params (incl. the learnable ``zero_dist``)
        normalizer.npz   running obs stats: mean/var and critic_mean/critic_var
        meta.json        constructor args + leaf paths, everything needed to rebuild

Params are stored as *flat* leaf arrays keyed by their tree path. Reload rebuilds a
fresh module with the recorded constructor args -- which yields an identical treedef --
and unflattens the saved leaves into it. ``meta.json`` carries the ordered path list so
the loader can assert the structure matches instead of silently mis-assigning weights.
"""

from __future__ import annotations

import json
import os

import jax
import numpy as np
from flax import nnx

# Effective actor min_std. `make_init` (src/jaxrl/reppo.py) constructs SACActorNetworks
# without passing min_std, so the class default 0.1 is what actually trains --
# `hyperparameters.actor_min_std` is never plumbed through. Recording the effective
# value, not the config value, keeps reloaded stochastic sampling faithful.
EFFECTIVE_ACTOR_MIN_STD = 0.1

# NormalizeVec uses this epsilon, not the usual 1e-8 (src/env_utils/jax_wrappers.py).
NORMALIZER_EPS = 1e-2


def _path_str(path) -> str:
    """Render a jax tree path as a stable string key."""
    parts = []
    for p in path:
        if isinstance(p, jax.tree_util.DictKey):
            parts.append(str(p.key))
        elif isinstance(p, jax.tree_util.SequenceKey):
            parts.append(str(p.idx))
        elif isinstance(p, jax.tree_util.GetAttrKey):
            parts.append(p.name)
        elif isinstance(p, jax.tree_util.FlattenedIndexKey):
            parts.append(str(p.key))
        else:  # pragma: no cover - defensive
            parts.append(str(p))
    return "/".join(parts)


def flatten_with_paths(tree) -> tuple[list[str], list[np.ndarray]]:
    """Flatten a pytree into (ordered path strings, ordered numpy leaves)."""
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(tree)
    paths = [_path_str(p) for p, _ in leaves_with_paths]
    leaves = [np.asarray(v) for _, v in leaves_with_paths]
    return paths, leaves


def _drop_seed_axis(tree, seed_index: int = 0):
    """Strip the leading ``num_seeds`` axis that ``jax.vmap(make_init)`` adds."""
    return jax.tree.map(lambda x: np.asarray(x)[seed_index], tree)


def _save_flat(path: str, keys: list[str], leaves: list[np.ndarray]) -> None:
    np.savez(path, **{f"{i:04d}|{k}": v for i, (k, v) in enumerate(zip(keys, leaves))})


def _meta_dict(
    hp, env_name, obs_dim, critic_obs_dim, action_dim, seed, seed_index,
    time_steps, iteration, actor_paths, critic_paths,
):
    """Metadata block shared by final-state and mid-training snapshot exports."""
    return {
        "env_name": env_name,
        "seed": seed,
        "seed_index": seed_index,
        "time_steps": int(time_steps),
        "iteration": int(iteration),
        "obs_dim": int(obs_dim),
        "critic_obs_dim": int(critic_obs_dim),
        "action_dim": int(action_dim),
        "normalizer_eps": NORMALIZER_EPS,
        "actor_kwargs": {
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "hidden_dim": int(hp.actor_hidden_dim),
            "ent_start": float(hp.ent_start),
            "kl_start": float(hp.kl_start),
            "use_norm": bool(hp.use_actor_norm),
            "layers": int(hp.num_actor_layers),
            "min_std": EFFECTIVE_ACTOR_MIN_STD,
            "use_skip": bool(hp.use_actor_skip),
            # eta exists only in the weighted_mle arm; absent from checkpoints
            # exported before it existed, where the loader default False is correct
            "with_eta": bool(
                getattr(hp, "actor_update_mode", "pathwise") == "weighted_mle"
            ),
            "with_betas": bool(getattr(hp, "mstep_decoupled", False)),
            # absent from checkpoints exported before the covariance-freeze
            # mechanism existed, where the loader default None is correct
            "freeze_sigma": getattr(hp, "freeze_sigma", None),
        },
        "critic_kwargs": {
            "obs_dim": int(critic_obs_dim),
            "action_dim": int(action_dim),
            "hidden_dim": int(hp.critic_hidden_dim),
            "num_bins": int(hp.num_bins),
            "vmin": float(hp.vmin),
            "vmax": float(hp.vmax),
            "use_norm": bool(hp.use_critic_norm),
            "encoder_layers": int(hp.num_critic_encoder_layers),
            "head_layers": int(hp.num_critic_head_layers),
            "pred_layers": int(hp.num_critic_pred_layers),
            "use_simplical_embedding": bool(hp.use_simplical_embedding),
            "use_skip": bool(hp.use_critic_skip),
        },
        "actor_update_mode": str(getattr(hp, "actor_update_mode", "pathwise")),
        "eps_e": float(getattr(hp, "eps_e", float("nan"))),
        "hl_gauss": bool(hp.hl_gauss),
        "gamma": float(hp.gamma),
        "lmbda": float(hp.lmbda),
        "max_episode_steps": int(hp.max_episode_steps),
        "num_eval": int(hp.num_eval),
        "normalize_env": bool(hp.normalize_env),
        "actor_leaf_paths": actor_paths,
        "critic_leaf_paths": critic_paths,
    }


def export_train_state(
    state,
    hp,
    env_name: str,
    obs_dim: int,
    critic_obs_dim: int,
    action_dim: int,
    out_dir: str,
    seed: int,
    seed_index: int = 0,
    extra_meta: dict | None = None,
) -> str:
    """Write a standalone checkpoint from a finished ``SACTrainState``.

    Args:
        state: the ``SACTrainState`` returned by ``train_fn`` (still carrying the
            leading ``num_seeds`` axis).
        hp: the ``ReppoConfig`` the run was trained with.
        env_name: mujoco_playground registry id, e.g. ``"WalkerRun"``.
        obs_dim / critic_obs_dim / action_dim: network shapes.
        out_dir: destination directory, created if absent.
        seed: the training seed, recorded for provenance.
        seed_index: which vmap seed slice to export.
    """
    os.makedirs(out_dir, exist_ok=True)

    actor_params = _drop_seed_axis(state.actor.params, seed_index)
    critic_params = _drop_seed_axis(state.critic.params, seed_index)

    actor_paths, actor_leaves = flatten_with_paths(actor_params)
    critic_paths, critic_leaves = flatten_with_paths(critic_params)

    _save_flat(os.path.join(out_dir, "actor.npz"), actor_paths, actor_leaves)
    _save_flat(os.path.join(out_dir, "critic.npz"), critic_paths, critic_leaves)

    # The obs normalizer lives in the env wrapper state, not in the networks. Without
    # it the reloaded policy sees unnormalized inputs and is useless.
    norm = state.last_env_state
    np.savez(
        os.path.join(out_dir, "normalizer.npz"),
        mean=np.asarray(norm.mean)[seed_index],
        var=np.asarray(norm.var)[seed_index],
        critic_mean=np.asarray(norm.critic_mean)[seed_index],
        critic_var=np.asarray(norm.critic_var)[seed_index],
        count=np.asarray(norm.count)[seed_index],
    )

    meta = _meta_dict(
        hp, env_name, obs_dim, critic_obs_dim, action_dim, seed, seed_index,
        int(np.asarray(state.time_steps)[seed_index]),
        int(np.asarray(state.iteration)[seed_index]),
        actor_paths, critic_paths,
    )
    if extra_meta:
        meta.update(extra_meta)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return out_dir


def summarize(state, seed_index: int = 0) -> dict:
    """Pull the two learned duals out of the actor for logging."""
    actor = nnx.merge(state.actor.graphdef, state.actor.params)
    return {
        "alpha_entropy": float(np.asarray(actor.temperature())[seed_index].item()),
        "alpha_kl": float(np.asarray(actor.lagrangian())[seed_index].item()),
    }


def _duals_from_params(actor_params):
    """Read alpha_entropy / alpha_kl straight out of a params tree.

    Snapshots are raw params without a graphdef handy, so the duals are located by
    leaf path instead of by merging a module. Both live inside the actor
    (src/networks/jax_models.py:354-357).
    """
    paths, leaves = flatten_with_paths(actor_params)
    out = {}
    for p, v in zip(paths, leaves):
        if "temperature_log_param" in p:
            out["alpha_entropy"] = float(np.exp(np.asarray(v)).ravel()[0])
        elif "lagrangian_log_param" in p:
            out["alpha_kl"] = float(np.exp(np.asarray(v)).ravel()[0])
        elif "eta_param" in p:
            x = float(np.asarray(v).ravel()[0])
            out["eta"] = float(min(max(np.log1p(np.exp(-abs(x))) + max(x, 0.0),
                                       1e-4), 10.0))
    return out


def export_snapshot(
    snap,
    idx,
    hp,
    env_name,
    obs_dim,
    critic_obs_dim,
    action_dim,
    out_dir,
    seed,
    seed_index=0,
    extra_meta=None,
):
    """Export one mid-training snapshot produced by make_train_fn(return_snapshots=True).

    Snapshot leaves are stacked (num_iterations, num_seeds, ...), so both axes are
    indexed here.
    """
    os.makedirs(out_dir, exist_ok=True)
    pick = lambda t: jax.tree.map(lambda x: np.asarray(x)[idx][seed_index], t)  # noqa: E731

    actor_params = pick(snap["actor"])
    critic_params = pick(snap["critic"])
    actor_paths, actor_leaves = flatten_with_paths(actor_params)
    critic_paths, critic_leaves = flatten_with_paths(critic_params)
    _save_flat(os.path.join(out_dir, "actor.npz"), actor_paths, actor_leaves)
    _save_flat(os.path.join(out_dir, "critic.npz"), critic_paths, critic_leaves)

    np.savez(
        os.path.join(out_dir, "normalizer.npz"),
        **{k: np.asarray(snap[k])[idx][seed_index]
           for k in ("mean", "var", "critic_mean", "critic_var", "count")},
    )

    meta = _meta_dict(
        hp, env_name, obs_dim, critic_obs_dim, action_dim, seed, seed_index,
        int(np.asarray(snap["time_steps"])[idx][seed_index]), int(idx),
        actor_paths, critic_paths,
    )
    meta.update(_duals_from_params(actor_params))
    if extra_meta:
        meta.update(extra_meta)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return out_dir


if __name__ == "__main__":
    raise SystemExit(
        "export_ckpt is a library, not a standalone entry point.\n"
        "The JAX training path keeps no parameters on disk, so export has to happen\n"
        "in-process at the end of training. Run scripts/train_and_export.py instead."
    )
