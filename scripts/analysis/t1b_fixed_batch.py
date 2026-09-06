#!/usr/bin/env python3
"""T1b: fixed-batch gradient no-op test.

G1 is not run-to-run deterministic (identical code, identical config, diverges), so an
end-to-end bitwise T1 cannot discriminate there. This exercises the REAL actor and
critic loss through reppo's own `learn_step` on a FIXED batch with a FIXED key, so no
environment stepping enters the comparison.

  mode=dump  : build cfg/env, init the train state, collect ONE rollout, save the batch
  mode=step  : rebuild the SAME train state, load that saved batch, run one learn_step,
               save the resulting actor and critic parameters

Run `step` under the pre-change commit and under the new commit with both flags off; the
parameters must match bitwise. The batch is shared, so rollout nondeterminism is excluded.
"""
import os, sys, pickle
import numpy as np
import jax, jax.numpy as jnp
from flax import nnx
import hydra
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())

MODE = sys.argv[1]
OUT = sys.argv[2]
INIT_KEY, ROLL_KEY, STEP_KEY = 0, 1, 2


def find_fn(obj, name, depth=0):
    """Locate a closure-nested function by name."""
    if depth > 6 or obj is None:
        return None
    cl = getattr(obj, "__closure__", None) or ()
    fv = getattr(getattr(obj, "__code__", None), "co_freevars", ()) or ()
    for n, cell in zip(fv, cl):
        try:
            v = cell.cell_contents
        except ValueError:
            continue
        if n == name and callable(v):
            return v
        r = find_fn(v, name, depth + 1)
        if r is not None:
            return r
    return None


def build():
    from src.jaxrl import reppo as R
    from src.env_utils.jax_wrappers import (
        MjxGymnaxWrapper, LogWrapper, ClipAction, NormalizeVec)
    with hydra.initialize(version_base=None, config_path="../../config"):
        cfg = hydra.compose(config_name="reppo", overrides=[
            "env=mjx_humanoid", "env.name=G1JoystickFlatTerrain",
            "env.asymmetric_obs=false",
            "experiment_overrides=mjx_humanoid_large_data",
            "seed=301", "num_trials=1", "num_seeds=1", "wandb.mode=disabled",
            # tiny but real; only shapes and the executed code path matter here
            "hyperparameters.num_envs=8", "hyperparameters.num_steps=4",
            "hyperparameters.num_mini_batches=2", "hyperparameters.num_epochs=1",
            "hyperparameters.total_time_steps=64", "hyperparameters.num_eval=1",
            "hyperparameters.ent_start=0.00020752247655764222",
            "hyperparameters.update_entropy_lagrangian=false",
            "hyperparameters.actor_update_mode=weighted_mle",
        ])
    # env construction copied from reppo.run(), mjx branch
    env = MjxGymnaxWrapper(
        cfg.env.name,
        episode_length=cfg.env.max_episode_steps,
        reward_scale=cfg.env.reward_scaling,
        push_distractions=cfg.env.get("push_distractions", False),
        asymmetric_observation=cfg.env.get("asymmetric_obs", False),
    )
    hp = R.ReppoConfig(**OmegaConf.to_container(cfg.hyperparameters, resolve=True))
    # the same wrapper stack make_train_fn applies (reppo.py:546-550), so make_init
    # sees exactly the env the real code path gives it
    wenv = LogWrapper(env, hp.num_envs)
    wenv = ClipAction(wenv)
    if hp.normalize_env:
        wenv = NormalizeVec(wenv)
    init = R.make_init(hp, wenv)
    ts = init(jax.random.PRNGKey(INIT_KEY))
    train_fn = R.make_train_fn(cfg=hp, env=env, num_seeds=1,
                               reward_scale=1.0 / cfg.env.reward_scaling)
    return R, hp, wenv, None, ts, train_fn


def main():
    R, hp, env, env_params, ts, train_fn = build()
    collect = find_fn(train_fn, "collect_rollout")
    learn = find_fn(train_fn, "learn_step")
    print("  collect_rollout found:", collect is not None)
    print("  learn_step found     :", learn is not None)
    assert learn is not None, "could not reach learn_step"

    if MODE == "dump":
        assert collect is not None, "could not reach collect_rollout"
        batch, _ = collect(jax.random.PRNGKey(ROLL_KEY), ts)
        leaves, treedef = jax.tree_util.tree_flatten(batch)
        np.savez(OUT, *[np.asarray(l) for l in leaves])
        with open(OUT + ".tree", "wb") as f:
            pickle.dump(treedef, f)
        print("  dumped %d leaves to %s" % (len(leaves), OUT))
        for i, l in enumerate(leaves[:6]):
            print("    leaf %d shape %s dtype %s" % (i, np.asarray(l).shape, np.asarray(l).dtype))
    else:
        z = np.load(OUT + ".npz")
        with open(OUT + ".tree", "rb") as f:
            treedef = pickle.load(f)
        leaves = [jnp.asarray(z[k]) for k in sorted(z.files, key=lambda s: int(s.split("_")[1]))]
        batch = jax.tree_util.tree_unflatten(treedef, leaves)
        a0 = jax.tree_util.tree_leaves(ts.actor.params)
        c0 = jax.tree_util.tree_leaves(ts.critic.params)
        ts2, metrics = learn(jax.random.PRNGKey(STEP_KEY), ts, batch)
        a = jax.tree_util.tree_leaves(ts2.actor.params)
        c = jax.tree_util.tree_leaves(ts2.critic.params)
        np.savez(sys.argv[3],
                 **{"i_a%03d" % i: np.asarray(x) for i, x in enumerate(a0)},
                 **{"i_c%03d" % i: np.asarray(x) for i, x in enumerate(c0)},
                 **{"a%03d" % i: np.asarray(x) for i, x in enumerate(a)},
                 **{"c%03d" % i: np.asarray(x) for i, x in enumerate(c)})
        print("  wrote %d actor + %d critic leaves to %s" % (len(a), len(c), sys.argv[3]))


if __name__ == "__main__":
    main()
