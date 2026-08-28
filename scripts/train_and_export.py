"""Train REPPO (JAX) and keep the resulting parameters.

Upstream's ``run()`` in ``src/jaxrl/reppo.py`` throws the trained state away::

    _, metrics = jax.jit(train_fn, static_argnums=(1,))(key, ReppoConfig(...))

so nothing but ``metrics.npz`` survives a run. This driver mirrors that function's
env construction and training call but retains the returned ``SACTrainState`` and
hands it to ``scripts/export_ckpt.py``. Upstream files are left untouched.

Usage (from the repo root)::

    CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python scripts/train_and_export.py \
        env=mjx_dmc env.name=WalkerRun experiment_overrides=mjx_dmc_large_data \
        seed=0 num_trials=1 num_seeds=1 wandb.mode=disabled

``num_trials`` defaults to 10 upstream (sequential re-runs, each overwriting
``metrics.npz``); pass ``num_trials=1``. Task ids go verbatim to the
mujoco_playground registry, so they are CamelCase: ``WalkerRun``, ``HumanoidRun``.
"""

from __future__ import annotations

import logging
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import hydra  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from scripts.export_ckpt import (  # noqa: E402
    export_snapshot,
    export_train_state,
    summarize,
)
from src.env_utils.jax_wrappers import BraxGymnaxWrapper, MjxGymnaxWrapper  # noqa: E402
from src.jaxrl.reppo import ReppoConfig, make_train_fn  # noqa: E402


def build_env(cfg: DictConfig):
    """Mirror of the env construction in ``src/jaxrl/reppo.py`` run()."""
    if cfg.env.type == "brax":
        return BraxGymnaxWrapper(
            cfg.env.name,
            episode_length=cfg.env.max_episode_steps,
            reward_scaling=cfg.env.reward_scaling,
            terminate=cfg.env.terminate,
        )
    if cfg.env.type == "mjx":
        return MjxGymnaxWrapper(
            cfg.env.name,
            episode_length=cfg.env.max_episode_steps,
            reward_scale=cfg.env.reward_scaling,
            push_distractions=cfg.env.get("push_distractions", False),
            asymmetric_observation=cfg.env.get("asymmetric_obs", False),
        )
    raise ValueError(f"Unknown environment type: {cfg.env.type}")


@hydra.main(version_base=None, config_path="../config", config_name="reppo")
def main(cfg: DictConfig) -> None:
    cfg.hyperparameters = OmegaConf.merge(
        cfg.hyperparameters, cfg.experiment_overrides.hyperparameters
    )
    logging.info(OmegaConf.to_yaml(cfg))

    if cfg.num_trials != 1:
        logging.warning(
            "num_trials=%s: only the final trial's parameters would be exported. "
            "Pass num_trials=1 unless you mean it.",
            cfg.num_trials,
        )

    env = build_env(cfg)
    obs_space, critic_obs_space = env.observation_space(None)
    action_space = env.action_space(None)

    history: list[dict] = []

    def log_callback(state, metrics):
        metrics = dict(metrics)
        metrics["sys_time"] = time.perf_counter()
        if history:
            steps = state.time_steps[0] - history[-1]["time_step"][0]
            sps = steps / (metrics["sys_time"] - history[-1]["sys_time"])
        else:
            sps = 0.0
        history.append(metrics)
        def g(k, default=float("nan")):
            v = metrics.get(k)
            return float(np.asarray(v).mean()) if v is not None else default

        logging.info(
            "step=%s ret=%.3f len=%.1f sps=%.0f | ent=%.3f sigma=%.3f "
            "[%.3f,%.3f] temp=%.5f kl=%.4f | ess=%.2f w_max=%.3f qspread=%.2f eta=%.4f | b_mu=%.3g b_sg=%.3g kl_mu=%.4f kl_sg=%.2e pin=%.2f/%.2f | essP=%.1f/%.1f/%.1f/%.1f lt4=%.3f",
            state.time_steps[0],
            float(metrics["eval/episode_return"].mean()),
            float(metrics["eval/episode_length"].mean()),
            sps,
            g("train/entropy"),
            g("train/pi_sigma_mean"),
            g("train/pi_sigma_min"),
            g("train/pi_sigma_max"),
            g("train/temp"),
            g("train/kl"),
            g("train/ess"),
            g("train/w_max"),
            g("train/q_spread"),
            g("train/eta"),
            g("train/beta_mu"),
            g("train/beta_sigma"),
            g("train/kl_mu"),
            g("train/kl_sigma"),
            g("train/beta_mu_pinned"),
            g("train/beta_sigma_pinned"),
            g("train/ess_p5"),
            g("train/ess_p25"),
            g("train/ess_median"),
            g("train/ess_p75"),
            g("train/ess_frac_lt4"),
        )

    train_fn = make_train_fn(
        cfg=ReppoConfig(**cfg.hyperparameters),
        env=env,
        log_callback=log_callback,
        num_seeds=cfg.num_seeds,
        reward_scale=1.0 / cfg.env.reward_scaling,
        return_snapshots=True,
    )

    key = jax.random.PRNGKey(cfg.seed)
    start = time.perf_counter()
    # The one substantive difference from upstream: keep the state.
    state, metrics, snaps = jax.jit(train_fn, static_argnums=(1,))(
        key, ReppoConfig(**cfg.hyperparameters)
    )
    jax.block_until_ready(metrics)
    duration = time.perf_counter() - start
    logging.info("Training took %.2f seconds.", duration)

    # Written into the hydra run dir, same as upstream, for verification comparison.
    jnp.savez("metrics.npz", **metrics)

    eval_returns = np.asarray(metrics["eval/episode_return"])
    final_return = float(eval_returns[-1].mean())
    nan_seen = bool(np.isnan(eval_returns).any())
    if nan_seen:
        # The upstream README flags Humanoid playground tasks as NaN-prone and says
        # the authors simply reran. Refuse to write a checkpoint full of NaNs -- an
        # exported-but-poisoned directory is worse than no directory.
        logging.error(
            "NaN present in eval/episode_return (env=%s seed=%s). Skipping export; "
            "relaunch with a different seed and record which seed was kept.",
            cfg.env.name,
            cfg.seed,
        )
        raise SystemExit(2)

    hp = ReppoConfig(**cfg.hyperparameters)
    mode = cfg.hyperparameters.actor_update_mode
    # the decoupled M-step is still actor_update_mode="weighted_mle", so without this
    # suffix it silently overwrites the single-KL-clip runs' checkpoints
    # Disambiguate arms that share actor_update_mode:
    #   pathwise + learned alpha -> ""      (arm A)
    #   pathwise + frozen alpha  -> "_fa"   (A-frozen control)
    #   weighted_mle + single KL -> ""      (arm B)
    #   weighted_mle + decoupled -> "_dec"
    frozen_alpha = not cfg.hyperparameters.update_entropy_lagrangian
    if cfg.hyperparameters.mstep_decoupled:
        # eps_sigma is in the tag: runs that differ only in the trust-region bound
        # would otherwise overwrite each other's checkpoints
        bsf = cfg.hyperparameters.beta_sigma_fixed
        variant = (f"_dec_bs{bsf:g}" if bsf is not None
                   else f"_dec_es{cfg.hyperparameters.eps_sigma:g}")
    elif mode == "pathwise" and frozen_alpha:
        variant = "_fa"
    else:
        variant = ""
    tag = f"{cfg.env.name}_{mode}{variant}_s{cfg.seed}"
    duals = summarize(state)

    ess_curve = np.asarray(metrics.get("train/ess", np.zeros(1))).mean(axis=-1).ravel()
    qspread_curve = np.asarray(
        metrics.get("train/q_spread", np.zeros(1))
    ).mean(axis=-1).ravel()
    eta_curve = np.asarray(metrics.get("train/eta", np.zeros(1))).mean(axis=-1).ravel()
    cur = lambda k: np.asarray(  # noqa: E731
        metrics.get(k, np.zeros(1))
    ).mean(axis=-1).ravel().tolist()
    ess_final = float(ess_curve[-1]) if ess_curve.size else float("nan")
    ess_flag = mode == "weighted_mle" and ess_final < 2.0

    shapes = dict(
        env_name=cfg.env.name,
        obs_dim=obs_space.shape[0],
        critic_obs_dim=critic_obs_space.shape[0],
        action_dim=action_space.shape[0],
    )
    common_meta = dict(
        actor_update_mode=mode,
        estep_num_samples=int(cfg.hyperparameters.estep_num_samples),
        reward_scaling=float(cfg.env.reward_scaling),
        hydra_run_dir=os.getcwd(),
    )

    # Mid-training snapshots. The 25% one is the high-entropy checkpoint; the
    # converged one is known to be useless for the fidelity study.
    n_iter = int(np.asarray(snaps["time_steps"]).shape[0])
    curve = eval_returns.mean(axis=-1)
    for frac, name in ((0.25, "p25"), (0.50, "p50")):
        idx = min(max(int(round(frac * n_iter)) - 1, 0), n_iter - 1)
        out_dir = os.path.join(REPO_ROOT, "exports", f"{tag}_{name}")
        export_snapshot(
            snaps, idx, hp, out_dir=out_dir, seed=cfg.seed, **shapes,
            extra_meta=dict(
                common_meta,
                checkpoint_frac=frac,
                snapshot_index=idx,
                eval_return_at_snapshot=float(curve[idx]),
                ess_at_snapshot=float(ess_curve[idx]) if ess_curve.size > idx else None,
                q_spread_at_snapshot=(
                    float(qspread_curve[idx]) if qspread_curve.size > idx else None
                ),
                eta_at_snapshot=(
                    float(eta_curve[idx]) if eta_curve.size > idx else None
                ),
            ),
        )
        logging.info("Exported %s (iter %d/%d, return %.1f)", out_dir, idx + 1,
                     n_iter, float(curve[idx]))

    out_dir = os.path.join(REPO_ROOT, "exports", f"{tag}_final")
    export_train_state(
        state=state, hp=hp, out_dir=out_dir, seed=cfg.seed, **shapes,
        extra_meta=dict(
            common_meta,
            checkpoint_frac=1.0,
            final_eval_return=final_return,
            eval_return_curve=curve.tolist(),
            ess_curve=ess_curve.tolist(),
            q_spread_curve=qspread_curve.tolist(),
            eta_curve=eta_curve.tolist(),
            ess_p5_curve=cur("train/ess_p5"),
            ess_p25_curve=cur("train/ess_p25"),
            ess_median_curve=cur("train/ess_median"),
            ess_p75_curve=cur("train/ess_p75"),
            ess_frac_lt4_curve=cur("train/ess_frac_lt4"),
            beta_mu_curve=cur("train/beta_mu"),
            beta_sigma_curve=cur("train/beta_sigma"),
            kl_mu_curve=cur("train/kl_mu"),
            kl_sigma_curve=cur("train/kl_sigma"),
            beta_mu_pinned_curve=cur("train/beta_mu_pinned"),
            beta_sigma_pinned_curve=cur("train/beta_sigma_pinned"),
            alpha_curve=np.asarray(
                metrics.get("train/temp", np.zeros(1))
            ).mean(axis=-1).ravel().tolist(),
            kl_curve=np.asarray(
                metrics.get("train/kl", np.zeros(1))
            ).mean(axis=-1).ravel().tolist(),
            entropy_curve=np.asarray(
                metrics.get("train/entropy", np.zeros(1))
            ).mean(axis=-1).ravel().tolist(),
            pi_sigma_curve=np.asarray(
                metrics.get("train/pi_sigma_mean", np.zeros(1))
            ).mean(axis=-1).ravel().tolist(),
            train_seconds=duration,
            nan_in_eval=nan_seen,
            alpha_entropy=duals["alpha_entropy"],
            alpha_kl=duals["alpha_kl"],
            ess_final=ess_final,
            ess_degenerate=bool(ess_flag),
        ),
    )
    logging.info(
        "Exported %s | return %.3f | alpha_ent %.5f | alpha_kl %.5f | ess %.2f",
        out_dir, final_return, duals["alpha_entropy"], duals["alpha_kl"], ess_final,
    )
    if ess_flag:
        logging.error(
            "ESS=%.2f < 2 at end of training: the weighted_mle arm has degenerated "
            "into cloning a single sample; this run is uninterpretable.", ess_final
        )


if __name__ == "__main__":
    main()
