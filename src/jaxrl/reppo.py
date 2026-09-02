import logging
import time
import typing
from typing import Callable

import hydra
import jax
import numpy as np
import optax
import optuna
import plotly.graph_objs as go
from flax import nnx, struct
from flax.struct import PyTreeNode
from gymnax.environments.environment import Environment, EnvParams, EnvState
from jax import numpy as jnp
from jax.random import PRNGKey
from omegaconf import DictConfig, OmegaConf

import wandb
from src.env_utils.jax_wrappers import (
    BraxGymnaxWrapper,
    ClipAction,
    LogWrapper,
    MjxGymnaxWrapper,
    NormalizeVec,
)
from src.jaxrl import utils
from src.networks.jax_models import (
    CategoricalCriticNetwork,
    CriticNetwork,
    SACActorNetworks,
)

logging.basicConfig(level=logging.INFO)


class Policy(typing.Protocol):
    def __call__(
        self,
        key: jax.random.PRNGKey,
        obs: PyTreeNode,
    ) -> tuple[PyTreeNode, PyTreeNode]:
        pass


class Transition(struct.PyTreeNode):
    obs: jax.Array
    critic_obs: jax.Array
    action: jax.Array
    reward: jax.Array
    soft_reward: jax.Array
    next_emb: jax.Array
    value: jax.Array
    done: jax.Array
    truncated: jax.Array
    importance_weight: jax.Array
    info: dict[str, jax.Array]


class ReppoConfig(struct.PyTreeNode):
    lr: float
    gamma: float
    total_time_steps: int
    num_steps: int
    lmbda: float
    lmbda_min: float
    num_mini_batches: int
    num_envs: int
    num_epochs: int
    max_grad_norm: float | None
    normalize_env: bool
    polyak: float
    exploration_noise_min: float
    exploration_noise_max: float
    exploration_base_envs: int
    ent_start: float
    ent_target_mult: float
    kl_start: float
    eval_interval: int = 10
    num_eval: int = 25
    max_episode_steps: int = 1000
    critic_hidden_dim: int = 512
    actor_hidden_dim: int = 512
    vmin: int = -100
    vmax: int = 100
    num_bins: int = 250
    hl_gauss: bool = False
    kl_bound: float = 1.0
    aux_loss_mult: float = 0.0
    update_kl_lagrangian: bool = True
    update_entropy_lagrangian: bool = True
    use_critic_norm: bool = True
    num_critic_encoder_layers: int = 1
    num_critic_head_layers: int = 1
    num_critic_pred_layers: int = 1
    use_simplical_embedding: bool = False
    use_critic_skip: bool = False
    use_actor_norm: bool = True
    num_actor_layers: int = 2
    actor_min_std: float = 0.05
    use_actor_skip: bool = False
    reduce_kl: bool = True
    reverse_kl: bool = False
    anneal_lr: bool = False
    actor_kl_clip_mode: str = "clipped"
    # "pathwise" = original SAC-style loss (grad flows through the critic).
    # "weighted_mle" = MPO E-step: softmax-weighted MLE on pi_old samples, no
    # gradient through the critic. eta is tied to alpha by construction.
    actor_update_mode: str = "pathwise"
    estep_num_samples: int = 32
    # Diagnostic only. In the pathwise arm computing q_spread needs an extra critic
    # forward, and merely adding that op changes XLA fusion and hence float32
    # rounding -- which breaks bit-identity with the pre-E-step implementation.
    # Default off so the pathwise arm stays reproducible; the same quantity can be
    # measured offline from an exported checkpoint (scripts/q_spread_from_ckpt.py).
    log_q_spread: bool = False
    # Estimator diagnostics for algorithm development (Phase 3/4). Same bit-identity
    # caveat as log_q_spread: it adds critic ops and an extra autodiff pass, which
    # perturbs XLA fusion and hence float32 rounding. Default OFF so every
    # confirmatory run stays bit-identical to the pristine reference; development
    # runs in the 201+ seed namespace turn it on.
    log_estimator_diag: bool = False
    # Per-checkpoint evaluation IQM. The trainer otherwise stores only the MEAN over
    # ~num_envs episodes, and the ladder's normalization anchor is defined on the IQM
    # (docs/prereg_dimension_ladder.md, Sec. 2). Adds ops inside the eval branch, so
    # it is gated: default OFF keeps confirmatory runs bit-identical to the pristine
    # reference; the seed-901 anchor reruns turn it on.
    log_eval_iqm: bool = False
    # ---- faithful-repair flags (docs/faithful_repair_design.md) --------------
    # Repairs ONE upstream coding inconsistency: the old- and new-policy log
    # probabilities entering the sampled KL are evaluated at different action
    # points, because `jnp.clip` sits between them. With this on, the pre-squash
    # latent y_i is materialised and BOTH log probabilities are computed from that
    # same y_i, so no clip is needed and the tanh Jacobian cancels exactly.
    # The published gate and the published exponential multiplier are UNCHANGED.
    # Default off: with every faithful flag off the code is byte-identical to the
    # original implementation.
    faithful_same_point: bool = False
    # Split a fresh actor-sampling key inside the minibatch scan. Legacy behaviour
    # reuses one key for every minibatch of an epoch.
    fresh_minibatch_key: bool = False
    # Faithful-repair diagnostics. Computed, never zero-placeheld; disabled fields
    # are written as NaN.
    log_faithful_diag: bool = False
    # E-step KL budget. eta is solved against this by its own dual, rather than
    # being tied to the entropy dual alpha.
    eps_e: float = 0.5
    # MPO decoupled M-step: replaces (does not stack with) REPPO's single KL clip.
    mstep_decoupled: bool = False
    eps_mu: float = 0.1
    eps_sigma: float = 5e-5      # four orders below eps_mu -- that is the point
    # Hold beta_sigma at a CONSTANT instead of learning it (same move as freezing
    # alpha). None = learn it via the dual. eps_sigma is then irrelevant to the
    # sigma constraint, and beta_sigma becomes a clean, unconfounded width knob.
    beta_sigma_fixed: float | None = None
    # The SAC entropy error is a SUM over action dims, so the same per-dim mismatch
    # drives alpha d times harder. At d=21 that is 3.5x the d=6 pressure. Setting this
    # divides the error by d, making it a MEAN over dims.
    ent_loss_per_dim: bool = False
    # Covariance-freeze intervention (docs/covariance_freeze_design_note.md).
    # null  -> learned state-dependent sigma, exactly the corrected replication.
    # scalar or length-d vector -> that EFFECTIVE pre-squash sigma everywhere.
    freeze_sigma: float | list[float] | None = None


def estep_weights(q_i, eta):
    """MPO E-step weights over the sample axis (axis 0).

        w_i propto exp(Q(s, a_i) / eta)

    The E-step target is q*(a|s) propto pi_old(a|s) * exp(Q(s,a)/eta). Because
    pi_old sits INSIDE the target it cancels against the pi_old proposal the
    samples are drawn from, so the self-normalised weight is just exp(Q/eta) --
    there is no 1/pi_old importance factor.

    (An earlier version targeted the un-anchored exp(Q/eta) and therefore carried
    a 1/pi_old factor. Its spread, ~0.5*sqrt(2d) = 1.73 at d=6, swamped the Q
    signal: with Q zeroed that term alone predicts ESS 4.4 and the run measured
    4.2, i.e. the critic was contributing nothing to the weights.)

    eta is a learned dual solved against the E-step KL budget eps_e (see
    `eta_dual_loss`). Softmax subtracts the per-state max internally, so this is
    numerically safe.
    """
    return jax.nn.softmax(q_i / eta, axis=0)


def eta_dual_loss(q_i, eta, eps_e):
    """Standard MPO dual for the E-step temperature, per state then averaged.

        g(eta) = eta * eps_e + eta * mean_j log mean_i exp(q_ji / eta)

    The log-mean-exp is evaluated with the per-state max pulled out:

        log mean_i exp(q_ji/eta) = log mean_i exp((q_ji - qmax_j)/eta) + qmax_j/eta

    so the exponent is <= 0 everywhere. q is detached: only eta takes gradient.
    """
    q_d = jax.lax.stop_gradient(q_i)
    qmax = jnp.max(q_d, axis=0)
    lse = jnp.log(jnp.mean(jnp.exp((q_d - qmax) / eta), axis=0))
    return eta * eps_e + jnp.mean(eta * lse + qmax)


def decoupled_kls(mu_new, sg_new, mu_old, sg_old):
    """MPO's two decoupled Gaussian KLs, on the PRE-SQUASH Gaussian.

        kl_mu    = 0.5 * sum_d (mu_new - mu_old)^2 / sg_old^2
        kl_sigma = 0.5 * sum_d [ sg_old^2/sg_new^2 - 1 + 2 log(sg_new/sg_old) ]

    Each term is the full Gaussian KL with the other factor held at its old value, so
    the mean and the scale get separate trust regions and separate multipliers.
    """
    kl_mu = 0.5 * jnp.sum(((mu_new - mu_old) ** 2) / (sg_old**2), axis=-1)
    kl_sigma = 0.5 * jnp.sum(
        (sg_old**2) / (sg_new**2) - 1.0 + 2.0 * (jnp.log(sg_new) - jnp.log(sg_old)),
        axis=-1,
    )
    return kl_mu, kl_sigma


def gaussian_logp(x, loc, scale):
    """Diagonal-Gaussian log density, summed over action dims."""
    return jnp.sum(
        -0.5 * (((x - loc) / scale) ** 2) - jnp.log(scale) - 0.5 * jnp.log(2 * jnp.pi),
        axis=-1,
    )


def tanh_log_det_jacobian(y):
    """sum_j log(1 - tanh(y_j)^2), evaluated stably from the PRE-SQUASH latent.

    Uses log(1 - tanh(y)^2) = 2*(log 2 - y - softplus(-2y)), which is finite for
    every finite y and does not underflow as |y| grows. Never calls arctanh.
    """
    return jnp.sum(
        2.0 * (jnp.log(2.0) - y - jax.nn.softplus(-2.0 * y)), axis=-1
    )


def gaussian_kl_diag(mu0, sg0, mu1, sg1):
    """Exact KL( N(mu0, sg0^2) || N(mu1, sg1^2) ) for diagonal Gaussians, summed
    over action dimensions.

    The tanh push-forward is a smooth invertible reparameterisation, so the KL
    between the transformed laws equals the KL between the pre-squash Gaussians:
    the Jacobian term appears in both densities and cancels in the log ratio.
    This is therefore the exact KL of the tanh-Normal policies as well.
    """
    return jnp.sum(
        jnp.log(sg1 / sg0)
        + (sg0**2 + (mu0 - mu1) ** 2) / (2.0 * sg1**2)
        - 0.5,
        axis=-1,
    )


def effective_sample_size(w, axis=0):
    """1 / sum_i w_i^2 for weights summing to 1 along `axis`. Range [1, M]."""
    return 1.0 / jnp.sum(w**2, axis=axis)


class SACTrainState(struct.PyTreeNode):
    critic: nnx.TrainState
    actor: nnx.TrainState
    actor_target: nnx.TrainState
    iteration: int
    time_steps: int
    last_env_state: EnvState
    last_obs: jax.Array
    last_critic_obs: jax.Array


def make_policy(
    train_state: SACTrainState,
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, dict]]:
    def policy(key: PRNGKey, obs: jax.Array) -> tuple[jax.Array, dict]:
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        action: jax.Array = actor_model.det_action(obs)
        return action, {}

    return policy


def make_eval_fn(
    env: Environment, max_episode_steps: int, reward_scale: float = 1.0,
    log_iqm: bool = False,
) -> Callable[[jax.random.PRNGKey, Policy, PyTreeNode | None], dict[str, float]]:
    def evaluation_fn(
        key: jax.random.PRNGKey, policy: Policy, norm_state: PyTreeNode | None
    ):
        def step_env(carry, _):
            key, env_state, obs = carry
            key, act_key, env_key = jax.random.split(key, 3)
            action, _ = policy(act_key, obs)
            step_key = jax.random.split(env_key, env.num_envs)
            obs, _, env_state, reward, done, info = env.step(
                step_key, env_state, action
            )
            return (key, env_state, obs), info

        key, init_key = jax.random.split(key)
        init_key = jax.random.split(init_key, env.num_envs)
        obs, _, env_state = env.reset(init_key, norm_state)
        # randomize initial steps
        key, env_key = jax.random.split(key)
        _, infos = jax.lax.scan(
            f=step_env,
            init=(key, env_state, obs),
            xs=None,
            length=max_episode_steps,
        )

        if log_iqm:
            # Interquartile mean over COMPLETED episodes: the middle 50% of the
            # return distribution, which is what the ladder's U_t anchor is defined
            # on. The mean sits below it whenever the tail of failed episodes is
            # heavy, so the two are not interchangeable.
            _r = infos["returned_episode_returns"]
            _m = infos["returned_episode"]
            _v = jnp.where(_m, _r, jnp.nan)
            _q25 = jnp.nanquantile(_v, 0.25)
            _q75 = jnp.nanquantile(_v, 0.75)
            _in = _m & (_r >= _q25) & (_r <= _q75)
            _iqm = jnp.sum(jnp.where(_in, _r, 0.0)) / jnp.maximum(jnp.sum(_in), 1)
            _extra = {
                "episode_return_iqm": _iqm * reward_scale,
                "episode_return_q25": _q25 * reward_scale,
                "episode_return_q75": _q75 * reward_scale,
                "num_episodes_iqm": jnp.sum(_in),
            }
        else:
            _extra = {
                "episode_return_iqm": jnp.zeros(()),
                "episode_return_q25": jnp.zeros(()),
                "episode_return_q75": jnp.zeros(()),
                "num_episodes_iqm": jnp.zeros(()),
            }

        return {
            **_extra,
            "episode_return": infos["returned_episode_returns"].mean(
                where=infos["returned_episode"]
            )
            * reward_scale,
            "episode_return_std": infos["returned_episode_returns"].std(
                where=infos["returned_episode"]
            ),
            "episode_length": infos["returned_episode_lengths"].mean(
                where=infos["returned_episode"]
            ),
            "episode_length_std": infos["returned_episode_lengths"].std(
                where=infos["returned_episode"]
            ),
            "num_episodes": infos["returned_episode"].sum(),
        }

    return evaluation_fn


def make_init(
    cfg: ReppoConfig,
    env: Environment,
    env_params: EnvParams = None,
) -> Callable[[jax.Array], SACTrainState]:
    def init(key: jax.random.PRNGKey) -> SACTrainState:
        # Number of calls to train_step
        key, model_key = jax.random.split(key)
        actor_networks = SACActorNetworks(
            obs_dim=env.observation_space(env_params)[0].shape[0],
            action_dim=env.action_space(env_params).shape[0],
            hidden_dim=cfg.actor_hidden_dim,
            ent_start=cfg.ent_start,
            kl_start=cfg.kl_start,
            use_norm=cfg.use_actor_norm,
            layers=cfg.num_actor_layers,
            use_skip=cfg.use_actor_skip,
            with_eta=cfg.actor_update_mode == "weighted_mle",
            with_betas=cfg.mstep_decoupled,
            freeze_sigma=cfg.freeze_sigma,
            rngs=nnx.Rngs(model_key),
        )
        actor_target_networks = SACActorNetworks(
            obs_dim=env.observation_space(env_params)[0].shape[0],
            action_dim=env.action_space(env_params).shape[0],
            hidden_dim=cfg.actor_hidden_dim,
            ent_start=cfg.ent_start,
            kl_start=cfg.kl_start,
            use_norm=cfg.use_actor_norm,
            layers=cfg.num_actor_layers,
            use_skip=cfg.use_actor_skip,
            with_eta=cfg.actor_update_mode == "weighted_mle",
            with_betas=cfg.mstep_decoupled,
            freeze_sigma=cfg.freeze_sigma,
            rngs=nnx.Rngs(model_key),
        )

        if cfg.hl_gauss:
            critic_networks: nnx.Module = CategoricalCriticNetwork(
                obs_dim=env.observation_space(env_params)[1].shape[0],
                action_dim=env.action_space(env_params).shape[0],
                hidden_dim=cfg.critic_hidden_dim,
                num_bins=cfg.num_bins,
                vmin=cfg.vmin,
                vmax=cfg.vmax,
                use_norm=cfg.use_critic_norm,
                encoder_layers=cfg.num_critic_encoder_layers,
                use_simplical_embedding=cfg.use_simplical_embedding,
                head_layers=cfg.num_critic_head_layers,
                pred_layers=cfg.num_critic_pred_layers,
                use_skip=cfg.use_critic_skip,
                rngs=nnx.Rngs(model_key),
            )
        else:
            critic_networks: nnx.Module = CriticNetwork(
                obs_dim=env.observation_space(env_params)[1].shape[0],
                action_dim=env.action_space(env_params).shape[0],
                hidden_dim=cfg.critic_hidden_dim,
                use_norm=cfg.use_critic_norm,
                encoder_layers=cfg.num_critic_encoder_layers,
                use_simplical_embedding=cfg.use_simplical_embedding,
                head_layers=cfg.num_critic_head_layers,
                pred_layers=cfg.num_critic_pred_layers,
                use_skip=cfg.use_critic_skip,
                rngs=nnx.Rngs(model_key),
            )

        if not cfg.anneal_lr:
            lr = cfg.lr
        else:
            num_iterations = cfg.total_time_steps // cfg.num_steps // cfg.num_envs
            num_updates = num_iterations * cfg.num_epochs * cfg.num_mini_batches
            lr = optax.linear_schedule(cfg.lr, 0, num_updates)

        if cfg.max_grad_norm is not None:
            actor_optimizer = optax.chain(
                optax.clip_by_global_norm(cfg.max_grad_norm),
                optax.adam(lr)
            )
            critic_optimizer = optax.chain(
                optax.clip_by_global_norm(cfg.max_grad_norm),
                optax.adam(lr)
            )
        else:
            actor_optimizer = optax.adam(lr)
            critic_optimizer = optax.adam(lr)

        actor_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(actor_networks),
            params=nnx.state(actor_networks),
            tx=actor_optimizer,
        )
        actor_target_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(actor_target_networks),
            params=nnx.state(actor_target_networks),
            tx=optax.set_to_zero(),
        )
        critic_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(critic_networks),
            params=nnx.state(critic_networks),
            tx=critic_optimizer,
        )

        key, env_key = jax.random.split(key)
        env_key = jax.random.split(env_key, cfg.num_envs)
        obs, critic_obs, env_state = env.reset(key=env_key, params=env_params)

        # randomize initial time step to prevent all envs stepping in tandem
        _env_state = env_state.unwrapped()
        key, randomize_steps_key = jax.random.split(key)
        _env_state.info["steps"] = jax.random.randint(
            randomize_steps_key,
            _env_state.info["steps"].shape,
            0,
            cfg.max_episode_steps,
        ).astype(jnp.float32)
        env_state.set_env_state(_env_state)

        return SACTrainState(
            actor=actor_trainstate,
            actor_target=actor_target_trainstate,
            critic=critic_trainstate,
            iteration=0,
            time_steps=0,
            last_env_state=env_state,
            last_obs=obs,
            last_critic_obs=critic_obs,
        )

    return init


def make_train_fn(
    cfg: ReppoConfig,
    env: Environment,
    env_params: EnvParams = None,
    log_callback: Callable[[SACTrainState, dict[str, jax.Array]], None] | None = None,
    num_seeds: int = 1,
    reward_scale: float = 1.0,
    return_snapshots: bool = False,
):
    env_params = env_params  # or env.default_params
    env = LogWrapper(env, cfg.num_envs)
    env = ClipAction(env)
    # env = VecEnv(env, cfg.num_envs)
    if cfg.normalize_env:
        env = NormalizeVec(env)
    eval_fn = make_eval_fn(
        env, cfg.max_episode_steps, reward_scale=reward_scale,
        log_iqm=cfg.log_eval_iqm,
    )
    action_size_target = (
        jnp.prod(jnp.array(env.action_space(env_params).shape)) * cfg.ent_target_mult
    )
    action_dim_f = jnp.prod(jnp.array(env.action_space(env_params).shape)).astype(
        jnp.float32
    )

    if cfg.freeze_sigma is not None and not (
        cfg.exploration_noise_min == 1.0 and cfg.exploration_noise_max == 1.0
    ):
        raise ValueError(
            "freeze_sigma requires exploration_noise_min == exploration_noise_max == 1.0: "
            "the rollout `scale` multiplies sigma, so any other value would make the "
            "effective width differ from the frozen value it is declared to be."
        )

    def collect_rollout(
        key: PRNGKey, train_state: SACTrainState
    ) -> tuple[Transition, SACTrainState]:
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)

        offset = (
            jnp.arange(cfg.num_envs - cfg.exploration_base_envs)[:, None]
            * (cfg.exploration_noise_max - cfg.exploration_noise_min)
            / (cfg.num_envs - cfg.exploration_base_envs)
        ) + cfg.exploration_noise_min
        offset = jnp.concatenate(
            [
                jnp.ones((cfg.exploration_base_envs, 1)) * cfg.exploration_noise_min,
                offset,
            ],
            axis=0,
        )

        def step_env(carry, _) -> tuple[tuple, Transition]:
            key, env_state, train_state, obs, critic_obs = carry
            key, act_key, step_key = jax.random.split(key, 3)
            step_key = jax.random.split(step_key, cfg.num_envs)

            # get policy action
            og_pi = actor_model.actor(obs)
            pi = actor_model.actor(obs, scale=offset)
            action = pi.sample(seed=act_key)

            next_obs, next_critic_obs, next_env_state, reward, done, info = env.step(
                step_key, env_state, action
            )

            # compute importance weights
            action = jnp.clip(action, -0.999, 0.999)
            raw_importance_weight = jnp.nan_to_num(
                og_pi.log_prob(action).sum(-1) - pi.log_prob(action).sum(-1),
                nan=jnp.log(cfg.lmbda_min),
            )
            importance_weight = jnp.clip(
                raw_importance_weight, min=jnp.log(cfg.lmbda_min), max=jnp.log(1.0)
            )

            # compute next state embedding and value
            next_action, log_prob = actor_model.actor(next_obs).sample_and_log_prob(
                seed=act_key
            )
            next_emb, _, _, value = critic_model.forward(next_critic_obs, next_action)
            soft_reward = (
                reward
                - cfg.gamma * log_prob.sum(-1).squeeze() * actor_model.temperature()
            )
            transition = Transition(
                obs=obs,
                critic_obs=critic_obs,
                action=action,
                next_emb=next_emb,
                reward=reward,
                soft_reward=soft_reward,
                value=value,
                done=done,
                truncated=next_env_state.truncated,
                info=info,
                importance_weight=importance_weight,
            )
            return (
                key,
                next_env_state,
                train_state,
                next_obs,
                next_critic_obs,
            ), transition

        rollout_state, transitions = jax.lax.scan(
            f=step_env,
            init=(
                key,
                train_state.last_env_state,
                train_state,
                train_state.last_obs,
                train_state.last_critic_obs,
            ),
            length=cfg.num_steps,
        )
        _, last_env_state, train_state, last_obs, last_critic_obs = rollout_state
        train_state = train_state.replace(
            last_env_state=last_env_state,
            last_obs=last_obs,
            last_critic_obs=last_critic_obs,
            time_steps=train_state.time_steps + cfg.num_steps * cfg.num_envs,
        )

        return transitions, train_state

    def learn_step(
        key: PRNGKey, train_state: SACTrainState, batch: Transition
    ) -> tuple[SACTrainState, dict[str, jax.Array]]:
        # compute n-step lambda estimates

        def compute_nstep_lambda(carry, transition):
            lambda_return, truncated, importance_weight = carry
            # combine importance_weights with TD lambda
            done = transition.done
            reward = transition.soft_reward
            value = transition.value
            lambda_sum = (
                jnp.exp(importance_weight) * cfg.lmbda * lambda_return
                + (1 - jnp.exp(importance_weight) * cfg.lmbda) * value
            )
            delta = cfg.gamma * jnp.where(truncated, value, (1.0 - done) * lambda_sum)
            lambda_return = reward + delta
            truncated = transition.truncated
            return (
                lambda_return,
                truncated,
                transition.importance_weight,
            ), lambda_return

        _, target_values = jax.lax.scan(
            compute_nstep_lambda,
            (
                batch.value[-1],
                jnp.ones_like(batch.truncated[0]),
                jnp.zeros_like(batch.importance_weight[0]),
            ),
            batch,
            reverse=True,
        )
        # Reshape data to (num_steps * num_envs, ...)
        data = (batch, target_values)
        data = jax.tree.map(
            lambda x: x.reshape((cfg.num_steps * cfg.num_envs, *x.shape[2:])), data
        )

        train_state = train_state.replace(
            actor_target=train_state.actor_target.replace(
                params=train_state.actor.params
            ),
        )
        actor_target_model = nnx.merge(
            train_state.actor_target.graphdef, train_state.actor_target.params
        )

        def update(train_state, key) -> tuple[SACTrainState, dict[str, jax.Array]]:
            def minibatch_update(carry, indices):
                idx, train_state = carry
                # Faithful-repair: a fresh actor-sampling key per minibatch. Legacy
                # reuses the epoch key for every minibatch, so the standard-normal
                # array is bit-identical across the whole epoch. Folding on `idx`
                # keeps this deterministic for a fixed root seed. Only the actor
                # stream is touched: env, rollout, permutation and eval keys are
                # derived elsewhere and are unaffected, so changing the actor sample
                # count cannot shift the environment realisation.
                akey = (
                    jax.random.fold_in(key, idx)
                    if cfg.fresh_minibatch_key
                    else key
                )
                # Sample data at indices from the batch
                minibatch, target_values = jax.tree.map(
                    lambda x: jnp.take(x, indices, axis=0), data
                )

                def critic_loss_fn(params):
                    critic_model = nnx.merge(train_state.critic.graphdef, params)
                    critic_pred = critic_model.critic_cat(
                        minibatch.critic_obs, minibatch.action
                    ).squeeze()
                    if cfg.hl_gauss:
                        target_cat = jax.vmap(
                            utils.hl_gauss, in_axes=(0, None, None, None)
                        )(target_values, cfg.num_bins, cfg.vmin, cfg.vmax)
                        critic_update_loss = optax.softmax_cross_entropy(
                            critic_pred, target_cat
                        )
                    else:
                        critic_update_loss = optax.squared_error(
                            critic_pred.reshape(-1,1),
                            target_values.reshape(-1,1),
                        )

                    # Aux loss
                    _, pred, pred_rew, value = critic_model.forward(
                        minibatch.critic_obs, minibatch.action
                    )
                    aux_loss = optax.squared_error(pred,  minibatch.next_emb)
                    aux_rew_loss = optax.squared_error(pred_rew, minibatch.reward.reshape(-1, 1))
                    aux_loss = jnp.mean(
                        (1 - minibatch.done.reshape(-1, 1))
                        * jnp.concatenate(
                            [aux_loss, aux_rew_loss], axis=-1
                        ), axis=-1)

                    # compute l2 error for logging
                    critic_loss = optax.squared_error(
                        value,
                        target_values,
                    )
                    critic_loss = jnp.mean(critic_loss)
                    loss = jnp.mean(
                        (1.0 - minibatch.truncated)
                        * (critic_update_loss + cfg.aux_loss_mult * aux_loss)
                    )
                    return loss, dict(
                        value_loss=critic_loss,
                        critic_update_loss=critic_update_loss,
                        loss=loss,
                        aux_loss=aux_loss,
                        rew_aux_loss= aux_rew_loss,
                        q=value.mean(),
                        abs_batch_action=jnp.abs(minibatch.action).mean(),
                        reward_mean=minibatch.reward.mean(),
                        target_values=target_values.mean(),
                    )

                def actor_loss(params):
                    critic_target_model = nnx.merge(
                        train_state.critic.graphdef,
                        train_state.critic.params,
                    )
                    actor_model = nnx.merge(train_state.actor.graphdef, params)

                    # SAC actor loss
                    pi = actor_model.actor(minibatch.obs)
                    # pre-squash Gaussian std: the policy's own scale parameter,
                    # which is what locates a run on the sigma axis
                    pi_sigma = pi.distribution.scale
                    pred_action, log_prob = pi.sample_and_log_prob(seed=akey)
                    value = critic_target_model.critic(
                        minibatch.critic_obs, pred_action
                    )
                    log_prob = log_prob.sum(-1)
                    entropy = -log_prob

                    # policy KL constraint
                    if cfg.reverse_kl:
                        pi_action, pi_act_log_prob = pi.sample_and_log_prob(
                            sample_shape=(16,), seed=key
                        )
                        pi_action = jnp.clip(pi_action, -1 + 1e-4, 1 - 1e-4)

                        old_pi = actor_target_model.actor(minibatch.obs)

                        old_pi_act_log_prob = old_pi.log_prob(pi_action).sum(-1).mean(0)
                        pi_act_log_prob = pi_act_log_prob.sum(-1).mean(0)
                        kl = pi_act_log_prob - old_pi_act_log_prob
                    else:
                        n_estep = (
                            cfg.estep_num_samples
                            if cfg.actor_update_mode == "weighted_mle"
                            else 16
                        )
                        if cfg.faithful_same_point:
                            # ---- same-point latent path -----------------------
                            # Materialise y_i and keep it. Both log probabilities,
                            # the critic action and the WML likelihood all use this
                            # one (y_i, a_i) pair. No clip: y_i is carried forward,
                            # so arctanh is never needed and the clip rate is zero
                            # by construction.
                            mu_old_f, sg_old_f = actor_target_model.gaussian(
                                minibatch.obs
                            )
                            mu_old_f = jax.lax.stop_gradient(mu_old_f)
                            sg_old_f = jax.lax.stop_gradient(sg_old_f)
                            mu_new_f, sg_new_f = actor_model.gaussian(minibatch.obs)
                            u_i = jax.random.normal(
                                akey, (n_estep, *mu_old_f.shape)
                            )
                            y_i = mu_old_f[None] + sg_old_f[None] * u_i
                            old_pi_action = jnp.tanh(y_i)

                            # The tanh Jacobian is identical in both terms and
                            # cancels in their difference; it is subtracted from
                            # both so each is a true transformed log density.
                            _ldj = tanh_log_det_jacobian(y_i)
                            logp_old_i = (
                                gaussian_logp(y_i, mu_old_f[None], sg_old_f[None])
                                - _ldj
                            )
                            logp_theta_i = (
                                gaussian_logp(y_i, mu_new_f[None], sg_new_f[None])
                                - _ldj
                            )
                            # exact analytic KL(pi_old || pi_new), an independent
                            # diagnostic for the sampled estimate below
                            kl_analytic = gaussian_kl_diag(
                                mu_old_f, sg_old_f, mu_new_f, sg_new_f
                            )
                        else:
                            old_pi_action, old_pi_act_log_prob = (
                                actor_target_model.actor(minibatch.obs)
                                .sample_and_log_prob(
                                    sample_shape=(n_estep,), seed=akey
                                )
                            )
                            old_pi_action = jnp.clip(
                                old_pi_action, -1 + 1e-4, 1 - 1e-4
                            )

                            # keep the per-sample arrays: the E-step needs them
                            # un-averaged. The .mean(0) below is the uniform-weight
                            # special case.
                            logp_old_i = old_pi_act_log_prob.sum(-1)            # (M, B)
                            logp_theta_i = pi.log_prob(old_pi_action).sum(-1)   # (M, B)
                            kl_analytic = jnp.full(logp_old_i.shape[1:], jnp.nan)

                        old_pi_act_log_prob = logp_old_i.mean(0)
                        pi_act_log_prob = logp_theta_i.mean(0)

                        # KL(pi_old || pi_theta), forward, orientation unchanged
                        kl = old_pi_act_log_prob - pi_act_log_prob

                    lagrangian = actor_model.lagrangian()
                    if not (cfg.actor_update_mode == "weighted_mle"
                            and cfg.mstep_decoupled):
                        kl_mu = jnp.zeros_like(kl)
                        kl_sigma = jnp.zeros_like(kl)
                        beta_mu = jnp.zeros(())
                        beta_sigma = jnp.zeros(())
                        beta_loss = jnp.zeros(())

                    # `objective` is what the KL clip modes wrap. For "pathwise" it is
                    # the original SAC term verbatim, so that arm stays bit-identical.
                    if cfg.actor_update_mode == "weighted_mle":
                        if cfg.reverse_kl:
                            raise ValueError(
                                "actor_update_mode='weighted_mle' needs reverse_kl=False: "
                                "the E-step reweights samples drawn from pi_old."
                            )
                        eta = jnp.squeeze(actor_model.eta())
                        # critic obs broadcast to the sample axis; actions are reused,
                        # never re-sampled
                        critic_obs_i = jnp.broadcast_to(
                            minibatch.critic_obs,
                            (n_estep, *minibatch.critic_obs.shape),
                        )
                        q_i = critic_target_model.critic(critic_obs_i, old_pi_action)
                        # eta is detached in the weights; it takes gradient only
                        # through its own dual below
                        w_i = jax.lax.stop_gradient(
                            estep_weights(q_i, jax.lax.stop_gradient(eta))
                        )
                        objective = -jnp.sum(w_i * logp_theta_i, axis=0)
                        ess = effective_sample_size(w_i, axis=0)
                        w_max = w_i.max(axis=0)
                        # the actual E-step signal: spread of the softmax exponent
                        q_spread = (q_i / jax.lax.stop_gradient(eta)).std(axis=0)
                        eta_loss = eta_dual_loss(q_i, eta, cfg.eps_e)

                        if cfg.mstep_decoupled:
                            # MPO's decoupled M-step. Everything below is on the
                            # PRE-SQUASH Gaussian, as MPO and V-MPO define it.
                            mu_old, sg_old = actor_target_model.gaussian(minibatch.obs)
                            mu_old = jax.lax.stop_gradient(mu_old)
                            sg_old = jax.lax.stop_gradient(sg_old)
                            mu_new, sg_new = actor_model.gaussian(minibatch.obs)

                            # E-step samples drawn from the old pre-squash Gaussian; the
                            # critic still sees the squashed action
                            u_i = mu_old + sg_old * jax.random.normal(
                                key, (n_estep, *mu_old.shape)
                            )
                            a_i = jnp.clip(jnp.tanh(u_i), -1 + 1e-4, 1 - 1e-4)
                            q_i = critic_target_model.critic(critic_obs_i, a_i)
                            w_i = jax.lax.stop_gradient(
                                estep_weights(q_i, jax.lax.stop_gradient(eta))
                            )
                            ess = effective_sample_size(w_i, axis=0)
                            w_max = w_i.max(axis=0)
                            q_spread = (q_i / jax.lax.stop_gradient(eta)).std(axis=0)
                            eta_loss = eta_dual_loss(q_i, eta, cfg.eps_e)

                            # mean moves against the OLD scale; scale moves about the
                            # OLD mean -- the two halves of the decoupled objective
                            logp_mu = gaussian_logp(u_i, mu_new[None], sg_old[None])
                            logp_sigma = gaussian_logp(u_i, mu_old[None], sg_new[None])
                            objective = -jnp.sum(w_i * (logp_mu + logp_sigma), axis=0)

                            kl_mu, kl_sigma = decoupled_kls(
                                mu_new, sg_new, mu_old, sg_old
                            )
                            beta_mu = jnp.squeeze(actor_model.beta_mu())
                            beta_loss = -beta_mu * jax.lax.stop_gradient(
                                jnp.mean(kl_mu) - cfg.eps_mu
                            )
                            if cfg.beta_sigma_fixed is None:
                                beta_sigma = jnp.squeeze(actor_model.beta_sigma())
                                beta_loss = beta_loss - beta_sigma * jax.lax.stop_gradient(
                                    jnp.mean(kl_sigma) - cfg.eps_sigma
                                )
                            else:
                                # constant: no dual term, so beta_sigma_param gets no
                                # gradient and the logged value is exactly the config
                                beta_sigma = jnp.asarray(
                                    cfg.beta_sigma_fixed, dtype=jnp.float32
                                )
                    else:
                        objective = (
                            log_prob * jax.lax.stop_gradient(actor_model.temperature())
                            - value
                        )
                        # 0 is out of range for a real ESS (min is 1), so it reads
                        # unambiguously as "not applicable to this arm"
                        ess = jnp.zeros_like(kl)
                        w_max = jnp.zeros_like(kl)
                        eta = jnp.zeros(())
                        eta_loss = jnp.zeros(())
                        # measured in this arm too, purely as a diagnostic: it is what
                        # the E-step weights WOULD see. Under stop_gradient and reusing
                        # already-drawn samples, so it perturbs neither loss nor RNG.
                        if cfg.reverse_kl or not cfg.log_q_spread:
                            q_spread = jnp.zeros_like(kl)
                        else:
                            _alpha_d = jnp.squeeze(
                                jax.lax.stop_gradient(actor_model.temperature())
                            )
                            _cobs = jnp.broadcast_to(
                                minibatch.critic_obs,
                                (n_estep, *minibatch.critic_obs.shape),
                            )
                            _q = jax.lax.stop_gradient(
                                critic_target_model.critic(_cobs, old_pi_action)
                            )
                            q_spread = (_q / _alpha_d).std(axis=0)

                    if cfg.actor_update_mode == "weighted_mle" and cfg.mstep_decoupled:
                        # replaces the single KL clip entirely -- three simultaneous
                        # constraints would not be the experiment
                        actor_loss = (
                            objective
                            + jax.lax.stop_gradient(beta_mu) * kl_mu
                            + jax.lax.stop_gradient(beta_sigma) * kl_sigma
                        )
                    elif cfg.actor_kl_clip_mode == "full":
                        actor_loss = (
                            objective
                            + kl * jax.lax.stop_gradient(lagrangian) * cfg.reduce_kl
                        )
                    elif cfg.actor_kl_clip_mode == "clipped":
                        actor_loss = jnp.where(
                            kl < cfg.kl_bound,
                            objective,
                            kl * jax.lax.stop_gradient(lagrangian) * cfg.reduce_kl,
                        )
                    elif cfg.actor_kl_clip_mode == "value":
                        actor_loss = objective
                    else:
                        raise ValueError(
                            f"Unknown actor loss mode: {cfg.actor_kl_clip_mode}"
                        )

                    # SAC target entropy loss
                    target_entropy = action_size_target + entropy
                    if cfg.ent_loss_per_dim:
                        target_entropy = target_entropy / action_dim_f
                    target_entropy_loss = (
                        actor_model.temperature()
                        * jax.lax.stop_gradient(target_entropy)
                    )

                    # Lagrangian constraint (follows temperature update)
                    lagrangian_loss = -lagrangian * jax.lax.stop_gradient(
                        kl - cfg.kl_bound
                    )

                    # total loss
                    loss = jnp.mean(actor_loss)
                    if cfg.update_entropy_lagrangian:
                        loss += jnp.mean(target_entropy_loss)
                    decoupled = (
                        cfg.actor_update_mode == "weighted_mle" and cfg.mstep_decoupled
                    )
                    # With the decoupled M-step the old KL lagrangian no longer appears
                    # in the loss, so updating it would leave a second dual chasing a
                    # target it cannot actuate -- the exact failure mode that made the
                    # entropy dual destabilise this arm.
                    if cfg.update_kl_lagrangian and not decoupled:
                        loss += jnp.mean(lagrangian_loss)
                    if cfg.actor_update_mode == "weighted_mle":
                        loss += eta_loss
                    if decoupled:
                        loss += beta_loss

                    # ---- estimator diagnostics (development only) -----------------
                    # All under stop_gradient and reusing already-drawn samples, so
                    # neither `loss` nor the RNG stream is touched. Quantities follow
                    # docs/wasted_step_fraction_proposition.md: everything lives in the
                    # WHITENED pre-tanh metric u, where Sigma is diagonal, so
                    # Sigma^{1/2} is elementwise multiplication by sigma.
                    if cfg.log_estimator_diag and not cfg.reverse_kl:
                        _mu, _sg = actor_model.gaussian(minibatch.obs)
                        _mu = jax.lax.stop_gradient(_mu)
                        _sg = jax.lax.stop_gradient(_sg)

                        # h = Sigma^{1/2} grad_y Q(s, tanh(y)) at y = mu.  eq (7)/(19).
                        def _q_of_y(y):
                            return critic_target_model.critic(
                                minibatch.critic_obs, jnp.tanh(y)
                            ).sum()

                        _h = _sg * jax.lax.stop_gradient(jax.grad(_q_of_y)(_mu))
                        _h_norm = jnp.linalg.norm(_h, axis=-1)

                        # Recover the whitened draws behind the M reused samples.
                        _u_i = (jnp.arctanh(jax.lax.stop_gradient(old_pi_action))
                                - _mu[None]) / _sg[None]
                        _cobs_i = jnp.broadcast_to(
                            minibatch.critic_obs,
                            (n_estep, *minibatch.critic_obs.shape),
                        )
                        _q_i = jax.lax.stop_gradient(
                            critic_target_model.critic(
                                _cobs_i, jax.lax.stop_gradient(old_pi_action)
                            )
                        )
                        # canonical centred ZO estimator, eq (13)/(16):
                        #   a_hat_M = (1/M) sum_i (Q_i - Qbar) u_i,  E[a_hat_M] = (1-1/M) h
                        _qc = _q_i - _q_i.mean(axis=0, keepdims=True)
                        _terms = _qc[..., None] * _u_i
                        _a_hat = _terms.mean(axis=0)
                        _a_norm = jnp.linalg.norm(_a_hat, axis=-1)

                        _M = jnp.float32(n_estep)
                        # de-attenuate the known (1 - 1/M) shrinkage before comparing
                        _a_deatt = _a_hat * (_M / (_M - 1.0))
                        _den = jnp.maximum(_h_norm, 1e-12)
                        _cos = jnp.sum(_a_hat * _h, axis=-1) / jnp.maximum(
                            _a_norm * _h_norm, 1e-12
                        )
                        _err2 = jnp.square(
                            jnp.linalg.norm(_a_deatt - _h, axis=-1)
                        )
                        _rel_l2 = jnp.linalg.norm(_a_deatt - _h, axis=-1) / _den
                        # sampling-noise energy of the mean over M: tr(Cov)/M
                        _var = _terms.var(axis=0).sum(axis=-1) / _M
                        # squared bias with the sampling-noise energy removed
                        _bias2 = _err2 - _var * jnp.square(_M / (_M - 1.0))

                        _est = dict(
                            # M differs by arm: the pathwise branch draws 16
                            # samples, weighted_mle draws estep_num_samples. Logged so
                            # the decomposition stays checkable without inferring it.
                            est_M=jnp.float32(n_estep),
                            est_h_norm=_h_norm.mean(),
                            est_a_norm=_a_norm.mean(),
                            est_cos=_cos.mean(),
                            est_rel_l2=_rel_l2.mean(),
                            # Self-check: est_rel_l2_sq must equal
                            #   est_bias2_proxy + est_var_proxy * (M/(M-1))^2
                            # state by state. est_rel_l2 is a mean OF A RATIO and by
                            # Jensen sits below sqrt(est_rel_l2_sq); do not mix them.
                            est_rel_l2_sq=(_err2 / jnp.square(_den)).mean(),
                            est_var_proxy=(_var / jnp.square(_den)).mean(),
                            est_bias2_proxy=(_bias2 / jnp.square(_den)).mean(),
                            est_nonfinite=(
                                1.0 - jnp.isfinite(_cos).astype(jnp.float32)
                            ).mean(),
                        )
                        if cfg.actor_update_mode == "weighted_mle":
                            # what the E-step ACTUALLY moves the mean by, in the same
                            # whitened metric: argmax_mu sum_i w_i log N(u_i; mu, I).
                            _d = jnp.sum(w_i[..., None] * _u_i, axis=0)
                            _d_norm = jnp.linalg.norm(_d, axis=-1)
                            _est["est_wdisp_norm"] = _d_norm.mean()
                            _est["est_wdisp_cos"] = (
                                jnp.sum(_d * _h, axis=-1)
                                / jnp.maximum(_d_norm * _h_norm, 1e-12)
                            ).mean()
                        else:
                            _est["est_wdisp_norm"] = jnp.zeros(())
                            _est["est_wdisp_cos"] = jnp.zeros(())
                    else:
                        _est = {
                            k: jnp.zeros(())
                            for k in (
                                "est_M",
                                "est_h_norm", "est_a_norm", "est_cos", "est_rel_l2",
                                "est_rel_l2_sq",
                                "est_var_proxy", "est_bias2_proxy", "est_nonfinite",
                                "est_wdisp_norm", "est_wdisp_cos",
                            )
                        }

                    # ---- faithful-repair diagnostics -------------------------
                    # Computed, never zero-placeheld. When disabled every field is
                    # NaN so a reader cannot mistake a placeholder for a measurement.
                    _nan = jnp.float32(jnp.nan)
                    if cfg.log_faithful_diag and not cfg.reverse_kl:
                        _kl_d = jax.lax.stop_gradient(kl)
                        _gate_open = (_kl_d < cfg.kl_bound).astype(jnp.float32)
                        _q = jnp.quantile(
                            _kl_d,
                            jnp.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]),
                        )
                        _lag_raw = actor_model.lagrangian_log_param.value.squeeze()
                        _kl_an = jax.lax.stop_gradient(kl_analytic)
                        _fd = dict(
                            fr_gate_operator=_gate_open.mean(),
                            fr_gate_kl_only=1.0 - _gate_open.mean(),
                            fr_kl_q10=_q[0], fr_kl_q25=_q[1], fr_kl_q50=_q[2],
                            fr_kl_q75=_q[3], fr_kl_q90=_q[4], fr_kl_q95=_q[5],
                            fr_kl_q99=_q[6],
                            fr_kl_min=_kl_d.min(), fr_kl_max=_kl_d.max(),
                            fr_kl_analytic_med=jnp.nanmedian(_kl_an),
                            fr_kl_sampled_minus_analytic_med=jnp.nanmedian(
                                _kl_d - _kl_an
                            ),
                            fr_kl_sampled_minus_analytic_mean=jnp.nanmean(
                                _kl_d - _kl_an
                            ),
                            fr_lag_raw=_lag_raw,
                            fr_lag_eff=jax.lax.stop_gradient(lagrangian).squeeze(),
                            fr_lag_finite=jnp.isfinite(
                                jax.lax.stop_gradient(lagrangian)
                            ).all().astype(jnp.float32),
                            fr_action_sat=(
                                jnp.abs(jax.lax.stop_gradient(old_pi_action))
                                > 1.0 - 1e-4
                            ).mean().astype(jnp.float32),
                            fr_sigma_mean=pi_sigma.mean(),
                            fr_sigma_min=pi_sigma.min(),
                            fr_sigma_max=pi_sigma.max(),
                        )
                    else:
                        _fd = dict(
                            (k, _nan)
                            for k in (
                                "fr_gate_operator", "fr_gate_kl_only",
                                "fr_kl_q10", "fr_kl_q25", "fr_kl_q50", "fr_kl_q75",
                                "fr_kl_q90", "fr_kl_q95", "fr_kl_q99",
                                "fr_kl_min", "fr_kl_max",
                                "fr_kl_analytic_med",
                                "fr_kl_sampled_minus_analytic_med",
                                "fr_kl_sampled_minus_analytic_mean",
                                "fr_lag_raw", "fr_lag_eff", "fr_lag_finite",
                                "fr_action_sat", "fr_sigma_mean", "fr_sigma_min",
                                "fr_sigma_max",
                            )
                        )

                    return loss, dict(
                        **_fd,
                        actor_loss=actor_loss,
                        loss=loss,
                        temp=actor_model.temperature(),
                        abs_batch_action=jnp.abs(minibatch.action).mean(),
                        abs_pred_action=jnp.abs(pred_action).mean(),
                        reward_mean=minibatch.reward.mean(),
                        kl=kl.mean(),
                        lagrangian=lagrangian,
                        lagrangian_loss=lagrangian_loss,
                        entropy=entropy,
                        entropy_loss=target_entropy_loss,
                        target_values=target_values.mean(),
                        pi_sigma_mean=pi_sigma.mean(),
                        pi_sigma_min=pi_sigma.min(),
                        pi_sigma_max=pi_sigma.max(),
                        ess=ess.mean(),
                        # ESS DISTRIBUTION: the mean hides a population of states
                        # running at ESS ~2, which by the shrinkage law lose ~50% of
                        # their covariance per update. Percentiles are taken per
                        # minibatch and then averaged across minibatches.
                        ess_p5=jnp.percentile(ess, 5),
                        ess_p25=jnp.percentile(ess, 25),
                        ess_median=jnp.percentile(ess, 50),
                        ess_p75=jnp.percentile(ess, 75),
                        ess_frac_lt4=jnp.mean((ess < 4.0).astype(jnp.float32)),
                        w_max=w_max.mean(),
                        q_spread=q_spread.mean(),
                        eta=eta,
                        eta_loss=eta_loss,
                        kl_mu=kl_mu.mean(),
                        kl_sigma=kl_sigma.mean(),
                        beta_mu=beta_mu,
                        beta_sigma=beta_sigma,
                        beta_mu_pinned=(beta_mu >= 1000.0 * (1 - 1e-6)).astype(
                            jnp.float32
                        ),
                        beta_sigma_pinned=(beta_sigma >= 1000.0 * (1 - 1e-6)).astype(
                            jnp.float32
                        ),
                        **_est,
                    )

                critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
                output, grads = critic_grad_fn(train_state.critic.params)
                critic_train_state = train_state.critic.apply_gradients(grads)
                train_state = train_state.replace(
                    critic=critic_train_state,
                )
                critic_metrics = output[1]
                # Gated for the same reason as log_q_spread: an extra reduction over
                # the gradient tree perturbs XLA fusion and hence bit-identity.
                if cfg.log_estimator_diag:
                    critic_grad_norm = optax.global_norm(grads)

                actor_grad_fn = jax.value_and_grad(actor_loss, has_aux=True)
                output, grads = actor_grad_fn(train_state.actor.params)
                actor_train_state = train_state.actor.apply_gradients(grads)
                train_state = train_state.replace(
                    actor=actor_train_state,
                )
                actor_metrics = output[1]
                if cfg.log_estimator_diag:
                    grad_metrics = dict(
                        grad_norm_actor=optax.global_norm(grads),
                        grad_norm_critic=critic_grad_norm,
                    )
                else:
                    grad_metrics = dict(
                        grad_norm_actor=jnp.zeros(()),
                        grad_norm_critic=jnp.zeros(()),
                    )
                return (idx + 1, train_state), {
                    **critic_metrics,
                    **actor_metrics,
                    **grad_metrics,
                }

            # Shuffle data and split into mini-batches
            key, shuffle_key = jax.random.split(key)
            mini_batch_size = (cfg.num_steps * cfg.num_envs) // cfg.num_mini_batches
            indices = jax.random.permutation(shuffle_key, cfg.num_steps * cfg.num_envs)
            minibatch_idxs = jax.tree.map(
                lambda x: x.reshape(
                    (cfg.num_mini_batches, mini_batch_size, *x.shape[1:])
                ),
                indices,
            )

            # Run model update for each mini-batch
            train_state, metrics = jax.lax.scan(
                minibatch_update, train_state, minibatch_idxs
            )
            # Compute mean metrics across mini-batches
            metrics = jax.tree.map(lambda x: x.mean(0), metrics)
            return train_state, metrics

        # Update the model for a number of epochs
        key, train_key = jax.random.split(key)
        (_, train_state), update_metrics = jax.lax.scan(
            f=update,
            init=(1, train_state),
            xs=jax.random.split(train_key, cfg.num_epochs),
        )
        # Get metrics from the last epoch
        update_metrics = jax.tree.map(lambda x: x[-1], update_metrics)

        return train_state, update_metrics

    def train_fn(key: PRNGKey, cfg: ReppoConfig) -> tuple[SACTrainState, dict]:
        def train_eval_step(key, train_state):
            def train_step(
                state: SACTrainState, key: PRNGKey
            ) -> tuple[SACTrainState, dict[str, jax.Array]]:
                key, rollout_key, learn_key = jax.random.split(key, 3)
                transitions, state = collect_rollout(key=rollout_key, train_state=state)
                state, update_metrics = learn_step(
                    key=learn_key, train_state=state, batch=transitions
                )
                metrics = {**update_metrics, **update_metrics}
                state = state.replace(iteration=state.iteration + 1)
                return state, metrics

            train_key, eval_key = jax.random.split(key)
            eval_interval = int(
                (cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval
            )
            train_state, train_metrics = jax.lax.scan(
                f=train_step,
                init=train_state,
                xs=jax.random.split(train_key, eval_interval),
            )
            train_metrics = jax.tree.map(lambda x: x[-1], train_metrics)
            policy = make_policy(train_state)
            if cfg.normalize_env:
                norm_state = train_state.last_env_state
            else:
                norm_state = None
            eval_metrics = eval_fn(eval_key, policy, norm_state)
            train_returns = {
                "train/episode_return": train_state.last_env_state.info[
                    "returned_episode_returns"
                ].mean(),
                "train/episode_length": train_state.last_env_state.info[
                    "returned_episode_lengths"
                ].mean(),
            }
            metrics = {
                "time_step": train_state.time_steps,
                **utils.prefix_dict("train", train_metrics),
                **utils.prefix_dict("eval", eval_metrics),
                **train_returns,
            }
            return train_state, metrics

        def loop_body(
            train_state: SACTrainState, key: PRNGKey
        ) -> tuple[SACTrainState, dict]:
            key, subkey = jax.random.split(key)
            train_state, metrics = jax.vmap(train_eval_step)(
                jax.random.split(subkey, num_seeds), train_state
            )
            jax.debug.callback(log_callback, train_state, metrics)
            if return_snapshots:
                # Everything needed to export a standalone checkpoint at this
                # iteration: params plus the frozen obs-normalizer statistics.
                snap = dict(
                    actor=train_state.actor.params,
                    critic=train_state.critic.params,
                    mean=train_state.last_env_state.mean,
                    var=train_state.last_env_state.var,
                    critic_mean=train_state.last_env_state.critic_mean,
                    critic_var=train_state.last_env_state.critic_var,
                    count=train_state.last_env_state.count,
                    time_steps=train_state.time_steps,
                )
                return train_state, (metrics, snap)
            return train_state, metrics

        eval_interval = int(
            (cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval
        )
        num_train_steps = cfg.total_time_steps // (cfg.num_steps * cfg.num_envs)
        num_iterations = num_train_steps // eval_interval + int(
            num_train_steps % eval_interval != 0
        )
        key, init_key = jax.random.split(key)
        train_state = jax.vmap(make_init(cfg, env, env_params))(
            jax.random.split(init_key, num_seeds)
        )
        keys = jax.random.split(key, num_iterations)
        if return_snapshots:
            state, (metrics, snaps) = jax.lax.scan(
                f=loop_body, init=train_state, xs=keys
            )
            return state, metrics, snaps
        state, metrics = jax.lax.scan(f=loop_body, init=train_state, xs=keys)
        return state, metrics

    return train_fn


def plot_history(history: list[dict[str, jax.Array]]):
    steps = jnp.array([m["time_step"][0] for m in history])
    eval_return = jnp.array([m["eval/episode_return"].mean() for m in history])
    eval_return_std = jnp.array([m["eval/episode_return"].std() for m in history])
    fig = go.Figure(
        [
            go.Scatter(
                x=steps,
                y=eval_return,
                name="Mean Episode Return",
                mode="lines",
                line=dict(color="blue"),
                showlegend=False,
            ),
            go.Scatter(
                x=steps,
                y=eval_return + eval_return_std,
                name="Upper Bound",
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            ),
            go.Scatter(
                x=steps,
                y=eval_return - eval_return_std,
                name="Lower Bound",
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(50, 127, 168, 0.3)",
                showlegend=False,
            ),
        ]
    )
    fig.update_layout(
        xaxis=dict(title=dict(text="Environment Steps")),
    )

    return fig


# type object
def _get_optuna_type(trial: optuna.Trial, name, values: list):
    if all(isinstance(v, int) for v in values):
        return trial.suggest_int(name, low=min(values), high=max(values))
    elif all(isinstance(v, float) for v in values):
        return trial.suggest_float(name, low=min(values), high=max(values))
    elif all(isinstance(v, str) for v in values):
        return trial.suggest_categorical(name, values)
    elif all(isinstance(v, bool) for v in values):
        return trial.suggest_categorical(name, [True, False])
    else:
        raise ValueError("Values must be of the same type (int, float, or str).")


def run(cfg: DictConfig, trial: optuna.Trial | None) -> float:
    """
    Run a single trial of the SAC training process with hyperparameter tuning.
    Args:
        cfg (DictConfig): Configuration for the SAC training.
        trial (optuna.Trial | None): Optuna trial object for hyperparameter tuning.
    Returns:
        float: The mean episode return from the trial.
    """
    sweep_metrics = []

    if trial is not None:
        # Set hyperparameters from the trial
        for name, values in cfg.trial_spec.items():
            if name in cfg.hyperparameters:
                sampled_value = _get_optuna_type(trial, name, values)
                # TODO: Why the fuck is this happening
                if isinstance(sampled_value, np.float64):
                    sampled_value = float(sampled_value)
                cfg.hyperparameters[name] = sampled_value
            else:
                raise ValueError(f"Hyperparameter {name} not found in config.")

    try:
        with open("completed_trials.txt", "r") as f:
            completed_trials = int(f.read())
    except FileNotFoundError:
        completed_trials = 0

    metric_history = []

    def log_callback(state, metrics):
        metrics["sys_time"] = time.perf_counter()
        if len(metric_history) > 0:
            num_env_steps = state.time_steps[0] - metric_history[-1]["time_step"][0]
            seconds = metrics["sys_time"] - metric_history[-1]["sys_time"]
            sps = num_env_steps / seconds
        else:
            sps = 0

        metric_history.append(metrics)
        episode_return = metrics["eval/episode_return"].mean()
        eval_length = metrics["eval/episode_length"].mean()
        logging.info(
            f"step={state.time_steps[0]} episode_return={episode_return:.3f}, episode_length={eval_length:.3f} sps={sps:.2f}"
        )
        log_data = {
            "eval/episode_return": episode_return,
            "eval/episode_length": eval_length,
            **jax.tree.map(jnp.mean, utils.filter_prefix("train", metrics)),
        }
        wandb.log(log_data, step=state.time_steps[0])

    # Set up the experiment
    if cfg.env.type == "brax":
        env = BraxGymnaxWrapper(
            cfg.env.name,
            episode_length=cfg.env.max_episode_steps,
            reward_scaling=cfg.env.reward_scaling,
            terminate=cfg.env.terminate,
        )
    elif cfg.env.type == "mjx":
        env = MjxGymnaxWrapper(
            cfg.env.name,
            episode_length=cfg.env.max_episode_steps,
            reward_scale=cfg.env.reward_scaling,
            push_distractions=cfg.env.get("push_distractions", False),
            asymmetric_observation=cfg.env.get("asymmetric_obs", False),
        )
    else:
        raise ValueError(f"Unknown environment type: {cfg.env.type}")

    # build algo config with overrides

    train_fn = make_train_fn(
        cfg=ReppoConfig(**cfg.hyperparameters),
        env=env,
        log_callback=log_callback,
        num_seeds=cfg.num_seeds,
        reward_scale=1.0 / cfg.env.reward_scaling,
    )

    for i in range(completed_trials, cfg.num_trials):
        cfg.seed = cfg.seed + i

        wandb.init(
            mode=cfg.wandb.mode,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            tags=[
                cfg.name,
                cfg.env.name,
                cfg.env.type,
                "hp_tune" if trial is not None else "val",
                *cfg.tags,
            ],
            config=OmegaConf.to_container(cfg),
            name=f"{cfg.name}-{cfg.env.name.lower()}",
            save_code=True,
        )

        logging.info(OmegaConf.to_yaml(cfg))

        key = jax.random.PRNGKey(cfg.seed)
        start = time.perf_counter()
        _, metrics = jax.jit(train_fn, static_argnums=(1,))(
            key, ReppoConfig(**cfg.hyperparameters)
        )
        jax.block_until_ready(metrics)
        duration = time.perf_counter() - start

        # Save metrics and finish the run
        logging.info(f"Training took {duration:.2f} seconds.")
        jnp.savez("metrics.npz", **metrics)
        wandb.finish()

        sweep_metrics.append(metrics["eval/episode_return"])

        with open("completed_trials.txt", "w") as f:
            f.write(str(i))

    sweep_metrics_array = jnp.array(sweep_metrics)
    return (0.1 * sweep_metrics_array.mean() + sweep_metrics_array[:, -1].mean()).item()


@hydra.main(version_base=None, config_path="../../config", config_name="reppo")
def main(cfg: DictConfig):
    print(cfg)
    cfg.hyperparameters = OmegaConf.merge(cfg.hyperparameters, cfg.experiment_overrides.hyperparameters)
    run(cfg, trial=None)


if __name__ == "__main__":
    main()
