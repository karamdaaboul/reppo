"""mjlab adapter for the REPPO torch trainer.

mjlab's ManagerBasedRlEnv clones IsaacLab's API, so this mirrors
isaaclab_env.py but (a) consumes a PRE-BUILT env handed over by the launcher
(the mjlab task registry lives in a different top-level `src` package that
cannot be imported simultaneously with this repo's `src`), and (b) returns the
5-tuple step the trainer/evaluator actually unpack (the isaaclab wrapper's
4-tuple is stale relative to reppo.py).
"""

import torch

_PREBUILT = {}


def register_prebuilt(name, env):
    _PREBUILT[name] = env


class MjlabEnv:
    def __init__(
        self,
        task_name,
        device,
        num_envs,
        seed,
        reward_scaling=1.0,
        action_bounds=None,
    ):
        if task_name not in _PREBUILT:
            raise RuntimeError(
                f"MjlabEnv expects a pre-built env registered for {task_name!r} "
                "(see run_mjlab_ant.py launcher)."
            )
        self.envs = _PREBUILT.pop(task_name)
        self.device = device
        self.reward_scaling = float(reward_scaling)
        self.action_bounds = action_bounds

        self.num_envs = self.envs.num_envs
        self.max_episode_steps = int(self.envs.max_episode_length)
        self.num_actions = self.envs.action_manager.total_action_dim

        obs_dict, _ = self.envs.reset()
        pol, crit = self._split(obs_dict, probe=True)
        self.num_obs = pol.shape[-1]
        self.asymmetric_obs = crit is not None
        self.num_privileged_obs = crit.shape[-1] if crit is not None else 0

    def _split(self, obs_dict, probe=False):
        pol = obs_dict.get("actor", obs_dict.get("policy"))
        if pol is None:
            raise KeyError(f"no actor/policy obs group; keys={list(obs_dict.keys())}")
        crit = obs_dict.get("critic")
        if not probe and not self.asymmetric_obs:
            crit = None
        return pol, crit

    def reset(self, random_start_init=True):
        obs_dict, _ = self.envs.reset()
        if random_start_init:
            self.envs.episode_length_buf = torch.randint_like(
                self.envs.episode_length_buf, high=self.max_episode_steps
            )
        pol, _ = self._split(obs_dict)
        return pol, {}

    def reset_with_critic_obs(self):
        obs_dict, _ = self.envs.reset()
        pol, crit = self._split(obs_dict)
        return pol, (crit if crit is not None else pol)

    def step(self, actions):
        if self.action_bounds is not None:
            actions = torch.clamp(actions, -1.0, 1.0) * self.action_bounds
        obs_dict, rew, terminated, truncated, info = self.envs.step(actions)
        pol, crit = self._split(obs_dict)
        terminated = terminated.bool().view(-1)
        truncated = truncated.bool().view(-1)
        dones = terminated | truncated
        infos = {
            "time_outs": truncated,
            "observations": {"critic": crit if crit is not None else pol},
        }
        return pol, rew.view(-1) * self.reward_scaling, dones, truncated, infos

    def render(self):
        raise NotImplementedError
