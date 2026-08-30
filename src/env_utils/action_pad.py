"""Redundant-action padding: the policy acts in R^(d+k); the env receives the first d.

The extra k coordinates never reach the simulator and have no effect on dynamics
or reward. They ARE visible to everything above the environment boundary: the actor
emits d+k means/log-stds, the critic consumes the full d+k action, the E-step samples
all d+k coordinates, and the entropy bonus covers all d+k dims. That is the point --
only the action dimension the algorithm sees changes; the task does not.

Sits innermost, directly around ``MjxGymnaxWrapper``, so LogWrapper / ClipAction /
NormalizeVec all see the padded action space and nothing has to know about it.
"""

from __future__ import annotations

import jax.numpy as jnp
from gymnax.environments.spaces import Box
from mujoco_playground._src.wrapper import Wrapper


class ActionPad(Wrapper):
    def __init__(self, env, k: int):
        super().__init__(env)
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        self.k = int(k)
        self.real_dim = int(env.action_space(None).shape[0])

    def action_space(self, params):
        return Box(low=-1.0, high=1.0, shape=(self.real_dim + self.k,))

    def observation_space(self, params):
        return self.env.observation_space(params)

    def reset(self, key):
        return self.env.reset(key)

    def step(self, key, state, action):
        action = jnp.asarray(action)
        assert action.shape[-1] == self.real_dim + self.k, (
            f"expected action dim {self.real_dim + self.k}, got {action.shape}"
        )
        return self.env.step(key, state, action[..., : self.real_dim])
