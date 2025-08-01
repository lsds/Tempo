import time
from typing import Dict, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces


class TrivialEnv(gym.vector.VectorEnv):
    def __init__(
        self,
        num_envs: int = 1,
        max_ep_len: int = 100,
        observation_shape: Tuple[int, ...] = (3, 256, 256),
        continuous: bool = True,
        dev: str = "gpu",
        auto_reset: bool = False,
        seed: int = 0,
    ):
        single_observation_space = spaces.Box(
            low=-1, high=1, shape=observation_shape, dtype=np.float32
        )
        batch_observation_space = spaces.Box(
            low=-1, high=1, shape=(num_envs, *observation_shape), dtype=np.float32
        )
        self.obs_shape = observation_shape
        if continuous:
            single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            batch_action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(num_envs, 1), dtype=np.float32
            )
        else:
            single_action_space = spaces.Discrete(2)  # Actions: 0 -> decrease, 1 -> increase
            batch_action_space = spaces.MultiDiscrete([2] * num_envs)

        # self.single_action_space = single_action_space
        # self.single_observation_space = single_observation_space
        # self.action_space = batch_action_space
        # self.observation_space = batch_observation_space
        self.num_envs = num_envs
        self.single_action_space = single_action_space
        self.single_observation_space = single_observation_space
        self.action_space = batch_action_space
        self.observation_space = batch_observation_space
        self.reward_range = (-1, 0)

        super(TrivialEnv, self).__init__(
            num_envs, self.single_observation_space, self.single_action_space
        )

        self.continuous = continuous
        self.max_ep_len = max_ep_len
        self.dev = jax.devices(dev)[0] if isinstance(dev, str) else dev

        self.target_state = jnp.full((self.num_envs,), 0.0, device=self.dev)
        self.state = self.target_state.copy()

        self.step_count = 0

        self.not_done = jnp.full((self.num_envs,), False, device=self.dev)
        self.done = jnp.full((self.num_envs,), True, device=self.dev)
        self.auto_reset = auto_reset

        seed = int(time.time())
        self.random_key = jax.random.PRNGKey(seed)

        def _step(
            state: jnp.ndarray, action: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            if self.continuous:
                change = jnp.clip(action, -1.0, 1.0).reshape((-1,))
            else:
                change = (jnp.clip(action, 0, 1) * 2 - 1).reshape((-1,))

            # New code
            state += change * 0.05
            state = jnp.clip(state, -1, 1)
            reward = -(jnp.abs(state))  # - self.target_state
            # reward = jnp.zeros_like(state)

            # Expand state to observation shape with batch dim
            obs = jnp.broadcast_to(
                jnp.expand_dims(state, list(range(1, len(self.obs_shape) + 1))),
                (*state.shape, *self.obs_shape),
            )

            return state, obs, reward  # , done

        self._step = jax.jit(_step, device=self.dev, donate_argnums=(0,))

    def reset(self, seed: int = 0) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        # if seed:
        #    self.random_key = jax.random.PRNGKey(seed)

        self.step_count = 0
        # Initial state is random between -1 and 1
        self.random_key, subkey = jax.random.split(self.random_key)
        self.state = jax.random.uniform(subkey, self.state.shape, jnp.float32, -1.0, 1.0)
        obs = jnp.broadcast_to(
            jnp.expand_dims(self.state, list(range(1, len(self.obs_shape) + 1))),
            (*self.state.shape, *self.obs_shape),
        )
        return obs, {}

    def step(
        self, action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        self.state, obs, reward = self._step(self.state, action)
        self.step_count += 1

        # if self.auto_reset and self.step_count >= self.max_ep_len:
        #    self.reset()

        done = self.not_done if self.step_count < self.max_ep_len else self.done

        if self.auto_reset and self.step_count >= self.max_ep_len:
            self.reset()

        return obs, reward, done, done, {}

    def render(self, mode: str = "human") -> None:
        pass
