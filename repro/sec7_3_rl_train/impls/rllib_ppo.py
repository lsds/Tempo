from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import ray
import torch
import torch.nn as nn
from gymnasium import spaces
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

from tempo.core.configs import get_default_path

default_path = get_default_path()

DISABLE_VF_CLIP = 9e6
DISABLE_KL_TARGET = 9e6
DISABLE_KL_COEFF = 0.0


class TrivialEnvGym(gym.Env):
    """
    Because RLlib does not support GPU-accelerated environments, we reimplement the trivial
    environment in CPU gym.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        cfg: Dict = None,
    ):
        ep_len: int = cfg["ep_len"]
        observation_shape = cfg["obs_shape"]
        continuous: bool = cfg["continuous"]
        seed: int = cfg["seed"]
        self.ep_len = ep_len

        self.auto_reset = True
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=observation_shape, dtype=np.float32
        )
        self.obs_shape = observation_shape

        if continuous:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(2)  # Actions: 0 -> decrease, 1 -> increase

        self.continuous = continuous
        self.ep_len = ep_len
        self.seed(seed)

        self.target_state = 0.0  # Target state to reach
        self.state = None  # Will be initialized in reset()
        self.step_count = 0
        self.done = False

    def seed(self, seed=None):
        return super().reset(seed=seed)

    def reset(self, seed: int = None, options=None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.seed(seed)
        self.step_count = 0
        self.done = False

        # Initial state is random between -1 and 1
        self.state = self.np_random.uniform(-1.0, 1.0)
        obs = np.full(self.obs_shape, self.state, dtype=np.float32)
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # if self.done:
        #    raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        if self.continuous:
            change = np.clip(action, -1.0, 1.0).reshape(-1)
        else:
            change = (np.clip(action, 0, 1) * 2 - 1).reshape(-1)
        change = change[0]  # Extract scalar value from array

        # Update state
        self.state += change * 0.05
        self.state = np.clip(self.state, -1.0, 1.0)

        # Compute reward
        reward = -abs(self.state)  # - self.target_state

        # Prepare observation
        obs = np.full(self.obs_shape, self.state, dtype=np.float32)

        # Update step count and check if episode is done
        self.step_count += 1
        terminated = self.step_count >= self.ep_len  # No early termination condition
        truncated = False
        # self.done = terminated or truncated

        if self.auto_reset and (terminated or truncated):
            obs, _ = self.reset()

        return obs, reward, terminated, truncated, {}

    def render(self, mode: str = "human") -> None:
        pass  # No rendering implemented

    def close(self):
        pass  # No resources to clean up


# Define layer initialization function
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# Custom Model matching your Agent class
class CustomPPOModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        num_layers,
        num_params_per_layer,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # assert num_layers == 2
        # assert num_params_per_layer == 64

        self.num_outputs = num_outputs

        # Shared backbone with an encoder layer
        layers = [
            layer_init(
                nn.Linear(
                    np.prod(obs_space.shape),
                    num_params_per_layer,
                )
            ),
            nn.Tanh(),
        ]

        # Add specified number of hidden layers
        for _ in range(num_layers):
            layers.append(layer_init(nn.Linear(num_params_per_layer, num_params_per_layer)))
            layers.append(nn.Tanh())

        # Define the backbone
        self.backbone = nn.Sequential(*layers)

        # Define critic using the shared backbone output
        self.critic = nn.Sequential(layer_init(nn.Linear(num_params_per_layer, 1), std=1.0))

        # Define actor mean using the shared backbone output
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(num_params_per_layer, action_space.shape[0]),
                std=0.01,
            ),
        )

        # Log standard deviation parameter for the actor
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_space.shape[0]))

        self._value = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        x = x.view(x.size(0), -1)  # Flatten the observation
        x = self.backbone(x)
        self._value_out = self.critic(x).squeeze(-1)

        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        # Concatenate mean and log_std for the distribution
        self._logits = torch.cat([action_mean, action_logstd], dim=1)
        return self._logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._value_out


def get_rllib_ppo_execute_fn(
    wandb_run: Any,
    dev: str = "cpu",
    results_path: str = default_path,
    env_name: str = "Pendulum-v1",  # Use a continuous action space env
    num_envs: int = 100,
    ep_len: int = 1000,
    seed: int = 0,
    params_per_layer: int = 64,
    num_layers: int = 2,
    iterations: int = 5,
    gamma: float = 0.99,
    start_lr: float = 1e-3,
    lambda_: float = 0.96,
    ent_coef: float = 0.01,
    vf_coef: float = 0.01,
    obs_shape: Tuple[int, ...] = (3, 256, 256),
    ## RL-games-specific parameters
    minibatch_size: int = 32,
    **kwargs,
):
    assert (num_envs * ep_len) % minibatch_size == 0
    # Give it all the cpus and ram, and 1 GPU
    ray.init(num_gpus=1, num_cpus=32)

    # We set to 32 because our machine has 24 cores/48 threads
    num_rollout_workers = 32

    # Register the custom model
    ModelCatalog.register_custom_model("custom_ppo_model", CustomPPOModel)

    assert env_name == "trivial.trivial", f"Only 'trivial.trivial' is supported, not {env_name}"

    config = (
        PPOConfig()
        .training(
            gamma=gamma,
            lr=start_lr,
            train_batch_size=num_envs * ep_len,
            sgd_minibatch_size=num_envs * ep_len,
            num_sgd_iter=1,
            vf_clip_param=DISABLE_VF_CLIP,  # 10.0
            kl_coeff=DISABLE_KL_COEFF,
            kl_target=DISABLE_KL_TARGET,
            clip_param=1_000_000,
            optimizer=dict(
                type="adam",
                lr=start_lr,
                eps=1e-8,
                weight_decay=0.0,
            ),
            lambda_=lambda_,
            entropy_coeff=ent_coef,
            vf_loss_coeff=vf_coef,
            use_gae=True,
            use_critic=True,
            # sample_async=False,
            grad_clip=None,  # Match your code's lack of grad clipping
            model={
                "vf_share_layers": True,
                "custom_model": "custom_ppo_model",
                "custom_model_config": {
                    "num_layers": num_layers,
                    "num_params_per_layer": params_per_layer,
                },
            },
        )
        .evaluation(evaluation_interval=None)
        .environment(
            env=TrivialEnvGym,
            env_config={
                "obs_shape": obs_shape,
                "ep_len": ep_len,
                "continuous": True,
                "seed": seed,
            },
        )
        .framework(framework="torch")
        .rollouts(
            num_rollout_workers=num_rollout_workers,
            num_envs_per_worker=num_envs // num_rollout_workers,
            # num=1,
            # num_gpus_per_env_runner=0,
            # max_in=1,
            rollout_fragment_length=ep_len,
            batch_mode="complete_episodes",
        )
        .resources(
            num_gpus=1 if (dev == "cuda" or dev == "cpu") else 0,
            num_cpus_per_worker=1,
        )
        .debugging(seed=seed)
    )
    config.max_num_worker_restarts = 5

    # Create trainer
    trainer = config.build()

    def train_fn():
        # Training loop
        for iteration in range(iterations):
            result = trainer.train()

            wandb_run.log(
                {
                    # "l_vf": result["info"]["learner"]["default_policy"]["learner_stats"]["vf_loss"],
                    # "l_pg": result["info"]["learner"]["default_policy"]["learner_stats"][
                    #    "policy_loss"
                    # ],
                    # "l_ent": result["info"]["learner"]["default_policy"]["learner_stats"]["entropy"]
                    # * -ent_coef,
                    "loss": int(
                        result["info"]["learner"]["default_policy"]["learner_stats"]["total_loss"]
                    ),
                    # "mean_episode_return": result["episode_reward_mean"],
                    "mean_episode_return": int(result["env_runners"]["episode_return_mean"]),
                    "iteration": iteration,
                }
            )

        # Cleanup
        trainer.stop()
        ray.shutdown()

    return train_fn
