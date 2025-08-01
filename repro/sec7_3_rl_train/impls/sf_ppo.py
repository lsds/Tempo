from math import prod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import fire
import gymnasium as gym
import numpy as np
import torch
import torch.utils.dlpack as tpack
from gymnasium.core import RenderFrame
from sample_factory.algo.runners.runner import AlgoObserver, Runner
from sample_factory.algo.utils.gymnasium_utils import convert_space
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.envs.env_utils import register_env
from sample_factory.train import make_runner
from sample_factory.utils.typing import Config, Env, PolicyID
from tensorboardX import SummaryWriter
from torch import Tensor

from repro.sec7_3_rl_train.shared import FakeWandBLogger
from tempo.core.configs import get_default_path

default_path = get_default_path()

# Based on https://github.com/alex-petrenko/sample-factory/blob/master/sf_examples/brax/train_brax.py


def jax_to_torch(tensor):
    # noinspection PyProtectedMember
    from jax._src.dlpack import to_dlpack

    # tensor = to_dlpack(jnp.array(tensor))
    tensor = to_dlpack(tensor)
    tensor = tpack.from_dlpack(tensor)
    return tensor


def torch_to_jax(tensor):
    # noinspection PyProtectedMember
    from jax._src.dlpack import from_dlpack

    # tensor = tpack.to_dlpack(torch.tensor(tensor))
    tensor = tpack.to_dlpack(tensor)
    tensor = from_dlpack(tensor)
    return tensor


class BraxEnvConverter(gym.Env):
    # noinspection PyProtectedMember
    def __init__(
        self,
        brax_env,
        num_actors,
        render_mode: Optional[str],
        render_res: int,
        clamp_actions: bool,
        clamp_rew_obs: bool,
    ):
        self.env = brax_env
        self.num_agents = num_actors
        self.env.closed = False
        self.env.viewer = None

        self.renderer = None
        self.render_mode = render_mode
        self.brax_video_res_px = render_res

        self.clamp_actions = False
        self.clamp_rew_obs = False

        if len(self.env.observation_space.shape) > 1:
            observation_size = self.env.observation_space.shape[1]
            action_size = self.env.action_space.shape[1]

            obs_high = np.inf * np.ones(observation_size)
            self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)

            action_high = np.ones(action_size)
            self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)
        else:
            self.observation_space = convert_space(self.env.observation_space)
            self.action_space = convert_space(self.env.action_space)

    def reset(self, *args, **kwargs) -> Tuple[Tensor, Dict]:
        obs = self.env.reset()
        obs = jax_to_torch(obs)
        return obs, {}

    def step(self, action):
        action_clipped = action
        if self.clamp_actions:
            action_clipped = torch.clamp(action, -1, 1)

        action_clipped = torch_to_jax(action_clipped)
        next_obs, reward, terminated, info = self.env.step(action_clipped)
        next_obs = jax_to_torch(next_obs)
        reward = jax_to_torch(reward)
        terminated = jax_to_torch(terminated).to(torch.bool)
        truncated = jax_to_torch(info["truncation"]).to(torch.bool)

        if self.clamp_rew_obs:
            reward = torch.clamp(reward, -100, 100)
            next_obs = torch.clamp(next_obs, -100, 100)

        return next_obs, reward, terminated, truncated, info

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        if self.renderer is None:
            from sf_examples.brax.brax_render import BraxRenderer

            self.renderer = BraxRenderer(self.env, self.render_mode, self.brax_video_res_px)
        return self.renderer.render()


class TrivialEnvConverter(gym.Env):
    # noinspection PyProtectedMember
    def __init__(
        self,
        trivial_env,
        num_actors,
    ):
        self.env = trivial_env
        self.num_agents = num_actors
        self.env.closed = False

        if len(self.env.single_observation_space.shape) > 1:
            observation_size = (int(prod(self.env.single_observation_space.shape)),)
            action_size = (int(prod(self.env.single_action_space.shape)),)

            obs_high = np.inf * np.ones(observation_size)
            self.observation_space = convert_space(
                gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)
            )

            action_high = np.ones(action_size)
            self.action_space = convert_space(
                gym.spaces.Box(-action_high, action_high, dtype=np.float32)
            )

            self.single_observation_space = self.observation_space
            self.single_action_space = self.action_space
        else:
            self.observation_space = convert_space(self.env.single_observation_space)
            self.action_space = convert_space(self.env.single_action_space)

    def reset(self, *args, **kwargs) -> Tuple[Tensor, Dict]:
        obs, info = self.env.reset()
        # NOTE: flatten to prevent SF from using a CNN encoder:
        # https://github.com/alex-petrenko/sample-factory/blob/abbc4591fcfa3f2cb20b98bb9b0f2a1ee83f47fa/sample_factory/model/encoder.py#L44
        obs = jax_to_torch(obs).flatten(1)
        return obs, info

    def step(self, action):
        action_clipped = action
        assert action_clipped.shape == (self.num_agents, 1), (
            f"Action shape {action_clipped.shape} not expected!"
        )

        action_clipped = torch_to_jax(action_clipped)
        next_obs, reward, terminated, truncated, info = self.env.step(action_clipped)
        next_obs = jax_to_torch(next_obs)
        reward = jax_to_torch(reward)
        terminated = jax_to_torch(terminated).to(torch.bool)
        truncated = jax_to_torch(truncated).to(torch.bool)

        next_obs = next_obs.flatten(1)

        # NOTE: flatten to prevent SF from using a CNN encoder:
        # https://github.com/alex-petrenko/sample-factory/blob/abbc4591fcfa3f2cb20b98bb9b0f2a1ee83f47fa/sample_factory/model/encoder.py#L44
        return next_obs, reward, terminated, truncated, info


def make_brax_env(
    full_env_name: str, cfg: Config, _env_config=None, render_mode: Optional[str] = None
) -> Env:
    batch_size = cfg.env_agents
    ep_len = cfg.max_ep_len

    from brax import envs
    from brax.envs.wrappers import gym as gym_wrapper

    brax_env = envs.create(
        env_name=full_env_name,
        episode_length=ep_len,
        batch_size=batch_size,
        auto_reset=True,
    )
    gym_env = gym_wrapper.VectorGymWrapper(brax_env, backend="gpu", seed=0)
    env = BraxEnvConverter(
        gym_env,
        batch_size,
        render_mode,
        cfg.brax_render_res,
        cfg.clamp_actions,
        cfg.clamp_rew_obs,
    )
    return env


def make_trivial_env(
    full_env_name: str, cfg: Config, _env_config=None, render_mode: Optional[str] = None
) -> Env:
    batch_size = cfg.env_agents
    assert batch_size != 1
    ep_len = cfg.max_ep_len
    obs_shape = cfg.obs_shape
    dev = cfg.device

    from tempo.api.rl.env.trivial_env import TrivialEnv

    trivial_env = TrivialEnv(
        num_envs=batch_size,
        max_ep_len=ep_len,
        observation_shape=obs_shape,
        dev=dev,
        continuous=True,
        auto_reset=True,
    )
    env = TrivialEnvConverter(
        trivial_env,
        batch_size,
    )
    return env


def register_brax_custom_components(evaluation: bool = False) -> None:
    for env_name in [
        "reacher",
        "ant",
        "humanoid",
        "halfcheetah",
        "walker2d",
        "humanoidstandup",
    ]:
        register_env(env_name, make_brax_env)


def register_trivial_custom_components(evaluation: bool = False) -> None:
    for env_name in [
        "trivial",
    ]:
        register_env(env_name, make_trivial_env)


class WandbAlgoObserver(AlgoObserver):
    def __init__(self, cfg: Config, runner: Runner, wandb_run: Any):
        self.run = wandb_run

        self.last_loss = 0.0
        self.last_entropy_loss = 0.0
        self.last_value_loss = 0.0
        self.last_policy_loss = 0.0
        self.acc_returns = None
        # self.last_len = 0.

    def on_init(self, runner: Runner) -> None:
        """Called after ctor, but before signal-slots are connected or any processes are started."""

        def train_handler(runner: Runner, msg: Dict, policy_id: PolicyID) -> None:
            m = msg["train"]
            self.last_loss += m["loss"]
            self.last_entropy_loss = m["exploration_loss"]
            self.last_value_loss += m["value_loss"]
            self.last_policy_loss += m["policy_loss"]

        def episodic_handler(runner: Runner, msg: Dict, policy_id: PolicyID) -> None:
            m = msg["episodic"]
            rew = np.array(m["reward"])
            assert rew.ndim == 1
            # if self.acc_returns is None:
            #    self.acc_returns = rew
            # else:
            #    self.acc_returns += rew
            self.acc_returns = rew

        runner.policy_msg_handlers["train"].append(train_handler)
        runner.policy_msg_handlers["episodic"].append(episodic_handler)

    def on_connect_components(self, runner: Runner) -> None:
        """Connect additional signal-slot pairs in the observers if needed."""
        pass

    def on_start(self, runner: Runner) -> None:
        """Called right after sampling/learning processes are started."""
        pass

    def on_training_step(self, runner: Runner, training_iteration_since_resume: int) -> None:
        """Called after each training step."""
        iteration = training_iteration_since_resume - 1
        if iteration > 0:
            iteration -= 1
            self.run.log(
                {
                    "mean_episode_return": np.mean(self.acc_returns),
                    "iteration": iteration,
                    "loss": self.last_loss,
                }
            )
            self.last_loss = 0.0
            self.last_entropy_loss = 0.0
            self.last_value_loss = 0.0
            self.last_policy_loss = 0.0
            self.acc_returns = None

            # self.run.log({"l_pg": self.last_policy_loss, "iteration": iteration})
            # self.run.log({"l_vf": self.last_value_loss, "iteration": iteration})
            # self.run.log({"l_ent": self.last_entropy_loss, "iteration": iteration})

    def extra_summaries(
        self, runner: Runner, policy_id: PolicyID, env_steps: int, writer: SummaryWriter
    ) -> None:
        pass

    def on_stop(self, runner: Runner) -> None:
        pass


def get_sample_factory_ppo_execute_fn(
    wandb_run: Any,
    dev: str = "cpu",
    results_path: str = default_path,
    env_name: str = "gym.CartPole-v1",  # "brax.halfcheetah",
    num_envs: int = 100,
    ep_len: int = 1000,
    seed: int = 0,
    params_per_layer: int = 64,
    num_layers: int = 2,
    iterations: int = 5,
    gamma: float = 0.99,
    start_lr: float = 1e-3,
    lambda_: Optional[float] = 0.96,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    obs_shape: Tuple[int, ...] = (3, 256, 256),
    minibatch_size: int = 32,
    **kwargs,
):
    assert (num_envs * ep_len) % minibatch_size == 0
    if dev != "cpu":
        torch.ones(1, device="cuda")  # init torch cuda before jax
    register_brax_custom_components()
    register_trivial_custom_components()
    from sample_factory.cfg.arguments import cfg_dict, default_cfg

    cfg = default_cfg(algo="APPO", env=env_name.split(".")[1], experiment="sf_ppo")
    cfg = dict(cfg_dict(cfg))
    from sample_factory.utils.attr_dict import AttrDict

    # see https://www.samplefactory.dev/02-configuration/cfg-params/
    cfg.update(
        {
            "algo": "APPO",
            "experiment": "sf_ppo",
            "batched_sampling": True,
            "num_workers": 1,
            "num_envs_per_worker": 1,
            "worker_num_splits": 1,  # NOTE: this controls double buffering,
            "env": env_name.split(".")[1],
            "env_agents": num_envs,
            "obs_shape": obs_shape,
            "num_envs": num_envs,
            "clamp_actions": False,
            "clamp_rew_obs": False,
            "brax_render_res": 200,
            "device": "cpu" if dev == "cpu" else "gpu",
            "seed": seed,
            "optimizer": "adam",
            "adam_eps": 1e-8,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "actor_worker_gpus": [] if dev == "cpu" else [0],
            "train_for_env_steps": (iterations - 1) * ep_len * num_envs,
            "train_for_seconds": 0,
            "use_rnn": False,
            "adaptive_stddev": False,
            "policy_initialization": "orthogonal",
            "env_gpu_actions": dev != "cpu",
            "env_gpu_observations": dev != "cpu",
            "reward_scale": 1.0,  # 0.01,
            "max_grad_norm": 0.0,  # NOTE: 0 disables grad clipping,
            "rollout": ep_len,  # * num_envs
            "batch_size": ep_len * num_envs,
            "num_batches_to_accumulate": 1,  # default 2,
            "num_batches_per_epoch": 1,  # ,
            "num_epochs": 1,  # PPO epochs,
            "ppo_clip_ratio": 1000,  # NOTE: this cannot be disabled,
            "ppo_clip_value": 10000,  # NOTE: this cannot be disabled,
            "value_loss_coeff": 0.5 * vf_coef,  # NOTE: sf does not have the 0.5, so we do it here.
            "exploration_loss": "entropy",
            "exploration_loss_coeff": ent_coef,
            "nonlinearity": "tanh",
            # NOTE: +1 here because of the way the net is created. See:
            # https://github.com/alex-petrenko/sample-factory/blob/abbc4591fcfa3f2cb20b98bb9b0f2a1ee83f47fa/sample_factory/model/encoder.py#L76
            "encoder_mlp_layers": [params_per_layer] * (num_layers + 1),
            "decoder_mlp_layers": [],
            "actor_critic_share_weights": True,
            "learning_rate": start_lr,
            "lr_schedule": "constant",
            "kl_loss_coeff": 0.0,
            "lr_schedule_kl_threshold": 0.008,
            "lr_adaptive_max": 2e-3,
            "shuffle_minibatches": False,
            "gamma": gamma,
            "gae_lambda": lambda_,
            "with_vtrace": False,
            "value_bootstrap": True,
            "normalize_input": False,
            "decorrelate_envs_on_one_worker": False,
            "normalize_returns": True,
            "save_best_every_sec": -1,  # Try to disable checkpoints,
            "save_every_sec": 100000000000,
            "save_best_after": ep_len * num_envs * iterations * 100,
            "save_milestones_sec": -1,
            "keep_checkpoints": 0,
            "serial_mode": True,
            "async_rl": False,
            "experiment_summaries_interval": ep_len * num_envs * iterations * 10000,
            "max_ep_len": ep_len,
            "restart_behaviour": "overwrite",
            "restart_behavior": "overwrite",
            "with_wandb": False,
            "with_pbt": False,
            "initial_stddev": 1.0,
            "continuous_tanh_scale": 0.0,
            "policy_init_gain": 1.0,
            "obs_subtract_mean": False,
            "obs_scale": 1.0,
            "decorrelate_experience_max_seconds": 0,
            "stats_avg": 1,
        }
    )

    cfg = AttrDict(cfg)

    cfg, runner = make_runner(cfg)

    observer = WandbAlgoObserver(cfg, runner, wandb_run)
    runner.register_observer(observer)

    status = runner.init()

    if status == ExperimentStatus.SUCCESS:
        return runner.run
    else:
        raise RuntimeError(f"Runner init failed with status {status}")


def execute(
    dev: str = "cpu",
    results_path: str = default_path,
    env_name: str = "brax.reacher",  # "brax.halfcheetah", #"gym.CartPole-v1"
    num_envs: int = 2048,
    ep_len: int = 200,
    seed: int = 0,
    params_per_layer: int = 64,
    num_layers: int = 4,
    iterations: int = 20,
    gamma: float = 0.95,
    start_lr: float = 5e-2,
    lambda_: float = 0.96,
    ent_coef: float = 0.01,  # 0.01,
    vf_coef: float = 0.5,  # .5, # 0.5,
) -> None:
    wandb_run = FakeWandBLogger(str(Path(results_path) / "sf_ppo.csv"))
    get_sample_factory_ppo_execute_fn(
        wandb_run,
        dev,
        results_path,
        env_name,
        num_envs,
        ep_len,
        seed,
        params_per_layer,
        num_layers,
        iterations,
        gamma,
        start_lr,
        lambda_,
        ent_coef,
        vf_coef,
    )()


if __name__ == "__main__":
    fire.Fire(execute)
