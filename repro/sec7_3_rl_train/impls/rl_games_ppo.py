from functools import partial
from typing import Any

import fire
import gym
import numpy as np
import torch
import torch.utils.dlpack as tpack
import yaml
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common.algo_observer import AlgoObserver
from rl_games.common.env_configurations import register as register_env_cfg
from rl_games.common.ivecenv import IVecEnv
from rl_games.common.vecenv import register as register_vec_env_cfg
from rl_games.torch_runner import Runner

from tempo.core.configs import get_default_path
from tempo.utils.cprofiler import Profiler
from tempo.utils.resource_monitor import ResourceMonitorManager
from tempo.utils.torch_profiler import TorchProfiler


def jax_to_torch(tensor):
    from jax._src.dlpack import to_dlpack

    tensor = to_dlpack(tensor)
    tensor = tpack.from_dlpack(tensor)
    return tensor


def torch_to_jax(tensor):
    from jax._src.dlpack import from_dlpack

    tensor = tpack.to_dlpack(tensor)
    tensor = from_dlpack(tensor)
    return tensor


class BraxEnv(IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        from brax import envs

        self.batch_size = num_actors
        ep_len = kwargs.pop("ep_len", 200)
        env_name = kwargs.pop("env_name", "ant")

        env = envs.create(
            env_name=env_name,
            batch_size=self.batch_size,
            episode_length=ep_len,
            # seed = 0,
            # backend = 'gpu',
            auto_reset=True,
        )
        from brax.envs.wrappers import gym as gym_wrapper

        self.env = gym_wrapper.VectorGymWrapper(env, backend=kwargs.pop("dev", "gpu"))

        obs_high = np.inf * np.ones(self.env._env.unwrapped.observation_size)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)

        action_high = np.ones(self.env._env.unwrapped.action_size)
        self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)

    def step(self, action):
        action = torch_to_jax(action)
        next_obs, reward, is_done, info = self.env.step(action)
        next_obs = jax_to_torch(next_obs)
        reward = jax_to_torch(reward)
        is_done = jax_to_torch(is_done)
        return next_obs, reward, is_done, info

    def reset(self):
        obs = self.env.reset()
        return jax_to_torch(obs)

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {}
        info["action_space"] = self.action_space
        info["observation_space"] = self.observation_space
        return info


class TrivialIVecEnv(IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        from tempo.api.rl.env.trivial_env import TrivialEnv

        self.batch_size = num_actors
        ep_len = kwargs.pop("ep_len", 200)

        self.env = TrivialEnv(
            num_envs=self.batch_size,
            max_ep_len=ep_len,
            # seed = 0,
            dev=kwargs.pop("dev", "gpu"),
            continuous=True,
            observation_shape=kwargs.pop("obs_shape", (3, 256, 256)),
            auto_reset=True,
        )
        o = self.env.single_observation_space
        a = self.env.single_action_space
        self.observation_space = gym.spaces.Box(high=o.high, low=o.low, shape=o.shape)
        self.action_space = gym.spaces.Box(high=a.high, low=a.low, shape=a.shape)

    def step(self, action):
        action = torch_to_jax(action)
        next_obs, reward, is_done, trunc, info = self.env.step(action)
        next_obs = jax_to_torch(next_obs)
        reward = jax_to_torch(reward)
        is_done = jax_to_torch(is_done)
        return next_obs, reward, is_done, info

    def reset(self):
        obs, _ = self.env.reset()
        return jax_to_torch(obs)

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {}
        info["action_space"] = self.action_space
        info["observation_space"] = self.observation_space
        return info


register_vec_env_cfg(
    "BRAX",
    lambda config_name, num_actors, **kwargs: BraxEnv(config_name, num_actors, **kwargs),
)

register_vec_env_cfg(
    "TRIVIAL",
    lambda config_name, num_actors, **kwargs: TrivialIVecEnv(config_name, num_actors, **kwargs),
)


def create_trivial_env(**kwargs):
    return TrivialIVecEnv("", kwargs.pop("num_actors", 256), **kwargs)


register_env_cfg(
    "trivial",
    {
        "env_creator": lambda **kwargs: create_trivial_env(**kwargs),
        "vecenv_type": "TRIVIAL",
    },
)

default_path = get_default_path()


class EpochStatsAlgoObserver(AlgoObserver):
    def __init__(
        self,
        stats_logger: Any,
        vf_coef: float = 0.01,
        ent_coef: float = 0.01,
        ep_len: int = 200,
        num_envs: int = 1024,
    ):
        self.run = stats_logger
        self.algo: A2CAgent = None
        self.rewards = None
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.ep_len = ep_len
        self.num_envs = num_envs
        self.calls = 0

    def before_init(self, base_name, config, experiment_name): ...

    def after_init(self, algo: A2CAgent):
        self.algo = algo

    def process_infos(self, infos, done_indices):
        # all_done_indices = self.algo.dones.nonzero(as_tuple=False)
        # env_done_indices = all_done_indices[::self.algo.num_agents]

        # self.algo.current_rewards[env_done_indices]
        ...

    def after_steps(self): ...

    def after_print_stats(self, frame, epoch_num, total_time):
        """Called once per epoch"""

        iteration = int(epoch_num) - 1
        assert iteration == self.calls
        self.calls += 1

        mean_returns = round(
            self.algo.game_rewards.get_mean().item(),
            2,
        )
        # self.algo.game_rewards.clear()

        # steps_elapsed = frame - self.last_frame
        # mean_lengths = int(self.algo.game_lengths.get_mean().item())

        pg_loss, vf_loss, entropy = (
            -self.algo.train_result[0],
            0.5 * self.algo.train_result[1] * self.vf_coef,
            -self.algo.train_result[2] * self.ent_coef,
        )
        l = -float(pg_loss) + float(vf_loss) + float(entropy)
        self.run.log({"mean_episode_return": mean_returns, "iteration": iteration, "loss": l})
        # self.run.log({"l_pg": -float(pg_loss), "iteration": iteration})
        # self.run.log({"l_vf": float(vf_loss), "iteration": iteration})
        # self.run.log({"l_ent": float(entropy), "iteration": iteration})

    def after_clear_stats(self): ...


def get_rlgames_ppo_execute_fn(
    stats_logger: Any,
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
    lambda_: float | None = 0.96,
    ent_coef: float = 0.01,
    vf_coef: float = 0.01,
    obs_shape: tuple[int, ...] = (3, 256, 256),
    ## RL-games-specific parameters
    minibatch_size: int = 32,
    **kwargs,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    experiment_config = locals()
    experiment_config["system"] = "rlgames"

    minibatch_size = num_envs * ep_len

    brax_net = {
        "name": "actor_critic",
        "separate": False,
        "mlp": {
            "activation": "tanh",
            "d2rl": False,
            "initializer": {
                "name": "orthogonal_initializer",
                "gain": np.sqrt(2),
            },
            "regularizer": {"name": "None"},
            # +1 because: https://github.com/Denys88/rl_games/blob/master/rl_games/algos_torch/network_builder.py#L91
            "units": [params_per_layer for _ in range(num_layers + 1)],
        },
        "space": {
            "continuous": {
                "fixed_sigma": True,
                "mu_activation": "None",
                "mu_init": {
                    "name": "orthogonal_initializer",
                    "gain": np.sqrt(2),
                },
                "sigma_activation": "None",
                "sigma_init": {"name": "const_initializer", "val": 0},
            }
        },
    }
    # cule_net_orig = {
    #    "name": "actor_critic",
    #    "separate": False,
    #    "space": {
    #        "discrete": {}
    #    },
    #    "cnn": {
    #        "permute_input": True,
    #        "type": "conv2d",
    #        "activation": "elu",
    #        "initializer": {"name": "default"},
    #        "regularizer": {"name": "None"},
    #        "convs": [
    #            {"filters": 32,
    #             "kernel_size": 8,
    #             "strides": 4,
    #             "padding": 0},
    #            {"filters": 64,
    #             "kernel_size": 4,
    #             "strides": 2,
    #             "padding": 0},
    #            {"filters": 64,
    #             "kernel_size": 3,
    #             "strides": 1,
    #             "padding": 0}
    #        ]
    #    },
    #    "mlp": {
    #        "units": [512],
    #        "activation": "elu",
    #        "d2rl": False,
    #        "initializer": {
    #            "name": "orthogonal_initializer",
    #            "gain": np.sqrt(2),
    #        },
    #        "regularizer": {"name": "None"},
    #    },
    # }
    # cule_net_big = {
    #    "name": "actor_critic",
    #    "separate": False,
    #    "space": {
    #        "discrete": {}
    #    },
    #    "cnn": {
    #        "permute_input": True,
    #        "type": "conv2d",
    #        "activation": "elu",
    #        "initializer": {"name": "default"},
    #        "regularizer": {"name": "None"},
    #        "convs": [
    #            {"filters": 32,
    #             "kernel_size": 12,
    #             "strides": 5,
    #             "padding": 0},
    #            {"filters": 64,
    #             "kernel_size": 7,
    #             "strides": 3,
    #             "padding": 0},
    #            {"filters": 64,
    #             "kernel_size": 3,
    #             "strides": 1,
    #             "padding": 0}
    #        ]
    #    },
    #    "mlp": {
    #        "units": [512],
    #        "activation": "elu",
    #        "d2rl": False,
    #        "initializer": {
    #            "name": "orthogonal_initializer",
    #            "gain": np.sqrt(2),
    #        },
    #        "regularizer": {"name": "None"},
    #    },
    # }

    total_epochs = iterations

    env_group, env_id = env_name.split(".")

    config = {
        "params": {
            "algo": {
                "name": (
                    "a2c_continuous"
                    if (env_group == "brax" or env_group == "trivial")
                    else "a2c_discrete"
                )
            },
            "config": {
                "name": "test",
                "device": dev,
                "clip_value": False,
                "clip_actions": False,
                "critic_coef": vf_coef,
                "entropy_coef": ent_coef,
                "env_config": {
                    "env_name": env_id,
                    "ep_len": ep_len,
                    "dev": "gpu" if dev.startswith("cuda") else "cpu",
                    "obs_shape": obs_shape,
                },
                "env_name": env_group,
                "ppo": False,
                "e_clip": 10000.0,  # 0.2 # ppo=False disables this
                "full_experiment_name": "test",
                "gamma": gamma,
                "grad_norm": None,
                "lr_schedule": "identity",
                "kl_threshold": None,  # NOTE Disabled cause identity lr_scheduler
                "bounds_loss_coef": 0.0,
                "bound_loss_type": "none",
                "learning_rate": start_lr,
                "mixed_precision": False,
                "multi_gpu": False,
                "normalize_input": False,
                "normalize_advantage": True,
                "normalize_value": False,
                "normalize_rms_advantage": False,
                "games_to_track": num_envs,  # * ep_len??
                "horizon_length": ep_len,
                "value_bootstrap": True,
                "max_epochs": total_epochs,
                "mini_epochs": 1,  # NOTE ppo iterations
                "minibatch_size": minibatch_size,
                "weight_decay": 0.0,
                "num_actors": num_envs,
                "reward_shaper": {
                    "scale_value": 1.0,
                    "shift_value": 0.0,
                    "min_val": -np.inf,
                    "max_val": np.inf,
                    "log_val": False,
                    "is_torch": True,
                },  # NOTE: this is not doing anything to reward
                "save_best_after": total_epochs * 1000 * ep_len * num_envs * 10,  # NOTE never save
                "save_frequency": 0,
                # 'score_to_win': 20000,
                "tau": lambda_,
                "truncate_grads": False,  # bool(max_grad_norm)
                "use_smooth_clamp": False,
                "print_stats": False,
            },
            "model": {
                "name": (
                    "continuous_a2c_logstd"
                    if (env_group == "brax" or env_group == "trivial")
                    else "discrete_a2c"
                )
            },
            "network": brax_net,  # brax_net if env_group == "brax" else cule_net_big,
            "seed": seed,
        }
    }

    observer = EpochStatsAlgoObserver(stats_logger, vf_coef, ent_coef, ep_len, num_envs)
    runner = Runner(observer)

    try:
        runner.load(config)
    except yaml.YAMLError as exc:
        print(exc)

    return partial(runner.run_train, args={})


class StatsRun:
    def log(self, x):
        print(x)


def execute_rl_games_ppo(
    dev: str = "cuda:0",
    results_path: str = default_path,
    env_name: str = "trivial.trivial",  # "brax.halfcheetah",
    num_envs: int = 256,
    ep_len: int = 3,
    seed: int = 0,
    params_per_layer: int = 128,
    num_layers: int = 8,
    iterations: int = 10,
    gamma: float = 0.99,
    start_lr: float = 1e-3,
    lambda_: float | None = 0.96,
    ent_coef: float = 0.01,
    vf_coef: float = 0.01,
    torch_profile: bool = False,
    profile: bool = False,
    monitor_resources: bool = False,
    minibatch_size: int = 32,
) -> None:
    assert (num_envs * ep_len) % minibatch_size == 0

    execute_fn = get_rlgames_ppo_execute_fn(
        StatsRun(),  # type: ignore
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
        minibatch_size,
    )
    # create the path
    import os

    os.makedirs(results_path, exist_ok=True)

    with TorchProfiler.get(torch_profile, results_path):
        with Profiler.get(profile, results_path):
            with ResourceMonitorManager.get(monitor_resources, results_path, fps=5):
                execute_fn()


if __name__ == "__main__":
    fire.Fire(execute_rl_games_ppo)
