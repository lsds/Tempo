import random
from functools import partial
from typing import Any, Callable, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.dlpack as tpack
from torch.distributions.normal import Normal

from tempo.api.rl.env.wrappers import DoneToTermTruncAPIConverterWrapper
from tempo.core.configs import get_default_path

default_path = get_default_path()


# Minimally modified version of
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
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


# class ToTorchWrapper(gym.vector.VectorWrapper):
class ToTorchWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
    ) -> None:
        super().__init__(env)

    def reset(self, **kwargs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        o, info = self.env.reset(**kwargs)
        o = jax_to_torch(o)
        # info = optree.tree_map(self.to_backend_tensor, info)  # type: ignore
        info = {}
        return o, info

    def step(
        self, action: Any
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict,
    ]:
        action = torch_to_jax(action)
        o, r, term, trunc, info = self.env.step(action)
        o = jax_to_torch(o)
        r = jax_to_torch(r)
        term = jax_to_torch(term)
        trunc = jax_to_torch(trunc)
        # info = optree.tree_map(self.to_backend_tensor, info)  # type: ignore
        info = {}
        return o, r, term, trunc, info


def make_env(
    env_name: str,
    num_envs: int,
    ep_len: int,
    obs_shape: Tuple[int, ...],
    dev: str,
    seed: int = 0,
) -> gym.vector.SyncVectorEnv:
    group, name = env_name.split(".")

    if group == "gym":
        import gym

        env = gym.vector.make(env_name, num_envs, asynchronous=False)
    if group == "brax":
        from brax import envs
        from brax.envs.wrappers import gym as gym_wrapper

        env = envs.create(
            env_name,
            auto_reset=False,
            batch_size=num_envs,
            episode_length=ep_len,
            disable_env_checker=True,
        )

        env = gym_wrapper.VectorGymWrapper(env, backend=dev if dev == "cpu" else "gpu")
        env = DoneToTermTruncAPIConverterWrapper(env, None)  # type: ignore
    if group == "trivial":
        from tempo.api.rl.env.trivial_env import TrivialEnv

        env = TrivialEnv(
            num_envs=num_envs,
            max_ep_len=ep_len,
            observation_shape=obs_shape,
            dev=dev if dev == "cpu" else "gpu",
            # seed=seed
        )
    if group == "brax" or group == "trivial":
        env = ToTorchWrapper(env)
    return env  # type: ignore


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, num_layers: int, num_params_per_layer: int):
        super().__init__()

        # Shared backbone with an encoder layer
        layers = [
            nn.Flatten(),
            layer_init(
                nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(),
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
                nn.Linear(num_params_per_layer, np.prod(envs.single_action_space.shape)),
                std=0.01,
            ),
        )

        # Log standard deviation parameter for the actor
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        x = self.backbone(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = self.backbone(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


def get_cleanrl_ppo_execute_fn(
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
    lambda_: float = 0.96,
    ent_coef: float = 0.01,
    vf_coef: float = 0.01,
    obs_shape: tuple = (3, 256, 256),
    minibatch_size: int = 100 * 1000,
    **kwargs,
) -> Callable[[], None]:
    assert (num_envs * ep_len) % minibatch_size == 0

    return partial(  # type: ignore
        execute_cleanrl_ppo,
        wandb_run=wandb_run,
        dev=dev,
        results_path=results_path,
        env_name=env_name,
        num_envs=num_envs,
        ep_len=ep_len,
        seed=seed,
        params_per_layer=params_per_layer,
        num_layers=num_layers,
        iterations=iterations,
        gamma=gamma,
        start_lr=start_lr,
        lambda_=lambda_,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        obs_shape=obs_shape,
        minibatch_size=minibatch_size,
    )


def execute_cleanrl_ppo(
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
    lambda_: float = 0.96,
    ent_coef: float = 0.01,
    vf_coef: float = 0.01,
    obs_shape: tuple = (3, 256, 256),
    minibatch_size: int = 100 * 1000,
) -> None:
    batch_size = num_envs * ep_len

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(dev)

    # env setup
    env = make_env(env_name, num_envs, ep_len, obs_shape, dev)
    assert isinstance(env.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    agent = Agent(env, num_layers, params_per_layer).to(device)

    print(f"CLEANRN NUM AGENT PARAMETERS: {len(list(agent.parameters()))}")

    optimizer = optim.Adam(agent.parameters(), lr=start_lr)  # , eps=1e-8

    # ALGO Logic: Storage setup
    obs = torch.zeros((ep_len, num_envs) + env.single_observation_space.shape).to(device)
    actions = torch.zeros((ep_len, num_envs) + env.single_action_space.shape).to(device)
    logprobs = torch.zeros((ep_len, num_envs)).to(device)
    rewards = torch.zeros((ep_len, num_envs)).to(device)
    dones = torch.zeros((ep_len, num_envs)).to(device)
    values = torch.zeros((ep_len, num_envs)).to(device)
    next_done = torch.zeros(num_envs).to(device)

    # TRY NOT TO MODIFY: start the game
    for iteration in range(1, iterations + 1):
        next_obs, _ = env.reset(seed=seed)
        for step in range(0, ep_len):
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, _ = env.step(action)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1)  # type: ignore

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(ep_len)):
                if t == ep_len - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * lambda_ * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + env.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        np.random.shuffle(b_inds)
        optimizer.zero_grad()
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                b_obs[mb_inds], b_actions[mb_inds]
            )

            mb_advantages = b_advantages[mb_inds]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * newlogprob
            pg_loss = pg_loss1.mean()

            # Value loss
            newvalue = newvalue.view(-1)
            v_loss = 0.5 * vf_coef * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = -ent_coef * entropy.mean()
            loss = pg_loss + entropy_loss + v_loss

            loss.backward()
        optimizer.step()

        wandb_run.log(
            {
                # "l_vf": v_loss.item(),
                # "l_pg": pg_loss.item(),
                # "l_ent": entropy_loss.item(),
                "loss": loss.item(),
                "mean_episode_return": rewards.sum(0).mean(0).item(),
                "iteration": iteration - 1,
            }
        )

    env.close()
