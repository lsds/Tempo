import random
from functools import partial
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from repro.sec7_3_rl_train.impls.cleanrl_ppo import Agent, make_env
from tempo.core.configs import get_default_path

""" Cleanrl implementation of PPO with caching of activations for backpropagation.
Because of the lack of symbolic autodiff in cleanrl, this leads to slowdowns due to
large backprop graphs.
"""

default_path = get_default_path()


def get_cleanrl_caching_ppo_execute_fn(
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
    lambda_: float = 0.96,
    ent_coef: float = 0.01,
    vf_coef: float = 0.01,
    obs_shape: tuple = (3, 256, 256),
    minibatch_size: int = 100 * 1000,
    **kwargs,
) -> Callable[[], None]:
    assert (num_envs * ep_len) % minibatch_size == 0

    return partial(  # type: ignore
        execute_cleanrl_caching_ppo,
        stats_logger=stats_logger,
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


def execute_cleanrl_caching_ppo(
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

    rewards = torch.zeros((ep_len, num_envs)).to(device)
    dones = torch.zeros((ep_len, num_envs)).to(device)
    values = torch.zeros((ep_len, num_envs)).to(device)
    # entropies = torch.zeros((ep_len, num_envs)).to(device)
    # logprobs = torch.zeros((ep_len, num_envs)).to(device)
    next_done = torch.zeros(num_envs).to(device)

    # TRY NOT TO MODIFY: start the game
    for iteration in range(1, iterations + 1):
        logprobs = []
        entropies = []
        values = []
        next_obs, _ = env.reset(seed=seed)
        for step in range(0, ep_len):
            dones[step] = next_done

            # ALGO LOGIC: action logic
            action, logprob, ent, value = agent.get_action_and_value(next_obs)
            logprobs.append(logprob)
            entropies.append(ent.flatten())
            values.append(value.flatten())
            # logprobs[step] = logprob
            # entropies[step] = ent.flatten()

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
            returns = advantages + torch.stack(values, dim=0)

        # flatten the batch
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        stackedlogprob = torch.stack(logprobs, dim=0).reshape(-1)
        stackedentropies = torch.stack(entropies, dim=0).reshape(-1)
        stackedvalues = torch.stack(values, dim=0).reshape(-1)
        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        np.random.shuffle(b_inds)
        optimizer.zero_grad()
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            # _, newlogprob, entropy, newvalue = agent.get_action_and_value(
            #    b_obs[mb_inds], b_actions[mb_inds]
            # )
            newlogprob = stackedlogprob[mb_inds]
            newvalue = stackedvalues[mb_inds]
            entropy = stackedentropies[mb_inds]

            mb_advantages = b_advantages[mb_inds]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages.detach() * newlogprob
            pg_loss = pg_loss1.mean()

            # Value loss
            newvalue = newvalue.view(-1)
            v_loss = 0.5 * vf_coef * ((newvalue - b_returns[mb_inds].detach()) ** 2).mean()

            entropy_loss = -ent_coef * entropy.mean()
            loss = pg_loss + entropy_loss + v_loss

            loss.backward(retain_graph=True)
        optimizer.step()

        stats_logger.log(
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
