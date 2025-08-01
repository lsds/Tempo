from typing import Any

from repro.launch_lib import FakeWandBLogger
from tempo.api import rl
from tempo.api.optim.optim import Adam
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.api.rl.env.env import Env
from tempo.api.rl.networks.model_builder import ModelBuilder
from tempo.api.tempo_context_manager import TempoContext
from tempo.core.configs import ExecutionConfig, get_default_path
from tempo.runtime.executor.executor import Executor

default_path = get_default_path()

DEFAULT_EXEC_CFG = ExecutionConfig.default()

""" Reinforce implementation with Tempo. Supports both Monte Carlo and n-step Temporal Difference.
Demonstrates the use and importance of algorithm-specific scheduling.
"""


def get_tempo_rl_train_config(
    **kwargs,
) -> ExecutionConfig:
    sys_cfg = kwargs["sys_cfg"]

    cfg = ExecutionConfig.default()

    cfg.dev = kwargs["dev"]
    cfg.seed = kwargs["seed"]
    cfg.path = kwargs["results_path"]

    # NOTE: These are used for debugging purposes.
    cfg.render_schedule = True
    if kwargs.get("vizualize", False):
        cfg.visualize_pipeline_stages = True
    if kwargs.get("validate", False):
        cfg.validate_pipeline_stages = True

    backend = sys_cfg.split("-")[1]
    cfg.backend = backend

    cfg.enable_swap = True

    # NOTE: Currently, the TD computation is fused very aggressively by Tempo,
    # leading to unschedulable graphs.
    cfg.enable_conservative_grouping = kwargs["objective"] is not None

    return cfg


def get_tempo_reinforce_executor(  # noqa: C901
    wandb_run: Any,
    env_name: str = "gym.CartPole-v1",  # "brax.halfcheetah",
    num_envs: int = 1024,
    ep_len: int = 1000,
    params_per_layer: int = 64,
    num_layers: int = 4,
    iterations: int = 20,
    gamma: float = 0.99,
    start_lr: float = 1e-3,
    # NOTE: None means Monte Carlo, and other values are n-step Temporal Difference
    **kwargs,
) -> Executor:
    cfg = get_tempo_rl_train_config(**kwargs)
    objective = kwargs["objective"]

    ctx = TempoContext(cfg, num_dims=3)

    b, i, t = ctx.variables
    B, I, T = ctx.upper_bounds

    with ctx:
        obs_shape = kwargs.get("obs_shape", (3, 4, 4))
        env = Env.make_env(
            env_name,
            max_episode_steps=ep_len,
            observation_shape=obs_shape,
            continuous=True,
        )
        with ctx.tag_region("network_def"):
            net = ModelBuilder.from_env(
                hidden=[params_per_layer] * num_layers, domain=(i,), env=env, actor_critic=False
            )

        with ctx.tag_region("optim_def"):
            optimizer = Adam(
                net.parameters,
                net.buffers,
                start_lr,
            )

        with ctx.tag_region("acting"):
            o = RecurrentTensor.placeholder(
                env.observation_shape, env.observation_dtype, domain=(b, i, t)
            )
            o[b, i, 0] = env.reset(domain=(b, i))

            with ctx.tag_region("fwd"):
                action, _ = net(o.unsqueeze())
                o[b, i, t + 1], r, term, trunc = env.step(action)

        # NOTE: This is where we do the "one-line-change" to the algorithm
        with ctx.tag_region("returns"):
            if objective is not None:
                g = rl.n_step_returns(r, t, T, objective, gamma)
            else:
                g = r[b, i, 0:T].discounted_cum_sum(gamma)

        with ctx.tag_region("losses"):
            with ctx.tag_region("log_prob"):
                log_pr = net.log_prob(action)
            l_pg = (-log_pr) * g.detach()

            l_pg_avg = l_pg[0:B, i, 0:T].mean()
            loss = l_pg_avg

        with ctx.tag_region("backward"):
            loss.backward()

        mean_ep_ret = r[0:B, i, 0:T].sum(1).mean(0)  # type: ignore

        RecurrentTensor.sink_many_with_ts_udf(
            [mean_ep_ret, loss],
            lambda xs, ts: wandb_run.log(
                {
                    "iteration": ts[i],
                    "mean_episode_return": xs[0].item(),
                    "loss": xs[1].item(),
                }
            ),
        )

        with ctx.tag_region("optim_step"):
            optimizer.step()

        return ctx.compile(bounds={B: num_envs, I: iterations, T: ep_len})


if __name__ == "__main__":
    params = {
        "env_name": "trivial.trivial",
        # NOTE: Default obs shape for trivial env
        "obs_shape": (3, 256, 256),
        "seed": 0,
        "dev": "fake-gpu",
        "iterations": 50,
        # PPO hyperparams
        "gamma": 0.99,
        "start_lr": 1e-4,
        "lambda_": 0.96,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        # NOTE: Fixed param base used in large obs experiments (overwritten in small_to_med_scale)
        "num_envs": 256,
        "ep_len": 1000,
        "params_per_layer": 64,
        "num_layers": 2,
        "sys_cfg": "tempo-jax",
        "results_path": "./results/minimal_test_reinforce",
        "vizualize": True,
        "objective": None,
    }

    exe = get_tempo_reinforce_executor(
        wandb_run=FakeWandBLogger("./results/minimal_test_reinforce/tempo_reinforce.csv"),
        **params,
    )

    # exe.execute()
