from typing import Any

from repro.sec7_3_rl_train.shared import (
    FakeWandBLogger,
    is_large_obs,
)
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

    obs_shape = kwargs.get("obs_shape", (3, 4, 4))

    # NOTE: Enforce original submission's behaviour
    if obs_shape[-1] < 64:
        # NOTE: RL fairs better with only point storage, since intermediate activations
        # are fairly small. Thus, doing in-place writes ends up being expensive.
        # Ultimately it is preferable to do a single stack operation on 1000 small tensors,
        # rather than doing 1000 in-place writes to avoid the stack.
        cfg.enable_hybrid_tensorstore = False
        cfg.enable_incrementalization = False

    if is_large_obs(obs_shape):
        # NOTE: Enable swap for large obs experiments.
        cfg.enable_swap = True

    #if backend == "torch":
    #    # This optimization leads to worse performance on PyTorch backend
    #    cfg.enable_inplace_writes = False

    return cfg


def get_tempo_ppo_executor(  # noqa: C901
    wandb_run: Any,
    env_name: str = "gym.CartPole-v1",
    num_envs: int = 1024,
    ep_len: int = 1000,
    params_per_layer: int = 64,
    num_layers: int = 4,
    iterations: int = 20,
    gamma: float = 0.99,
    start_lr: float = 1e-3,
    lambda_: float = 0.96,
    ent_coef: float = 0.01,
    vf_coef: float = 0.01,
    **kwargs,
) -> Executor:
    # Create config from params
    cfg = get_tempo_rl_train_config(**kwargs)

    ctx = TempoContext(cfg, num_dims=3)

    b, i, t = ctx.variables
    B, I, T = ctx.upper_bounds

    obs_shape = kwargs.get("obs_shape", (3, 4, 4))

    # NOTE: The ctx.tag_region context managers are used purely for debugging purposes.
    # They help visualize different regions in a rendered PDG.
    # NOTE: Can be removed if desired.

    with ctx:
        env = Env.make_env(
            env_name,
            max_episode_steps=ep_len,
            observation_shape=obs_shape,
            continuous=True,
        )
        with ctx.tag_region("network_def"):
            net = ModelBuilder.from_env(
                hidden=[params_per_layer] * num_layers, domain=(i,), env=env, actor_critic=True
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
                action, value = net(o.unsqueeze())
                assert value is not None
                action, value = action, value.squeeze(0)
                o[b, i, t + 1], r, term, trunc = env.step(action)

        with ctx.tag_region("gae"):
            advantage = rl.gae(r, value.detach(), t, T, gamma, lambda_)

        advantage_norm = advantage.normalize(
            advantage[0:B, i, 0:T].mean(),
            advantage[0:B, i, 0:T].std(),
            1e-8,
        ).squeeze()

        returns = advantage.detach() + value.detach()

        with ctx.tag_region("losses"):
            with ctx.tag_region("entropy"):
                entropy = net.entropy()
                if len(entropy.domain) > 1:
                    entropy = entropy[0:B, i, 0:T].mean()
            with ctx.tag_region("log_prob"):
                log_pr = net.log_prob(action)
            l_pg = (-log_pr) * advantage_norm.detach()

            l_vf = (value - returns) ** 2
            l_vf = (0.5 * vf_coef) * l_vf

            l_ent = ent_coef * -entropy

            l_pg_avg = l_pg[0:B, i, 0:T].mean()
            l_vf_avg = l_vf[0:B, i, 0:T].mean()
            loss = l_pg_avg + l_ent + l_vf_avg

        with ctx.tag_region("backward"):
            loss.backward()

        mean_ep_ret = r[0:B, i, 0:T].sum(1).mean(0)  # type: ignore

        # NOTE: Can log other metrics if desired, such as commented example
        RecurrentTensor.sink_many_with_ts_udf(
            [mean_ep_ret, loss],  # , l_pg_avg, l_vf_avg, l_ent
            lambda xs, ts: wandb_run.log(
                {
                    "iteration": ts[i],
                    "mean_episode_return": xs[0].item(),
                    "loss": xs[1].item(),
                    # "l_pg": xs[2].item(),
                    # "l_vf": xs[3].item(),
                    # "l_ent": xs[4].item(),
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
        "sys_cfg": "tempo-torch",
        "results_path": "./results/minimal_test",
        "vizualize": True,
    }

    exe = get_tempo_ppo_executor(
        wandb_run=FakeWandBLogger("./results/minimal_test/tempo_ppo.csv"),
        **params,
    )

    #exe.execute()
