from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, Optional, Sequence, Type

import gymnasium as gym

from tempo.api.rl.env.env import EnvDesc
from tempo.api.rl.env.env_desc import RuntimeEnvBuilder
from tempo.api.rl.env.wrappers import RuntimeEnv, TempoEnvWrapper
from tempo.core.configs import ExecutionConfig
from tempo.runtime.backends.backend import DLBackend
from tempo.utils import logger

log = logger.get_logger(__name__)


@dataclass(frozen=True)
class EnvBuildCtx:
    name: str
    num_envs: Optional[int]
    max_episode_steps: Optional[int]
    kwargs: Dict[str, Any]
    exec_cfg: ExecutionConfig
    backend: DLBackend


EnvBuilder = Callable[[EnvBuildCtx], gym.Env]


class EnvRegistry:
    registry: ClassVar[Dict[str, EnvBuilder]] = {}

    @staticmethod
    def register_env_set(env_set_name: str, builder: Callable[[EnvBuildCtx], gym.Env]) -> None:
        if env_set_name in EnvRegistry.registry:
            raise ValueError(f"Tried overwriting environment set {env_set_name} with new builder")
        EnvRegistry.registry[env_set_name] = builder

    @staticmethod
    def _env_desc_from_runtime_builder(
        builder: RuntimeEnvBuilder,
        num_envs: Optional[int] = None,
    ) -> EnvDesc:
        env = builder(None)
        reward_range = (float(env.reward_range[0]), float(env.reward_range[1]))
        if hasattr(env, "single_observation_space"):
            obs_space = env.single_observation_space  # type:ignore
            act_space = env.single_action_space  # type:ignore
        else:
            obs_space = env.observation_space
            act_space = env.action_space
        env.close()

        desc = EnvDesc(num_envs, obs_space, act_space, reward_range, builder)

        return desc

    @staticmethod
    def get_env_description(
        name: str,
        env_build_ctx: EnvBuildCtx,
        wrappers: Optional[Sequence[Type[TempoEnvWrapper]]] = None,
    ) -> EnvDesc:
        env_set, name = name.split(".", maxsplit=1)

        # Runtime env builder requires only number of envs
        def runtime_env_builder(ne: Optional[int]) -> RuntimeEnv:
            env_build_ctx_ = dataclasses.replace(env_build_ctx, num_envs=ne)
            env_ = EnvRegistry.registry[env_set](env_build_ctx_)
            for wrapper in wrappers or []:
                log.debug("Wrapping env with %s", wrapper)
                env_ = wrapper(env_, env_build_ctx_.exec_cfg)
            return env_

        return EnvRegistry._env_desc_from_runtime_builder(
            runtime_env_builder, env_build_ctx.num_envs
        )
