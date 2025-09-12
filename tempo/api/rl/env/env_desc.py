from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, Optional

import gymnasium as gym

from tempo.api.rl.env import space_utils
from tempo.api.rl.env.wrappers import RuntimeEnv
from tempo.core.dtype import DataType, dtypes
from tempo.core.shape import StaticShape

RuntimeEnvBuilder = Callable[[Optional[int]], RuntimeEnv]


@dataclass(frozen=True)
class EnvDesc:
    _id_counter: ClassVar[int] = 0
    num_vector_envs: int | None
    single_observation_space: gym.spaces.Space
    single_action_space: gym.spaces.Space
    reward_range: tuple[float, float]
    builder: RuntimeEnvBuilder
    unique_id: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "unique_id", EnvDesc._id_counter)
        EnvDesc._id_counter += 1

    @property
    def state_id(self) -> str:
        return f"env_{self.unique_id}"

    @property
    def observation_shape(self) -> StaticShape:
        return StaticShape(
            (
                *((self.num_vector_envs,) if self.num_vector_envs is not None else ()),
                *space_utils.space_to_shape(self.single_observation_space),
            )
        )

    @property
    def observation_dtype(self) -> DataType:
        return space_utils.space_to_dtype(self.single_observation_space)  # type:ignore

    @property
    def rew_dtype(self) -> DataType:
        return dtypes.float32

    @property
    def term_dtype(self) -> DataType:
        return dtypes.bool_

    @property
    def trunc_dtype(self) -> DataType:
        return dtypes.bool_

    @property
    def action_shape(self) -> StaticShape:
        return StaticShape(
            (
                *((self.num_vector_envs,) if self.num_vector_envs is not None else ()),
                *space_utils.space_to_shape(self.single_action_space),
            )
        )

    @property
    def action_dtype(self) -> DataType:
        return space_utils.space_to_dtype(self.single_action_space)  # type:ignore

    @property
    def rew_shape(self) -> StaticShape:
        return StaticShape(
            (*((self.num_vector_envs,) if self.num_vector_envs is not None else ()),)
        )

    @property
    def term_shape(self) -> StaticShape:
        return StaticShape(
            (*((self.num_vector_envs,) if self.num_vector_envs is not None else ()),)
        )

    @property
    def trunc_shape(self) -> StaticShape:
        return StaticShape(
            (*((self.num_vector_envs,) if self.num_vector_envs is not None else ()),)
        )
