from typing import Any, ClassVar

from tempo.api.rl.replay_buffer.replay_buffer_desc import (
    ReplayBufferCtx,
    ReplayBufferDesc,
)
from tempo.api.rl.replay_buffer.runtime_replay_buffer_interface import (
    RuntimeReplayBufferBuilder,
)
from tempo.core.configs import ExecutionConfig
from tempo.core.dtype import DataType
from tempo.core.shape import Shape


class ReplayBufferRegistry:
    """NOTE: Registered implementations must support sampling with replacement.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_

    """

    registry: ClassVar[
        dict[
            str,
            RuntimeReplayBufferBuilder,
        ]
    ] = {}

    @staticmethod
    def register_replay_buffer(
        storage_key: str,
        builder: RuntimeReplayBufferBuilder,
    ) -> None:
        if storage_key in ReplayBufferRegistry.registry:
            raise ValueError(f"Tried overwriting replay memory {storage_key} with new builder")
        ReplayBufferRegistry.registry[storage_key.lower()] = builder

    @staticmethod
    def get_replay_buffer_description(
        storage_type: str,
        max_size: int,
        item_shapes: list[Shape],
        item_dtypes: list[DataType],
        exec_cfg: ExecutionConfig,
        **kwargs: Any,
    ) -> ReplayBufferDesc:
        builder = ReplayBufferRegistry.registry[storage_type.lower()]
        ctx = ReplayBufferCtx(max_size, item_shapes, item_dtypes, exec_cfg, [], kwargs)
        desc = ReplayBufferDesc(builder, ctx)
        return desc
