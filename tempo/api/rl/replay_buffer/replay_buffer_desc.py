from dataclasses import dataclass, field
from typing import ClassVar

from tempo.api.rl.replay_buffer.runtime_replay_buffer_interface import (
    ReplayBufferCtx,
    RuntimeReplayBufferBuilder,
)


@dataclass(frozen=True)
class ReplayBufferDesc:
    _id_counter: ClassVar[int] = 0
    builder: RuntimeReplayBufferBuilder
    ctx: ReplayBufferCtx
    unique_id: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "unique_id", ReplayBufferDesc._id_counter)
        ReplayBufferDesc._id_counter += 1

    @property
    def state_id(self) -> str:
        return f"replay_{self.unique_id}"
