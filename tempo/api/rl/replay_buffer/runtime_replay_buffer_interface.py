from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from tempo.core.configs import ExecutionConfig
from tempo.core.dtype import DataType
from tempo.core.shape import Shape


@dataclass(frozen=True)
class ReplayBufferCtx:
    max_size: int
    item_shapes: list[Shape]
    item_dtypes: list[DataType]
    exec_cfg: ExecutionConfig
    vectorizations: list[int]
    kwargs: dict[str, Any]


# TODO this any should be BackendTensorT
class RuntimeReplayBufferInterface:
    @abstractmethod
    def insert(self, data: Sequence[Any]) -> None: ...

    @abstractmethod
    def insert_batched(self, data: Sequence[Any]) -> None: ...

    @abstractmethod
    def sample(self) -> Sequence[Any]:
        raise NotImplementedError

    @abstractmethod
    def sample_batched(self, num_samples: int) -> Sequence[Any]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None: ...


RuntimeReplayBufferBuilder = Callable[[ReplayBufferCtx], RuntimeReplayBufferInterface]
