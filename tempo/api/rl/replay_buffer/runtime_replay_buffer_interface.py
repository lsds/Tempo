from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

from tempo.core.configs import ExecutionConfig
from tempo.core.dtype import DataType
from tempo.core.shape import Shape


@dataclass(frozen=True)
class ReplayBufferCtx:
    max_size: int
    item_shapes: List[Shape]
    item_dtypes: List[DataType]
    exec_cfg: ExecutionConfig
    vectorizations: List[int]
    kwargs: Dict[str, Any]


# TODO this any should be BackendTensorT
class RuntimeReplayBufferInterface:
    @abstractmethod
    def insert(self, data: Sequence[Any]) -> None:
        pass

    @abstractmethod
    def insert_batched(self, data: Sequence[Any]) -> None:
        pass

    @abstractmethod
    def sample(self) -> Sequence[Any]:
        raise NotImplementedError

    @abstractmethod
    def sample_batched(self, num_samples: int) -> Sequence[Any]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        pass


RuntimeReplayBufferBuilder = Callable[[ReplayBufferCtx], RuntimeReplayBufferInterface]
