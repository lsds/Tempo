from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generator, Generic, Tuple, Union

from tempo.core.analysis_ctx import AnalysisCtx
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import BackendTensorT, TensorId
from tempo.core.dependence_graph import PDG
from tempo.core.external_state_store import ExternalStateStore
from tempo.runtime.backends.backend import DLBackend
from tempo.runtime.backends.pytorch.pytorch_backend import PyTorchBackend
from tempo.runtime.tensor_store.tensor_store import TensorStore
from tempo.utils import logger

log = logger.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ExecutorCtx(Generic[BackendTensorT]):
    dg: PDG
    external_state_store: ExternalStateStore
    tensor_store: TensorStore[BackendTensorT]
    exec_cfg: ExecutionConfig
    analysis_ctx: AnalysisCtx
    backend: DLBackend


class Executor(Generic[BackendTensorT], ABC):
    def __init__(
        self,
        exec_ctx: ExecutorCtx,
    ) -> None:
        self.executor_ctx = exec_ctx
        self.dg = exec_ctx.dg
        self.backend = exec_ctx.backend

    @abstractmethod
    def tick(self) -> Generator[bool, None, None]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def execute_until_barrier(self, barrier_name: str) -> None:
        pass

    @abstractmethod
    def execute(self) -> None:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass

    def get_spatial_tensor(
        self, tensor_id: TensorId, index: Tuple[Union[int, slice], ...] = ()
    ) -> BackendTensorT:
        return self.executor_ctx.tensor_store[tensor_id][index]  # type: ignore

    def get_spatial_tensor_torch(
        self, tensor_id: TensorId, index: Tuple[Union[int, slice], ...] = ()
    ) -> BackendTensorT:
        t = self.executor_ctx.tensor_store[tensor_id][index]  # type: ignore
        return PyTorchBackend.from_dlpack(t)  # type: ignore

    def get_compilation_time_breakdown(self) -> Dict[str, Any]:
        return self.executor_ctx.analysis_ctx.compilation_time_breakdown
