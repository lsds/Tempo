from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Generic

from tempo.core.analysis_ctx import AnalysisCtx
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import BackendTensorT, TensorId
from tempo.core.dependence_graph import PDG
from tempo.core.dl_backend import DLBackend
from tempo.core.external_state_store import ExternalStateStore
from tempo.runtime.backends.pytorch.pytorch_backend import PyTorchBackend
from tempo.runtime.tensor_store.tensor_store import TensorStore
from tempo.utils import logger

log = logger.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ExecutorCtx(Generic[BackendTensorT]):
    external_state_store: ExternalStateStore
    tensor_store: TensorStore[BackendTensorT]
    compilation_ctx: CompilationCtx
    backend: DLBackend

    @property
    def dg(self) -> PDG:
        return self.compilation_ctx.dg

    @property
    def exec_cfg(self) -> ExecutionConfig:
        return self.compilation_ctx.exec_cfg

    @property
    def analysis_ctx(self) -> AnalysisCtx:
        return self.compilation_ctx.analysis_ctx


class Executor(Generic[BackendTensorT], ABC):
    def __init__(
        self,
        exec_ctx: ExecutorCtx,
    ) -> None:
        self.executor_ctx = exec_ctx
        self.dg = exec_ctx.dg
        self.backend = exec_ctx.backend

    @abstractmethod
    def tick(self) -> Generator[bool]: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def execute_until_barrier(self, barrier_name: str) -> None: ...

    @abstractmethod
    def execute(self) -> None: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    def get_spatial_tensor(
        self, tensor_id: TensorId, index: tuple[int | slice, ...] = ()
    ) -> BackendTensorT:
        return self.executor_ctx.tensor_store[tensor_id][index]  # type: ignore

    def get_spatial_tensor_torch(
        self, tensor_id: TensorId, index: tuple[int | slice, ...] = ()
    ) -> BackendTensorT:
        t = self.executor_ctx.tensor_store[tensor_id][index]  # type: ignore
        return PyTorchBackend.from_dlpack(t)  # type: ignore

    def get_compilation_time_breakdown(self) -> dict[str, Any]:
        return self.executor_ctx.analysis_ctx.compilation_profile_ms
