from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Generic

from tempo.core import index_expr as ie
from tempo.core.analysis_ctx import AnalysisCtx
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import BackendTensorT
from tempo.core.dependence_graph import PDG
from tempo.core.device import DeviceGroup
from tempo.core.external_state_store import ExternalStateStore
from tempo.core.tensor_op import TensorOp
from tempo.core.thunk import Thunk
from tempo.utils import logger

log = logger.get_logger(__name__)

# Forward reference to avoid circular imports
if TYPE_CHECKING:
    from tempo.core import tensor_ops as top

# Define the type alias here to avoid circular imports
OpToThunkTranslationFn = Callable[
    ["top.TensorOp", "ThunkEmissionCtx[BackendTensorT]"],
    Thunk[BackendTensorT],
]


class ThunkEmitter(Generic[BackendTensorT], ABC):
    """Interface for thunk emitters that convert tensor operations to executable thunks."""

    @abstractmethod
    def emit_thunk_for_op(
        self,
        op: TensorOp,
        ctx: ThunkEmissionCtx[BackendTensorT],
    ) -> Thunk[BackendTensorT]: ...


# NOTE: Prevent circular import with DLBackend
from tempo.core.dl_backend import DLBackend  # noqa: E402


@dataclasses.dataclass
class ThunkEmissionCtx(Generic[BackendTensorT]):
    domain_map: Mapping[ie.Symbol, ie.IntIndexValue]
    loop_counters: Mapping[ie.Symbol, int]
    dev: DeviceGroup
    compile_time_known_symbol_values: Mapping[ie.Symbol, int]
    _external_state_store: ExternalStateStore | None
    # TODO: use backend directly in thunk emitters.
    backend: DLBackend[BackendTensorT]
    compilation_ctx: CompilationCtx

    @property
    def dg(self) -> PDG:
        return self.compilation_ctx.dg

    @property
    def exec_cfg(self) -> ExecutionConfig:
        return self.compilation_ctx.exec_cfg

    @property
    def analysis_ctx(self) -> AnalysisCtx:
        return self.compilation_ctx.analysis_ctx

    @property
    def external_state_store(self) -> ExternalStateStore:
        assert self._external_state_store is not None
        return self._external_state_store

    @property
    def is_in_dataflow(self) -> bool:
        return self.dg.parent_graph is not None

    def replace_dg(self, dg: PDG) -> ThunkEmissionCtx[BackendTensorT]:
        return dataclasses.replace(
            self, compilation_ctx=dataclasses.replace(self.compilation_ctx, dg=dg)
        )
