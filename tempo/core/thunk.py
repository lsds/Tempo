from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from tempo.core import index_expr as ie
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import BackendTensorT
from tempo.core.external_state_store import ExternalStateStore


@dataclass(frozen=True, slots=True)
class ThunkExecutionCtx:
    symbol_values: Mapping[ie.Symbol, int]
    exec_cfg: ExecutionConfig
    universe_basis_expr: ie.IndexSequence
    external_state_store: ExternalStateStore | None


Thunk = Callable[[tuple[BackendTensorT, ...], ThunkExecutionCtx], tuple[BackendTensorT, ...]]
