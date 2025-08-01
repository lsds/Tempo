from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generic, Mapping, Optional, Sequence, Tuple

from tempo.core import index_expr as ie
from tempo.core import tensor_op as top
from tempo.core.analysis_ctx import AnalysisCtx
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import BackendTensorT
from tempo.core.dependence_graph import PDG
from tempo.core.device import DeviceGroup
from tempo.core.dtype import DataType
from tempo.core.external_state_store import ExternalStateStore, StateStoreKey
from tempo.core.shape import Shape


@dataclass(frozen=True, slots=True)
class ThunkExecutionCtx:
    symbol_values: Mapping[ie.Symbol, int]
    exec_cfg: ExecutionConfig
    universe_basis_expr: ie.IndexSequence
    external_state_store: Optional[ExternalStateStore]


Thunk = Callable[[Tuple[BackendTensorT, ...], ThunkExecutionCtx], Tuple[BackendTensorT, ...]]


@dataclass
class ThunkEmissionCtx(Generic[BackendTensorT]):
    domain_map: Mapping[ie.Symbol, ie.IntIndexValue]
    loop_counters: Mapping[ie.Symbol, int]
    dev: DeviceGroup
    dg: PDG
    compile_time_known_symbol_values: Mapping[ie.Symbol, int]
    _external_state_store: Optional[ExternalStateStore]
    exec_cfg: ExecutionConfig
    analysis_ctx: AnalysisCtx
    # backend: DLBackend #TODO: runtime_ctx??

    @property
    def external_state_store(self) -> ExternalStateStore:
        assert self._external_state_store is not None
        return self._external_state_store

    @property
    def is_in_dataflow(self) -> bool:
        return self.dg.parent_graph is not None


OpToThunkTranslationFn = Callable[
    [top.TensorOp, ThunkEmissionCtx],
    Thunk[BackendTensorT],
]


@dataclass
class UDFVectorizationCtx:
    current_udf_desc: UserDefinedThunkDesc
    vec_dim: ie.Symbol
    vec_dim_bound: ie.Symbol
    vec_size: ie.IntIndexValueLike
    prior_vectorizations: Tuple[ie.IntIndexValueLike, ...]


@dataclass(frozen=True)
class UserDefinedThunkDesc:
    thunk_translation: OpToThunkTranslationFn
    num_inputs: int
    num_outputs: int
    infer_output_shapes: Callable[[Sequence[Shape]], Sequence[Shape]]
    infer_output_dtypes: Callable[[Sequence[DataType]], Sequence[DataType]]
    state_store_access_keys: Optional[Tuple[StateStoreKey, ...]] = None
    require_simultaneous_vectorization: bool = True
    vectorize: Optional[
        Callable[
            [UDFVectorizationCtx],
            UserDefinedThunkDesc,
        ]
    ] = None
    clean_up_state: Optional[Dict[StateStoreKey, Callable[[ExternalStateStore], None]]] = None
    needs_symbol_setter: bool = True

    thunk_name: str = "anonymous_udf"

    def __post_init__(self) -> None:
        assert self.num_inputs >= 0 or self.num_outputs >= 0
        if self.state_store_access_keys is not None:
            assert len(self.state_store_access_keys) > 0
            assert self.clean_up_state is not None

            # Must have a cleanup function for each state store access key
            assert all(
                self.clean_up_state.get(k, None) is not None for k in self.state_store_access_keys
            )

    # incrementalize: Callable[[Domain], UDFDesc[BackendTensorT]]
    @property
    def is_stateful(self) -> bool:
        return self.state_store_access_keys is not None and len(self.state_store_access_keys) > 0
