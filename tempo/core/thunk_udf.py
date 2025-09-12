from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from tempo.core import index_expr as ie
from tempo.core.dtype import DataType
from tempo.core.external_state_store import ExternalStateStore, StateStoreKey
from tempo.core.shape import Shape
from tempo.core.thunk_emitter import OpToThunkTranslationFn  # noqa: E402


@dataclass
class UDFVectorizationCtx:
    current_udf_desc: UserDefinedThunkDesc
    vec_dim: ie.Symbol
    vec_dim_bound: ie.Symbol
    vec_size: ie.IntIndexValueLike
    prior_vectorizations: tuple[ie.IntIndexValueLike, ...]


@dataclass(frozen=True)
class UserDefinedThunkDesc:
    thunk_translation: OpToThunkTranslationFn
    num_inputs: int
    num_outputs: int
    infer_output_shapes: Callable[[Sequence[Shape]], Sequence[Shape]]
    infer_output_dtypes: Callable[[Sequence[DataType]], Sequence[DataType]]
    state_store_access_keys: tuple[StateStoreKey, ...] | None = None
    require_simultaneous_vectorization: bool = True
    vectorize: (
        None
        | (
            Callable[
                [UDFVectorizationCtx],
                UserDefinedThunkDesc,
            ]
        )
    ) = None
    clean_up_state: dict[StateStoreKey, Callable[[ExternalStateStore], None]] | None = None
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
