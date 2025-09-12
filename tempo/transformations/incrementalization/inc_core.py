import dataclasses
import uuid
from collections.abc import Callable, Sequence
from enum import IntEnum, auto
from typing import Union

from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.datatypes import OpInId
from tempo.core.symbolic_tensor import (
    SymbolicTensor,
)
from tempo.utils import logger

log = logger.get_logger(__name__)

# TODO: Do we really need both of these?
IncStartOp = Union[top.ReduceOp, top.MatMulOp]
ALLOWED_START_OPS = (top.SumOp, top.MaxOp, top.MatMulOp)


class IncKind(IntEnum):
    MEMORY = auto()
    STATIFYING = auto()
    # INTRA_OP_PARALLEL = auto() #TODO
    # INTER_OP_PARALLEL = auto()


@dataclasses.dataclass(frozen=True)
class PadInfo:
    padding: tuple[ie.IntIndexValueLike, ie.IntIndexValueLike]
    how_to_index_padding: ie.IndexSequence
    pad_idx_index_expr: ie.IntIndexValueLike
    src_inc_dim_access_expr: ie.IndexAtom
    pad_id: str = dataclasses.field(default_factory=lambda: f"{uuid.uuid4()}")


@dataclasses.dataclass(frozen=True)
class NonIncedDepyAccessInfo:
    src_op: top.TensorOp
    access_expr: ie.IndexSequence
    dim_access_expr: ie.IndexAtom
    num_slices_before: int
    pad_info: PadInfo | None


@dataclasses.dataclass(frozen=True)
class IncRoundCtx:
    kind: IncKind
    inc_start_ops: set[IncStartOp]
    start_op_inputs_and_dims: dict[IncStartOp, Sequence[tuple[OpInId, int]]]
    inc_var: ie.Symbol
    dim_size: ie.IntIndexValueLike
    block_size: int
    block_idx: SymbolicTensor
    num_blocks: ie.IntIndexValue
    comp_ctx: CompilationCtx
    needs_incrementalization: Callable[[top.TensorOp, CompilationCtx, bool], bool]
    finalize_incremental: bool = False
    # The following are filled in by the incrementalization pass itself
    op_mapping: dict[top.TensorOp, top.TensorOp] = dataclasses.field(default_factory=dict)
    dim_position_map: dict[top.TensorOp, int] = dataclasses.field(default_factory=dict)
    padding_applied: dict[top.TensorOp, PadInfo] = dataclasses.field(default_factory=dict)
    all_dynamic_ops: set[top.TensorOp] = dataclasses.field(default_factory=set)
    max_depth: int | None = None

    def __post_init__(self) -> None:
        if (not self.finalize_incremental) and self.kind == IncKind.STATIFYING:
            raise ValueError("Statifying inc does not support all at once finalizing.")

    def __str__(self) -> str:
        return (
            f"IncRoundCtx("
            f"\tinc_start_ops={self.inc_start_ops}, "
            f"\tstart_op_inputs_and_dims={self.start_op_inputs_and_dims}, "
            f"\tinc_var={self.inc_var}, "
            f"\tdim_size={self.dim_size}, "
            f"\tblock_size={self.block_size}, "
            f"\tblock_idx={self.block_idx}, "
            f"\tfinalize_incremental={self.finalize_incremental}, "
            f"\tall_dynamic_ops={self.all_dynamic_ops}"
            f")"
        )
