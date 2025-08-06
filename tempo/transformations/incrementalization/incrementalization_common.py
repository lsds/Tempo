import dataclasses
import uuid
from enum import IntEnum, auto
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union

from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.datatypes import OpInId, OpOutId
from tempo.core.dependence_graph import PDG, DependencyData, OpData
from tempo.core.domain import Domain
from tempo.core.dtype import dtypes
from tempo.core.op_tags import STATIFY_PAD_ID_TAG
from tempo.core.shape import Shape
from tempo.core.symbolic_tensor import (
    SymbolicTensor,
    _get_symbolic_tensor_for_op_output,
)
from tempo.core.tensor_op import TensorOp
from tempo.runtime.backends.backend import DLBackendName
from tempo.transformations.optimizer.dead_code_elimination import DeadCodeElimination
from tempo.utils import isl as isl_utils
from tempo.utils import logger
from tempo.utils.dg_utils import (
    get_padding_for_slice,
    is_all_future_access,
    is_all_past_access,
    is_expanded_dim,
    is_matmul_contracting_dim,
    is_non_dimensional_param,
    is_window_access,
    propagate_dim_through_op_in_to_out,
    propagate_dim_through_op_out_to_in,
    recursively_follow_op_out_dim_through_dependencies,
)

log = logger.get_logger(__name__)


# NOTE: The following is a description of the incrementalization mechanism.
# Once a starting op and dimension is decided, the incrementalization is done recursively,
# following the current dims position.
# The recursion stops when the incrementalization dimension is "created" by either an
# expand or a symbolic index operation.
# Then, edges are created connecting the new incrementalized ops to the remaining ops that
# were not incrementalized.
# This is divided into dependents and dependencies.

# Non-incrementalized dependents dep -- e --> inc_op are handled by replacing e
# with e'=(0:INCN, *e),
# and then inserting a permutation and reshape to ensure the input shape matches what dep expects:
# dep --> reshape(such_that_shape_matches) --> permute(place_inc_dim_where_needed) -- e' --> inc_op.
# An edge case is when the inc_op is a flip on inc_dim, in which case we flip it back:
# dep --> reshape(such_that_shape_matches) --> permute(place_inc_dim_where_needed)
# --> flip(inc_dim) -- e' --> inc_op.

# Non-incrementalized dependencies inc_op -- e --> depy are handled by a few cases:
# - If depy is a index, kernel or slice parameter, we keep it as is.
# - If depy was also inc (internal), we just fix the edge by appending inc_var.
# - If depy was an expand that creates our dim of interest, we expand to a smaller (block) size.
# - Else, we use a spatial indexing to access the desired block of data from the depy.

# NOTE: Data-dependent operations, such as IndexSelect, IndexSlice, Gather, Scatter,
# (data dependent due to runtime index parameters) are not incrementalizable on the dimension
# which they affect.
# For similar reasons, it does not make sense to incrementalize the dimension
# which is being padded in a padding op and such, especially if the padding is dynamic.

# TODO: Discuss padding.

# TODO do we want to add support for inc'ing MergeOps?


class IncKind(IntEnum):
    MEMORY = auto()
    STATIFYING = auto()
    # INTRA_OP_PARALLEL = auto() #TODO
    # INTER_OP_PARALLEL = auto()


@dataclasses.dataclass(frozen=True)
class PadInfo:
    padding: Tuple[ie.IntIndexValueLike, ie.IntIndexValueLike]
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
    pad_info: Optional[PadInfo]


def get_or_create_inc_symbol(
    dg: PDG,
    inc_round: int,
    upper_bound: ie.IntIndexValueLike,
    inc_var_name: str = "di",
    allow_reuse_symbol: bool = True,
) -> ie.Symbol:
    if allow_reuse_symbol:
        for d in dg.universe.variables:
            if d.name.startswith(inc_var_name) and ie.lift_to_int_ie(
                dg.bound_defs[d.as_bound()]
            ).logical_eq(ie.lift_to_int_ie(upper_bound)):
                return d

    inc_var, _ = dg.extend_universe(f"{inc_var_name}{inc_round}", upper_bound)
    return inc_var


def create_inc_symbol_and_block_idxs(
    dg: PDG,
    inc_round: int,
    block_size: int,
    upper_bound: ie.IntIndexValueLike,
    inc_var_name: str = "di",
    allow_reuse_symbol: bool = True,
) -> Tuple[ie.Symbol, SymbolicTensor]:
    inc_var = get_or_create_inc_symbol(dg, inc_round, upper_bound, inc_var_name, allow_reuse_symbol)

    block_idxs = create_block_idxs(block_size, inc_var)

    return inc_var, block_idxs


def create_block_idxs(block_size: int, inc_var: ie.Symbol) -> SymbolicTensor:
    block_shift = SymbolicTensor.arange(start=0, stop=block_size)
    if block_shift.shape == ():
        block_shift = block_shift.unsqueeze(0)
    block_idxs = (SymbolicTensor.eval_symbol(inc_var) * block_size).unsqueeze(0).expand(
        block_shift.shape
    ) + block_shift

    return block_idxs


# TODO: Do we really need both of these?
IncStartOp = Union[top.ReduceOp, top.MatMulOp]
ALLOWED_START_OPS = (top.SumOp, top.MaxOp, top.MatMulOp)


@dataclasses.dataclass(frozen=True)
class IncRoundCtx:
    kind: IncKind
    inc_start_ops: Set[IncStartOp]
    start_op_inputs_and_dims: Dict[IncStartOp, Sequence[Tuple[OpInId, int]]]
    inc_var: ie.Symbol
    dim_size: ie.IntIndexValueLike
    block_size: int
    block_idx: SymbolicTensor
    num_blocks: ie.IntIndexValue
    comp_ctx: CompilationCtx
    needs_incrementalization: Callable[[top.TensorOp, CompilationCtx, bool], bool]
    finalize_incremental: bool = False
    # The following are filled in by the incrementalization pass itself
    op_mapping: Dict[top.TensorOp, top.TensorOp] = dataclasses.field(default_factory=dict)
    dim_position_map: Dict[top.TensorOp, int] = dataclasses.field(default_factory=dict)
    padding_applied: Dict[top.TensorOp, PadInfo] = dataclasses.field(default_factory=dict)
    all_dynamic_ops: Set[top.TensorOp] = dataclasses.field(default_factory=set)
    max_depth: Optional[int] = None

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


def _finalize_reduce_op(
    new_dg: PDG,
    op: top.ReduceOp,
    start_op_depys: Sequence[Tuple[TensorOp, DependencyData, int]],
    inc_round_ctx: IncRoundCtx,
    inc_var_UB_max_val: int,
    inp_inc_dim0: int,
    old_op_data: OpData,
) -> Tuple[SymbolicTensor, SymbolicTensor]:
    """Handle finalization of reduce operations during incrementalization."""
    assert len(start_op_depys) == 1, (
        f"Expected exactly one dependency for {op}, got {start_op_depys}"
    )
    depy, depy_data, depy_out_inc_dim = start_op_depys[0]

    # 1. Create a new reduce op, which is exactly the same, but has block dim in domain
    new_initial_red_op = dataclasses.replace(
        op,
        op_id=new_dg.get_next_op_id(),
        domain=op.domain.append_dim(inc_round_ctx.inc_var, inc_round_ctx.inc_var.as_bound()),
    )
    new_op_data = dataclasses.replace(old_op_data, op=new_initial_red_op)
    new_dg.insert_op(new_op_data)

    _handle_dependency_edge(
        new_dg,
        inc_round_ctx,
        op,
        new_initial_red_op,
        depy,
        depy_data,
        inp_inc_dim0,
        None,
    )

    # 3. Apply finalizing reduction manually
    first_red_symb_t = _get_symbolic_tensor_for_op_output(new_dg, new_initial_red_op, OpOutId(0))
    if inc_var_UB_max_val == 1:
        inc_var_idx = first_red_symb_t.domain.find_variable_index(inc_round_ctx.inc_var)
        final_op_symb_t = first_red_symb_t.symbolic_index(
            first_red_symb_t.domain.basis_expr.replace_idx(inc_var_idx, 0)
        ).ident()
    else:
        if inc_round_ctx.finalize_incremental:
            inc_var_idx = first_red_symb_t.domain.find_variable_index(inc_round_ctx.inc_var)
            curr_expr = first_red_symb_t.domain.basis_expr
            prev_expr = curr_expr.replace_idx(inc_var_idx, inc_round_ctx.inc_var - 1)

            final_op_symb_t = SymbolicTensor.merge_like(first_red_symb_t)
            final_op_symb_t.add_merge_branch(
                ie.Equal(inc_round_ctx.inc_var, ie.ConstInt(0)), first_red_symb_t
            )

            if isinstance(op, top.SumOp):
                second_branch = first_red_symb_t + final_op_symb_t.symbolic_index(prev_expr)
            elif isinstance(op, top.MaxOp):
                second_branch = SymbolicTensor.cat(
                    first_red_symb_t.unsqueeze(0),
                    final_op_symb_t.symbolic_index(prev_expr).unsqueeze(0),
                    dim=0,
                ).max(0)[0]

            final_op_symb_t.add_merge_branch(
                ie.Not(ie.Equal(inc_round_ctx.inc_var, ie.ConstInt(0))), second_branch
            )
            final_op_symb_t = final_op_symb_t.symbolic_index(
                curr_expr.replace_idx(inc_var_idx, inc_round_ctx.inc_var.as_bound() - 1)
            ).ident()
        else:
            first_red_symb_t_ = first_red_symb_t.symbolic_index(
                new_initial_red_op.domain.basis_expr.replace_idx(
                    len(new_initial_red_op.domain) - 1,
                    ie.slice_(ie.ConstInt(0), inc_var_UB_max_val),  # inc_var_UB)
                )
            )
            if isinstance(op, top.SumOp):
                final_op_symb_t = first_red_symb_t_.sum(dims=(0,), keepdim=False)
            elif isinstance(op, top.MaxOp):
                final_op_symb_t = first_red_symb_t_.max(dim=0, keepdim=False)[0]

    return first_red_symb_t, final_op_symb_t


def _finalize_matmul_contracting(
    new_dg: PDG,
    op: top.MatMulOp,
    start_op_depys: Sequence[Tuple[TensorOp, DependencyData, int]],
    inc_round_ctx: IncRoundCtx,
    inc_var_UB_max_val: int,
    old_op_data: OpData,
) -> Tuple[SymbolicTensor, SymbolicTensor]:
    """Handle finalization of matmul operations with
    contracting dimensions during incrementalization."""
    assert len(start_op_depys) == 2, (
        f"Expected exactly two dependencies for {op}, got {start_op_depys}"
    )
    depy_0, depy_data_0, depy_out_inc_dim0 = start_op_depys[0]
    depy_1, depy_data_1, depy_out_inc_dim1 = start_op_depys[1]

    new_initial_matmul_op = dataclasses.replace(
        op,
        op_id=new_dg.get_next_op_id(),
        domain=op.domain.append_dim(inc_round_ctx.inc_var, inc_round_ctx.inc_var.as_bound()),
    )

    new_dg.insert_op(dataclasses.replace(old_op_data, op=new_initial_matmul_op))
    first_red_symb_t = _get_symbolic_tensor_for_op_output(new_dg, new_initial_matmul_op, OpOutId(0))
    inc_var_idx = first_red_symb_t.domain.find_variable_index(inc_round_ctx.inc_var)

    for d, dd, ddim in [
        (depy_0, depy_data_0, depy_out_inc_dim0),
        (depy_1, depy_data_1, depy_out_inc_dim1),
    ]:
        _handle_dependency_edge(
            new_dg,
            inc_round_ctx,
            op,
            new_initial_matmul_op,
            d,
            dd,
            ddim,
            None,
        )
    if inc_var_UB_max_val == 1:
        final_op_symb_t = first_red_symb_t.symbolic_index(
            first_red_symb_t.domain.basis_expr.replace_idx(inc_var_idx, 0)
        ).ident()
    else:
        curr_expr = first_red_symb_t.domain.basis_expr
        prev_expr = curr_expr.replace_idx(inc_var_idx, inc_round_ctx.inc_var - 1)

        if inc_round_ctx.finalize_incremental:
            final_op_symb_t = SymbolicTensor.merge_like(first_red_symb_t)
            cond = ie.Equal(inc_round_ctx.inc_var, ie.ConstInt(0))
            final_op_symb_t.add_merge_branch(
                cond,
                first_red_symb_t,
            )

            add_matmul_symb_t = first_red_symb_t + final_op_symb_t.symbolic_index(prev_expr)
            final_op_symb_t.add_merge_branch(ie.Not(cond), add_matmul_symb_t)
            final_op_symb_t = final_op_symb_t.symbolic_index(
                curr_expr.replace_idx(inc_var_idx, inc_round_ctx.inc_var.as_bound() - 1)
            ).ident()
        else:
            expr = first_red_symb_t.domain.basis_expr.replace_idx(
                inc_var_idx,
                ie.slice_(ie.ConstInt(0), inc_var_UB_max_val),  # inc_var_UB)
            )
            final_op_symb_t = first_red_symb_t.symbolic_index(expr).sum(dims=(0,), keepdim=False)

    return first_red_symb_t, final_op_symb_t


def _finalize_matmul_non_contracting(
    new_dg: PDG,
    op: top.MatMulOp,
    start_op_depys: Sequence[Tuple[TensorOp, DependencyData, int]],
    inc_round_ctx: IncRoundCtx,
    inc_var_UB_max_val: int,
    inp_inc_dim0: int,
    inps_inc_dim_and_idx: Sequence[Tuple[OpInId, int]],
    old_op_data: OpData,
) -> Tuple[SymbolicTensor, SymbolicTensor]:
    """Handle finalization of matmul operations with
    non-contracting dimensions during incrementalization."""
    assert len(start_op_depys) == 2, (
        f"Expected exactly two dependencies for {op}, got {start_op_depys}"
    )
    depy_0, depy_data_0, depy_out_inc_dim0 = start_op_depys[0]
    depy_1, depy_data_1, depy_out_inc_dim1 = start_op_depys[1]

    new_initial_matmul_op = dataclasses.replace(
        op,
        op_id=new_dg.get_next_op_id(),
        domain=op.domain.append_dim(inc_round_ctx.inc_var, inc_round_ctx.inc_var.as_bound()),
    )

    # NOTE: Idea is to cat. Because of symbolic dims, just need to permute + reshape.
    orig_shape = new_dg.get_output_shapes_list(op)[OpOutId(0)]
    out_inc_dim = propagate_dim_through_op_in_to_out(
        new_dg, op, new_dg.get_input_shapes_list(op), inp_inc_dim0, inps_inc_dim_and_idx[0][0]
    )[OpOutId(0)]
    new_shape = orig_shape.resize_dim(out_inc_dim, inc_round_ctx.block_size)
    new_op_data = dataclasses.replace(
        old_op_data, op=new_initial_matmul_op, output_shapes={OpOutId(0): new_shape}
    )
    new_dg.insert_op(new_op_data)
    first_red_symb_t = _get_symbolic_tensor_for_op_output(new_dg, new_initial_matmul_op, OpOutId(0))
    inc_var_idx = first_red_symb_t.domain.find_variable_index(inc_round_ctx.inc_var)

    for i, d, dd in [(0, depy_0, depy_data_0), (1, depy_1, depy_data_1)]:
        ddim_ = inps_inc_dim_and_idx[i][1] if i < len(inps_inc_dim_and_idx) else None
        _handle_dependency_edge(  # NOTE: this function understands ddim is None
            new_dg,
            inc_round_ctx,
            op,
            new_initial_matmul_op,
            d,
            dd,
            ddim_,
            out_inc_dim,
        )

    if inc_var_UB_max_val == 1:
        final_op_symb_t = first_red_symb_t.symbolic_index(
            first_red_symb_t.domain.basis_expr.replace_idx(inc_var_idx, 0)
        ).ident()
    else:
        # NOTE: This perm should work for all input 0/1 incrementalizations.
        # E.g. (B, M, K) @ (B, K, N) -> (B, M, N) =inc(dim=2) =>
        #      (B, M, K) @ (B, K, N//4) -> (B, M, N//4)
        # When indexed on 0:INC_VAR, we get (4, B, M, N//4)
        #      -perm-> (B, M, 4, N//4) -reshape-> (B, M, N)
        perm = list(range(len(orig_shape) + 1))  # +1 for inc_var slice added
        perm.pop(0)
        perm.insert(0, inp_inc_dim0)
        final_op_symb_t = first_red_symb_t.permute(dims=perm).reshape(orig_shape)

    return first_red_symb_t, final_op_symb_t


def finalize_incrementalization_start_point(
    new_dg: PDG,
    op: IncStartOp,
    start_op_depys: Sequence[Tuple[TensorOp, DependencyData, int]],
    inc_round_ctx: IncRoundCtx,
) -> None:
    inc_var_UB = inc_round_ctx.inc_var.as_bound()
    op_mapping = inc_round_ctx.op_mapping

    inps_inc_dim_and_idx = inc_round_ctx.start_op_inputs_and_dims[op]

    inp_inc_dim0 = inps_inc_dim_and_idx[0][1]
    inp_inc_dim_idx0 = inps_inc_dim_and_idx[0][0]

    old_op_data = new_dg.ops_by_id[op.op_id]

    inc_var_UB_max_val = get_int_inc_var_ub(new_dg, inc_var_UB)

    first_red_symb_t: Optional[SymbolicTensor] = None

    if isinstance(op, top.ReduceOp):
        first_red_symb_t, final_op_symb_t = _finalize_reduce_op(
            new_dg, op, start_op_depys, inc_round_ctx, inc_var_UB_max_val, inp_inc_dim0, old_op_data
        )

    elif isinstance(op, top.MatMulOp):
        is_contract_dim = is_matmul_contracting_dim(new_dg, op, inp_inc_dim_idx0, inp_inc_dim0)
        if is_contract_dim:
            first_red_symb_t, final_op_symb_t = _finalize_matmul_contracting(
                new_dg, op, start_op_depys, inc_round_ctx, inc_var_UB_max_val, old_op_data
            )
        else:
            first_red_symb_t, final_op_symb_t = _finalize_matmul_non_contracting(
                new_dg,
                op,
                start_op_depys,
                inc_round_ctx,
                inc_var_UB_max_val,
                inp_inc_dim0,
                inps_inc_dim_and_idx,
                old_op_data,
            )

    else:
        raise ValueError(f"Unexpected start op {op}")

    assert first_red_symb_t is not None
    # print(f"Registering op mapping: {op} -> {first_red_symb_t.op}")
    op_mapping[op] = first_red_symb_t.op
    # op_mapping[op] = final_op_symb_t.op
    new_dg.move_dependents(op, final_op_symb_t.op)


def get_int_inc_var_ub(new_dg: PDG, inc_var_UB: ie.Symbol) -> int:
    if inc_var_UB not in new_dg.dynamic_bounds:
        inc_var_UB_max_val = new_dg.static_bounds[inc_var_UB]
    else:
        real_ub = new_dg.dynamic_bounds[inc_var_UB]
        inc_var_UB_max_val_ = isl_utils.int_index_val_max(
            real_ub,
            # domain=new_dg.universe,
            known_symbols=new_dg.static_bounds,
        )
        assert isinstance(inc_var_UB_max_val_, ie.ConstInt)
        inc_var_UB_max_val = inc_var_UB_max_val_.const
    assert inc_var_UB_max_val is not None
    return inc_var_UB_max_val


def cumsum_rewrite(
    dg: PDG,
    op: top.TensorOp,
    out_inc_dim: int,
    inc_round_ctx: IncRoundCtx,
) -> OpData:
    nb = inc_round_ctx.num_blocks.partial_eval(dg.static_bounds)
    assert isinstance(nb, ie.ConstInt) and nb.const == 1, (
        "Cumsum rewrite only supported for 1 block. TODO: advanced cumsum requires chaining."
    )

    # The new operation now has the incrementalizing dimension added to its domain
    new_op = dataclasses.replace(
        op,
        op_id=dg.get_next_op_id(),
        domain=op.domain.append_dim(inc_round_ctx.inc_var, inc_round_ctx.inc_var.as_bound()),
    )

    curr_op_data = dg.ops_by_id[op.op_id]

    new_out_shapes = {}
    for out_id, out_shape in curr_op_data.output_shapes.items():
        resized = out_shape.resize_dim(out_inc_dim, inc_round_ctx.block_size)
        new_out_shapes[out_id] = resized

    new_op_data = dataclasses.replace(curr_op_data, op=new_op, output_shapes=new_out_shapes)

    return new_op_data


def elementwise_rewrite(
    dg: PDG,
    op: top.TensorOp,
    out_inc_dim: int,
    inc_round_ctx: IncRoundCtx,
) -> OpData:
    # The new operation now has the incrementalizing dimension added to its domain
    new_op = dataclasses.replace(
        op,
        op_id=dg.get_next_op_id(),
        domain=op.domain.append_dim(inc_round_ctx.inc_var, inc_round_ctx.inc_var.as_bound()),
    )

    curr_op_data = dg.ops_by_id[op.op_id]

    new_out_shapes = {}
    for out_id, out_shape in curr_op_data.output_shapes.items():
        resized = out_shape.resize_dim(out_inc_dim, inc_round_ctx.block_size)
        new_out_shapes[out_id] = resized

    new_op_data = dataclasses.replace(curr_op_data, op=new_op, output_shapes=new_out_shapes)

    return new_op_data


def index_slice_rewrite(
    dg: PDG,
    op: top.TensorOp,
    out_inc_dim: int,
    inc_round_ctx: IncRoundCtx,
) -> OpData:
    assert isinstance(op, top.IndexSliceOp)

    if out_inc_dim != op.dim:
        return elementwise_rewrite(dg, op, out_inc_dim, inc_round_ctx)

    # NOTE: it is possible we will be incrementalizing on the slicing dim
    # (e.g. when doing statification)
    # In this case, we need to replace the length with the static block size.
    # TODO: However, it is also very important that we replace the start_idx
    # with a shifted start idx, and that proper masking is in place.
    # Start idx needs to be shifted by the amount missing from the right to fill the block.
    # I believe currently, since there is no "pad" op, the need for a mask is not counted.

    raise NotImplementedError(
        "TODO: implement index slice rewrite by shifting start_idx and adding masks"
    )

    new_op = dataclasses.replace(
        op,
        length=inc_round_ctx.block_size,
        op_id=dg.get_next_op_id(),
        domain=op.domain.append_dim(inc_round_ctx.inc_var, inc_round_ctx.inc_var.as_bound()),
    )

    curr_op_data = dg.ops_by_id[op.op_id]

    new_out_shapes = {}
    for out_id, out_shape in curr_op_data.output_shapes.items():
        new_out_shapes[out_id] = out_shape.resize_dim(out_inc_dim, inc_round_ctx.block_size)

    new_op_data = dataclasses.replace(curr_op_data, op=new_op, output_shapes=new_out_shapes)

    return new_op_data


# Simply moves a dim from spatial to symbolic.
def expand_rewrite(
    dg: PDG,
    op: top.TensorOp,
    out_inc_dim: int,
    inc_round_ctx: IncRoundCtx,
) -> OpData:
    assert isinstance(op, top.ExpandOp)
    # The new operation now has the incrementalizing dimension added to its domain
    new_op = dataclasses.replace(
        op,
        sizes=op.sizes.resize_dim(out_inc_dim, inc_round_ctx.block_size),
        op_id=dg.get_next_op_id(),
        domain=op.domain.append_dim(inc_round_ctx.inc_var, inc_round_ctx.inc_var.as_bound()),
    )

    curr_op_data = dg.ops_by_id[op.op_id]

    new_out_shapes = {}
    for out_id, out_shape in curr_op_data.output_shapes.items():
        new_out_shapes[out_id] = out_shape.resize_dim(out_inc_dim, inc_round_ctx.block_size)

    new_op_data = dataclasses.replace(curr_op_data, op=new_op, output_shapes=new_out_shapes)

    return new_op_data


def index_select_rewrite(
    dg: PDG,
    op: top.TensorOp,
    out_inc_dim: int,
    inc_round_ctx: IncRoundCtx,
) -> OpData:
    assert isinstance(op, top.IndexSelectOp)

    assert out_inc_dim != op.dim, (
        f"Cannot incrementalize on the indexing dimension {op.dim} of {op}"
    )

    new_op = dataclasses.replace(
        op,
        op_id=dg.get_next_op_id(),
        domain=op.domain.append_dim(inc_round_ctx.inc_var, inc_round_ctx.inc_var.as_bound()),
    )

    curr_op_data = dg.ops_by_id[op.op_id]

    new_out_shapes = {}
    for out_id, out_shape in curr_op_data.output_shapes.items():
        new_out_shapes[out_id] = out_shape.resize_dim(out_inc_dim, inc_round_ctx.block_size)

    new_op_data = dataclasses.replace(curr_op_data, op=new_op, output_shapes=new_out_shapes)

    return new_op_data


def reshape_rewrite(
    dg: PDG,
    op: top.TensorOp,
    out_inc_dim: int,
    inc_round_ctx: IncRoundCtx,
) -> OpData:
    assert isinstance(op, top.ReshapeOp)
    # The new operation now has the incrementalizing dimension added to its domain
    new_op = dataclasses.replace(
        op,
        shape=Shape.from_(
            op.shape._shape[:out_inc_dim]
            + (inc_round_ctx.block_size,)
            + op.shape._shape[out_inc_dim + 1 :]
        ),
        op_id=dg.get_next_op_id(),
        domain=op.domain.append_dim(inc_round_ctx.inc_var, inc_round_ctx.inc_var.as_bound()),
    )

    curr_op_data = dg.ops_by_id[op.op_id]

    new_out_shapes = {}
    for out_id, out_shape in curr_op_data.output_shapes.items():
        new_out_shapes[out_id] = out_shape.resize_dim(out_inc_dim, inc_round_ctx.block_size)

    new_op_data = dataclasses.replace(curr_op_data, op=new_op, output_shapes=new_out_shapes)

    return new_op_data


def const_rewrite(
    dg: PDG,
    op: top.TensorOp,
    out_inc_dim: int,
    inc_round_ctx: IncRoundCtx,
) -> OpData:
    assert isinstance(op, top.ConstOp)

    # NOTE: can also be uniform across the dimension
    # assert op.is_uniform, f"Cannot incrementalize non-uniform constant {op}"

    curr_op_data = dg.ops_by_id[op.op_id]

    id_ = dg.get_next_op_id()
    new_shape = op.shape.resize_dim(out_inc_dim, inc_round_ctx.block_size)
    new_op = dataclasses.replace(
        op,
        op_id=id_,
        shape=new_shape,
    )

    new_op_data = dataclasses.replace(
        curr_op_data,
        op=new_op,
        output_shapes={OpOutId(0): new_shape},
    )

    return new_op_data


def rand_rewrite(
    dg: PDG,
    op: top.TensorOp,
    out_inc_dim: int,
    inc_round_ctx: IncRoundCtx,
) -> OpData:
    assert isinstance(op, top.RandOp)

    curr_op_data = dg.ops_by_id[op.op_id]

    id_ = dg.get_next_op_id()
    new_domain = op.domain.append_dim(inc_round_ctx.inc_var, inc_round_ctx.inc_var.as_bound())
    new_shape = op.shape.resize_dim(out_inc_dim, inc_round_ctx.block_size)
    new_op = top.RandOp(
        op_id=id_,
        domain=new_domain,
        tags=op.tags,
        dtype=op.dtype,
        shape=new_shape,
    )

    new_op_data = dataclasses.replace(
        curr_op_data,
        op=new_op,
        output_shapes={OpOutId(0): new_shape},
    )

    return new_op_data


IncRuleFn = Callable[
    [
        PDG,
        top.TensorOp,
        int,
        IncRoundCtx,
    ],
    OpData,
]

incrementalization_rules: Dict[
    Type[top.TensorOp],
    IncRuleFn,
] = {
    top.ConstOp: const_rewrite,
    top.RandOp: rand_rewrite,
    # Elementwise
    ## Unary
    top.CastOp: elementwise_rewrite,
    top.SqrtOp: elementwise_rewrite,
    top.NegOp: elementwise_rewrite,
    top.NotOp: elementwise_rewrite,
    top.LnOp: elementwise_rewrite,
    top.ExpOp: elementwise_rewrite,
    top.SinOp: elementwise_rewrite,
    top.IdentOp: elementwise_rewrite,
    top.ValToValOp: elementwise_rewrite,
    # Binary
    top.AddOp: elementwise_rewrite,
    top.SubOp: elementwise_rewrite,
    top.MulOp: elementwise_rewrite,
    top.DivOp: elementwise_rewrite,
    top.PowOp: elementwise_rewrite,
    top.OrOp: elementwise_rewrite,
    top.AndOp: elementwise_rewrite,
    top.EqualOp: elementwise_rewrite,
    top.LessThanOp: elementwise_rewrite,
    # Ternary
    top.WhereOp: elementwise_rewrite,
    # Source Ops
    # top.RandOp: elementwise_rewrite,
    # top.EvalSymbolOp:  # we ignore EvalSymbolOps
    top.MergeOp: elementwise_rewrite,
    # Reductions
    top.SumOp: elementwise_rewrite,
    top.MaxOp: elementwise_rewrite,
    # Movement
    top.FlipOp: elementwise_rewrite,
    top.SqueezeOp: elementwise_rewrite,
    top.UnsqueezeOp: elementwise_rewrite,
    top.PermuteOp: elementwise_rewrite,
    top.ExpandOp: expand_rewrite,
    top.IndexSliceOp: index_slice_rewrite,
    top.PadOp: elementwise_rewrite,
    # Scans
    top.CumSumOp: cumsum_rewrite,
    # Gather/Scatter
    top.GatherOp: elementwise_rewrite,
    # top.ScatterAddOp: scatter_rewrite,
    # Shaped
    # NOTE: These can be incrementalized on unnaffected dims using the elementwise rule
    top.ReshapeOp: reshape_rewrite,
    top.CatOp: elementwise_rewrite,
    top.SplitOp: elementwise_rewrite,
    top.MatMulOp: elementwise_rewrite,
    # Indexing
    top.IndexSelectOp: index_select_rewrite,  # should exist?
    # top.IndexAdd: single_dim_rewrite,  # should exist?
    # Convolution
    # top.ConvOp: conv_rewrite,
    # top.ConvBwdOp: conv_rewrite,  # can this be treated the same as ConvOp?
    # UDF
    # top.UDFOp: udf_rewrite,
}


def get_inc_common_should_recurr_and_effect(
    inc_round_ctx: IncRoundCtx,
) -> Tuple[
    Callable[[PDG, TensorOp, DependencyData, TensorOp, Optional[int], int, Any], bool],
    Callable[[PDG, TensorOp, int, Dict[OpInId, int], Any], Tuple[PDG, Any]],
]:
    def inc_common_should_recurr(
        dg: PDG,
        snk: TensorOp,
        depy_data: DependencyData,
        src: TensorOp,
        snk_out_inc_dim: Optional[int],
        src_out_inc_dim: int,
        ctx: Any,
    ) -> bool:
        depth = ctx
        not_already_inced = True
        if src in inc_round_ctx.op_mapping:
            assert src_out_inc_dim == inc_round_ctx.dim_position_map[src], (
                f"{src=} already inced on dim={inc_round_ctx.dim_position_map[src]},"
                + f"ours={src_out_inc_dim}"
            )
            not_already_inced = False

        can_proceed = True

        if is_expanded_dim(dg, snk, snk_out_inc_dim):
            can_proceed = False

        is_initial = False
        needs_inc = inc_round_ctx.needs_incrementalization(src, inc_round_ctx.comp_ctx, is_initial)
        can_inc = _can_incrementalize(
            dg, src, src_out_inc_dim, inc_round_ctx.block_size, inc_round_ctx.num_blocks
        )
        exceeds_max_depth = (
            depth > inc_round_ctx.max_depth
            if (inc_round_ctx.max_depth is not None and depth is not None)
            else False
        )

        # if exceeds_max_depth:
        #    print(f"Exceeded max depth {depth} for {src}")

        return can_proceed and needs_inc and can_inc and not_already_inced and not exceeds_max_depth

    def inc_common_effect(
        dg: PDG, op_: TensorOp, out_dim: int, in_dims: Dict[OpInId, int], ctx: Any
    ) -> Tuple[PDG, Any]:
        inc_round_ctx.dim_position_map[op_] = out_dim
        new_op_data = incrementalization_rules[op_.__class__](dg, op_, out_dim, inc_round_ctx)
        dg.insert_op(new_op_data)
        # print(f"Incrementalized {op_} to {new_op_data.op}")
        inc_round_ctx.op_mapping[op_] = new_op_data.op
        return dg, ctx + 1

    return inc_common_should_recurr, inc_common_effect


def incrementalize_ops_recursively(
    pdg: PDG,
    start_op: TensorOp,
    start_inc_out_dim: int,
    start_inc_out_id: OpOutId,
    inc_round_ctx: IncRoundCtx,
) -> Tuple[PDG, int]:
    """
    Incrementalizes an operation and recursively processes all its dependencies.
    """

    if start_op in inc_round_ctx.op_mapping:
        # print(f"Already inced {start_op}")
        return pdg, 0

    inc_common_should_recurr, inc_common_effect = get_inc_common_should_recurr_and_effect(
        inc_round_ctx
    )

    pdg, count = recursively_follow_op_out_dim_through_dependencies(
        pdg, start_op, start_inc_out_dim, inc_common_should_recurr, inc_common_effect, 0
    )
    return pdg, count


def _handle_edges(
    new_dg: PDG,
    inc_round_ctx: IncRoundCtx,
) -> None:
    for old_op, new_op in inc_round_ctx.op_mapping.items():
        inc_dim = inc_round_ctx.dim_position_map[old_op]

        _handle_dependent_edges(new_dg, inc_round_ctx, old_op, inc_dim)

        _handle_dependency_edges(
            new_dg,
            inc_round_ctx,
            old_op,
            new_op,
        )


def _handle_dependency_edges(
    new_dg: PDG,
    inc_round_ctx: IncRoundCtx,
    old_op: top.TensorOp,
    new_op: top.TensorOp,
) -> None:
    inc_dim = inc_round_ctx.dim_position_map[old_op]
    inc_dim_in = propagate_dim_through_op_out_to_in(
        new_dg, old_op, new_dg.get_input_shapes_list(old_op), inc_dim
    )
    for depy, depy_data in new_dg.get_flat_direct_dependencies(old_op):
        _handle_dependency_edge(
            new_dg,
            inc_round_ctx,
            old_op,
            new_op,
            depy,
            depy_data,
            inc_dim_in[depy_data.sink_in_idx] if depy_data.sink_in_idx in inc_dim_in else None,
            inc_dim,
        )


def get_inced_padding_and_src_access_expr(
    dg: PDG,
    src_op: TensorOp,
    e: ie.IndexSequence,
    # TODO these were switched... May lead to errors now.
    src_dom: Domain,
    snk_dom: Domain,
    inc_dim_in: int,
    inc_round_ctx: IncRoundCtx,
) -> NonIncedDepyAccessInfo:
    inc_var = inc_round_ctx.inc_var
    block_size = inc_round_ctx.block_size

    slices = [(m, k) for k, m in enumerate(e.members) if not m.is_point()]
    # NOTE: Find inc_dim_in'th slice
    slice_and_index_of_interest = slices[inc_dim_in]
    slice_of_interest = slice_and_index_of_interest[0]
    assert isinstance(slice_of_interest, ie.Slice)
    expr_idx_of_interest = slice_and_index_of_interest[1]

    orig_var = src_dom.variables[expr_idx_of_interest]

    is_block = inc_round_ctx.kind == IncKind.STATIFYING
    slice_of_interest = isl_utils.simplify_slice(slice_of_interest, known_symbols=dg.static_bounds)
    padding = get_padding_for_slice(slice_of_interest, block_size, orig_var, is_block=is_block)
    slice_lb = slice_of_interest.start  # type: ignore
    slice_ub = slice_of_interest.stop  # type: ignore

    assert isinstance(slice_of_interest, ie.Slice)

    num_slices_before = sum(
        not m.is_point() for i, m in enumerate(e.members) if i < expr_idx_of_interest
    )

    # inc_var_UB = inc_var.as_bound()
    # orig_var_UB = orig_var.as_bound()

    how_to_index_padding: Optional[ie.IndexSequence] = None

    if slice_of_interest.is_constant():
        # TODO: this will evaluate to True, even when T is dynamic. We need to handle this properly
        # by checking if the upper bound is static. Otherwise, we will need padding.
        if any(b in dg.dynamic_bounds for b in slice_of_interest.bound_symbols_used()):
            raise ValueError("Dynamic bounds are not supported")

        # NOTE: We know for sure, that the block size was picked such that it divided ub - lb.
        # Thus, we can just use the lower bound and add the block size * inc_var to it.
        src_inc_dim_access_expr = ie.slice_(
            slice_lb + inc_var * block_size,
            slice_lb + (inc_var + 1) * block_size,
        )
    elif is_window_access(
        slice_of_interest
    ):  # max(t-w, 0):t  or t:min(t+w, T)  or max(t-w1, 0):min(t+w2, T)
        # slice_lb = slice_lb.simplify_mins_and_maxes()
        # slice_ub = slice_ub.simplify_mins_and_maxes()

        # NOTE: keep expression as is, we just want to pad it.
        src_inc_dim_access_expr = ie.slice_(slice_lb, slice_ub)

        how_to_index_padding = src_dom.basis_expr
    elif is_all_past_access(slice_of_interest):  # c:t or c:t+1 case
        # NOTE: high confidence in correctness.
        # NOTE: In order to get a pad op with domain eq to src, for each t, we grab and pad
        # only the last block. Then, the src_inc_dim_access_expr will get index the padding
        # at the unpadded block locations for inc_var < slice_ub//block_size.
        # Thus, only the last block processed is padded.
        assert padding is not None
        src_inc_dim_access_expr = ie.slice_(
            slice_ub - (block_size - padding[1]),
            slice_ub,
        )
        src_inc_dim_access_expr = isl_utils.simplify_expr(
            src_inc_dim_access_expr,
            known_symbols=dg.static_bounds,
        )

        # NOTE: for t's within full blocks, we want to index pad op to fetch a non-padded block
        how_to_index_padding_idx = ie.min((inc_var + 1) * block_size - 1, orig_var)
        how_to_index_padding = src_dom.basis_expr.replace_idx(
            expr_idx_of_interest, how_to_index_padding_idx
        )

    elif is_all_future_access(slice_of_interest):  # t:T case
        assert padding is not None
        src_inc_dim_access_expr = ie.slice_(
            slice_lb,
            slice_lb + (block_size - padding[0]),
        )

        src_inc_dim_access_expr = isl_utils.simplify_expr(
            src_inc_dim_access_expr, known_symbols=dg.static_bounds
        )

        # NOTE: access at t, if t is in first block, otherwise, access inc_var non-padded block.
        how_to_index_padding_idx_ = ie.max(
            orig_var, ((slice_lb // block_size) * block_size) + inc_var * block_size
        )
        how_to_index_padding = src_dom.basis_expr.replace_idx(
            expr_idx_of_interest, how_to_index_padding_idx_
        )

    elif isinstance(
        isl_utils.simplify_int_index_value((slice_ub - slice_lb), known_symbols=dg.static_bounds),
        ie.ConstInt,
    ) and slice_ub.logical_eq(
        slice_lb.remap(
            {
                (
                    symb := (
                        list(slice_lb.vars_used())[0]
                        if not isinstance(slice_lb, ie.Add)
                        else list(slice_lb.right_operand.vars_used())[0]
                    )
                ): symb + 1
            }
        )
    ):
        # NOTE handle (recursive) block_access exprs themselves:
        # (b1*40):(((b1+1)*40))
        # (b1*c1)+(b2*c2):(b1*c1)+((b2+1)*c2) as a pattern

        # NOTE: A block access is a non-overlapping slice of a fixed size.
        # The fixed size can be determined by symbolic subtraction.
        # The non-overlapping can be determined by noticing that the lb and ub differ by the +1
        # prior_block_size = simplified.const
        src_inc_dim_access_expr = ie.slice_(
            slice_lb + inc_var * block_size,
            slice_lb + (inc_var + 1) * block_size,
        )

    else:
        raise ValueError(
            f"TODO: No inc handler for expr {e} {slice_of_interest} \
                  with src_dom {src_dom} and inc_dim_in {inc_dim_in}"
        )

    pad_idx_index_expr = (
        ie.lift_to_int_ie(how_to_index_padding.members[expr_idx_of_interest])
        if how_to_index_padding is not None
        else None
    )

    if padding is None:
        pad_info = None
    else:
        assert how_to_index_padding is not None
        assert pad_idx_index_expr is not None
        pad_info = PadInfo(
            padding,
            how_to_index_padding,
            pad_idx_index_expr,
            src_inc_dim_access_expr,
        )

    return NonIncedDepyAccessInfo(
        src_op=src_op,
        access_expr=e.replace_idx(expr_idx_of_interest, src_inc_dim_access_expr),
        # TODO: could we not get the following with just "access_expr[num_slices_before]???"
        dim_access_expr=src_inc_dim_access_expr,
        num_slices_before=num_slices_before,
        pad_info=pad_info,
    )


def _get_pad_value_for_symbolic_tensor(
    symb_t: SymbolicTensor,
) -> Union[float, int, bool]:
    dtype = symb_t.dtype
    if dtypes.is_integer(dtype):
        return 0
    elif dtypes.is_float(dtype):
        return 0.0
    elif dtypes.is_bool(dtype):
        return False
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _handle_dependency_edge(  # noqa: C901
    new_dg: PDG,
    inc_round_ctx: IncRoundCtx,
    old_snk_op: top.TensorOp,
    new_snk_op: top.TensorOp,
    old_src_op: top.TensorOp,
    depy_data: DependencyData,
    inc_dim_in: Optional[int],
    inc_dim_out: Optional[int],
) -> None:
    depy_was_inc = old_src_op in inc_round_ctx.op_mapping
    #    or old_src_op.domain.has_dim(inc_round_ctx.inc_var)

    maybe_new_src_op = (
        inc_round_ctx.op_mapping[old_src_op]
        if old_src_op in inc_round_ctx.op_mapping
        else old_src_op
    )
    e = depy_data.expr

    depy_sym_t = _get_symbolic_tensor_for_op_output(new_dg, old_src_op, depy_data.src_out_idx)

    if isinstance(old_snk_op, top.MatMulOp) and inc_dim_in is None:
        # NOTE: This is the case where the depy is a matmul on non-batch non-contracting dims
        # and we are adjusting the edge to the depy that does not get inced.
        # We can just add the edge as is.
        new_dg.add_edge(new_snk_op, old_src_op, depy_data)

    elif is_non_dimensional_param(new_dg, old_src_op):
        # NOTE: This is the case where the depy is a kernel or index and should not be inc
        # We keep the same edge and continue
        new_dg.add_edge(new_snk_op, old_src_op, depy_data)

    # NOTE: This is where we handle the internal edges, where both the source and sink inc.
    elif depy_was_inc:
        depy_inc_dim = inc_round_ctx.dim_position_map[old_src_op]
        _handle_internal_edge(
            new_dg,
            inc_round_ctx.inc_var,
            inc_round_ctx.inc_var.as_bound(),
            new_snk_op,
            old_src_op,
            depy_data,
            depy_inc_dim,
            maybe_new_src_op,
        )
    elif is_expanded_dim(new_dg, old_snk_op, inc_dim_in or inc_dim_out):
        # NOTE: Special case where the dimension was created by an expand op,
        # so we can just add the edge as is: the expand op will expand to a smaller
        # size.
        new_dg.add_edge(new_snk_op, old_src_op, depy_data)
    # NOTE: This is where we handle the external edges, where only the sink is inc.
    else:
        slices = [(m, k) for k, m in enumerate(e.members) if not m.is_point()]
        # In this case, depy_ is not inc, so does not matter.
        # depy_dom_vars = depy.domain.variables
        assert inc_dim_in is not None, (
            f"Should be handled by another case {old_snk_op=} {old_src_op=} {depy_data.expr=}"
        )

        inc_dim_is_created_by_symb_indexing = inc_dim_in < len(slices)

        if inc_dim_is_created_by_symb_indexing:
            access_info = get_inced_padding_and_src_access_expr(
                new_dg,
                old_src_op,
                depy_data.expr,
                old_src_op.domain,
                old_snk_op.domain,
                inc_dim_in,
                inc_round_ctx,
            )
            new_src_access_expr = access_info.access_expr
            src_inc_dim_access_expr = access_info.dim_access_expr
            num_slices_before = access_info.num_slices_before
            pad_info = access_info.pad_info

            assert depy_data.cond is None, (
                "Should not have a condition here, since sink cannot be merge op"
            )

            is_slice = isinstance(src_inc_dim_access_expr, ie.Slice)
            if not is_slice:
                # NOTE: so if we have gone from a slice to a point (bs=1), the output shape
                # of the tensor will have lost an expected dimension, so we need to recover it
                # by doing unsqueeze(num_slices_before)
                depy_sym_t = depy_sym_t.unsqueeze(num_slices_before)

            depy_sym_t = depy_sym_t.symbolic_index(new_src_access_expr).ident()
            if pad_info is not None:
                padding = pad_info.padding
                how_to_index_padding = pad_info.how_to_index_padding
                depy_sym_t = depy_sym_t.pad_dim(
                    padding, inc_dim_in, value=_get_pad_value_for_symbolic_tensor(depy_sym_t)
                )
                depy_sym_t = depy_sym_t.symbolic_index(how_to_index_padding)
                pad_op: top.PadOp = depy_sym_t.op  # type: ignore
                pad_op.tags[STATIFY_PAD_ID_TAG] = pad_info.pad_id
                inc_round_ctx.padding_applied[pad_op] = pad_info
                # print(f"Inserted padding {pad_info} on src_op {old_src_op}")
                # inc_round_ctx.op_mapping[old_src_op] = pad_op

            depy_sym_t = depy_sym_t.ident()

            new_dg.add_edge(
                new_snk_op,
                depy_sym_t.op,
                DependencyData(depy_sym_t.domain.basis_expr, OpOutId(0), depy_data.sink_in_idx),
            )

        elif is_expanded_dim(new_dg, old_snk_op, inc_dim_in):
            # NOTE: Special case where the dimension was created by an expand op,
            # so we can just add the edge as is: the expand op will expand to a smaller
            # size.
            new_dg.add_edge(new_snk_op, old_src_op, depy_data)
        else:
            # NOTE: Base case, we need to access the depy ourselves at the input block
            # of interest using a spatial slice.

            assert inc_dim_in >= len(slices), "Should be handled by another case"
            # TODO still need to check this stuff
            hypothetical_inc_dim_depy = inc_dim_in  #  - len(slices)

            symb_t = depy_sym_t.symbolic_index(depy_data.expr)
            # print(f"{old_src_op=} {depy_data.expr=} {symb_t.shape=}")
            assert hypothetical_inc_dim_depy < len(symb_t.shape), (
                f"{hypothetical_inc_dim_depy=} {symb_t.shape=}"
            )

            # NOTE: For torch backend, we may want to use index_select here, though
            # it is less efficient than slicing, to allow for use of torch.compile
            if (
                DLBackendName.str_to_enum(inc_round_ctx.comp_ctx.exec_cfg.backend)
                == DLBackendName.TORCH
            ):
                block_idx = inc_round_ctx.block_idx or SymbolicTensor.lift(inc_round_ctx.inc_var)
                indexed_src = symb_t.index_select(
                    hypothetical_inc_dim_depy, block_idx, keepdim=True
                )
            else:
                indexed_src = symb_t.index_slice(
                    hypothetical_inc_dim_depy,
                    inc_round_ctx.inc_var * inc_round_ctx.block_size,
                    inc_round_ctx.block_size,
                )

            new_depy_data = DependencyData(
                indexed_src.domain.basis_expr,
                OpOutId(0),
                depy_data.sink_in_idx,
                depy_data.cond,
            )

            new_dg.add_edge(new_snk_op, indexed_src.op, new_depy_data)

    new_dg.remove_edge(old_snk_op, old_src_op, depy_data)


def _handle_internal_edge(
    new_dg: PDG,
    inc_var: ie.Symbol,
    inc_var_UB: ie.Symbol,
    new_op: top.TensorOp,
    depy: top.TensorOp,
    old_depy_data: DependencyData,
    depy_inc_dim: int,
    inc_depy: top.TensorOp,
) -> None:
    # Depending on the operation, we change the symbolic indexing expression
    # NOTE: if depy flips in smaller blocks now, then block 0 of sink should
    # use block INC_VAR - 1, block 1 should use INC_VAR - 2, etc.
    # This is because the overall flip is equivalent to flipping blocks, plus accessing
    # blocks in flipped order.
    if isinstance(depy, top.FlipOp) and depy.dim == depy_inc_dim:
        new_e = old_depy_data.expr.append_member((inc_var_UB - 1) - inc_var)
    elif isinstance(depy, top.ConstOp):
        new_e = old_depy_data.expr
    else:
        new_e = old_depy_data.expr.append_member(inc_var)

    new_depy_data = DependencyData(
        new_e,
        old_depy_data.src_out_idx,
        old_depy_data.sink_in_idx,
        old_depy_data.cond,
    )

    new_dg.add_edge(new_op, inc_depy, new_depy_data)


def _handle_dependent_edges(
    new_dg: PDG,
    inc_round_ctx: IncRoundCtx,
    old_op: top.TensorOp,
    inc_dim: int,
) -> None:
    new_op = inc_round_ctx.op_mapping[old_op]
    for dependent, dep_data in new_dg.get_flat_direct_dependents(old_op):
        # Skip if dependent is incrementalized
        if dependent in inc_round_ctx.op_mapping:
            continue

        # NOTE: Do not insert these edges as they are handled in the finalize step.
        if dependent in inc_round_ctx.inc_start_ops:
            continue

        symb_t_dep = _get_symbolic_tensor_for_op_output(new_dg, new_op, dep_data.src_out_idx)
        expected_shape = new_dg.get_input_shape(dependent, dep_data.sink_in_idx)

        # Our new dim will always be rightmost, so we can just append the expr
        new_e = dep_data.expr.append_member(
            ie.slice_(ie.ConstInt(0), inc_round_ctx.inc_var.as_bound())
        )
        indexed = symb_t_dep.symbolic_index(new_e)

        # NOTE: Similar to the internal flip case, in order to restore the correct layout,
        # we need to flip the inc dim, since each block is flipped on its own.
        # Without this additional flip, the reshape would not restore the correct order.
        # NOTE: assume 1,2,3,4,5,6,7,8 is the already flipped order.
        # 1,2,3,4,5,6,7,8 -inc-> (1,2,), (3,4,), (5,6,), (7,8,) -flip-> (2,1), (4,3), (6,5), (8,7)
        # LATER (without flip): -0:INC_VAR-> ((2,1), (4,3), (6,5), (8,7))
        #   -reshape-> (2,1,4,3,6,5,8,7)
        # LATER (with flip): -0:INC_VAR-> ((2,1), (4,3), (6,5), (8,7))
        # -flip-> ((1,2), (3,4), (5,6), (7,8)) -reshape-> (1,2,3,4,5,6,7,8)

        if isinstance(new_op, top.FlipOp) and new_op.dim == inc_dim:
            indexed = indexed.flip(inc_dim)

        permutation = list(range(len(indexed.shape)))
        num_slices = sum(not m.is_point() for m in new_e.members)
        # We are the last slice, so pop the perm index at that position
        # (which is num_slices-1 since permutation is currently ordered)
        permutation.pop(num_slices - 1)
        # Then insert it back in at the left of the inc_dim, so we restore the original layout
        # after a reshape
        permutation.insert(inc_dim, num_slices - 1)
        permutation = tuple(permutation)

        permuted = indexed.permute(permutation)
        dependent_not_inc_but_dynamic = (
            dependent not in inc_round_ctx.op_mapping and dependent in inc_round_ctx.all_dynamic_ops
        )

        if inc_round_ctx.kind == IncKind.STATIFYING and dependent_not_inc_but_dynamic:
            div_ = expected_shape.at(inc_dim) / inc_round_ctx.block_size
            expected_shape_with_padding = Shape.from_(
                expected_shape._shape[:inc_dim]
                + (
                    (
                        inc_round_ctx.block_size * ie.Ceil(div_)  # type: ignore
                    ),
                )
                + expected_shape._shape[inc_dim + 1 :]
            )

            reshaped = permuted.reshape(expected_shape_with_padding)
            # TODO: could try to get it from padding applied, but still not perfect...
            # inc_round_ctx.padding_applied
            # TODO: this 0 is wrong. We need to know the amount of pad-left
            reshaped = reshaped.index_slice(inc_dim, start=0, length=inc_round_ctx.dim_size)
        else:
            reshaped = permuted.reshape(expected_shape)

        # new_dep_data = dataclasses.replace(dep_data, src_out_idx=OpOutId(0))
        new_dep_data = DependencyData(
            reshaped.domain.basis_expr,
            OpOutId(0),
            dep_data.sink_in_idx,
            dep_data.cond,
        )

        new_dg.add_edge(dependent, reshaped.op, new_dep_data)
        new_dg.remove_edge(dependent, old_op, dep_data)


def _can_incrementalize_base(
    dg: PDG,
    op: TensorOp,
) -> bool:
    if op.__class__ not in incrementalization_rules:
        # log.warning("Cannot incrementalize %s due to missing rule", op)
        return False

    if isinstance(op, top.MaxOp):
        dependents = dg.get_flat_direct_dependents(op)

        # Can't incrementalize if anyone depends on the indexes of the max op
        if any(d.src_out_idx == 1 for _, d in dependents):
            return False

    return True


def _can_incrementalize(
    dg: PDG,
    op: TensorOp,
    inc_dim: int,
    block_size: int,
    num_blocks: ie.IntIndexValue,
) -> bool:
    result = True

    if not _can_incrementalize_base(dg, op):
        result = False

    if is_non_dimensional_param(dg, op):
        result = False

    # TODO: could we not remove this? Pretty sure we could
    if isinstance(op, top.MergeOp):
        result = False

    # TODO remove this block of comments if unnecessary
    # NOTE: it works, it's fine. We just need to be careful of propagating back to the inputs.
    # if isinstance(op, top.MatMulOp):
    #    len_of_input_shapes = len(dg.get_input_shape(op, OpInId(0)))
    #    # NOTE: inc_dim cannot be in last 2 dims
    #    result &= len_of_input_shapes - 2 > inc_dim

    if isinstance(op, top.ConstOp):
        result &= op.is_uniform or op.is_uniform_on_dim(inc_dim)

    if isinstance(op, (top.ReshapeOp, top.CatOp, top.IndexSliceOp, top.PadOp)):
        old_input_shapes = dg.get_input_shapes_list(op)
        result &= inc_dim not in op.dims_affected(tuple(old_input_shapes))

    if isinstance(op, top.IndexSelectOp):
        result &= inc_dim != op.dim

    if isinstance(op, top.GatherOp):
        result &= inc_dim != op.dim

    # TODO: add support for cumsum inc when num_blocks is not 1: need to get last result of prev
    # block and add it to the current block, then continue the cumsum
    if isinstance(op, top.CumSumOp):
        # print(f"num_blocks: {num_blocks}")
        partial_eval_nb = num_blocks.partial_eval(dg.static_bounds)
        # print(f"partial_eval_nb: {partial_eval_nb}")
        res = isinstance(partial_eval_nb, ie.ConstInt) and partial_eval_nb.const == 1
        # print(f"res: {res}")
        result &= res

    return result


def perform_incrementalization(
    new_dg: PDG,
    inc_round_ctx: IncRoundCtx,
) -> PDG:
    op_mapping = inc_round_ctx.op_mapping

    inc_common_should_recurr, inc_common_effect = get_inc_common_should_recurr_and_effect(
        inc_round_ctx
    )

    start_op_depys_dict: Dict[TensorOp, Sequence[Tuple[TensorOp, DependencyData, int]]] = {}
    for start_op in inc_round_ctx.inc_start_ops:
        if start_op in op_mapping:
            continue
        if start_op.op_id not in new_dg.ops_by_id:
            # NOTE: op might have been removed
            continue

        inps_id_and_inc_dim = list(set(inc_round_ctx.start_op_inputs_and_dims[start_op]))

        start_op_depys = [
            (op, data, inp_id_and_inc_dim[1])
            for op, data in new_dg.get_flat_direct_dependencies(start_op)
            for inp_id_and_inc_dim in inps_id_and_inc_dim
            if data.sink_in_idx == inp_id_and_inc_dim[0]
        ]
        start_op_depys_dict[start_op] = start_op_depys

        for depy, depy_data, in_dim in start_op_depys:
            num_slices = sum(not m.is_point() for m in depy_data.expr.members)
            out_dim = in_dim - num_slices
            if (out_dim >= 0) and inc_common_should_recurr(
                new_dg, start_op, depy_data, depy, None, out_dim, None
            ):
                (new_dg, _) = incrementalize_ops_recursively(
                    new_dg,
                    depy,
                    out_dim,
                    start_op_depys[0][1].src_out_idx,
                    inc_round_ctx,
                )

    # NOTE: We handle the edges with start ops in the finalize step.
    _handle_edges(
        new_dg,
        inc_round_ctx,
    )
    for start_op in inc_round_ctx.inc_start_ops:
        if start_op in op_mapping:
            continue
        round_start_op_depys = start_op_depys_dict[start_op]
        finalize_incrementalization_start_point(
            new_dg,
            start_op,
            round_start_op_depys,
            inc_round_ctx,
        )

    # TODO perform the masking here.

    for op in op_mapping.keys():
        new_dg.remove_op(op)

    # DGRenderer(
    #    CompilationCtx(
    #        new_dg, inc_round_ctx.comp_ctx.analysis_ctx, inc_round_ctx.comp_ctx.exec_cfg
    #    ),
    #    f"{inc_round_ctx.comp_ctx.exec_cfg.path}/after_incbbbb.dot",
    # ).render()

    new_ctx_, _, _ = DeadCodeElimination(
        CompilationCtx(new_dg, inc_round_ctx.comp_ctx.analysis_ctx, inc_round_ctx.comp_ctx.exec_cfg)
    ).run()
    new_dg = new_ctx_.dg

    return new_dg  # type: ignore
