import dataclasses
from collections.abc import Callable

from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.datatypes import OpOutId
from tempo.core.dependence_graph import PDG, OpData
from tempo.core.shape import Shape
from tempo.transformations.incrementalization.inc_core import IncRoundCtx
from tempo.utils import logger

log = logger.get_logger(__name__)


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

INC_RULES: dict[
    type[top.TensorOp],
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
    top.SortOp: elementwise_rewrite,
    # top.IndexAdd: single_dim_rewrite,  # should exist?
    # Convolution
    # top.ConvOp: conv_rewrite,
    # top.ConvBwdOp: conv_rewrite,  # can this be treated the same as ConvOp?
    # UDF
    # top.UDFOp: udf_rewrite,
}
