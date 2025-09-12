import dataclasses
from collections.abc import Callable

from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.datatypes import OpOutId
from tempo.core.dependence_graph import OpData
from tempo.core.dtype import dtypes
from tempo.core.shape import Shape
from tempo.core.thunk_udf import UDFVectorizationCtx
from tempo.transformations.vectorization.core import OpVecCtx
from tempo.utils import logger

log = logger.get_logger(__name__)


def register_vec_op(
    op_vec_ctx: OpVecCtx,
    new_op: top.TensorOp,
) -> None:
    old_op_data = op_vec_ctx.dg.ops_by_id[op_vec_ctx.op.op_id]
    # Insert the new op into op_vectorizations, copying the old op's existing vectorizations
    # and inserting the new vectorizations' information alongside it
    vecs = op_vec_ctx.op_vectorizations[op_vec_ctx.op]
    op_vec_ctx.op_vectorizations[new_op] = (
        list(vecs[0]) + [op_vec_ctx.vec_dim_symbol],
        list(vecs[1]) + [op_vec_ctx.dim_size],
    )
    # new_op.tags[f"vectorized_{op_vec_ctx.vec_dim_symbol}"] = True

    # Update the output shapes of the new op to include the dim we are vectorizing over
    new_out_shapes = {
        OpOutId(i): old_op_data.output_shapes[OpOutId(i)].prepend_dim(op_vec_ctx.dim_size)
        for i in range(len(old_op_data.output_shapes))
    }

    # Now we have all the information to insert the new op in the graph
    new_op_data = OpData(new_op, new_out_shapes, dict(old_op_data.output_dtypes))
    op_vec_ctx.dg.insert_op(new_op_data)

    op_vec_ctx.op_mapping[op_vec_ctx.op] = new_op


def rand_rewrite(
    op_vec_ctx: OpVecCtx,
) -> top.TensorOp:
    assert isinstance(op_vec_ctx.op, top.RandOp)
    new_op = dataclasses.replace(
        op_vec_ctx.op,
        op_id=op_vec_ctx.dg.get_next_op_id(),
        domain=op_vec_ctx.op.domain.remove_dim(op_vec_ctx.vec_dim_symbol),
        shape=op_vec_ctx.op.shape.prepend_dim(op_vec_ctx.dim_size),
    )
    register_vec_op(op_vec_ctx, new_op)
    return new_op


def eval_symbol_rewrite(
    op_vec_ctx: OpVecCtx,
) -> top.TensorOp:
    assert isinstance(op_vec_ctx.op, top.EvalSymbolOp)
    # TODO is this assertion allowed?
    # TODO: basically, for dynamic bounds, we will need a different vec approach.
    assert isinstance(op_vec_ctx.dim_size, (int, ie.ConstInt)), (
        f"Expected int, got {op_vec_ctx.dim_size}"
    )
    dim_size_int = int(op_vec_ctx.dim_size)

    import numpy as np

    default_int_np_dtype = dtypes.to_np(dtypes.default_int)

    const_ = np.arange(dim_size_int, dtype=default_int_np_dtype)

    new_op = top.ConstOp(
        op_vec_ctx.dg.get_next_op_id(),
        op_vec_ctx.op.domain.remove_dim(op_vec_ctx.vec_dim_symbol),
        op_vec_ctx.op.tags,
        shape=Shape.from_((dim_size_int,)),
        dtype=dtypes.default_int,
        value=const_,
    )
    register_vec_op(op_vec_ctx, new_op)
    return new_op


def merge_rewrite(
    op_vec_ctx: OpVecCtx,
) -> top.TensorOp:
    assert isinstance(op_vec_ctx.op, top.MergeOp)

    new_op = dataclasses.replace(
        op_vec_ctx.op,
        op_id=op_vec_ctx.dg.get_next_op_id(),
        domain=op_vec_ctx.op.domain.remove_dim(op_vec_ctx.vec_dim_symbol),
        shape=op_vec_ctx.op.shape.prepend_dim(op_vec_ctx.dim_size),
    )
    register_vec_op(op_vec_ctx, new_op)
    return new_op


def elementwise_rewrite(
    op_vec_ctx: OpVecCtx,
) -> top.TensorOp:
    assert isinstance(op_vec_ctx.op, (top.ElementWiseOp, top.MatMulOp, top.ConvOp))
    # elementwise ops are rather simple to vectorize. Just remove the dim from the domain
    new_op = dataclasses.replace(
        op_vec_ctx.op,
        op_id=op_vec_ctx.dg.get_next_op_id(),
        domain=op_vec_ctx.op.domain.remove_dim(op_vec_ctx.vec_dim_symbol),
    )
    register_vec_op(op_vec_ctx, new_op)
    return new_op


def single_dim_rewrite(
    op_vec_ctx: OpVecCtx,
) -> top.TensorOp:
    accepted = (
        top.FlipOp,
        top.SqueezeOp,
        top.UnsqueezeOp,
        top.CumSumOp,
        top.GatherOp,
        top.ScatterAddOp,
        top.CatOp,
        top.IndexSelectOp,
        top.IndexAddOp,
        top.IndexSliceOp,
        top.SplitOp,
        top.PadOp,
        top.SortOp,
    )
    assert isinstance(
        op_vec_ctx.op,
        accepted,
    ), f"Expected one of {accepted}, got {type(op_vec_ctx.op)}"

    new_dim = op_vec_ctx.op.dim + 1
    new_op = dataclasses.replace(
        op_vec_ctx.op,
        op_id=op_vec_ctx.dg.get_next_op_id(),
        domain=op_vec_ctx.op.domain.remove_dim(op_vec_ctx.vec_dim_symbol),
        dim=new_dim,
    )
    register_vec_op(op_vec_ctx, new_op)
    return new_op


# def index_add_rewrite(
#    op_vec_ctx: OpVecCtx,
# ) -> top.TensorOp:
#    assert isinstance(op_vec_ctx.op, top.IndexAddOp)
#    # TODO: There are probably opportunities for nuance like below, but these
#    deps = list(op_vec_ctx.dg.get_flat_direct_dependencies(op_vec_ctx.op))
#    TENSOR_IDX = 0
#    INDEX_IDX = 1
#    SRC_IDX = 2
#    tensor = deps[TENSOR_IDX]
#    index = deps[INDEX_IDX]
#    src = deps[SRC_IDX]
#    tensor_s = _get_symbolic_tensor_for_op_output(
#        op_vec_ctx.dg, tensor[0], tensor[1].src_out_idx
#    ).symbolic_index(tensor[1].expr)
#    index_s = _get_symbolic_tensor_for_op_output(
#        op_vec_ctx.dg, index[0], index[1].src_out_idx
#    ).symbolic_index(index[1].expr)
#    src_s = _get_symbolic_tensor_for_op_output(
#        op_vec_ctx.dg, src[0], src[1].src_out_idx
#    ).symbolic_index(src[1].expr)
#
#    scatter_s = tensor_s._scatter_based_index_add(dim=op.dim, index=index_s, src=src_s)


class GoToBackOfQueueError(Exception): ...


def index_select_rewrite(
    op_vec_ctx: OpVecCtx,
) -> top.TensorOp:
    assert isinstance(op_vec_ctx.op, top.IndexSelectOp)
    # dg = op_vec_ctx.dg
    deps = list(op_vec_ctx.dg.get_flat_direct_dependencies(op_vec_ctx.op))
    SRC_IDX = 0
    INDEX_IDX = 1
    src_op, src_op_data = deps[SRC_IDX]
    index_op, index_op_data = deps[INDEX_IDX]

    src_tensor_is_vec = src_op in op_vec_ctx.ops_to_vectorize
    index_is_vec = index_op in op_vec_ctx.ops_to_vectorize
    index_is_symbol = isinstance(index_op, top.EvalSymbolOp) and index_op.symbol.struct_eq(
        op_vec_ctx.vec_dim_symbol
    )

    # INDEX SELECT CASES
    # | Ops Vectorized | Can use `index_select`? | NOTES
    # | -------------- | ----------------------- | ------------------------
    # | index          | Yes                     | ️ Needs reshape of index and out if index is ≥1D
    # | tensor         | Yes (needs dim++)       |  N/A
    # | tensor, index  | No                  | If index_is_symbol, redundant. Else, promote gather

    # assert not (
    #    src_tensor_is_vec and index_is_vec
    # ), "Vec tensor and index is not supported. Should have been promoted to gather first."
    if (src_op in op_vec_ctx.ops_to_vectorize and src_op not in op_vec_ctx.op_vectorizations) or (
        index_op in op_vec_ctx.ops_to_vectorize and index_op not in op_vec_ctx.op_vectorizations
    ):
        raise GoToBackOfQueueError(
            "Source and index need to be vectorized before index select can be vectorized"
        )

    if src_tensor_is_vec and index_is_vec:
        if index_is_symbol:
            # NOTE: The index was redundant, so we return the vectorized source.
            return op_vec_ctx.op_mapping[src_op]
        else:
            raise ValueError(
                "Attempted to vectorize index select with both source and index vectorized."
                "This should have been promoted to gather first."
                f"op_vec_ctx.op: {op_vec_ctx.op}"
                f"src_op: {src_op}, index_op: {index_op}"
                f"src_op_data: {src_op_data}, index_op_data: {index_op_data}"
            )
            # index_s: SymbolicTensor = get_symbolic_tensor_for_op_output(
            #    dg, op_vec_ctx.op_mapping[index_op], index_op_data.src_out_idx
            # ).symbolic_index(
            #    index_op_data.expr.skip_idx(
            #        index_op.domain.find_variable_index(op_vec_ctx.vec_dim_symbol)
            #    )
            # )
            # source_s: SymbolicTensor = get_symbolic_tensor_for_op_output(
            #    dg, op_vec_ctx.op_mapping[src_op], src_op_data.src_out_idx
            # ).symbolic_index(
            #    src_op_data.expr.skip_idx(
            #        src_op.domain.find_variable_index(op_vec_ctx.vec_dim_symbol)
            #    )
            # )

            ## Promote to gather-based index select. Gather will introduce needed unsqueezes.
            # gathered_s = source_s.gather(dim=op_vec_ctx.op.dim + 1, index=index_s)

            # register_vec_op(op_vec_ctx, gathered_s.op)

    elif src_tensor_is_vec:
        return single_dim_rewrite(op_vec_ctx)
    else:  # Then only index is vec
        index_shape = op_vec_ctx.dg.get_input_shapes_list(op_vec_ctx.op)[1]
        curr_index_is_1D = len(index_shape) == 1

        new_op = dataclasses.replace(
            op_vec_ctx.op,
            op_id=op_vec_ctx.dg.get_next_op_id(),
            domain=op_vec_ctx.op.domain.remove_dim(op_vec_ctx.vec_dim_symbol),
            # NOTE: keep dim in same place on purpose since src shape does not change
        )

        if curr_index_is_1D:
            # TODO: gotta reshape the output back to expected shape.
            # TODO: will also need to flatten the index in vec_edge_both
            # TODO: for that, gonna have to register op first,
            # so we can get a symb_tensor for the output
            raise NotImplementedError("TODO")
        else:
            register_vec_op(op_vec_ctx, new_op)
            return new_op


def tuple_dim_rewrite(
    op_vec_ctx: OpVecCtx,
) -> top.TensorOp:
    assert isinstance(op_vec_ctx.op, (top.SumOp, top.MaxOp))

    new_dims = tuple(dim_ + 1 for dim_ in op_vec_ctx.op.dims)
    new_op = dataclasses.replace(
        op_vec_ctx.op,
        op_id=op_vec_ctx.dg.get_next_op_id(),
        domain=op_vec_ctx.op.domain.remove_dim(op_vec_ctx.vec_dim_symbol),
        dims=new_dims,
    )
    register_vec_op(op_vec_ctx, new_op)
    return new_op


def permute_rewrite(
    op_vec_ctx: OpVecCtx,
) -> top.TensorOp:
    assert isinstance(op_vec_ctx.op, top.PermuteOp)

    new_dims = (0,) + tuple(dim_ + 1 for dim_ in op_vec_ctx.op.dims)
    new_op = dataclasses.replace(
        op_vec_ctx.op,
        op_id=op_vec_ctx.dg.get_next_op_id(),
        domain=op_vec_ctx.op.domain.remove_dim(op_vec_ctx.vec_dim_symbol),
        dims=new_dims,
    )
    register_vec_op(op_vec_ctx, new_op)
    return new_op


def reshape_rewrite(
    op_vec_ctx: OpVecCtx,
) -> top.TensorOp:
    assert isinstance(op_vec_ctx.op, top.ReshapeOp)
    new_op = dataclasses.replace(
        op_vec_ctx.op,
        op_id=op_vec_ctx.dg.get_next_op_id(),
        domain=op_vec_ctx.op.domain.remove_dim(op_vec_ctx.vec_dim_symbol),
        shape=op_vec_ctx.op.shape.prepend_dim(op_vec_ctx.dim_size),
    )
    register_vec_op(op_vec_ctx, new_op)
    return new_op


def expand_rewrite(
    op_vec_ctx: OpVecCtx,
) -> top.TensorOp:
    assert isinstance(op_vec_ctx.op, top.ExpandOp)
    new_op = dataclasses.replace(
        op_vec_ctx.op,
        op_id=op_vec_ctx.dg.get_next_op_id(),
        domain=op_vec_ctx.op.domain.remove_dim(op_vec_ctx.vec_dim_symbol),
        sizes=op_vec_ctx.op.sizes.prepend_dim(op_vec_ctx.dim_size),
    )
    register_vec_op(op_vec_ctx, new_op)
    return new_op


def udf_rewrite(
    op_vec_ctx: OpVecCtx,
) -> top.TensorOp:
    assert isinstance(op_vec_ctx.op, top.UDFOp)
    assert op_vec_ctx.op.desc.vectorize is not None, "Tried vec UDF with no user-defined vec rule"

    new_udf_desc = op_vec_ctx.op.desc.vectorize(
        UDFVectorizationCtx(
            op_vec_ctx.op.desc,
            op_vec_ctx.vec_dim_symbol,
            op_vec_ctx.vec_dim_symbol.as_bound(),
            # TODO: These evals should not be needed when we add dynamic bounds
            ie.evaluate_int(op_vec_ctx.dim_size, op_vec_ctx.dg.static_bounds),
            tuple(
                ie.evaluate_int(d, op_vec_ctx.dg.static_bounds)
                for d in op_vec_ctx.past_vectorizations
            ),
        )
    )
    new_op = dataclasses.replace(
        op_vec_ctx.op,
        op_id=op_vec_ctx.dg.get_next_op_id(),
        domain=op_vec_ctx.op.domain.remove_dim(op_vec_ctx.vec_dim_symbol),
        desc=new_udf_desc,
    )
    register_vec_op(op_vec_ctx, new_op)
    return new_op


OP_VEC_RULES: dict[
    type[top.TensorOp],
    Callable[[OpVecCtx], top.TensorOp],
] = {
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
    top.MatMulOp: elementwise_rewrite,
    # Source Ops
    # top.ConstOp: Consts never have domain, so not needed,
    top.RandOp: rand_rewrite,
    top.EvalSymbolOp: eval_symbol_rewrite,
    top.MergeOp: merge_rewrite,
    # Reductions
    top.SumOp: tuple_dim_rewrite,
    top.MaxOp: tuple_dim_rewrite,
    # Movement
    top.CatOp: single_dim_rewrite,
    top.SplitOp: single_dim_rewrite,
    top.FlipOp: single_dim_rewrite,
    top.SqueezeOp: single_dim_rewrite,
    top.UnsqueezeOp: single_dim_rewrite,
    top.PermuteOp: permute_rewrite,
    top.IndexSliceOp: single_dim_rewrite,
    top.PadOp: single_dim_rewrite,
    # Scans
    top.CumSumOp: single_dim_rewrite,
    # Gather/Scatter
    top.GatherOp: single_dim_rewrite,
    top.ScatterAddOp: single_dim_rewrite,
    # Shaped
    top.ReshapeOp: reshape_rewrite,
    top.ExpandOp: expand_rewrite,
    # Indexing
    top.IndexSelectOp: index_select_rewrite,
    top.IndexAddOp: single_dim_rewrite,
    # Convolution
    # TODO
    top.ConvOp: elementwise_rewrite,
    # top.ConvBwdOp: conv_bwd_rewrite,
    # UDF
    top.UDFOp: udf_rewrite,
    top.SortOp: single_dim_rewrite,
}
