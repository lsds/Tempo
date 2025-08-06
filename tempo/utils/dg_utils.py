import dataclasses
from collections.abc import Callable
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.datatypes import OpInId, OpOutId
from tempo.core.dependence_graph import PDG, DependencyData
from tempo.core.dim_utils import normalize_negative_dim
from tempo.core.shape import Shape, dim_is_one
from tempo.core.tensor_op import TensorOp
from tempo.utils import isl as isl_utils


def is_scalar_index_select(dg: PDG, op: top.TensorOp) -> bool:
    if not isinstance(op, top.IndexSelectOp):
        return False
    (idx_op, idx_data) = dg.get_flat_direct_dependencies(op)[1]
    in_shape = dg.get_input_shape(op, idx_data.sink_in_idx)
    return in_shape.is_scalar()


def is_index_src_tensor_param(dg: PDG, op: top.TensorOp) -> bool:
    dependents = dg.get_flat_direct_dependents(op)
    for dep, dep_data in dependents:
        dep_is_indexop = isinstance(dep, top.IndexSelectOp) or isinstance(dep, top.IndexAddOp)
        is_idx_param = dep_data.sink_in_idx == OpInId(
            (0 if isinstance(dep, top.IndexSelectOp) else 2)
        )
        if dep_is_indexop and is_idx_param:
            return True
    return False


def is_index_index_param(dg: PDG, op: top.TensorOp) -> bool:
    dependents = dg.get_flat_direct_dependents(op)
    for dep, dep_data in dependents:
        dep_is_indexop = isinstance(dep, top.IndexSelectOp) or isinstance(dep, top.IndexAddOp)
        is_idx_param = dep_data.sink_in_idx == OpInId(1)
        if dep_is_indexop and is_idx_param:
            return True
    return False


def is_slice_start_param(dg: PDG, op: top.TensorOp) -> bool:
    dependents = dg.get_flat_direct_dependents(op)
    for dep, dep_data in dependents:
        dep_is_sliceop = isinstance(dep, top.IndexSliceOp)
        is_start_param = dep_data.sink_in_idx == OpInId(1)
        if dep_is_sliceop and is_start_param:
            return True
    return False


def is_conv_kernel(dg: PDG, op: top.TensorOp) -> bool:
    dependents = dg.get_flat_direct_dependents(op)
    for dep, dep_data in dependents:
        dep_is_convop = isinstance(dep, top.ConvOp)
        is_idx_param = dep_data.sink_in_idx == OpInId(1)
        if dep_is_convop and is_idx_param:
            return True
    return False


def is_non_dimensional_param(dg: PDG, op: top.TensorOp) -> bool:
    return is_index_index_param(dg, op) or is_slice_start_param(dg, op) or is_conv_kernel(dg, op)


def is_expanded_dim(dg: PDG, op: top.TensorOp, dim: Optional[int]) -> bool:
    if isinstance(op, top.ExpandOp):
        assert dim is not None, "dim was not provided"
        original_shape = dg.get_input_shape(op, OpInId(0))
        new_shape = op.sizes

        if dim_is_one(original_shape, dim) and not dim_is_one(new_shape, dim):
            return True
    return False


def is_matmul_contracting_dim(dg: PDG, op: top.MatMulOp, in_idx: OpInId, dim: int) -> bool:
    assert isinstance(op, top.MatMulOp)

    contracting_dim = -1 if in_idx == 0 else -2  # NOTE: This is torch way
    contracting_dim = normalize_negative_dim(contracting_dim, dg.get_input_shape(op, in_idx))

    return dim == contracting_dim


def is_matmul_batch_dim(dg: PDG, op: top.MatMulOp, in_idx: OpInId, dim: int) -> bool:
    assert isinstance(op, top.MatMulOp)

    in_shape = dg.get_input_shape(op, in_idx)

    return dim < len(in_shape) - 2


def propagate_dim_through_op(  # noqa: C901
    dg: PDG,
    op: TensorOp,
    input_shapes: Sequence[Shape],
    dim: int,
    in_id: Optional[OpInId],
    in_to_out: bool = False,
) -> Dict[Union[OpInId, OpOutId], int]:
    """
    Propagates a dimension of interest
    through an operation, adjusting it based on how the operation
    transforms dimensions.

    The direction of the propagation is determined by the in_to_out flag.
    If out_to_in, we expect an out_id to be provided, and we return per-input dim dict.
    If in_to_out, we expect an in_id to be provided, and we return per-output dim dict.
    """
    out_cls = OpOutId if in_to_out else OpInId

    assert dim >= 0, f"dim={dim} given is negative"

    if isinstance(op, top.PermuteOp):
        if in_to_out:
            return {out_cls(0): op.dims.index(dim)}
        else:  # TODO check.
            return {out_cls(0): op.dims[dim]}
    elif isinstance(op, top.IndexSelectOp):  # Behaves like a squeeze
        if is_scalar_index_select(dg, op):
            if in_to_out:
                return {out_cls(0): dim - 1 if dim >= op.dim else dim}
            else:
                return {out_cls(0): dim + 1 if dim >= op.dim else dim}
    # TODO: this code was here, but never used because the above matches first.
    # TODO: we should probably merge it in.
    # elif isinstance(op, top.IndexSelectOp):
    #    if in_id == OpInId(0):  # src tensor
    #        if in_to_out:
    #            if dim == op.dim:
    #                return {}  # NOTE: dim was selected out of existence
    #            else:
    #                return {out_cls(0): (dim - 1 if dim >= op.dim else dim)}
    #        else:
    #            return {out_cls(0): (dim + 1 if dim >= op.dim else dim)}
    #    elif in_id == OpInId(1):  # index
    #        # The index dim just stops existing?
    #        return {}
    elif isinstance(op, top.SqueezeOp):
        # Squeeze removes a dimension; if dim is after the removed dimension, shift it.
        # assert dim != op.dim
        if in_to_out:
            return {out_cls(0): (dim - 1 if dim >= op.dim else dim)}
        else:
            return {out_cls(0): (dim + 1 if dim >= op.dim else dim)}
    elif isinstance(op, top.UnsqueezeOp):
        # Unsqueeze adds a dimension; if that occurs before dim, shift it.
        if in_to_out:
            return {out_cls(0): (dim + 1 if dim >= op.dim else dim)}
        else:
            return {out_cls(0): (dim - 1 if dim > op.dim else dim)}

    # TODO a merge + split approach would not need input_shapes
    elif isinstance(op, top.ReshapeOp):
        dims_affected = op.dims_affected(input_shapes)
        if dim in dims_affected:
            raise ValueError("Propagating through affected dims of reshape is not supported")
        # NOTE: In this case, our dim of interest stays put
        elif dim < min(dims_affected):
            return {out_cls(0): dim}

        else:  # NOTE: Finally, if it comes after the affected dims, it may increase or decrease.
            out_shape = op.infer_output_shapes(input_shapes)[0]
            diff = len(out_shape) - len(input_shapes[0])
            # NOTE: If out_shape smaller, diff will be negative, and our dim will decrease.
            # NOTE: If out_shape larger, diff will be positive, and our dim will increase.
            return {out_cls(0): dim + (-diff if in_to_out else diff)}
    elif isinstance(op, top.ReduceOp):
        dims = op.dims
        num_outputs = len(dg.get_output_shapes_list(op))
        # NOTE: If we keepdims, the dim of interest will stay in the same place.
        if not in_to_out:  # out_to_in
            # NOTE: This is how many we have to add
            dim_shift = 0 if op.keepdim else sum(d <= dim for d in dims)
            return {out_cls(i): dim + dim_shift for i in range(num_outputs)}
        else:  # in_to_out
            # assert dim not in dims, "Cannot propagate through reduce if dim is reduced"
            if dim in dims:
                return {}

            # NOTE: This is how many we have to subtract
            dim_shift = 0 if op.keepdim else sum(d < dim for d in dims)
            return {out_cls(i): dim - dim_shift for i in range(num_outputs)}

    elif isinstance(op, top.MatMulOp):
        if in_to_out:
            assert in_id is not None, "in_to_out requires an in_id"
            is_contracting_dim = is_matmul_contracting_dim(dg, op, in_id, dim)
            if is_contracting_dim:
                # raise ValueError("Cannot propagate contracting dim of matmul")
                return {}
            else:
                # NOTE: In this case, it stays in the same place
                return {out_cls(0): dim}
        else:  # out_to_in
            # NOTE: will never be contracting dim, we can just move the dim around
            out_shape = dg.get_output_shapes(op)[OpOutId(0)]
            if dim == len(out_shape) - 1:
                # NOTE: only continues over the second input
                return {OpInId(1): dim}
            elif dim == len(out_shape) - 2:
                # NOTE: only continues over the first input
                return {OpInId(0): dim}
            else:
                return {OpInId(i): dim for i in range(2)}

    elif isinstance(op, top.GatherOp):
        # NOTE: applies to both src and index
        # since shape of index ~= shape of src, except for op.dim, we eliminate op.dim
        # when it matches
        if in_to_out and dim == op.dim:
            return {}  # NOTE: dim was selected out of existence
    elif isinstance(op, top.IndexSliceOp):
        if dim == op.dim:  # NOTE: dim is sliced. It is effectively no longer the same dim.
            return {}
    elif isinstance(op, top.ExpandOp):
        if is_expanded_dim(dg, op, dim):  # Dim is expanded into existance here
            return {}
    elif isinstance(op, top.PadOp):
        if dim == op.dim:
            return {}  # NOTE: dim is padded, so it is effectively no longer the same dim.

    # Default case: propagate same dim to all inputs/outputs
    num_needed = (
        len(dg.get_output_shapes_list(op)) if in_to_out else len(dg.get_input_shapes_list(op))
    )
    return {out_cls(i): dim for i in range(num_needed)}


def propagate_dim_through_op_out_to_in(
    dg: PDG,
    op: TensorOp,
    input_shapes: Sequence[Shape],
    dim: int,
) -> Dict[OpInId, int]:
    return propagate_dim_through_op(  # type: ignore
        dg, op, input_shapes, dim, in_id=None, in_to_out=False
    )


def propagate_dim_through_op_in_to_out(
    dg: PDG,
    op: TensorOp,
    input_shapes: Sequence[Shape],
    dim: int,
    in_id: OpInId,
) -> Dict[OpOutId, int]:
    return propagate_dim_through_op(  # type: ignore
        dg, op, input_shapes, dim, in_id=in_id, in_to_out=True
    )


def recursively_follow_op_in_dim_through_dependencies(
    dg: PDG,
    op: TensorOp,
    op_in_id: OpInId,
    op_in_dim: int,
    should_recurr: Callable[[PDG, TensorOp, DependencyData, TensorOp, int, int, Any], bool],
    effect: Callable[[PDG, TensorOp, int, Dict[OpInId, int], Any], Tuple[PDG, Any]],
    initial_ctx: Any = None,
) -> Tuple[PDG, int]:
    """
    Recursively traverses the dependency graph, adjusting the dimension,
    applying the effect function to each op, and recursing into its dependencies until:
    - a. The user-defined should_recurr returns False.
    - b. The dimension is created through symbolic indexing.
    - c. The dimension is eliminated by the current op (e.g. MatMul has some dims only on one input)

    This requires a should_recurr function which takes:
      pdg, current_op, edge data, dependency to recurr to, snk_out_dim, src_out_dim
    and an effect function which takes:
      pdg, current_op, snk_out_dim, snk_in_dim

    # NOTE: This will not call effect on op. It immediately goes to it's dependency.
    # NOTE: This function is essentially a wrapper around
    # recursively_follow_op_out_dim_through_dependencies, useful for when the current op
    # eliminates the dimension of interest.
    # TODO: simplify inc_common by using this function instead of the op_out_dim version.
    """

    depy_op, depy_data = dg.get_flat_direct_dependencies(op)[op_in_id]

    effect_count = 0
    ctx = initial_ctx

    num_slices = sum(not m.is_point() for m in depy_data.expr.members)

    # NOTE: If in_dim < num_slices, then the dim of interest was just created
    # by symbolic indexing.
    # In this case, we should not recurr, as we cannot find a dim.
    depy_out_dim = op_in_dim - num_slices
    dim_created_by_symb_index = op_in_dim < num_slices

    if not dim_created_by_symb_index:
        dg, count = recursively_follow_op_out_dim_through_dependencies(
            dg, depy_op, depy_out_dim, should_recurr, effect, ctx
        )
        effect_count += count

    return dg, effect_count


def recursively_follow_op_out_dim_through_dependencies(
    dg: PDG,
    op: TensorOp,
    op_out_dim: int,
    should_recurr: Callable[[PDG, TensorOp, DependencyData, TensorOp, int, int, Any], bool],
    effect: Callable[[PDG, TensorOp, int, Dict[OpInId, int], Any], Tuple[PDG, Any]],
    initial_ctx: Any = None,
) -> Tuple[PDG, int]:
    """
    Recursively traverses the dependency graph, adjusting the dimension,
    applying the effect function to each op, and recursing into its dependencies until:
    - a. The user-defined should_recurr returns False.
    - b. The dimension is created through symbolic indexing.
    - c. The dimension is eliminated by the current op (e.g. MatMul has some dims only on one input)

    This requires a should_recurr function which takes:
      pdg, current_op, edge data, dependency to recurr to, snk_out_dim, src_out_dim
    and an effect function which takes:
      pdg, current_op, snk_out_dim, snk_in_dim
    """

    # print(f"Rec follow on {op} with {out_dim=}")
    # print(f"    Input shapes: {new_dg.get_input_shapes_list(op)}")

    op_in_dims = propagate_dim_through_op_out_to_in(
        dg, op, dg.get_input_shapes_list(op), op_out_dim
    )

    # print(f"    {in_dims=}")
    dg, ctx = effect(dg, op, op_out_dim, op_in_dims, initial_ctx)
    effect_count = 1

    for depy_op, depy_data in dg.get_flat_direct_dependencies(op):
        dim_eliminated = depy_data.sink_in_idx not in op_in_dims
        # print(f"    depy_op={depy_op}, expr={depy_data.expr}, {dim_eliminated=}")
        if not dim_eliminated:
            in_dim = op_in_dims[depy_data.sink_in_idx]
            num_slices = sum(not m.is_point() for m in depy_data.expr.members)

            # NOTE: If in_dim < num_slices, then the dim of interest was just created
            # by symbolic indexing.
            # In this case, we should not recurr, as we cannot find a dim.
            depy_out_dim = in_dim - num_slices
            dim_created_by_symb_index = in_dim < num_slices

            user_should = should_recurr(dg, op, depy_data, depy_op, op_out_dim, depy_out_dim, ctx)
            # print(f"    dim_created={dim_created_by_symb_index} {user_should=}")

            if (not dim_created_by_symb_index) and user_should:
                dg, count = recursively_follow_op_out_dim_through_dependencies(
                    dg, depy_op, depy_out_dim, should_recurr, effect, ctx
                )
                effect_count += count

    return dg, effect_count


# TODO keep going when we hit pad/shrink?
def recursively_follow_dim_through_dependents_until_elimination(
    new_dg: PDG,
    op: TensorOp,
    edge_data: DependencyData,
    src_op: TensorOp,
    in_dim: int,
    # TODO: add ctx to should_recurr
    should_recurr: Callable[[PDG, TensorOp, DependencyData, TensorOp, int, int], bool],
    effect: Callable[
        [PDG, TensorOp, DependencyData, TensorOp, Dict[OpOutId, int], int, Any], Tuple[PDG, Any]
    ],
    initial_ctx: Any = None,
) -> Tuple[PDG, int]:
    """We are propagating towards the dependents of op, and came from src_op through edge_data.

    Should recurr is a function taking:
        pdg, op to recurr to, edge data, curr op, curr op input-side dim, recurr op input-side dim
    Effect is a function taking:
        pdg, op to effect on, origin edge data, origin src_op, op output-side dim, op input-side dim
    """

    effect_count = 0

    op_out_dims = propagate_dim_through_op_in_to_out(
        new_dg, op, new_dg.get_input_shapes_list(op), in_dim, edge_data.sink_in_idx
    )

    new_dg, ctx = effect(new_dg, op, edge_data, src_op, op_out_dims, in_dim, initial_ctx)
    effect_count += 1

    if len(op_out_dims) == 0:
        return new_dg, effect_count

    for dep_op, dep_data in new_dg.get_flat_direct_dependents(op):
        num_slices = sum(not m.is_point() for m in dep_data.expr.members)

        dep_op_in_dim = op_out_dims[dep_data.src_out_idx] + num_slices

        if should_recurr(new_dg, dep_op, dep_data, op, in_dim, dep_op_in_dim):
            new_dg, count = recursively_follow_dim_through_dependents_until_elimination(
                new_dg, dep_op, dep_data, op, dep_op_in_dim, should_recurr, effect, ctx
            )
            effect_count += count

    return new_dg, effect_count


def is_window_access(e: ie.IndexAtom) -> bool:
    # Captures accesses where lb and ub are moving, with a constant window size, except
    # for when near bounds.
    if not isinstance(e, ie.Slice):
        return False
    if e.start.is_constant() or e.stop.is_constant():
        return False
    # NOTE: To catch statifyied 0:t+1 and t:T-1 cases,
    # which produce something in between a window and block access
    # NOTE: We only want t:min(t+w, T-1) and max(t-w, 0):t like cases.
    num_vars = len(e.start.vars_used())
    if num_vars > 1:
        return False
    if is_block_access(e):
        return False
    if is_all_past_access(e) or is_all_future_access(e):
        return False
    return True


def is_all_past_access(e: ie.IndexAtom) -> bool:
    # 0:t, 0:t+1, 0:t+2, ...
    # Captures accesses where lb is constant and ub is moving.
    if not isinstance(e, ie.Slice):
        return False

    if e.start.is_constant() and not e.stop.is_constant() and len(e.stop.vars_used()) == 1:
        return True
    return False


def is_all_future_access(e: ie.IndexAtom) -> bool:
    # t:T, t+1:T, t+2:T, ... t:T-5
    # Captures accesses where lb is moving and ub is constant.
    if not isinstance(e, ie.Slice):
        return False
    # slice_ub.is_constant() and slice_lb.equivalent(domain_slice_var)
    if e.stop.is_constant() and len(e.start.vars_used()) == 1:
        return True
    return False


def is_const_block_access(e: ie.IndexAtom) -> bool:
    if not isinstance(e, ie.Slice):
        return False
    if not (e.start.is_constant() and e.stop.is_constant()):
        return False
    return True


def is_range_access(e: ie.IndexAtom) -> bool:
    if isinstance(e, ie.Slice):
        return True
    return False


def is_block_access(e: ie.IndexAtom) -> bool:
    # Basic:
    # b*BS:((b+1)*BS)
    # or deeply nested:
    # b1*BS1 + b2*BS2: b1*BS1 + (b2+1)*BS2
    if not isinstance(e, ie.Slice):
        return False

    # print(f"is_block_access: {e}")
    size = e.stop - e.start
    # print(f"    size: {size}")
    simplified_size = isl_utils.simplify_int_index_value(size)
    # print(f"    simplified_size: {simplified_size}")
    is_const_int_sized = isinstance(simplified_size, ie.ConstInt)
    if not is_const_int_sized:
        return False

    # NOTE: For one of the variables in the start expression, it should be the case that
    # if we increment it, we will get the stop expression. Except for t:t+1 case
    start_vars = e.start.vars_used()
    for var in start_vars:
        if ie.struct_eq(e, ie.Slice(var, var + 1)):
            continue
        remapped_start = e.start.remap({var: var + 1})
        if ie.logical_eq(e.stop, remapped_start):
            return True
        else:
            return False

    return False


def get_block_access_var(e: ie.IndexAtom) -> Optional[ie.Symbol]:
    if not is_block_access(e):
        return None

    assert isinstance(e, ie.Slice)

    start_vars = e.start.vars_used()
    for var in start_vars:
        remapped_start = e.start.remap({var: var + 1})
        if ie.logical_eq(e.stop, remapped_start):
            return var

    return None


def is_initialization_merge(dg: PDG, op: top.MergeOp) -> bool:
    # We're looking for patterns like:
    # x[0] = ...
    # x[anything_else] = ...

    dependencies = dg.get_flat_direct_dependencies(op)
    if len(dependencies) == 2:
        branch_0_cond = dependencies[0][1].cond
        branch_1_cond = dependencies[1][1].cond
        if branch_cond_is_eq_0(dg, branch_0_cond) and (
            branch_1_cond is None
            or branch_1_cond.partial_eval(dg.static_bounds).struct_eq(ie.ConstBool(True))
        ):
            return True
    return False


def branch_cond_is_eq_0(dg: PDG, branch_0_cond: Optional[ie.IndexAtom]) -> bool:
    if branch_0_cond is None:
        return False
    return isinstance(branch_0_cond, ie.Equal) and (
        (
            branch_0_cond.right_operand.partial_eval(dg.static_bounds).struct_eq(ie.ConstInt(0))
            and isinstance(branch_0_cond.left_operand, ie.Symbol)
        )
        or (
            branch_0_cond.left_operand.partial_eval(dg.static_bounds).struct_eq(ie.ConstInt(0))
            and isinstance(branch_0_cond.right_operand, ie.Symbol)
        )
    )


def get_padding_for_slice(
    slice_of_interest: ie.Slice,
    ub_size: ie.IntIndexValueLike,
    domain_slice_var: ie.Symbol,
    # TODO: we probably don't need this as the expressions should simplify to same thing?
    is_block: bool = False,
) -> Optional[Tuple[ie.IntIndexValueLike, ie.IntIndexValueLike]]:
    """Calculate padding values for a slice based on its type and goal padded size.

    Args:
        slice_of_interest: The slice to calculate padding for
        ub_size: The goal padded size
        domain_slice_var: The domain variable (of source) being sliced

    Returns:
        Tuple of (pad_left, pad_right) values
    """
    slice_lb = slice_of_interest.start
    slice_ub = slice_of_interest.stop
    # TODO: step support
    slice_size = slice_ub - slice_lb

    slice_size = isl_utils.simplify_int_index_value(
        slice_size,
    )
    if slice_size.is_constant():
        # NOTE: This should be sufficient to cover 0:T, 5:15 and block-access cases.
        # NOTE: If the slice is of constant size, we don't need to pad to a constant size.
        return None

    if is_window_access(slice_of_interest):
        # For max/min cases like max(t-w, 0):t or t:min(t+w, T)
        pad_amount = ub_size - slice_size

        # NOTE: quick hack to handle t:min(t+w-1, T-1)+1 case
        if isinstance(slice_ub, ie.Add) and slice_ub.right_operand.is_constant():
            slice_ub = slice_ub.left_operand
        if isinstance(slice_ub, ie.Add) and slice_ub.left_operand.is_constant():
            slice_ub = slice_ub.right_operand

        if isinstance(slice_lb, ie.Max) and not isinstance(slice_ub, ie.Min):
            return (0, pad_amount)
        elif not isinstance(slice_lb, ie.Max) and isinstance(slice_ub, ie.Min):
            return (pad_amount, 0)
        else:
            # TODO when it is both, we will need to pad both sides,
            # computing how much to pad on each side based on proximity to the edge.
            raise ValueError(
                f"TODO: max(t-w1, 0):min(t+w2, T) case not handled yet: {slice_of_interest}"
            )

    elif is_all_past_access(slice_of_interest):
        # For cases like c:t or c:t+1
        if is_block:
            pad_amount = (ub_size - ((slice_size - 1) % ub_size)) - 1
        else:
            pad_amount = ub_size - slice_size
        return (0, pad_amount)

    elif is_all_future_access(slice_of_interest):
        # For t:T case
        if is_block:
            pad_amount = (ub_size - ((slice_size - 1) % ub_size)) - 1
        else:
            pad_amount = ub_size - slice_size
        return (pad_amount, 0)

    else:
        raise ValueError(f"TODO: No padding handler for slice {slice_of_interest}")


def remove_dim_from_op_domain(dg: PDG, op: top.TensorOp, dim_to_remove: ie.Symbol) -> top.TensorOp:
    # Create a new domain without the dimension to remove
    new_domain = op.domain.remove_dim(dim_to_remove)

    # Create a new operation with the updated domain
    new_op = dataclasses.replace(op, domain=new_domain, op_id=dg.get_next_op_id())

    # Create new operation data
    old_op_data = dg.ops_by_id[op.op_id]
    new_op_data = dataclasses.replace(old_op_data, op=new_op)

    # Add new op to dependency graph
    dg.insert_op(new_op_data)

    # For each dependency, create a new edge to the new op
    dim_to_remove_idx = op.domain.find_variable_index(dim_to_remove)
    # print(f"Removing {dim_to_remove} from {op}")
    # print(f"    dim_to_remove_idx: {dim_to_remove_idx}")
    # print(f"    Domain: {op.domain}")
    # print(f"    New domain: {new_domain}")
    for dep_op, dep_data in dg.get_flat_direct_dependents(op):
        e_idx = dep_data.expr.members[dim_to_remove_idx]
        assert e_idx.is_point(), f"Expected idx={dim_to_remove_idx} of {dep_data.expr} to be point."
        # Create new dependency data with updated expressions
        new_dep_data = DependencyData(
            dep_data.expr.skip_idx(dim_to_remove_idx),
            dep_data.src_out_idx,
            dep_data.sink_in_idx,
            dep_data.cond,
        )
        dg.add_edge(dep_op, new_op, new_dep_data)

    dg.move_dependencies(op, new_op)

    # Remove the old op from the graph
    dg.remove_op(op)

    return new_op


def move_dependents_two_steps(
    ctx: CompilationCtx, from_: TensorOp, to: TensorOp, through: DependencyData
) -> Sequence[DependencyData]:
    """Moves dependents of `from_` to `to`, through connecting edge `through`.
    Maintains edge correctness by combining edges."""
    dependents_to_move = list(ctx.dg.get_flat_direct_dependents(from_))

    return move_select_dependents_two_steps(ctx, from_, to, through, dependents_to_move)


def move_select_dependents_two_steps(
    ctx: CompilationCtx,
    from_: TensorOp,
    to: TensorOp,
    through: DependencyData,
    dependents_to_move: Sequence[Tuple[TensorOp, DependencyData]],
) -> Sequence[DependencyData]:
    """Moves dependents of `from_` to `to`, through connecting edge `through`.
    Maintains edge correctness by combining edges."""
    new_dep_datas = []
    for dep, dep_data in dependents_to_move:
        new_dep_data = isl_utils.combine_edges(
            dep, dep_data, from_, through, to, ctx.dg.static_bounds, ctx.analysis_ctx.isl_ctx
        )
        ctx.dg.add_edge(dep, to, new_dep_data)
        ctx.dg.remove_edge(dep, from_, dep_data)
        new_dep_datas.append(new_dep_data)
    return new_dep_datas
