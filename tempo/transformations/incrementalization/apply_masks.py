from typing import List, Tuple

from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.datatypes import OpInId, OpOutId
from tempo.core.dependence_graph import PDG, DependencyData
from tempo.core.op_tags import STATIFY_PAD_ID_TAG
from tempo.core.symbolic_tensor import SymbolicTensor, _get_symbolic_tensor_for_op_output
from tempo.transformations.incrementalization.incrementalization_common import PadInfo
from tempo.utils import isl as isl_utils
from tempo.utils import logger
from tempo.utils.dg_utils import (
    is_matmul_contracting_dim,
)

log = logger.get_logger(__name__)


def op_needs_mask(dg: PDG, op: top.TensorOp, edge_data: DependencyData, dim: int) -> bool:
    return (
        (isinstance(op, top.SumOp) and dim in op.dims)
        or (isinstance(op, top.MaxOp) and dim in op.dims)
        or (isinstance(op, top.CumSumOp) and dim == op.dim)
        or (
            isinstance(op, top.MatMulOp)
            and is_matmul_contracting_dim(dg, op, edge_data.sink_in_idx, dim)
        )
    )


def get_mask_value(op: top.TensorOp) -> float:
    target_num = None

    if isinstance(op, (top.SumOp, top.CumSumOp)):
        target_num = 0.0
    elif isinstance(op, top.MaxOp):
        target_num = -float("inf")
    elif isinstance(op, top.MatMulOp):
        target_num = 0.0
    else:
        raise NotImplementedError(f"No mask conversion implemented for {op}")
    return target_num


def apply_mask_to_edge(
    dg: PDG,
    snk_op: top.TensorOp,
    edge_data: DependencyData,
    src_op: top.TensorOp,
    dom_var: ie.Symbol,
    masked_op_in_dim: int,
    pad_infos: List[PadInfo],
) -> None:
    log.info(
        "Applying mask to edge %s -%s-> %s on dim %d, with pad_infos: %s",
        snk_op,
        edge_data,
        src_op,
        masked_op_in_dim,
        pad_infos,
    )
    mask_value = get_mask_value(snk_op)

    # apply mask by modifying pdg
    symb_t = _get_symbolic_tensor_for_op_output(dg, src_op, OpOutId(edge_data.src_out_idx))

    mask_val_symb_t = symb_t.full(mask_value, dtype=symb_t.dtype)
    for i in range(len(symb_t.shape)):
        mask_val_symb_t = mask_val_symb_t.unsqueeze(i)
    snk_in_shape = dg.get_input_shape(snk_op, edge_data.sink_in_idx)
    dim_size = snk_in_shape.at(masked_op_in_dim)
    mask_val_symb_t_expanded = mask_val_symb_t.expand(snk_in_shape)

    idxs = SymbolicTensor.arange(dim_size)

    left_right_cond_exprs: List[Tuple[ie.IntIndexValue, ie.IntIndexValue]] = []
    for pad_info in pad_infos:
        lifted_how_padding_indexed = ie.lift_to_int_ie(pad_info.pad_idx_index_expr)

        # TODO: could get this from pad_op_in_shape[pad_op.dim]
        new_slice_size = pad_info.src_inc_dim_access_expr.evaluate_shape(dg.static_bounds)[0]
        lifted_new_slice_size = ie.lift_to_int_ie(new_slice_size)
        left_padding_remapped = ie.lift_to_int_ie(pad_info.padding[0]).remap(
            {dom_var: lifted_how_padding_indexed}
        )

        new_slice_size_remapped = lifted_new_slice_size.remap({dom_var: lifted_how_padding_indexed})

        # padding_zero_remapped_simplified = isl_utils.simplify_int_index_value(
        #    padding_zero_remapped,
        #    tuple(padding_zero_remapped.vars_used()),
        #    known_symbols=dg.static_bounds,
        # )
        # new_slice_size_remapped_simplified = isl_utils.simplify_int_index_value(
        #    new_slice_size_remapped,
        #    tuple(new_slice_size_remapped.vars_used()),
        #    known_symbols=dg.static_bounds,
        # )
        left_right_cond_exprs.append((left_padding_remapped, new_slice_size_remapped))

    max_of_all_lefts = ie.max(*[x[0] for x in left_right_cond_exprs])
    simplified_max_of_all_lefts = isl_utils.simplify_int_index_value(
        max_of_all_lefts,
        known_symbols=dg.static_bounds,
    )
    # TODO: if max_of_all_lefts is just 0, we could skip it
    min_of_all_rights = ie.min(*[x[1] for x in left_right_cond_exprs])
    simplified_min_of_all_rights = isl_utils.simplify_int_index_value(
        min_of_all_rights,
        known_symbols=dg.static_bounds,
    )
    # with dg.new_temp_var() as (tau, TAU):
    #    is_pad_cond_ie = (tau < simplified_max_of_all_lefts) & (
    #        tau >= simplified_max_of_all_lefts + simplified_min_of_all_rights
    #    )
    #    #log.info("is_pad_cond_ie: %s", is_pad_cond_ie)

    # print(f"simplified_max_of_all_lefts: {simplified_max_of_all_lefts}")
    # print(f"simplified_min_of_all_rights: {simplified_min_of_all_rights}")

    lt = idxs.less_than(simplified_max_of_all_lefts)
    gt = idxs.greater_than_or_equal(simplified_max_of_all_lefts + simplified_min_of_all_rights)

    is_padding_cond = lt | gt
    for i in range(len(snk_in_shape)):
        if i != masked_op_in_dim:
            is_padding_cond = is_padding_cond.unsqueeze(i)
    is_padding_cond_expanded = is_padding_cond.expand(snk_in_shape)

    masked_symb_t = is_padding_cond_expanded.where(mask_val_symb_t_expanded, symb_t)
    # elif not ie.lift_to_int_ie(padding[1]).struct_eq(ie.ConstInt(0)):
    #    # ValToVal-based mask
    #    # TODO: in the future, we will need better condition checks for whether we can apply this
    #    # type of mask. Right now, we are assuming that just because something is padded on the
    #    # right, it is safe to apply a mask. This may not be the case.
    #    # TODO: so apparently the val-to-val masks are problematic if the schedule ends up
    #    # filling all values of the padded tensor. A fix is to add a control edge, used only in
    #    # scheduling, to ensure the padded tensor is not fully filled before the mask is applied.

    #    masked_symb_t = symb_t.nan_to_num(mask_value)

    masked_symb_t.op.tags[STATIFY_PAD_ID_TAG] = tuple(p.pad_id for p in pad_infos)

    dg.remove_edge(snk_op, src_op, edge_data)
    dg.add_edge(
        snk_op,
        masked_symb_t.op,
        DependencyData(
            edge_data.expr,
            # masked_symb_t.op.domain.basis_expr,
            OpOutId(0),
            OpInId(edge_data.sink_in_idx),
        ),
    )


def apply_val_to_val_mask_to_edge(
    dg: PDG,
    snk_op: top.TensorOp,
    edge_data: DependencyData,
    src_op: top.TensorOp,
) -> None:
    target_num = get_mask_value(snk_op)

    # apply mask by modifying pdg
    symb_t = _get_symbolic_tensor_for_op_output(dg, src_op, OpOutId(edge_data.src_out_idx))
    masked_symb_t = symb_t.nan_to_num(target_num)

    dg.remove_edge(snk_op, src_op, edge_data)
    dg.add_edge(
        snk_op,
        masked_symb_t.op,
        DependencyData(
            edge_data.expr,
            # masked_symb_t.op.domain.basis_expr,
            OpOutId(0),
            OpInId(edge_data.sink_in_idx),
        ),
    )


## def apply_mask_through_dependents_up_to_elim(
##    dg: PDG,
##    snk: top.TensorOp,
##    dep_data: DependencyData,
##    src_op: top.TensorOp,  # Should probably be the padding op.
##    dim: int,
##    start_mask: SymbolicTensor,
## ) -> Tuple[PDG, int]:
##    # Tracks the ops that eliminate dim and the mask they need.
##
##    def should_recurr(
##        dg: PDG,
##        op: top.TensorOp,
##        dep: DependencyData,
##        depy_op: top.TensorOp,
##        out_dim: int,
##        in_dim: int,
##    ) -> bool:
##        # Keep going until dim is eliminated.
##        return True
##
##    def effect(
##        dg: PDG,
##        op: top.TensorOp,
##        edge_data: DependencyData,
##        src_op: top.TensorOp,
##        op_out_dim: Dict[OpOutId, int],
##        op_in_dim: int,
##        mask: SymbolicTensor,
##    ) -> Tuple[PDG, SymbolicTensor]:
##        # NOTE: Ensures the mask is broadcastable with op.
##        for slice_ in reversed([s for s in edge_data.expr.members if not s.is_point()]):
##            assert isinstance(slice_, ie.Slice)
##            mask = mask.unsqueeze(0).expand_dim(0, slice_.evaluate_shape(dg.static_bounds)[0])
##
##        if isinstance(op, top.MovementOp):
##            if isinstance(op, top.CatOp):
##                raise NotImplementedError("CatOp not implemented")
##
##            # Apply any movement ops to the mask
##            mov_op_copy = dataclasses.replace(op, op_id=dg.get_next_op_id())
##            op_data_copy = dataclasses.replace(dg.ops_by_id[op.op_id], op=mov_op_copy)
##            dg.insert_op(op_data_copy)
##            dg.add_edge(
##                mov_op_copy,
##                mask.op,
##                DependencyData(
##                    mask.op.domain.basis_expr, OpOutId(0), OpInId(edge_data.sink_in_idx)
##                ),
##            )
##            mask = _get_symbolic_tensor_for_op_output(dg, mov_op_copy, OpOutId(0))
##        else:
##            eliminates = len(op_out_dim) == 0
##            uses_dim = (
##                eliminates
##                or (hasattr(op, "dim") and op.dim == dim)
##                or hasattr(op, "dims")
##                and dim in op.dims
##            )
##
##            if uses_dim:
##                # NOTE: We expect at this point that the mask is 1s and NaNs.
##                if isinstance(op, (top.SumOp, top.CumSumOp)):
##                    converted_mask = mask.nan_to_zero()
##                elif isinstance(op, top.MaxOp):
##                    converted_mask = mask.nan_to_neg_inf()
##                elif isinstance(op, top.MatMulOp):
##                    assert eliminates and is_matmul_contracting_dim(
##                        dg, op, edge_data.sink_in_idx, op_in_dim
##                    ), "Non-contracting matmul dims should not need mask"
##                    converted_mask = mask.nan_to_zero()
##                else:
##                    raise NotImplementedError(f"No mask conversion implemented for {op}")
##
##                # apply mask by modifying pdg
##                symb_t = _get_symbolic_tensor_for_op_output(
##                    dg, src_op, OpOutId(edge_data.src_out_idx)
##                )
##                masked_symb_t = symb_t * converted_mask
##
##                dg.remove_edge(op, src_op, edge_data)
##                dg.add_edge(
##                    op,
##                    masked_symb_t.op,
##                    DependencyData(
##                        masked_symb_t.op.domain.basis_expr,
##                        OpOutId(0),
##                        OpInId(edge_data.sink_in_idx),
##                    ),
##                )
##
##        return dg, mask
##
##    return recursively_follow_dim_through_dependents_until_elimination(
##        dg, snk, dep_data, src_op, dim, should_recurr, effect, start_mask
##    )
##
#
