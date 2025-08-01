import dataclasses
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from tempo.core import global_objects as glob
from tempo.core import index_expr as ie
from tempo.core import isl_types as islt
from tempo.core import tensor_ops as top
from tempo.core.datatypes import OpInId, OpOutId, TensorId
from tempo.core.dependence_graph import PDG, DependencyData, OpData
from tempo.core.op_tags import STATIFY_PAD_ID_TAG
from tempo.core.symbolic_tensor import _get_symbolic_tensor_for_op_output
from tempo.transformations.compilation_pass import CompilationCtx, CompilationPass
from tempo.transformations.optimizer.dead_code_elimination import DeadCodeElimination
from tempo.utils import isl as isl_utils
from tempo.utils import logger
from tempo.utils.dg_utils import (
    is_expanded_dim,
    recursively_follow_dim_through_dependents_until_elimination,
    recursively_follow_op_in_dim_through_dependencies,
)

log = logger.get_logger(__name__)

MASK_TYPE = Union[top.ValToValOp, top.WhereOp]

MASK_VALUE_TYPE = Union[float, int, bool]

# NOTE: The main idea is to find convertible pads, ie where all affected masks share the same value.
# Then based on that, we compute no conflict masks, masks where all affecting pads have a
# consistent value.
# We can always remove all pads by folding.
# But masks with non-convertible pads cannot be removed, as the folded pad may not match
# the mask value.

# TODO: and even more so, we can convert the remaining masks to val-to-val ops,
# if they are valid for adding control edges.

# TODO: Need to add to can_remove_mask,
# a way to ensure the path from the mask to the pad does not involve unsupported ops:
# adds with dynamic values and such.
# For constant adds, we could try to statically infer the correct prealloc pad value by subtracting
# the const from the default mask. This affects only zero pad, not inf pad.

# TODO: for all unique paths between mask and pad, ensure only valid ops. Invalid ops:
# - dynami


# TODO: instead of using tags, use this
def find_mask_affecting_pads(dg: PDG, mask_op: MASK_TYPE) -> List[top.PadOp]:
    affecting_pads: Set[top.PadOp] = set()
    masked_dim = 0  # TODO how to find this??

    # TODO: one idea, though brittle, is to get the recursive dependencies on inID(0), reverse them,
    # find the first expandOp, and infer from the position of the non-1 dim
    # Let's come back to this after trying the other thing.
    def should_recurr(
        dg: PDG,
        op: top.TensorOp,
        depy_data: DependencyData,
        depy_op: top.TensorOp,
        op_out_dim: int,
        depy_out_dim: int,
        ctx: Any,
    ) -> bool:
        if isinstance(op, top.PadOp) and op.dim == op_out_dim and op.is_mask_pad():
            return False
        if is_expanded_dim(dg, op, op_out_dim):
            return False
        return True

    # PDG, TensorOp, int, Dict[OpInId, int]
    def effect(
        dg: PDG,
        op: top.TensorOp,
        op_out_dim: int,
        op_in_dims: Dict[OpInId, int],
        ctx: Any,
    ) -> Tuple[PDG, Any]:
        if isinstance(op, top.PadOp) and op.dim == op_out_dim and op.is_mask_pad():
            affecting_pads.add(op)

        return dg, ctx

    recursively_follow_op_in_dim_through_dependencies(
        dg,
        mask_op,
        OpInId(2),  # Original input
        masked_dim,
        should_recurr,
        effect,
    )

    return list(affecting_pads)


def _get_sched_and_phys_expr_for_combinable_pad_op(
    new_dg: PDG, pad_op: top.PadOp, isl_ctx: islt.Context
) -> Tuple[ie.IndexSequence, ie.IndexSequence, DependencyData]:
    ((dep_op, dep_data),) = new_dg.get_flat_direct_dependents(pad_op)
    ((depy_op, depy_data),) = new_dg.get_flat_direct_dependencies(pad_op)
    t_index = _get_pad_dim_temporal_index(depy_data, pad_op)
    t_var = depy_op.domain.variables[t_index]

    pad_left = pad_op.padding[0]
    pad_right = pad_op.padding[1]

    # NOTE: this is going to be our "isl_expr"
    combined_edge = isl_utils.combine_edges(
        dep_op, dep_data, pad_op, depy_data, depy_op, ctx=isl_ctx
    )
    assert combined_edge.cond is None
    isl_sched_expr = combined_edge.expr

    slice_access = combined_edge.expr.members[t_index]
    assert isinstance(slice_access, ie.Slice)
    lb = slice_access.start
    ub = slice_access.stop

    remmapped_lb = lb.remap({t_var: t_var - pad_left})
    remmapped_ub = ub.remap({t_var: t_var + pad_right})

    new_slice = ie.slice_(remmapped_lb, remmapped_ub)
    if isinstance(new_slice, ie.Slice):
        new_slice = isl_utils.simplify_slice(
            new_slice,
            known_symbols=new_dg.static_bounds,
            ctx=isl_ctx,
        )
    physical_access_expr = combined_edge.expr.replace_idx(t_index, new_slice)

    return isl_sched_expr, physical_access_expr, combined_edge


def _get_pad_dim_temporal_index(pad_depy_data: DependencyData, pad_op: top.PadOp) -> int:
    pad_dim = pad_op.dim
    num_slices_in_src_expr = sum(1 for m in pad_depy_data.expr if isinstance(m, ie.Slice))
    assert pad_dim < num_slices_in_src_expr, (
        f"PadOp must pad a temporal slice, {pad_dim=}, {pad_depy_data.expr=}"
    )

    t_index = None
    slice_count = 0
    for i, m in enumerate(pad_depy_data.expr):
        if isinstance(m, ie.Slice):
            if slice_count == pad_dim:
                t_index = i
                break
            slice_count += 1
    assert t_index is not None, f"Could not find T index in {pad_depy_data.expr=}"
    return t_index


def _get_mask_value(new_dg: PDG, mask_op: top.TensorOp) -> float:
    if isinstance(mask_op, top.ValToValOp):
        return float(mask_op.out_val)
    elif isinstance(mask_op, top.WhereOp):
        rec_depys = new_dg.get_flat_recursive_dependencies(
            mask_op,
            start_from_input=OpInId(1),  # 1->mask_value
        )
        mask_const_op = rec_depys[-1][0]
        assert isinstance(mask_const_op, top.ConstOp) and mask_const_op.is_uniform
        mask_val = mask_const_op.uniform_value
        return mask_val
    else:
        raise ValueError(f"Unknown mask op type: {mask_op}")


def is_mask_op(op: top.TensorOp) -> bool:
    return (isinstance(op, top.ValToValOp) or isinstance(op, top.WhereOp)) and op.tags.get(
        STATIFY_PAD_ID_TAG, None
    ) is not None


def is_valid_for_control_edge(mask_op: MASK_TYPE, affecting_pad_ops: Set[top.PadOp]) -> bool:
    for pad_op in affecting_pad_ops:
        if not ie.struct_eq(pad_op.padding[0], 0):
            return False
    return True


def has_invalid_ops_between_mask_and_pads(
    dg: PDG, mask_op: MASK_TYPE, affecting_pad_ops: Set[top.PadOp]
) -> bool:
    for pad_op in affecting_pad_ops:
        paths = dg.get_paths_between(mask_op, pad_op)
        for path in paths:
            for op in path[1:-1]:  # Skip mask_op and pad_op
                if not isinstance(op, (top.MovementOp)):
                    return True
    return False


def can_remove_mask(
    dg: PDG,
    mask_op: MASK_TYPE,
    masks_with_consistent_pad_conversions: List[MASK_TYPE],
    affecting_pad_ops: Set[top.PadOp],
    pad_conversions_and_values: Dict[top.PadOp, MASK_VALUE_TYPE],
) -> bool:
    # NOTE: All affecting pads must have a consistent pad conversion.
    if mask_op not in masks_with_consistent_pad_conversions:
        return False

    # NOTE: All affecting pads must be converted, ie, they will prealloc the needed value
    if not all(pad_op in pad_conversions_and_values for pad_op in affecting_pad_ops):
        return False

    if not is_valid_for_control_edge(mask_op, affecting_pad_ops):
        return False

    if has_invalid_ops_between_mask_and_pads(dg, mask_op, affecting_pad_ops):
        return False

    return True


def _remove_mask(
    new_dg: PDG, mask_op: MASK_TYPE, affecting_pad_ops: Set[top.PadOp], isl_ctx: islt.Context
) -> None:
    if isinstance(mask_op, top.ValToValOp):
        ((non_masked_op, non_masked_data),) = new_dg.get_flat_direct_dependencies(mask_op)
    elif isinstance(mask_op, top.WhereOp):
        (_, _, (non_masked_op, non_masked_data)) = new_dg.get_flat_direct_dependencies(mask_op)
    else:
        raise ValueError(f"Unknown mask op type: {mask_op}")

    for mask_dep_op, mask_dep_data in new_dg.get_flat_direct_dependents(mask_op):
        combined_edge = isl_utils.combine_edges(
            mask_dep_op,
            mask_dep_data,
            mask_op,
            non_masked_data,
            non_masked_op,
            ctx=isl_ctx,
        )
        for pad_op in affecting_pad_ops:
            ((pad_depy_op, pad_depy_data),) = new_dg.get_flat_direct_dependencies(pad_op)

            # TODO: do we even need this? Could we just use pad_depy_op.basis_expr?
            _, physical_access_expr, _ = _get_sched_and_phys_expr_for_combinable_pad_op(
                new_dg, pad_op, isl_ctx
            )
            t_index = _get_pad_dim_temporal_index(pad_depy_data, pad_op)
            t_var = pad_depy_op.domain.variables[t_index]

            ub_eval = int(t_var.as_bound().partial_eval(new_dg.static_bounds))
            padded_access = pad_depy_data.expr.members[t_index]
            assert isinstance(padded_access, ie.Slice)
            must_be_mask_val_slice = ie.slice_(padded_access.stop, ub_eval)

            e = physical_access_expr.replace_idx(t_index, must_be_mask_val_slice)

            e_rev = isl_utils.reverse_dependence_expr(e, mask_dep_op.domain, pad_depy_op.domain)

            control_edge = DependencyData.make_control(e_rev)

            # NOTE: snk is pad_depy_op
            log.info("Adding control edge from %s to %s with %s", pad_depy_op, mask_dep_op, e_rev)
            new_dg.add_edge(pad_depy_op, mask_dep_op, control_edge)
        new_dg.add_edge(mask_dep_op, non_masked_op, combined_edge)

    new_dg.remove_op(mask_op)


def find_masks_for_pad_op(
    dg: PDG,
    snk: top.TensorOp,
    dep_data: DependencyData,
    pad_op: top.PadOp,
    dim: int,
    pad_mask_uuid: str,
) -> List[MASK_TYPE]:
    masks: List[MASK_TYPE] = []

    def should_recurr(
        dg: PDG,
        op: top.TensorOp,
        dep: DependencyData,
        depy_op: top.TensorOp,
        out_dim: int,
        in_dim: int,
    ) -> bool:
        # Keep going until dim is eliminated.
        return True

    def effect(
        dg: PDG,
        op: top.TensorOp,
        edge_data: DependencyData,
        src_op: top.TensorOp,
        op_out_dim: Dict[OpOutId, int],
        op_in_dim: int,
        state: Any,
    ) -> Tuple[PDG, Any]:
        print(f"Path from {pad_op}: {op}")
        if is_mask_op(src_op) and pad_mask_uuid in src_op.tags.get(STATIFY_PAD_ID_TAG, []):
            assert isinstance(src_op, (top.ValToValOp, top.WhereOp))
            masks.append(src_op)
        return dg, state

    recursively_follow_dim_through_dependents_until_elimination(
        dg, snk, dep_data, pad_op, dim, should_recurr, effect
    )
    return masks


class FoldPadsNMasksIntoStorage(CompilationPass):
    """The goal with this transformation is to identify opportunities to remove padding and masking
    operations that can already be fulfilled by a source which will be preallocated with
    the padded size and filled with the mask value.

    E.g.:
    elim_op --> mask --> snk --PW--> pad() --block-access--> src
                 =>
    elim_op --> snk  --big-block-access--> src (alloc with mask value)

    """

    def __init__(self, ctx: CompilationCtx):
        super().__init__(ctx)

    def _run(self) -> Tuple[CompilationCtx, bool]:
        new_dg = self.ctx.dg
        masks_removed = 0
        pads_removed = 0
        pads_converted = 0

        if self.ctx.analysis_ctx._tensor_prealloc_value is None:
            self.ctx.analysis_ctx._tensor_prealloc_value = {}
        ops = list(new_dg.nodes)

        masking_pads = [op for op in ops if isinstance(op, top.PadOp) and op.is_mask_pad()]
        mask_ops: List[MASK_TYPE] = [op for op in ops if is_mask_op(op)]  # type: ignore

        (
            pad_conversions_and_values,
            mask_ops_with_all_affecting_pads_converted,  # type: ignore
            pad_op_to_affected_mask_ops,
        ) = self._get_pad_conversions_and_no_conflict_masks(new_dg, masking_pads, mask_ops)

        # Just invert pad_op_to_affected_mask_ops
        mask_op_to_affecting_pad_ops: Dict[MASK_TYPE, Set[top.PadOp]] = {}
        for pad_op, mask_ops in pad_op_to_affected_mask_ops.items():
            for mask_op in mask_ops:
                mask_op_to_affecting_pad_ops.setdefault(mask_op, set()).add(pad_op)

        # NOTE: These are masks where
        for mask_op in mask_ops_with_all_affecting_pads_converted:
            if self.ctx.exec_cfg.enable_pad_mask_removal and can_remove_mask(
                new_dg,
                mask_op,
                mask_ops_with_all_affecting_pads_converted,
                mask_op_to_affecting_pad_ops[mask_op],
                pad_conversions_and_values,
            ):
                _remove_mask(
                    new_dg,
                    mask_op,
                    mask_op_to_affecting_pad_ops[mask_op],
                    self.ctx.analysis_ctx.isl_ctx,
                )
                masks_removed += 1

        if self.ctx.exec_cfg.enable_fold_pads_into_storage:
            # - 1. Remove any pads.
            any_pads = [
                op for op in ops if isinstance(op, top.PadOp) and op.mode == top.PadMode.ANY
            ]
            for pad_op in any_pads:
                self._remove_pad(new_dg, pad_op)
                pads_removed += 1

            # - 2. For each convertible pad
            for pad_op, mask_val in pad_conversions_and_values.items():
                # - 3. Remove pad, ensuring tensor preallocated with mask value.
                self._remove_pad(new_dg, pad_op, mask_val)
                pads_removed += 1

            # - 4. Separately, remove non-convertible pads: affected masks cannot be removed.
            non_convertible_pads = [
                op for op in masking_pads if op not in pad_conversions_and_values.keys()
            ]
            for pad_op in non_convertible_pads:
                self._remove_pad(new_dg, pad_op)
                pads_removed += 1
        else:
            conversions_done = self._convert_pads(new_dg, pad_conversions_and_values)
            pads_converted = len(conversions_done)

        new_ctx, _, _ = DeadCodeElimination(self.ctx).run()
        new_dg = new_ctx.dg

        glob.set_active_dg(new_dg)

        log.info(
            "Removed %d masks, %d pads, %d pads converted",
            masks_removed,
            pads_removed,
            pads_converted,
        )

        new_ctx = CompilationCtx(new_dg, self.ctx.analysis_ctx, self.ctx.exec_cfg)
        return new_ctx, masks_removed > 0 or pads_removed > 0

    def _convert_pads(
        self, new_dg: PDG, pad_conversions: Dict[top.PadOp, MASK_VALUE_TYPE]
    ) -> Dict[top.PadOp, top.PadOp]:
        pad_op_conversions = {}
        for pad_op, mask_val in pad_conversions.items():
            replacement_pad_op = dataclasses.replace(
                pad_op, op_id=new_dg.get_next_op_id(), value=mask_val
            )
            pad_op_conversions[pad_op] = replacement_pad_op
            new_dg.replace_op(pad_op, replacement_pad_op)
        return pad_op_conversions

    def _get_pad_conversions_and_no_conflict_masks(
        self, new_dg: PDG, mask_pads: List[top.PadOp], mask_ops: List[MASK_TYPE]
    ) -> Tuple[Dict[top.PadOp, MASK_VALUE_TYPE], List[MASK_TYPE], Dict[top.PadOp, Set[MASK_TYPE]]]:
        # print("--------------------------------")
        # print(f"All mask pads ({len(mask_pads)}): {mask_pads}")
        # print("--------------------------------")
        # print(f"All mask ops ({len(mask_ops)}): {mask_ops}")
        # print("--------------------------------")
        id_to_pad = {pad_op.tags[STATIFY_PAD_ID_TAG]: pad_op for pad_op in mask_pads}
        pad_id_to_mask_values: Dict[str, Set[MASK_VALUE_TYPE]] = defaultdict(set)
        pad_id_to_mask_value_counts: Dict[str, Dict[MASK_VALUE_TYPE, int]] = defaultdict(
            lambda: defaultdict(lambda: 0)
        )
        pad_id_to_mask_ops: Dict[str, List[MASK_TYPE]] = defaultdict(list)
        pad_op_to_affected_mask_ops: Dict[top.PadOp, Set[MASK_TYPE]] = defaultdict(set)

        for mask_op in mask_ops:
            for pad_id in mask_op.tags[STATIFY_PAD_ID_TAG]:
                mask_val = _get_mask_value(new_dg, mask_op)
                pad_id_to_mask_values[pad_id].add(mask_val)
                pad_id_to_mask_value_counts[pad_id][mask_val] += 1
                pad_id_to_mask_ops[pad_id].append(mask_op)
                pad_op_to_affected_mask_ops[id_to_pad[pad_id]].add(mask_op)

        # print(f"pad_id_to_mask_values: {pad_id_to_mask_values}")
        # print(f"pad_id_to_mask_value_counts: {pad_id_to_mask_value_counts}")

        pad_conversions_and_values: Dict[top.PadOp, MASK_VALUE_TYPE] = {}
        for pad_id, mask_value_counts in pad_id_to_mask_value_counts.items():
            if len(mask_value_counts) == 1:
                pad_conversions_and_values[id_to_pad[pad_id]] = next(iter(mask_value_counts.keys()))
            else:
                most_common_value = max(mask_value_counts.items(), key=lambda x: x[1])[0]
                pad_conversions_and_values[id_to_pad[pad_id]] = most_common_value

        mask_ops_with_all_affecting_pads_converted_to_mask_val = []
        for mask_op in mask_ops:
            affecting_pad_ids = mask_op.tags[STATIFY_PAD_ID_TAG]
            mask_val = _get_mask_value(new_dg, mask_op)
            all_affecting_pads_converted_to_mask_val = all(
                id_to_pad[pad_id] in pad_conversions_and_values
                and pad_conversions_and_values[id_to_pad[pad_id]] == mask_val
                for pad_id in affecting_pad_ids
            )
            if all_affecting_pads_converted_to_mask_val:
                # assert all(
                #    pad_conversions_and_values[id_to_pad[pad_id]] == mask_val
                #    for pad_id in affecting_pad_ids
                # )
                mask_ops_with_all_affecting_pads_converted_to_mask_val.append(mask_op)
        return (
            pad_conversions_and_values,
            mask_ops_with_all_affecting_pads_converted_to_mask_val,
            pad_op_to_affected_mask_ops,
        )

    def _remove_pad(
        self, new_dg: PDG, pad_op: top.PadOp, mask_value: Optional[float] = None
    ) -> None:
        ((dep_op, dep_data),) = new_dg.get_flat_direct_dependents(pad_op)
        ((depy_op, depy_data),) = new_dg.get_flat_direct_dependencies(pad_op)

        pad_left = pad_op.padding[0]
        pad_right = pad_op.padding[1]

        t_index = _get_pad_dim_temporal_index(depy_data, pad_op)
        t_var = depy_op.domain.variables[t_index]

        if isl_utils.can_combine_edges(dep_op, dep_data, pad_op, depy_data, depy_op):
            isl_sched_expr, physical_access_expr, combined_edge = (
                _get_sched_and_phys_expr_for_combinable_pad_op(
                    new_dg, pad_op, self.ctx.analysis_ctx.isl_ctx
                )
            )

            new_slice = physical_access_expr.members[t_index]
            sh = new_slice.evaluate_shape(new_dg.static_bounds)[0]
            sh = isl_utils.simplify_int_index_value(sh, known_symbols=new_dg.static_bounds)

            assert isinstance(sh, ie.ConstInt), f"Shape is not static: {sh}"

            new_data = dataclasses.replace(
                combined_edge, _isl_expr=isl_sched_expr, expr=physical_access_expr
            )
            new_dg.add_edge(dep_op, depy_op, new_data)
        else:
            # Cannot combine edges
            # Will need to insert an ident op
            isl_sched_expr = depy_data.expr
            slice_access = depy_data.expr.members[t_index]
            assert isinstance(slice_access, ie.Slice)
            lb = slice_access.start
            ub = slice_access.stop
            remmapped_lb = lb.remap({t_var: t_var - pad_left})
            remmapped_ub = ub.remap({t_var: t_var + pad_right})

            new_slice = ie.slice_(remmapped_lb, remmapped_ub)
            if isinstance(new_slice, ie.Slice):
                new_slice = isl_utils.simplify_slice(
                    new_slice,
                    known_symbols=new_dg.static_bounds,
                )
            physical_access_expr = depy_data.expr.replace_idx(t_index, new_slice)
            new_data = dataclasses.replace(
                depy_data, _isl_expr=isl_sched_expr, expr=physical_access_expr
            )
            ident_op = top.IdentOp(new_dg.get_next_op_id(), depy_op.domain, pad_op.tags)
            pad_symb_t = _get_symbolic_tensor_for_op_output(new_dg, pad_op, OpOutId(0))
            ident_op_data = OpData(
                ident_op,
                output_shapes={OpOutId(0): pad_symb_t.spatial_shape},
                output_dtypes={OpOutId(0): pad_symb_t.dtype},
            )
            new_dg.insert_op(ident_op_data)
            new_dg.add_edge(ident_op, depy_op, new_data)
            new_dg.add_edge(dep_op, ident_op, dep_data)

        new_dg.remove_op(pad_op)

        if mask_value is not None:
            tensor_id = TensorId(depy_op.op_id, depy_data.src_out_idx)
            self.ctx.analysis_ctx.tensor_prealloc_value[tensor_id] = mask_value
