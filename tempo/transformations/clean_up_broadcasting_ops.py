from dataclasses import replace
from typing import Tuple

import numpy as np

from tempo.core import global_objects as glob
from tempo.core import tensor_ops as top
from tempo.core.datatypes import OpInId, OpOutId
from tempo.core.dependence_graph import PDG, OpData
from tempo.core.domain import Domain
from tempo.core.shape import Shape
from tempo.core.symbolic_tensor import _get_symbolic_tensor_for_op_output
from tempo.transformations.compilation_pass import CompilationCtx, Transformation
from tempo.utils import logger

log = logger.get_logger(__name__)

# TODO support non-basis expressions


def _rem_elementwise_bcast(dg: PDG, op: top.TensorOp) -> bool:  # noqa: C901
    op_data = dg.ops_by_id[op.op_id]
    original_out_shape = dg.get_output_shapes(op)[OpOutId(0)]
    for depy_op, depy_data in dg.get_flat_direct_dependencies(op):
        op_input_shapes = list(dg.get_input_shapes_list(op))
        if (type(depy_op) is top.ExpandOp) and depy_data.is_unconditional_basis():
            # If a depy is expand, let's check if we can skip it and still get the same out shape
            op_input_shapes[depy_data.sink_in_idx] = dg.get_input_shape(depy_op, OpInId(0))
            # First, check if it broadcasts
            if Shape.can_broadcast(*op_input_shapes):
                bcast_shape = Shape.broadcast(*op_input_shapes)

                if bcast_shape != original_out_shape:
                    # NOTE: If the shape does not match the original, but it still broadcasts,
                    #  then we still skip the expand dependency, but now expand the output
                    # of op to match the original shape.
                    # This is because it reduces the load on this elementwise op, and may
                    # allow for further optimizations.
                    op_data.output_shapes[OpOutId(0)] = bcast_shape
                    op_symb_t = _get_symbolic_tensor_for_op_output(dg, op, OpOutId(0))
                    expanded = op_symb_t.expand(original_out_shape)
                    dg.move_dependents(op, expanded.op)
                    log.debug(
                        "Expanding output of op=%s to original shape by creating new op %s.",
                        op,
                        expanded.op,
                    )

                # Skip the expand op
                expand_depy_op, expand_depy_data = dg.get_flat_direct_dependencies(depy_op)[0]
                dg.remove_edge(op, depy_op, depy_data)
                dg.add_edge(
                    op,
                    expand_depy_op,
                    replace(expand_depy_data, sink_in_idx=depy_data.sink_in_idx),
                )
                log.debug(
                    "Skipping depy expand=%s for op=%s, connecting directly to %s",
                    depy_op,
                    op,
                    expand_depy_op,
                )
                return True

        elif (type(depy_op) is top.UnsqueezeOp) and depy_data.is_unconditional_basis():
            op_input_shapes[depy_data.sink_in_idx] = dg.get_input_shape(depy_op, OpInId(0))
            if (
                Shape.can_broadcast(*op_input_shapes)
                and Shape.broadcast(*op_input_shapes) == original_out_shape
            ):
                expand_depy_op, expand_depy_data = dg.get_flat_direct_dependencies(depy_op)[0]
                dg.remove_edge(op, depy_op, depy_data)
                dg.add_edge(
                    op,
                    expand_depy_op,
                    replace(expand_depy_data, sink_in_idx=depy_data.sink_in_idx),
                )
                log.debug("Skipping depy unsqueeze=%s for op=%s.", depy_op, op)
                return True
        elif (
            (type(depy_op) is top.ConstOp)
            and depy_data.is_unconditional_basis()
            and depy_op.is_uniform
            and not depy_op.shape.is_scalar()
        ):
            op_input_shapes[depy_data.sink_in_idx] = Shape.scalar()
            if Shape.can_broadcast(*op_input_shapes):
                bcast_shape = Shape.broadcast(*op_input_shapes)
                if bcast_shape != original_out_shape:
                    op_data.output_shapes[OpOutId(0)] = bcast_shape
                    op_symb_t = _get_symbolic_tensor_for_op_output(dg, op, OpOutId(0))

                    ## NOTE: if expand is not needed, this will return the original op
                    expanded = op_symb_t.expand(original_out_shape)

                    dg.move_dependents(op, expanded.op)
                    log.debug(
                        "Expanding output of op=%s to original shape by creating new op %s.",
                        op,
                        expanded.op,
                    )
                # We can skip this op.
                new_op = top.ConstOp(
                    dg.get_next_op_id(),
                    Domain.empty(),
                    depy_op.tags,
                    Shape.scalar(),
                    depy_op.dtype,
                    np.array(depy_op.uniform_value),
                    is_uniform=True,
                )
                log.debug("Turning uniform const op %s into scalar const op %s.", depy_op, new_op)
                op_data = OpData(new_op, {OpOutId(0): new_op.shape}, {OpOutId(0): new_op.dtype})
                dg.insert_op(op_data)
                dg.remove_edge(op, depy_op, depy_data)
                dg.add_edge(op, new_op, depy_data)

                return True

    return False


def _rem_matmul_bcast(dg: PDG, op: top.MatMulOp) -> bool:
    op_op_data = dg.ops_by_id[op.op_id]
    for depy_op, depy_data in dg.get_flat_direct_dependencies(op):
        original_out_shape = dg.get_output_shapes(op)[OpOutId(0)]
        if (
            (type(depy_op) is top.ExpandOp)
            and depy_data.is_unconditional_basis()
            and all(
                d < len(original_out_shape) - 2
                for d in depy_op.dims_affected(tuple(dg.get_input_shapes_list(depy_op)))
            )
        ):
            in_shapes = dg.get_input_shapes(op)
            new_input_shapes = [in_shapes[OpInId(i)] for i in range(op.num_inputs)]
            new_input_shapes[depy_data.sink_in_idx] = dg.get_input_shape(depy_op, OpInId(0))
            inferred_shape = op.infer_output_shapes(tuple(new_input_shapes))[0]
            if inferred_shape != original_out_shape:
                ## NOTE: Have to change shape b4 get_symbolic_tensor_for_op_output
                op_op_data.output_shapes[OpOutId(0)] = inferred_shape
                op_symb_t = _get_symbolic_tensor_for_op_output(dg, op, OpOutId(0))
                expanded = op_symb_t.expand(original_out_shape)
                dg.move_dependents(op, expanded.op)

            true_dep_op, true_dep_data = dg.get_flat_direct_dependencies(depy_op)[0]
            dg.remove_edge(op, depy_op, depy_data)
            dg.add_edge(
                op,
                true_dep_op,
                replace(true_dep_data, sink_in_idx=depy_data.sink_in_idx),
            )
            return True

    return False


class CleanUpBroadcastingOps(Transformation):
    """This transformations removes unnecessary broadcasting ops in the DG."""

    def __init__(self, ctx: CompilationCtx) -> None:
        super().__init__(ctx)

    def _run(self) -> Tuple[PDG, bool]:  # noqa: C901
        new_dg = self.ctx.dg
        glob.set_active_dg(new_dg)

        elem_count = 0
        matmul_count = 0
        has_changed = True
        while has_changed:
            has_changed = False

            for op in list(new_dg.nodes):
                if op.num_inputs > 1:
                    if isinstance(op, top.ElementWiseOp):
                        if _rem_elementwise_bcast(new_dg, op):
                            has_changed = True
                            elem_count += 1
                    elif isinstance(op, top.MatMulOp):
                        if _rem_matmul_bcast(new_dg, op):
                            has_changed = True
                            matmul_count += 1

        log.info("Removed %s unnecessary broadcasting connections.", elem_count)
        log.info("Removed %s unnecessary broadcasting connections for MatMulOps.", matmul_count)
        self.ctx.analysis_ctx._broadcast_elim_has_run = True
        return new_dg, has_changed
