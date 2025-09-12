import dataclasses

import numpy as np

from tempo.core import index_expr as ie
from tempo.core.datatypes import OpInId, OpOutId
from tempo.core.dependence_graph import PDG, OpData
from tempo.core.domain import Domain
from tempo.core.shape import Shape, StaticShape
from tempo.core.tensor_ops import ConstOp, EvalSymbolOp, ExpandOp, IndexSliceOp, PadOp, ReshapeOp
from tempo.transformations.compilation_pass import Transformation
from tempo.utils import isl, logger

log = logger.get_logger(__name__)


class Statify(Transformation):
    """This transformation finds ops that have index exprs in fields,
    and tries to resolve them to static shapes, or integer constants.

    Args:
        Transformation (_type_): _description_
    """

    def _run(self) -> tuple[PDG, bool]:  # noqa: C901
        new_dg = self.ctx.dg  # self.copy_dg()
        count = 0

        for op_data in list(new_dg.node_datas):
            op = op_data.op
            if isinstance(op, EvalSymbolOp) and op.symbol in new_dg.static_bounds.keys():
                val = new_dg.static_bounds[op.symbol]
                op_id = new_dg.get_next_op_id()
                new_op = ConstOp(
                    op_id,
                    Domain.empty(),
                    op.tags,
                    value=np.asarray(val),
                    shape=op.shape,
                    dtype=op.dtype,
                    is_uniform=True,
                )
                new_op_data = dataclasses.replace(op_data, op=new_op)
                new_dg.insert_op(new_op_data)
                new_dg.move_connections(op, new_op)
                new_dg.remove_op(op)
                op_data = new_op_data
                count += 1
            if isinstance(op, ReshapeOp):
                if not isinstance(op.shape, StaticShape):
                    new_shape = op.shape.try_resolve(new_dg.static_bounds)
                    if isinstance(new_shape, StaticShape):
                        op_id = new_dg.get_next_op_id()
                        new_reshape = ReshapeOp(
                            op_id,
                            op.domain.copy(),
                            op.tags,
                            new_shape,
                        )
                        new_op_data = OpData(
                            new_reshape,
                            {OpOutId(0): new_shape},
                            op_data.output_dtypes,
                        )
                        new_dg.insert_op(new_op_data)
                        new_dg.move_connections(op, new_reshape)
                        new_dg.remove_op(op)
                        op_data = new_op_data
                        count += 1
                        log.debug("Statified reshape %s to %s", op, new_reshape)
            if isinstance(op, ExpandOp):
                if not isinstance(op.sizes, StaticShape):
                    new_sizes = op.sizes.try_resolve(new_dg.static_bounds)
                    if isinstance(new_sizes, StaticShape):
                        op_id = new_dg.get_next_op_id()
                        new_expand = ExpandOp(
                            op_id,
                            op.domain.copy(),
                            op.tags,
                            new_sizes,
                        )
                        new_op_data = OpData(
                            new_expand,
                            {OpOutId(0): new_sizes},
                            op_data.output_dtypes,
                        )
                        new_dg.insert_op(new_op_data)
                        new_dg.move_connections(op, new_expand)
                        new_dg.remove_op(op)
                        op_data = new_op_data
                        count += 1
                        log.debug("Statified expand %s to %s", op, new_expand)
            if isinstance(op, IndexSliceOp):
                slice_op = op
                if not isinstance(slice_op.length, int):
                    length = (
                        Shape.from_((slice_op.length,))
                        .simplify()
                        .try_resolve(new_dg.static_bounds)
                        ._shape[0]
                    )

                    if length != slice_op.length:
                        op_id = new_dg.get_next_op_id()
                        new_slice = dataclasses.replace(slice_op, op_id=op_id, length=length)
                        input_shape = new_dg.get_input_shape(slice_op, OpInId(0))
                        input_shape_start = new_dg.get_input_shape(slice_op, OpInId(1))
                        new_op_data = OpData(
                            new_slice,
                            {
                                OpOutId(0): new_slice.infer_output_shapes(
                                    (input_shape, input_shape_start)
                                )[0]
                            },
                            op_data.output_dtypes,
                        )
                        new_dg.insert_op(new_op_data)
                        new_dg.move_connections(slice_op, new_slice)
                        new_dg.remove_op(slice_op)
                        op_data = new_op_data
                        count += 1
                        log.debug("Statified index slice %s to %s", op, new_slice)
            if isinstance(op, PadOp):
                pad_op = op
                if not pad_op.is_static():
                    simplified_pad_left = pad_op.padding[0]
                    if not isinstance(simplified_pad_left, int):
                        simplified_pad_left = simplified_pad_left.partial_eval(new_dg.static_bounds)
                        simplified_pad_left = isl.simplify_int_index_value(
                            simplified_pad_left,
                            known_symbols=new_dg.static_bounds,
                        )
                    simplified_pad_right = pad_op.padding[1]
                    if not isinstance(simplified_pad_right, int):
                        simplified_pad_right = simplified_pad_right.partial_eval(
                            new_dg.static_bounds
                        )
                        simplified_pad_right = isl.simplify_int_index_value(
                            simplified_pad_right,
                            known_symbols=new_dg.static_bounds,
                        )
                    if not ie.lift_to_int_ie(simplified_pad_left).struct_eq(
                        ie.lift_to_int_ie(pad_op.padding[0])
                    ) or not ie.lift_to_int_ie(simplified_pad_right).struct_eq(
                        ie.lift_to_int_ie(pad_op.padding[1])
                    ):
                        input_shape = new_dg.get_input_shape(pad_op, OpInId(0))

                        op_id = new_dg.get_next_op_id()
                        new_pad = PadOp(
                            op_id,
                            pad_op.domain.copy(),
                            pad_op.tags,
                            (
                                int(simplified_pad_left)
                                if isinstance(simplified_pad_left, int)
                                else simplified_pad_left,
                                int(simplified_pad_right)
                                if isinstance(simplified_pad_right, int)
                                else simplified_pad_right,
                            ),
                            pad_op.dim,
                            pad_op.mode,
                            pad_op.value,
                        )
                        new_op_data = OpData(
                            new_pad,
                            {OpOutId(0): new_pad.infer_output_shapes((input_shape,))[0]},
                            op_data.output_dtypes,
                        )
                        new_dg.insert_op(new_op_data)
                        new_dg.move_connections(pad_op, new_pad)
                        new_dg.remove_op(pad_op)
                        op_data = new_op_data
                        count += 1
                        log.debug("Statified pad %s to %s", op, new_pad)
            for k, s in op_data.output_shapes.items():
                if not isinstance(s, StaticShape):
                    new_shape = s.try_resolve(new_dg.static_bounds)
                    if isinstance(new_shape, StaticShape):
                        op_data.output_shapes[k] = new_shape
                        count += 1
                        log.debug("Statified output shape %s of %s to %s", s, op, new_shape)
        statified_edges = 0
        for snk, src, data in new_dg.get_all_edges():
            expr = data.expr
            if len(expr.bound_symbols_used()) == 0:
                continue

            new_expr = isl.simplify_dependence_expr(
                expr,
                snk.domain,
                src.domain,
                data.cond,
                new_dg.static_bounds,
                self.ctx.analysis_ctx.isl_ctx,
            )
            if new_expr.struct_eq(data.expr):
                continue

            statified_edges += 1
            log.debug("Statified edge %s -> %s from %s to %s", snk, src, data.expr, new_expr)
            new_edge_data = dataclasses.replace(data, expr=new_expr)

            new_dg.remove_edge(snk, src, data)
            new_dg.add_edge(snk, src, new_edge_data)

        log.info(
            "Resolved %s dynamic objects to static. Statified %s edges.", count, statified_edges
        )

        return new_dg, count + statified_edges > 0
