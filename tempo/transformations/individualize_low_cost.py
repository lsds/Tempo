import dataclasses

from tempo.core import tensor_ops as top
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.dependence_graph import PDG, OpData
from tempo.transformations.compilation_pass import Transformation
from tempo.utils import logger

log = logger.get_logger(__name__)

DEFAULT_OPS_TO_INDIVIDUALIZE = (
    top.ConstOp,
    top.MovementOp,
    top.IndexSelectOp,
    # NOTE: To allow for more vectorization opportunities, we want to individualize
    # EvalSymbolOp. But, when we go into code-gen, we do not want them individualized.
)

DEFAULT_EXCEPTION_OPS = (top.SplitOp,)


# def is_static_op(dg: PDG, op: top.TensorOp) -> bool:
#    all_shapes = list(dg.get_input_shapes_list(op)) + list(dg.get_output_shapes_list(op))
#    result = all(shape.is_static() for shape in all_shapes)
#    return result


class IndividualizeLowCost(Transformation):
    def __init__(
        self,
        ctx: CompilationCtx,
        additional_ops_to_individualize: tuple[type[top.TensorOp], ...] = (),
        additional_exception_ops: tuple[type[top.TensorOp], ...] = (),
    ):
        super().__init__(ctx)
        self.ops_to_individualize = DEFAULT_OPS_TO_INDIVIDUALIZE + additional_ops_to_individualize
        self.exception_ops = DEFAULT_EXCEPTION_OPS + additional_exception_ops

    def _individualize(self, dg: PDG, op: top.TensorOp) -> int:
        dependents = list(dg.get_flat_direct_dependents(op))
        if len(dependents) < 2:
            return 0

        # print(f"Individualizing {op} with {dependents} dependents")

        for dep_op, dep_data in dependents[1:]:
            new_op = dataclasses.replace(op, op_id=dg.get_next_op_id())

            curr_op_data = dg.ops_by_id[op.op_id]
            op_data = OpData(
                new_op,
                dict(curr_op_data.output_shapes),
                dict(curr_op_data.output_dtypes),
            )
            dg.remove_edge(dep_op, op, dep_data)
            dg.insert_op(op_data)
            dg.add_edge(dep_op, new_op, dep_data.copy())

            for depy, depy_data in dg.get_flat_direct_dependencies(op):
                dg.add_edge(new_op, depy, depy_data.copy())

        return 1

    def should_individualize(self, op: top.TensorOp) -> bool:
        """Determine if an operation should be individualized.

        Args:
            op: The operation to check

        Returns:
            bool: True if the operation should be individualized, False otherwise
        """
        threshold = self.ctx.exec_cfg.individualize_consts_smaller_than_x_bytes

        # NOTE: We do not individualize constants that are too large (e.g. embedding tables)
        if isinstance(op, top.ConstOp):
            const_numpy_array = op.value
            if const_numpy_array.nbytes > threshold:
                return False

        return (
            isinstance(op, self.ops_to_individualize) and not isinstance(op, self.exception_ops)
            # and is_static_op(new_dg, op)  # Commented out as in original code
        )

    def _run(self) -> tuple[PDG, bool]:
        new_dg = self.ctx.dg
        count = 0

        round_count = 1
        while round_count > 0:
            round_count = 0

            for op in list(new_dg.nodes):
                if self.should_individualize(op):
                    round_count += self._individualize(new_dg, op)

            count += round_count

        log.info("Individualized %d low-cost ops.", count)
        return new_dg, count > 0
