from tempo.core import tensor_ops as top
from tempo.core.analysis_ctx import AnalysisCtx
from tempo.core.datatypes import OpOutId
from tempo.core.dependence_graph import PDG, OpData
from tempo.transformations.compilation_pass import Analysis
from tempo.utils import logger

log = logger.get_logger(__name__)


class ValidatePDG(Analysis):
    def _run(self) -> AnalysisCtx:
        dg: PDG = self.ctx.dg

        log.info("Validating PDG...")

        for op_id in dg.node_ids:
            # Check the output shapes match the expected output_shapes
            try:
                op_data: OpData = dg.ops_by_id[op_id]
                self._validate_op(dg, op_data)
            except ValueError as e:
                log.error("Validation failed for op %s: %s", op_id, e)
                log.error("Creation Traceback:\n%s", op_data.op.creation_traceback)
                raise e
        for _, src, data in dg.get_all_edges():
            assert len(data.expr) == len(src.domain)
            # TODO assert snk dom = src dom + indexing
            # TODO assert expr makes sense?
            # TODO assert cond makes sense?

        # TODO: check dim and dims parameters to see if any negative values introduced

        return self.ctx.analysis_ctx

    def _validate_op(self, dg: PDG, op_data: OpData) -> None:
        self._validate_shapes(dg, op_data)
        self._validate_dtypes(dg, op_data)

        assert len(dg.get_flat_direct_dependencies(op_data.op)) == op_data.op.num_inputs, (
            f"{op_data.op=}: expected {op_data.op.num_inputs} inputs, "
            f" got {len(dg.get_flat_direct_dependencies(op_data.op))}"
        )

    def _validate_dtypes(self, dg: PDG, op_data: OpData) -> None:
        op: top.TensorOp = op_data.op
        registered_dtypes = op_data.output_dtypes
        registered_dtypes_list = [
            registered_dtypes[OpOutId(i)] for i in range(len(registered_dtypes))
        ]
        input_dtypes = dg.get_input_dtypes_list(op)
        inferred_dtypes = op.infer_output_dtypes(input_dtypes)

        # Assert lens match
        assert len(registered_dtypes_list) == len(inferred_dtypes)

        for out_idx, (reg, inf) in enumerate(
            zip(registered_dtypes_list, inferred_dtypes, strict=True)
        ):
            if reg != inf:
                raise ValueError(f"Output dtype mismatch for {op=},{out_idx=}. {reg=}, {inf=}")

    def _validate_shapes(self, dg: PDG, op_data: OpData) -> None:
        op: top.TensorOp = op_data.op
        registered_output_shapes = op_data.output_shapes
        registered_output_shapes_list = [
            registered_output_shapes[OpOutId(i)] for i in range(len(registered_output_shapes))
        ]

        input_shapes = dg.get_input_shapes_list(op)
        if (
            (not self.ctx.analysis_ctx._broadcast_elim_has_run)
            and isinstance(op, top.ElementWiseOp)
            and not all(is_ == input_shapes[0] for is_ in input_shapes)
        ):
            print(f"{op} creation traceback:")
            print(op.creation_traceback)
            print("\n" * 5)
            raise ValueError(f"Expected all input shapes to be equal, got {input_shapes}")

        # TODO: This is causing some warnings to be printed like:
        # Error when creating pw_aff: [] -> { [] -> [D0]: true and (true) and (true) }
        inferred_output_shapes = op.infer_output_shapes(input_shapes)

        # Assert lens match
        assert len(registered_output_shapes_list) == len(inferred_output_shapes)

        for out_idx, (reg, inf) in enumerate(
            zip(registered_output_shapes_list, inferred_output_shapes, strict=True)
        ):
            reg_static = reg.try_resolve(dg.static_bounds)
            inf_static = inf.try_resolve(dg.static_bounds)
            if reg_static != inf_static:
                msg = "Output shape mismatch for "
                msg += f"{op=}, {out_idx=}. {reg_static=}, {inf_static=}, {input_shapes=}"
                raise ValueError(msg)
