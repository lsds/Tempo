from typing import Tuple

from tempo.core.dependence_graph import PDG
from tempo.core.utils import bytes_to_human_readable
from tempo.transformations.compilation_pass import CompilationCtx, Transformation
from tempo.transformations.incrementalization.incrementalization_common import (
    perform_incrementalization,
)
from tempo.transformations.incrementalization.incrementalization_policy import (  # type: ignore
    IncTemporalOnce,
)
from tempo.utils import logger
from tempo.utils.memory_estimator import MemoryEstimator

log = logger.get_logger(__name__)


class Incrementalize(Transformation):
    def __init__(self, ctx: CompilationCtx):
        super().__init__(ctx)
        self.current_compilation_ctx = ctx

    def _run(self) -> Tuple[PDG, bool]:
        inc_ops_count = 0
        pad_ops_count = 0
        new_dg = self.ctx.dg
        from tempo.core import global_objects as glob

        inc_rounds = 0

        limit = self.ctx.exec_cfg.incrementalize_mem_threshold

        while True:
            glob.set_active_dg(new_dg)
            mem_est = MemoryEstimator(self.current_compilation_ctx)
            ## inc_policy = PreferTemporalDimsFirstBatchSecond()
            # if inc_rounds == 0:
            inc_policy = IncTemporalOnce()
            # elif (
            #    inc_rounds == 1
            #    and DLBackendName.str_to_enum(self.ctx.exec_cfg.backend) == DLBackendName.TORCH
            # ):
            #    max_mem_op = max(new_dg.nodes, key=lambda op: mem_est.estimate_op_size_bytes(op.op_id))
            #    max_mem_op_size = mem_est.estimate_op_size_bytes(max_mem_op.op_id)
            #    print(f"Max mem op: {max_mem_op} with size {max_mem_op_size}")
            #    if max_mem_op_size > limit / 2:
            #        inc_policy = PreferTemporalDimsFirstBatchSecond()
            #    else:
            #        break
            # else:
            #    break

            inc_round_ctx = inc_policy.get_round_info(self.current_compilation_ctx, inc_rounds)

            if inc_round_ctx is None:
                break

            log.info(
                """Inc round %s: on dim of size %s
                 with %s blocks of size %s, using inc var %s,  on start ops: \n %s""",
                inc_rounds,
                inc_round_ctx.dim_size,
                inc_round_ctx.num_blocks,
                inc_round_ctx.block_size,
                inc_round_ctx.inc_var,
                [
                    (op_, bytes_to_human_readable(mem_est.estimate_op_size_bytes(op_.op_id)))
                    for op_ in list(inc_round_ctx.inc_start_ops)
                ],
            )
            new_dg = perform_incrementalization(new_dg, inc_round_ctx)
            op_mapping = inc_round_ctx.op_mapping
            pad_ops_count += len(inc_round_ctx.padding_applied)
            inc_ops_count += len(op_mapping)

            self.current_compilation_ctx = CompilationCtx(
                new_dg, self.ctx.analysis_ctx, self.ctx.exec_cfg
            )

            inc_rounds += 1
            # DGRenderer(
            #    new_dg, f"{self.ctx.exec_cfg.path}incrementalize_round{inc_rounds}"
            # ).render()

            if inc_round_ctx.block_idx is None:
                # NOTE: This is how we currently communicate
                # that the incrementalization was full temporal,
                #  in which case it is typically
                # sufficient
                break

        log.info("In %s rounds, incrementalized %s ops.", inc_rounds, inc_ops_count)
        assert pad_ops_count == 0, "Padding ops should not be inserted in static inc"
        return new_dg, inc_ops_count > 0
