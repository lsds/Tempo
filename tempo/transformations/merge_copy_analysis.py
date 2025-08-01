from typing import Dict, Sequence, Tuple

from tempo.core import tensor_ops as top
from tempo.core.datatypes import OpId, TensorId
from tempo.core.dependence_graph import PDG
from tempo.core.storage_methods import PointStore
from tempo.transformations.compilation_pass import Transformation
from tempo.utils import logger

log = logger.get_logger(__name__)


class MergeCopyAnalysis(Transformation):
    def _run(self) -> Tuple[PDG, bool]:
        dg = self.ctx.dg
        analysis_ctx = self.ctx.analysis_ctx
        count = 0
        merge_copies: Dict[OpId, Sequence[bool]] = {}
        total_count = 0

        for op in list(dg.nodes):
            if isinstance(op, top.MergeOp):
                total_count += op.num_inputs
                copies = [False] * op.num_inputs
                # Basically, a copy is required in a merge when the source tensor is donated
                # to another op.
                donations = analysis_ctx.donatable_args
                is_donated_map = analysis_ctx.tensor_is_donated
                op_don = donations.get(op.op_id, ())

                for i, (depy, depy_data) in enumerate(dg.get_flat_direct_dependencies(op)):
                    tid = TensorId(depy.op_id, depy_data.src_out_idx)
                    expr = depy_data.expr
                    stor = analysis_ctx._tensor_storage_classes
                    src_stor = stor.get(tid) if stor is not None else PointStore()
                    # If it is donated, but not to us.
                    if (
                        is_donated_map.get(tid, False)  # Donated
                        and i not in op_don  # Not to us
                        # Being a dict point, it was not created just for us.
                        and (
                            (expr.is_point() and isinstance(src_stor, PointStore))
                            or (not isinstance(src_stor, PointStore))
                        )
                    ):
                        copies[i] = True
                        count += 1
                merge_copies[op.op_id] = copies

        log.info(
            "Found %s merge inputs needing copies (%s%%)",
            count,
            round(count / total_count * 100, 2) if total_count > 0 else 0,
        )
        analysis_ctx._needed_merge_copies = merge_copies
        return dg, count > 0
