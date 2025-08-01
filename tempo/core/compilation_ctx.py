from dataclasses import dataclass
from typing import List

from tempo.core import device
from tempo.core import tensor_op as top
from tempo.core.analysis_ctx import AnalysisCtx
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import TensorId
from tempo.core.dependence_graph import PDG


@dataclass(frozen=True)
class CompilationCtx:
    dg: PDG
    analysis_ctx: AnalysisCtx
    exec_cfg: ExecutionConfig

    def get_input_devices_list(self, op: top.TensorOp) -> List[device.DeviceGroup]:  # noqa: F821
        return [
            self.analysis_ctx.get_op_device(depy_op)
            for depy_op, _ in self.dg.get_flat_direct_dependencies(op)
        ]

    def get_tensor_device(self, tid: TensorId) -> device.DeviceGroup:
        op = self.dg.ops_by_id[tid.op_id].op
        return self.analysis_ctx.get_op_device(op)
