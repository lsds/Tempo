import tempo.core.index_expr as ie
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.datatypes import OpId, OpOutId
from tempo.core.op_tags import REGION_TAG
from tempo.core.schedule.execution_schedule import (
    DeallocInstruction,
    ExecInstruction,
    FetchInstruction,
    ForLoop,
    IfGuard,
    InstructionType,
    OffloadInstruction,
    ParallelBlock,
    ScheduleItem,
    SequentialBlock,
)
from tempo.core.utils import bytes_to_human_readable
from tempo.utils.memory_estimator import MemoryEstimator


def _get_indent_str(indent: int) -> str:
    return " " * (indent * 4)


def get_mem_usage_str(
    op_id: OpId, out_id: OpOutId, index: ie.IndexExpr, est: MemoryEstimator
) -> str:
    try:
        bytes_added, approx = est.estimate_tensor_expr_size_bytes(index, op_id, out_id)
        mem_usage = str(bytes_to_human_readable(bytes_added))
        if approx:
            mem_usage += " (approx)"
    except Exception:
        mem_usage = "?B"
    return mem_usage


class SchedulePrinter:
    def __init__(
        self,
        ctx: CompilationCtx,
    ):
        self.ctx = ctx
        self.est = MemoryEstimator(self.ctx)
        self.dg = ctx.dg
        self.schedule = ctx.analysis_ctx.execution_schedule
        self.exec_cfg = ctx.exec_cfg

    def _exec_print(self, item: ScheduleItem, indent: int) -> str:
        assert isinstance(item, ExecInstruction)
        indent_str = _get_indent_str(indent)
        exec_instr = f"thunk_{item.op_id}({str(item.domain_map)})"
        try:
            mem_usage = (
                f" +{bytes_to_human_readable(self.est.estimate_op_size_bytes_out(item.op_id))}"
            )
        except Exception:
            mem_usage = " +?B"
        tags = f" ; {self.dg.ops_by_id[item.op_id].op.tags.get(REGION_TAG, '')}"
        return indent_str + exec_instr + "# " + mem_usage + tags + "\n"

    def _dealloc_print(self, item: ScheduleItem, indent: int) -> str:
        assert isinstance(item, DeallocInstruction)
        indent_str = _get_indent_str(indent)

        op_id = item.tensor_id.op_id
        out_id = item.tensor_id.output_id

        mem_usage = "-" + get_mem_usage_str(op_id, out_id, item.index, self.est)
        dealloc_instr = f"del tensor_{op_id}_{out_id}[{str(item.index)}]"
        tags = f" ; {self.dg.ops_by_id[op_id].op.tags.get(REGION_TAG, '')}"
        return indent_str + dealloc_instr + "# " + mem_usage + tags + "\n"

    def _offload_print(self, item: ScheduleItem, indent: int) -> str:
        assert isinstance(item, OffloadInstruction)
        indent_str = _get_indent_str(indent)
        op_id = item.tensor_id.op_id
        out_id = item.tensor_id.output_id
        mem_usage = get_mem_usage_str(op_id, out_id, item.index, self.est)
        offload_instr = f"tensor_{op_id}_{out_id}.offload({str(item.index)})"
        tags = f" ; {self.dg.ops_by_id[op_id].op.tags.get(REGION_TAG, '')}"
        return indent_str + offload_instr + "# " + mem_usage + tags + "\n"

    def _fetch_print(self, item: ScheduleItem, indent: int) -> str:
        assert isinstance(item, FetchInstruction)
        indent_str = _get_indent_str(indent)
        op_id = item.tensor_id.op_id
        out_id = item.tensor_id.output_id
        fetch_instr = f"tensor_{op_id}_{out_id}.fetch({str(item.index)})"
        mem_usage = get_mem_usage_str(op_id, out_id, item.index, self.est)
        tags = f" ; {self.dg.ops_by_id[op_id].op.tags.get(REGION_TAG, '')}"
        return indent_str + fetch_instr + "# " + mem_usage + tags + "\n"

    def _sequential_block_print(self, item: ScheduleItem, indent: int) -> str:
        assert isinstance(item, SequentialBlock)
        res = ""
        for child in item.children:
            res += self._dispatch_print(child, indent)
        return res

    def _parallel_block_print(self, item: ScheduleItem, indent: int) -> str:
        assert isinstance(item, ParallelBlock)
        raise NotImplementedError

    def _if_guard_print(self, item: ScheduleItem, indent: int) -> str:
        assert isinstance(item, IfGuard)
        indent_str = _get_indent_str(indent)
        res = f"{indent_str}if {str(item.if_cond)}:\n"
        res += self._dispatch_print(item.then_inner, indent + 1)
        if item.else_inner is not None:
            res += f"{indent_str}else:\n"
            res += self._dispatch_print(item.else_inner, indent + 1)
        return res

    def _for_loop_print(self, item: ScheduleItem, indent: int) -> str:
        assert isinstance(item, ForLoop)
        indent_str = _get_indent_str(indent)
        loop_res = self._dispatch_print(item.inner, indent + 1)
        init_str = f"{indent_str}{str(item.counter)} = {str(item.init)}\n"
        increment_str = (
            f"{_get_indent_str(indent + 1)}" + f"{str(item.counter)} += {str(item.increment)}\n"
        )
        while_res = f"{indent_str}while {str(item.cond)}:\n"
        return init_str + while_res + loop_res + increment_str

    def _dispatch_print(self, item: ScheduleItem, indent: int) -> str:
        return {
            InstructionType.EXEC: self._exec_print,
            InstructionType.DEALLOC: self._dealloc_print,
            InstructionType.OFFLOAD: self._offload_print,
            InstructionType.FETCH: self._fetch_print,
            InstructionType.SEQUENTIAL_BLOCK: self._sequential_block_print,
            InstructionType.PARALLEL_BLOCK: self._parallel_block_print,
            InstructionType.IF_GUARD: self._if_guard_print,
            InstructionType.FOR_LOOP: self._for_loop_print,
        }[item.instr_type](item, indent)

    def print_schedule(self) -> str:
        return self._dispatch_print(self.schedule.schedule, 0)
