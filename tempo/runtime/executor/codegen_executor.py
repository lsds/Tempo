# from __future__ import annotations
#
# import time
# from typing import Dict, List
#
# from tempo.core import index_expr as ie
# from tempo.core.datatypes import BackendTensorT, OpId
# from tempo.core.schedule.execution_schedule import (
#    DeallocInstruction,
#    ExecInstruction,
#    FetchInstruction,
#    ForLoop,
#    IfGuard,
#    Instruction,
#    InstructionType,
#    OffloadInstruction,
#    ScheduleItem,
#    SequentialBlock,
# )
# from tempo.core.symbol_dict import SymbolDict
# from tempo.runtime.executor.executor import Executor, ExecutorCtx
# from tempo.runtime.tensor_store import TensorStore
# from tempo.runtime.wrapped_thunks import (
#    ThunkWrapCtx,
#    WrappedThunk,
#    WrappedThunkFactory,
# )
# from tempo.utils import logger
#
# log = logger.get_logger(__name__)
#
#
## @numba.jit()
# def _execute_op_instr(
#    schedule_item: ExecInstruction,
#    loop_counters_and_bound: Dict[ie.Symbol, int],
#    thunks: Dict[OpId, WrappedThunk],
#    tensor_store: TensorStore[BackendTensorT],
#    stack: List[ScheduleItem],
# ) -> None:
#    thunk = thunks[schedule_item.op_id]
#    symb_v = thunk.thunk_exec_ctx.symbol_values
#    for k, v in schedule_item.domain_map.items():
#        symb_v[k] = v.evaluate(loop_counters_and_bound)
#
#    try:
#        thunk.execute()  # type: ignore
#    except Exception as e:
#        log.error("Error executing thunk %s", thunk.op)
#        raise e
#
#
## @numba.jit()
# def _execute_dealloc_instr(
#    schedule_item: DeallocInstruction,
#    loop_counters_and_bounds: Dict[ie.Symbol, int],
#    thunks: Dict[OpId, WrappedThunk],
#    tensor_store: TensorStore[BackendTensorT],
#    stack: List[ScheduleItem],
# ) -> None:
#    index = schedule_item.index.evaluate(loop_counters_and_bounds)
#    tensor_store[schedule_item.tensor_id].deallocate(index)
#
#
## @numba.jit()
# def _execute_offload_instr(
#    schedule_item: OffloadInstruction,
#    loop_counters_and_bounds: Dict[ie.Symbol, int],
#    thunks: Dict[OpId, WrappedThunk],
#    tensor_store: TensorStore[BackendTensorT],
#    stack: List[ScheduleItem],
# ) -> None:
#    index = schedule_item.index.evaluate(loop_counters_and_bounds)
#    tensor_store[schedule_item.tensor_id].offload(index)
#
#
## @numba.jit()
# def _execute_fetch_instr(
#    schedule_item: FetchInstruction,
#    loop_counters_and_bounds: Dict[ie.Symbol, int],
#    thunks: Dict[OpId, WrappedThunk],
#    tensor_store: TensorStore[BackendTensorT],
#    stack: List[ScheduleItem],
# ) -> None:
#    index = schedule_item.index.evaluate(loop_counters_and_bounds)
#    tensor_store[schedule_item.tensor_id].fetch(index)
#
#
## @numba.jit()
# def _execute_sequential_block(
#    schedule_item: SequentialBlock,
#    loop_counters_and_bounds: Dict[ie.Symbol, int],
#    thunks: Dict[OpId, WrappedThunk],
#    tensor_store: TensorStore[BackendTensorT],
#    stack: List[ScheduleItem],
# ) -> None:
#    stack.extend(reversed(schedule_item.inner_block))
#
#
## @numba.jit()
# def _execute_if_guard(
#    schedule_item: IfGuard,
#    loop_counters_and_bounds: Dict[ie.Symbol, int],
#    thunks: Dict[OpId, WrappedThunk],
#    tensor_store: TensorStore[BackendTensorT],
#    stack: List[ScheduleItem],
# ) -> None:
#    cond: bool = schedule_item.if_cond.evaluate(loop_counters_and_bounds)
#    if cond:
#        stack.append(schedule_item.then_inner)
#    elif schedule_item.else_inner:
#        stack.append(schedule_item.else_inner)
#
#
## @numba.jit()
# def _execute_for_loop(
#    schedule_item: ForLoop,
#    loop_counters_and_bounds: Dict[ie.Symbol, int],
#    thunks: Dict[OpId, WrappedThunk],
#    tensor_store: TensorStore[BackendTensorT],
#    stack: List[ScheduleItem],
# ) -> None:
#    counter_symbol = schedule_item.counter
#    cond = schedule_item.cond
#    inc = schedule_item.increment
#    inner = schedule_item.inner
#
#    increment_static = isinstance(inc, ie.ConstInt)
#    cond_static = (
#        isinstance(cond, ie.LessThanOrEqual)
#        and isinstance(cond.right_operand, ie.ConstInt)
#        and isinstance(cond.left_operand, ie.Symbol)
#        and cond.left_operand.equivalent(counter_symbol)
#    )
#
#    init_val = schedule_item.init.evaluate(loop_counters_and_bounds)
#
#    if increment_static and cond_static:
#        inc_val = inc.evaluate(loop_counters_and_bounds)
#        bound = cond.right_operand.evaluate(loop_counters_and_bounds)
#        for counter_val in range(init_val, bound + 1, inc_val):
#            loop_counters_and_bounds[counter_symbol] = counter_val
#            stack.append(inner)
#    else:
#        counter_val = init_val
#        loop_counters_and_bounds[counter_symbol] = counter_val
#        while cond.evaluate(loop_counters_and_bounds):
#            stack.append(inner)
#            inc_amount = inc.evaluate(loop_counters_and_bounds)
#            counter_val += inc_amount
#            loop_counters_and_bounds[counter_symbol] = counter_val
#    del loop_counters_and_bounds[counter_symbol]
#
#
# def _dispatch(
#    schedule_item: Instruction,
#    loop_counters_and_bounds: Dict[ie.Symbol, int],
#    thunks: Dict[OpId, WrappedThunk],
#    tensor_store: TensorStore[BackendTensorT],
#    stack: List[ScheduleItem],
# ) -> None:
#    if schedule_item.instr_type == InstructionType.EXEC:
#        _execute_op_instr(
#            schedule_item, loop_counters_and_bounds, thunks, tensor_store, stack
#        )
#    if schedule_item.instr_type == InstructionType.DEALLOC:
#        _execute_dealloc_instr(
#            schedule_item, loop_counters_and_bounds, thunks, tensor_store, stack
#        )
#    if schedule_item.instr_type == InstructionType.OFFLOAD:
#        _execute_offload_instr(
#            schedule_item, loop_counters_and_bounds, thunks, tensor_store, stack
#        )
#    if schedule_item.instr_type == InstructionType.FETCH:
#        _execute_fetch_instr(
#            schedule_item, loop_counters_and_bounds, thunks, tensor_store, stack
#        )
#    if schedule_item.instr_type == InstructionType.SEQUENTIAL_BLOCK:
#        _execute_sequential_block(
#            schedule_item, loop_counters_and_bounds, thunks, tensor_store, stack
#        )
#    if schedule_item.instr_type == InstructionType.IF_GUARD:
#        _execute_if_guard(
#            schedule_item, loop_counters_and_bounds, thunks, tensor_store, stack
#        )
#    if schedule_item.instr_type == InstructionType.FOR_LOOP:
#        _execute_for_loop(
#            schedule_item, loop_counters_and_bounds, thunks, tensor_store, stack
#        )
#
#
# class StackInterpreterExecutor(Executor[BackendTensorT]):
#    def __init__(self, executor_ctx: ExecutorCtx) -> None:
#        super().__init__(executor_ctx)
#        wrap_thunks_start = time.perf_counter_ns()
#        ctx = ThunkWrapCtx(
#            self.dg,
#            self.executor_ctx.external_state_store,
#            self.executor_ctx.tensor_store,
#            self.executor_ctx.exec_cfg,
#            self.thunks,
#            self.backend,
#        )
#        factory = WrappedThunkFactory(ctx)
#        self.wrapped_thunks: Dict[OpId, WrappedThunk] = {
#            op_id: factory.wrap_thunk(op_id) for op_id in self.thunks
#        }
#        wrap_thunks_elapsed_sec = (time.perf_counter_ns() - wrap_thunks_start) / 1e9
#        log.info("Wrap thunks time: %s seconds.", wrap_thunks_elapsed_sec)
#        self.known_bounds = {**self.dg.static_bounds}
#        self.symbol_dict = SymbolDict(len(self.dg.universe) * 2)
#
#        program = self._generate_program()
#
#        self.compiled_program = compile(program, "<string>", "exec")
#
#    def _generate_program(self) -> str:
#        lines = []
#
#        # Define static constants
#        lines.append("# Define static bounds")
#        for name, value in self.executor_ctx.dg.static_bounds.items():
#            lines.append(f"{name} = {value}")
#
#        # Define dynamic bounds
#        lines.append("# Define dynamic bounds")
#        for name, value in self.executor_ctx.dg.static_bounds.items():
#            lines.append(f"{name} = {value}")
#
#        # Step 2: Initialize variables
#        # NOTE we may want to skip this actually.
#        # lines.append("\n# Initialize variables")
#        # for var in self.executor_ctx.dg.universe.variables:
#        #    lines.append(f"{var} = -1")
#
#        # Step 3: Generate code from the schedule
#        def process_instruction(instruction, indent=0):
#            indent_space = " " * indent
#            if isinstance(instruction, ExecInstruction):
#                # Simple execution instruction
#                lines.append(f"{indent_space}print('Executing {instruction.op_id}')")
#            elif isinstance(instruction, ForLoop):
#                # Generate a for loop
#                lines.append(
#                    f"{indent_space}for {instruction.counter} in range({instruction.start},
#                        {instruction.end}, {instruction.step}):"
#                )
#                for inner_instr in instruction.body:
#                    process_instruction(inner_instr, indent + 4)
#            elif isinstance(instruction, SequentialBlock):
#                # Sequential block: just generate code for each instruction in the block
#                for inner_instr in instruction.instructions:
#                    process_instruction(inner_instr, indent)
#
#        # Step 4: Walk through the schedule and generate the corresponding Python code
#        process_instruction(
#            self.executor_ctx.dg.analysis_ctx.execution_schedule.schedule
#        )
#
#        # Step 5: Return the generated Python code as a single string
#        return "\n".join(lines)
#
#    def shutdown(self) -> None:
#        pass
#
#    def reset(self) -> None:
#        self.executor_ctx.tensor_store.flush()
#
#    # @numba.jit()
#    def tick(self) -> bool:
#        raise NotImplementedError
#
#    def execute(self) -> None:
#        self.reset()
#
#        # Globals, locals
#        exec(
#            self.compiled_program,
#            {},
#            {
#                "thunks": self.wrapped_thunks,
#                "tensor_store": self.executor_ctx.tensor_store,
#            },
#        )
#
