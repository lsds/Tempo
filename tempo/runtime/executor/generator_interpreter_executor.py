from __future__ import annotations

from collections.abc import MutableMapping

# from multiprocessing.queues import Queue
from typing import Callable, Dict, Generator

from tempo.core import index_expr as ie
from tempo.core.datatypes import BackendTensorT
from tempo.core.schedule.execution_schedule import (
    DeallocInstruction,
    ExecInstruction,
    FetchInstruction,
    ForLoop,
    IfGuard,
    InstructionType,
    MemManInstr,
    OffloadInstruction,
    ScheduleItem,
    SequentialBlock,
)
from tempo.core.symbol_dict import SymbolDict
from tempo.runtime.executor.executor import Executor, ExecutorCtx
from tempo.runtime.tensor_store.tensor_store import TensorStore
from tempo.runtime.thunk_emitter import ThunkEmitterCtx
from tempo.runtime.thunk_launcher import ThunkLauncherFactory, ThunkLauncherFactoryCtx
from tempo.utils import logger

# from scalene import scalene_profiler


log = logger.get_logger(__name__)

dispatch_map: Dict[  # type: ignore
    InstructionType,
    Callable[
        [
            ScheduleItem,
            MutableMapping[ie.Symbol, int],
            TensorStore,
        ],
        Generator[bool, None, None],
    ],
] = {}


def _execute_op_instr(
    schedule_item: ExecInstruction,
    loop_counters_and_bound: MutableMapping[ie.Symbol, int],
    tensor_store: TensorStore[BackendTensorT],
) -> Generator[bool, None, None]:
    # log.info("Executing thunk with op_id=%s at %s",
    # schedule_item.op_id, dict(loop_counters_and_bound.items()))
    # try:
    schedule_item.thunk.launch()  # type: ignore
    # except Exception as e:
    #    log.error(
    #        "Error executing instr: %s at %s with %s",
    #        schedule_item.op_id,
    #        dict(schedule_item.domain_map.items()),
    #        dict(loop_counters_and_bound.items()),
    #    )
    #    raise e
    yield False


def _execute_dealloc_instr(
    schedule_item: DeallocInstruction,
    loop_counters_and_bounds: MutableMapping[ie.Symbol, int],
    tensor_store: TensorStore[BackendTensorT],
) -> Generator[bool, None, None]:
    # index = schedule_item.index.evaluate(loop_counters_and_bounds)
    # if schedule_item.is_point:
    #    tensor_store[schedule_item.tensor_id].deallocate_point(schedule_item.index.eval_fast())
    # else:
    #    tensor_store[schedule_item.tensor_id].deallocate_block(schedule_item.index.eval_fast())
    try:
        schedule_item.thunk(schedule_item.index.eval_fast())  # type: ignore
    except Exception as e:
        print(f"Sched item: {schedule_item}")
        print(f"Sched item: {schedule_item.index}")
        print(f"Sched item: {schedule_item.tensor_id}")
        print(f"{dict(loop_counters_and_bounds.items())}")

        print(f"{schedule_item.creation_traceback}")
        raise e
    yield False


def _execute_offload_instr(
    schedule_item: OffloadInstruction,
    loop_counters_and_bounds: MutableMapping[ie.Symbol, int],
    tensor_store: TensorStore[BackendTensorT],
) -> Generator[bool, None, None]:
    # index = schedule_item.index.evaluate(loop_counters_and_bounds)
    # log.debug("Offloading tensor %s at index %s", schedule_item.tensor_id, index)
    # if schedule_item.is_point:
    #    tensor_store[schedule_item.tensor_id].offload_point(schedule_item.index.eval_fast())
    # else:
    #    tensor_store[schedule_item.tensor_id].offload_block(schedule_item.index.eval_fast())
    schedule_item.thunk(schedule_item.index.eval_fast())  # type: ignore
    yield False


def _execute_fetch_instr(
    schedule_item: FetchInstruction,
    loop_counters_and_bounds: MutableMapping[ie.Symbol, int],
    tensor_store: TensorStore[BackendTensorT],
) -> Generator[bool, None, None]:
    # index = schedule_item.index.evaluate(loop_counters_and_bounds)
    # log.debug("Fetching tensor %s at index %s", schedule_item.tensor_id, index)
    # if schedule_item.is_point:
    #    tensor_store[schedule_item.tensor_id].fetch_point(schedule_item.index.eval_fast())
    # else:
    #    tensor_store[schedule_item.tensor_id].fetch_block(schedule_item.index.eval_fast())
    schedule_item.thunk(schedule_item.index.eval_fast())  # type: ignore
    yield False


def _execute_sequential_block(
    schedule_item: SequentialBlock,
    loop_counters_and_bounds: MutableMapping[ie.Symbol, int],
    tensor_store: TensorStore[BackendTensorT],
) -> Generator[bool, None, None]:
    for inner_item in schedule_item.inner_block:
        yield from dispatch_map[inner_item.instr_type](
            inner_item,
            loop_counters_and_bounds,
            tensor_store,
        )
    # yield False


def _execute_if_guard(
    schedule_item: IfGuard,
    loop_counters_and_bounds: MutableMapping[ie.Symbol, int],
    tensor_store: TensorStore[BackendTensorT],
) -> Generator[bool, None, None]:
    cond: bool = schedule_item.if_cond.eval_fast()
    if cond:
        then_inner = schedule_item.then_inner
        yield from dispatch_map[then_inner.instr_type](
            then_inner,
            loop_counters_and_bounds,
            tensor_store,
        )
    else:
        else_inner = schedule_item.else_inner
        if else_inner is not None:
            yield from dispatch_map[else_inner.instr_type](
                else_inner,
                loop_counters_and_bounds,
                tensor_store,
            )
        else:
            yield False


def _execute_for_loop(  # noqa: C901
    schedule_item: ForLoop,
    loop_counters_and_bounds: MutableMapping[ie.Symbol, int],
    tensor_store: TensorStore[BackendTensorT],
) -> Generator[bool, None, None]:
    # TODO logging
    counter_symbol = schedule_item.counter
    cond = schedule_item.cond
    inc = schedule_item.increment
    inner = schedule_item.inner
    inner_dispatch = dispatch_map[inner.instr_type]

    increment_static = type(inc) is ie.ConstInt
    cond_static = (
        type(cond) is ie.LessThanOrEqual
        and type(cond.right_operand) is ie.ConstInt
        and type(cond.left_operand) is ie.Symbol
        and cond.left_operand.struct_eq(counter_symbol)
    )

    init_val = schedule_item.init.eval_fast()

    if increment_static and cond_static:
        # In this case, we can skip over eval calls for faster execution.
        inc_val = inc.eval_fast()

        assert type(cond) is ie.LessThanOrEqual
        bound = cond.right_operand.eval_fast()
        for counter_val in range(init_val, bound + 1, inc_val):
            loop_counters_and_bounds[counter_symbol] = counter_val
            yield from inner_dispatch(
                inner,
                loop_counters_and_bounds,
                tensor_store,
            )
    else:
        if increment_static:
            inc_val = inc.eval_fast()
            counter_val = init_val
            loop_counters_and_bounds[counter_symbol] = counter_val
            while cond.eval_fast():
                yield from inner_dispatch(
                    inner,
                    loop_counters_and_bounds,
                    tensor_store,
                )
                counter_val += inc_val
                loop_counters_and_bounds[counter_symbol] = counter_val
        elif cond_static:
            counter_val = init_val
            loop_counters_and_bounds[counter_symbol] = counter_val
            assert type(cond) is ie.LessThanOrEqual
            bound = cond.right_operand.eval_fast()
            while counter_val <= bound:
                yield from inner_dispatch(
                    inner,
                    loop_counters_and_bounds,
                    tensor_store,
                )
                inc_amount: int = inc.eval_fast()
                counter_val += inc_amount
                loop_counters_and_bounds[counter_symbol] = counter_val
        else:  # Worst case scenario: both are dynamic
            counter_val = init_val
            loop_counters_and_bounds[counter_symbol] = counter_val
            while cond.eval_fast():
                yield from inner_dispatch(
                    inner,
                    loop_counters_and_bounds,
                    tensor_store,
                )
                inc_amount_: int = inc.eval_fast()
                counter_val += inc_amount_
                loop_counters_and_bounds[counter_symbol] = counter_val

    del loop_counters_and_bounds[counter_symbol]
    # yield False


dispatch_map.update(
    {
        InstructionType.EXEC: _execute_op_instr,  # type: ignore
        InstructionType.DEALLOC: _execute_dealloc_instr,  # type: ignore
        InstructionType.OFFLOAD: _execute_offload_instr,  # type: ignore
        InstructionType.FETCH: _execute_fetch_instr,  # type: ignore
        InstructionType.SEQUENTIAL_BLOCK: _execute_sequential_block,  # type: ignore
        InstructionType.IF_GUARD: _execute_if_guard,  # type: ignore
        InstructionType.FOR_LOOP: _execute_for_loop,  # type: ignore
    }
)


class InterpreterExecutor(Executor[BackendTensorT]):
    def __init__(self, exec_ctx: ExecutorCtx) -> None:
        super().__init__(exec_ctx)
        self.known_bounds = {**self.dg.static_bounds}

        self.loop_counters_and_bounds = self._initialize_loop_counters_and_bounds()
        self._emit_thunks(exec_ctx)
        self._cache_codegen_eval_for_instructions()

    def _cache_codegen_eval_for_instructions(self) -> None:
        # Cache the codegenerated evals for non-exec instructions
        for sched_item in self.dg.analysis_ctx.execution_schedule.schedule.flat_recursive_tree:
            if isinstance(sched_item, (OffloadInstruction, FetchInstruction, DeallocInstruction)):
                sched_item.index.cache_codegenerated_eval(self.loop_counters_and_bounds)
            if isinstance(sched_item, ForLoop):
                sched_item.init.cache_codegenerated_eval(self.loop_counters_and_bounds)
                sched_item.cond.cache_codegenerated_eval(self.loop_counters_and_bounds)
                if isinstance(sched_item.cond, ie.LessThanOrEqual):
                    sched_item.cond.right_operand.cache_codegenerated_eval(
                        self.loop_counters_and_bounds
                    )
                sched_item.increment.cache_codegenerated_eval(self.loop_counters_and_bounds)
            if isinstance(sched_item, IfGuard):
                sched_item.if_cond.cache_codegenerated_eval(self.loop_counters_and_bounds)

    def _emit_thunks(self, exec_ctx: ExecutorCtx) -> None:
        # Emit thunks for all ops
        thunk_emitter_ctx: ThunkEmitterCtx = ThunkEmitterCtx(
            dg=exec_ctx.dg,
            compile_time_known_symbol_values=exec_ctx.dg.static_bounds,
            external_state_store=exec_ctx.external_state_store,
            tensor_store=exec_ctx.tensor_store,
            exec_cfg=exec_ctx.exec_cfg,
        )
        thunk_emitter_class = exec_ctx.backend.get_thunk_emitter_cls()
        thunk_emitter = thunk_emitter_class(thunk_emitter_ctx)
        thunks = {}
        for op_id, op in exec_ctx.dg.ops_by_id.items():
            thunks[op_id] = thunk_emitter.emit_thunk_for_op(op.op)

        ctx = ThunkLauncherFactoryCtx(
            self.dg,
            self.executor_ctx.external_state_store,
            self.executor_ctx.tensor_store,
            self.executor_ctx.exec_cfg,
            thunks,
            self.backend,
            self.loop_counters_and_bounds,
            self.executor_ctx.analysis_ctx,
        )
        factory = ThunkLauncherFactory(ctx)

        for sched_item in self.dg.analysis_ctx.execution_schedule.schedule.flat_recursive_tree:
            if isinstance(sched_item, ExecInstruction):
                # sched_item.thunk = factory.wrap_thunk_in_launcher(sched_item)
                thunk = factory.emit_thunk_launcher(sched_item)
                # NOTE: This is a hack to get around the fact that we can't set attributes on the
                #      schedule item object.
                object.__setattr__(sched_item, "thunk", thunk)
            elif isinstance(sched_item, MemManInstr):
                tensor_ = self.executor_ctx.tensor_store[sched_item.tensor_id]
                if isinstance(sched_item, DeallocInstruction):
                    if sched_item.is_point:
                        thunk = tensor_.deallocate_point
                    else:
                        thunk = tensor_.deallocate_block
                elif isinstance(sched_item, OffloadInstruction):
                    if sched_item.is_point:
                        thunk = tensor_.offload_point
                    else:
                        thunk = tensor_.offload_block
                elif isinstance(sched_item, FetchInstruction):
                    if sched_item.is_point:
                        thunk = tensor_.fetch_point
                    else:
                        thunk = tensor_.fetch_block
                else:
                    raise ValueError("Unknown MemManInstr type")
                object.__setattr__(sched_item, "thunk", thunk)

    def _initialize_loop_counters_and_bounds(self) -> MutableMapping[ie.Symbol, int]:
        # Create the dict
        loop_counters_and_bounds = SymbolDict(len(self.dg.universe) * 2)

        # Walk the schedule to collect counters
        counters = set()
        for sched_item in self.dg.analysis_ctx.execution_schedule.schedule.flat_recursive_tree:
            if isinstance(sched_item, ForLoop):
                counters.add(sched_item.counter)

        # Load the counter & bound keys
        all_keys = list(self.dg.bound_defs.keys()) + list(counters)
        loop_counters_and_bounds.load_keys(all_keys)

        # -1 means not set, anything else indicates a compile-time known value (static bound)
        for c in counters:
            loop_counters_and_bounds[c] = -1
        for k in self.dg.bound_defs.keys():
            if k in self.known_bounds:
                loop_counters_and_bounds[k] = self.known_bounds[k]
            else:
                loop_counters_and_bounds[k] = -1

        return loop_counters_and_bounds

    def shutdown(self) -> None:
        pass

    def reset(self) -> None:
        # TODO fix the close issue with envs
        # self.executor_ctx.external_state_store.global_clean_up()
        self.executor_ctx.tensor_store.flush()

    def tick(self) -> Generator[bool, None, None]:
        with ie.StructEqCheckCtxManager():
            loop_counters_and_bounds = self.loop_counters_and_bounds
            for k, v in self.known_bounds.items():
                loop_counters_and_bounds[k] = v

            # Execute the schedule item incrementally
            item = self.dg.analysis_ctx.execution_schedule.schedule

            yield from dispatch_map[item.instr_type](  # type: ignore
                item,
                loop_counters_and_bounds,  # type: ignore
                self.executor_ctx.tensor_store,
            )

            # Yield True when done
            yield True

    def execute(self) -> None:
        with ie.StructEqCheckCtxManager():
            # log.info("================== Starting execution ====================")
            self.reset()

            # scalene_profiler.start()
            # Initialize the tick generator
            tick_generator = self.tick()

            # Execute ticks until it yields True
            for _ in tick_generator:
                pass

            # scalene_profiler.stop()
