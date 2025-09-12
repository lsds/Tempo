from __future__ import annotations

from collections.abc import Callable, Generator, MutableMapping

# from multiprocessing.queues import Queue
from functools import partial

from tempo.core import index_expr as ie
from tempo.core.datatypes import BackendTensorT, OpId
from tempo.core.dl_backend import DLBackend
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
from tempo.runtime.thunk_launcher import ThunkLauncherFactory, ThunkLauncherFactoryCtx
from tempo.utils import logger

# from scalene import scalene_profiler

log = logger.get_logger(__name__)


dispatch_map: dict[  # type: ignore
    InstructionType,
    Callable[
        [
            ScheduleItem,
            MutableMapping[ie.Symbol, int],
            type[DLBackend],
        ],
        Generator[bool],
    ],
] = {}


class ExecOpProfiler:
    def __init__(self, sync_after: bool = False, dl_backend: DLBackend | None = None) -> None:
        self.ts: dict[OpId, list[int]] = {}
        self.sync_after = sync_after
        self.dl_backend = dl_backend
        self._current_op: OpId | None = None

    def start(self, op_id: OpId) -> None:
        import time

        self._start_time = time.perf_counter_ns()
        self._current_op = op_id

    def stop(self, op_id: OpId) -> None:
        import time

        if self.sync_after:
            assert self.dl_backend is not None
            self.dl_backend.sync()

        elapsed = time.perf_counter_ns() - self._start_time
        if op_id not in self.ts:
            self.ts[op_id] = []
        self.ts[op_id].append(elapsed)
        self._current_op = None

    def dump(self, file_path: str | None = None) -> dict[OpId, str]:
        import json

        import numpy as np

        def f_(v: float) -> str:
            return f"{round(v / 1e6, 2)}ms"

        stats = {k: f"{f_(np.mean(v))} ± {f_(np.std(v))} X {len(v)}" for k, v in self.ts.items()}
        if file_path:
            with open(file_path, "w") as f:
                json.dump(stats, f, indent=2)
        else:
            import pprint

            print("ExecOpProfiler stats:")
            pprint.pprint(stats)
        return stats


def _execute_op_instr_with_profiler(
    schedule_item: ExecInstruction,
    loop_counters_and_bound: MutableMapping[ie.Symbol, int],
    bend: type[DLBackend],
    profiler: ExecOpProfiler,
) -> None:
    op_id = schedule_item.op_id
    profiler.start(op_id)
    _execute_op_instr(schedule_item, loop_counters_and_bound, bend)
    profiler.stop(op_id)


def _execute_op_instr_with_try_catch(
    schedule_item: ExecInstruction,
    loop_counters_and_bound: MutableMapping[ie.Symbol, int],
    bend: type[DLBackend],
) -> None:
    # log.info("Executing thunk with op_id=%s at %s",
    # schedule_item.op_id, dict(loop_counters_and_bound.items()))
    try:
        _execute_op_instr(schedule_item, loop_counters_and_bound, bend)
    except Exception as e:
        log.error(
            "Error executing instr: %s at %s with %s",
            schedule_item.op_id,
            dict(schedule_item.domain_map.items()),
            dict(loop_counters_and_bound.items()),
        )
        raise e


def _execute_op_instr(
    schedule_item: ExecInstruction,
    loop_counters_and_bound: MutableMapping[ie.Symbol, int],
    bend: type[DLBackend],
) -> None:
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


def _execute_dealloc_instr(
    schedule_item: DeallocInstruction,
    loop_counters_and_bounds: MutableMapping[ie.Symbol, int],
    bend: type[DLBackend],
) -> None:
    schedule_item.thunk(schedule_item.index.eval_fast())  # type: ignore


def _execute_offload_instr(
    schedule_item: OffloadInstruction,
    loop_counters_and_bounds: MutableMapping[ie.Symbol, int],
    bend: type[DLBackend],
) -> None:
    schedule_item.thunk(schedule_item.index.eval_fast())  # type: ignore


def _execute_fetch_instr(
    schedule_item: FetchInstruction,
    loop_counters_and_bounds: MutableMapping[ie.Symbol, int],
    bend: type[DLBackend],
) -> None:
    schedule_item.thunk(schedule_item.index.eval_fast())  # type: ignore


def _execute_sequential_block(
    schedule_item: SequentialBlock,
    loop_counters_and_bounds: MutableMapping[ie.Symbol, int],
    bend: type[DLBackend],
) -> None:
    for inner_item in schedule_item.inner_block:
        if inner_item.instr_type == InstructionType.EXEC:
            inner_item.thunk.launch()  # type: ignore
        elif inner_item.instr_type == InstructionType.DEALLOC:
            inner_item.thunk(inner_item.index.eval_fast())  # type: ignore
        elif inner_item.instr_type == InstructionType.OFFLOAD:
            inner_item.thunk(inner_item.index.eval_fast())  # type: ignore
        elif inner_item.instr_type == InstructionType.FETCH:
            inner_item.thunk(inner_item.index.eval_fast())  # type: ignore
        else:
            dispatch_map[inner_item.instr_type](
                inner_item,
                loop_counters_and_bounds,
                bend,
            )

    # if schedule_item.contains_inline_offload:
    #    bend.sync()


def _execute_if_guard(
    schedule_item: IfGuard,
    loop_counters_and_bounds: MutableMapping[ie.Symbol, int],
    bend: type[DLBackend],
) -> None:
    cond: bool = schedule_item.if_cond.eval_fast()
    if cond:
        then_inner = schedule_item.then_inner
        dispatch_map[then_inner.instr_type](
            then_inner,
            loop_counters_and_bounds,
            bend,
        )
    else:
        else_inner = schedule_item.else_inner
        if else_inner is not None:
            dispatch_map[else_inner.instr_type](
                else_inner,
                loop_counters_and_bounds,
                bend,
            )


def _execute_for_loop(  # noqa: C901
    schedule_item: ForLoop,
    loop_counters_and_bounds: MutableMapping[ie.Symbol, int],
    bend: type[DLBackend],
) -> None:
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
            inner_dispatch(
                inner,
                loop_counters_and_bounds,
                bend,
            )
    else:
        if increment_static:
            inc_val = inc.eval_fast()
            counter_val = init_val
            loop_counters_and_bounds[counter_symbol] = counter_val
            while cond.eval_fast():
                inner_dispatch(
                    inner,
                    loop_counters_and_bounds,
                    bend,
                )
                counter_val += inc_val
                loop_counters_and_bounds[counter_symbol] = counter_val
        elif cond_static:
            counter_val = init_val
            loop_counters_and_bounds[counter_symbol] = counter_val
            assert type(cond) is ie.LessThanOrEqual
            bound = cond.right_operand.eval_fast()
            while counter_val <= bound:
                inner_dispatch(
                    inner,
                    loop_counters_and_bounds,
                    bend,
                )
                inc_amount: int = inc.eval_fast()
                counter_val += inc_amount
                loop_counters_and_bounds[counter_symbol] = counter_val
        else:  # Worst case scenario: both are dynamic
            counter_val = init_val
            loop_counters_and_bounds[counter_symbol] = counter_val
            while cond.eval_fast():
                inner_dispatch(
                    inner,
                    loop_counters_and_bounds,
                    bend,
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


class NonGenInterpreterExecutor(Executor[BackendTensorT]):
    def __init__(self, exec_ctx: ExecutorCtx) -> None:
        super().__init__(exec_ctx)
        self.known_bounds = {**self.dg.static_bounds}

        self.loop_counters_and_bounds = self._initialize_loop_counters_and_bounds()
        self._emit_thunks(exec_ctx)
        self._cache_codegen_eval_for_instructions()

        cfg = exec_ctx.exec_cfg
        assert not (cfg.enable_exec_op_profiling and cfg.executor_debug_mode)
        if cfg.enable_exec_op_profiling:
            self.exec_op_profiler = ExecOpProfiler(
                sync_after=cfg.exec_op_profiling_sync_after_each,
                dl_backend=self.backend,
            )
            dispatch_map[InstructionType.EXEC] = partial(  # type: ignore
                _execute_op_instr_with_profiler, profiler=self.exec_op_profiler
            )

        elif cfg.executor_debug_mode:
            dispatch_map[InstructionType.EXEC] = _execute_op_instr_with_try_catch  # type: ignore

    def _cache_codegen_eval_for_instructions(self) -> None:
        # Cache the codegenerated evals for non-exec instructions
        for (
            sched_item
        ) in self.executor_ctx.analysis_ctx.execution_schedule.schedule.flat_recursive_tree:
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
        thunk_emitter_class = exec_ctx.backend.get_thunk_emitter_cls()
        thunk_emitter = thunk_emitter_class()

        ctx = ThunkLauncherFactoryCtx(
            thunk_emitter,
            self.executor_ctx.external_state_store,
            self.executor_ctx.tensor_store,
            self.backend,
            self.loop_counters_and_bounds,
            self.executor_ctx.compilation_ctx,
        )
        factory = ThunkLauncherFactory(ctx)

        for (
            sched_item
        ) in self.executor_ctx.analysis_ctx.execution_schedule.schedule.flat_recursive_tree:
            if isinstance(sched_item, ExecInstruction):
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
        log.info("Launcher percentages: %s", factory.get_launcher_percentages())

    def _initialize_loop_counters_and_bounds(self) -> MutableMapping[ie.Symbol, int]:
        # Create the dict
        loop_counters_and_bounds = SymbolDict(len(self.dg.universe) * 2)

        # Walk the schedule to collect counters
        counters = set()
        for (
            sched_item
        ) in self.executor_ctx.analysis_ctx.execution_schedule.schedule.flat_recursive_tree:
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

    def shutdown(self) -> None: ...

    def reset(self) -> None:
        # TODO fix the close issue with envs
        # self.executor_ctx.external_state_store.global_clean_up()
        self.executor_ctx.tensor_store.flush()

    def tick(self) -> Generator[bool]:
        raise NotImplementedError

    def execute(self) -> None:
        with ie.StructEqCheckCtxManager():
            loop_counters_and_bounds = self.loop_counters_and_bounds
            for k, v in self.known_bounds.items():
                loop_counters_and_bounds[k] = v

            item = self.executor_ctx.analysis_ctx.execution_schedule.schedule
            dispatch_map[item.instr_type](  # type: ignore
                item,
                loop_counters_and_bounds,
                self.backend,  # type: ignore
            )
        if self.executor_ctx.exec_cfg.enable_exec_op_profiling:
            self.exec_op_profiler.dump()

    def execute_until_barrier(self, barrier_name: str) -> None:
        # TODO
        ...
