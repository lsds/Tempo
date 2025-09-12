from abc import abstractmethod
from collections.abc import Callable
from typing import Generic

import numpy as np
import optree

from tempo.core import tensor_ops as top
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.datatypes import BackendTensorT
from tempo.core.dtype import dtypes
from tempo.core.global_objects import jit_kernel_cache
from tempo.core.tensor_op import TensorOp
from tempo.core.thunk import Thunk
from tempo.core.thunk_emitter import ThunkEmissionCtx, ThunkEmitter
from tempo.runtime.inplace_buffer_thunk_wrapper import (
    has_buffer_stored_outputs,
    make_inplace_write_wrapper,
)
from tempo.runtime.lazy_slice_thunk_wrapper import (
    has_lazy_sliced_outputs,
    make_lazy_slice_thunk_wrapper,
)
from tempo.utils import logger
from tempo.utils.common import Timer

log = logger.get_logger(__name__)


class ThunkEmitterBase(Generic[BackendTensorT], ThunkEmitter[BackendTensorT]):
    """Base implementation of ThunkEmitter with common functionality."""

    def emit_thunk_for_op(
        self,
        op: TensorOp,
        ctx: ThunkEmissionCtx[BackendTensorT],
    ) -> Thunk[BackendTensorT]:
        # Check if we have a cached thunk
        backend_name = ctx.exec_cfg.get_canonical_backend_name()
        cached_thunk = jit_kernel_cache.get((backend_name, ctx.dg.pdg_id, op.op_id), None)
        if cached_thunk is not None:
            log.debug("Using cached JIT kernel for %s", op.op_id)
            return cached_thunk  # type: ignore

        # Since both dataflow and UDFs are handled generically, we handle them here
        if type(op) is top.ExecDataflowOp:
            thunk = self._dataflow_op_translation(op, ctx)
        elif type(op) is top.UDFOp:
            thunk = op.desc.thunk_translation(op, ctx)
        else:
            # Otherwise, emit the thunk using the backend-specific method
            thunk = self._emit_thunk_for_op(op, ctx)

        # Cache the compiled thunk
        jit_kernel_cache[(backend_name, ctx.dg.pdg_id, op.op_id)] = thunk  # type: ignore
        return thunk

    @abstractmethod
    def _emit_thunk_for_op(
        self,
        op: TensorOp,
        ctx: ThunkEmissionCtx[BackendTensorT],
    ) -> Thunk[BackendTensorT]: ...

    def _dataflow_op_translation(
        self,
        op: top.ExecDataflowOp,
        emit_ctx: ThunkEmissionCtx[BackendTensorT],
    ) -> Thunk[BackendTensorT]:
        assert isinstance(op, top.ExecDataflowOp)

        internal_thunk_map = {
            o.op_id: self.emit_thunk_for_op(
                o,
                emit_ctx.replace_dg(op.dataflow.subgraph),
            )
            for o in op.dataflow.nodes
        }

        all_shapes = list(emit_ctx.dg.get_output_shapes(op).values()) + list(
            emit_ctx.dg.get_input_shapes(op, simplify=True).values()
        )
        is_dynamic_shapes = any(s.is_dynamic() for s in all_shapes)
        is_dynamic_ops = any(op.is_dynamic() for op in op.dataflow.nodes)

        should_codegen_base = emit_ctx.exec_cfg.enable_codegen_dataflows
        should_codegen_dynamic = emit_ctx.exec_cfg.enable_codegen_dynamic_dataflows
        should_codegen = should_codegen_base and (not is_dynamic_shapes or should_codegen_dynamic)

        # NOTE: While we can codegen dynamic shapes, dynamic ops require ctx access,
        # thus cannot be codegened.
        # NOTE: Also note that is_dynamic_ops should never be true,
        # since we do not group dynamic ops.
        if (not is_dynamic_ops) and should_codegen:
            lam = lambda op, ins: internal_thunk_map[op.op_id](ins, None)  # type: ignore
            return self._compile_dataflow_to_thunk(
                op, lambda xs: op.dataflow.execute(xs, thunk_map=lam), emit_ctx
            )
        else:
            log.info(
                "Not codegening dataflow %s. is_dynamic_shapes: %s, is_dynamic_ops: %s",
                op,
                is_dynamic_shapes,
                is_dynamic_ops,
            )
            compiled_exec = op.dataflow.get_compiled_executor(op.op_id, internal_thunk_map)

            return lambda inps, exec_ctx: compiled_exec(inps, exec_ctx)

    def _compile_dataflow_to_thunk(
        self,
        op: top.ExecDataflowOp,
        interp_exec_func_: Callable[[tuple[BackendTensorT, ...]], tuple[BackendTensorT, ...]],
        emit_ctx: ThunkEmissionCtx[BackendTensorT],
        is_dynamic: bool = False,
    ) -> Thunk[BackendTensorT]:
        exec_cfg = emit_ctx.exec_cfg
        analysis_ctx = emit_ctx.analysis_ctx
        backend = emit_ctx.backend

        op_id = op.op_id
        pg = op.dataflow.subgraph.parent_graph
        assert pg is not None

        exec_dev = analysis_ctx.get_op_device(pg.ops_by_id[op_id].op)

        input_shapes = pg.get_input_shapes_list(op)
        input_dtypes = pg.get_input_dtypes_list(op)
        output_types = pg.get_output_dtypes(op)

        input_devs = CompilationCtx(pg, analysis_ctx, exec_cfg).get_input_devices_list(op)

        log.debug(
            "Codegening dataflow %s with example inputs %s",
            op_id,
            tuple(zip(input_shapes, input_dtypes, strict=False)),
        )

        donatable_args: tuple[int, ...] = ()
        if analysis_ctx._donatable_args is not None:
            donatable_args = tuple(analysis_ctx.donatable_args[op_id])

        # A tuple of example inputs (of the same shape and type as the actual ones) to be passed
        # to the trace function
        # TODO: Once we add stride tracking, we should create example_inputs with the correct
        # strides.
        example_inputs = tuple(
            backend.zeros_tensor(
                backend.to_backend_shape(s.as_static()),
                dtype=backend.to_backend_datatype(dt),
                dev=backend.to_backend_device_obj(dev),
            )
            for (s, dt, dev) in zip(input_shapes, input_dtypes, input_devs, strict=True)
        )

        if exec_cfg.can_inplace_write() and has_buffer_stored_outputs(pg, op, analysis_ctx):
            log.info("Applying inplace write wrapper for %s", op_id)
            (
                interp_exec_func_,
                example_inputs,
                donatable_args,
            ) = make_inplace_write_wrapper(
                exec_cfg,
                analysis_ctx,
                pg,
                op,
                output_types,
                interp_exec_func_,
                example_inputs,
                donatable_args,
            )

        if exec_cfg.can_lazy_slice() and has_lazy_sliced_outputs(pg, op, analysis_ctx):
            # NOTE: We apply the lazy slice wrapper after the inplace write wrapper,
            # as the lazy slice wrapper only affects the original inputs.
            log.info("Applying lazy slice wrapper for %s", op_id)
            (
                interp_exec_func_,
                example_inputs,
                donatable_args,
            ) = make_lazy_slice_thunk_wrapper(
                exec_cfg,
                analysis_ctx,
                pg,
                op,
                output_types,
                interp_exec_func_,
                example_inputs,
                donatable_args,
            )

        example_inputs_shapes_and_dtypes = optree.tree_map(
            lambda x: (
                tuple(x.shape),
                backend.to_tpo_dtype(x.dtype),
            )
            if not isinstance(x, int)
            else ((), dtypes.default_int),
            example_inputs,
        )

        # Enhanced logging with donation source information
        log.info(
            "Jitting (%s) op %s with inputs %s and donations %s",
            str(exec_cfg.get_canonical_backend_name()),
            op_id,
            example_inputs_shapes_and_dtypes,
            donatable_args,
        )

        # TODO: We should not need to pass example inputs here.
        # TODO: All we want is to call jax.jit or torch.compile.
        # TODO: Only if we desire warmup or profile, should we have to pass example inputs.
        # TODO: However, for jitting dynamic ops, we cannot create example inputs (unless we were
        # to use the min size possible).
        # NOTE: However, torch.jit.trace does require example inputs.
        exec_func = backend.trace_codegen_thunk(
            interp_exec_func_,
            op_id,
            exec_dev,
            exec_cfg,
            example_inputs,
            donatable_args,
            analysis_ctx,
            pg,
        )

        # Warm-up the thunk
        if exec_cfg.enable_codegen_thunk_warmup:
            exec_func(example_inputs, None)  # type: ignore

        if exec_cfg.profile_codegen_kernels:
            iters = 10
            measurements = []
            for _ in range(iters):
                example_inputs_copy = tuple(
                    backend.copy(x) if not isinstance(x, int) else x for x in example_inputs
                )
                with Timer() as timer:
                    exec_func(example_inputs_copy, None)  # type: ignore
                    backend.sync()
                measurements.append(timer.elapsed_ms)
            avg_elapsed_ms = round(sum(measurements) / iters, 2)
            std_elapsed_ms = round(float(np.std(measurements)), 2)
            min_elapsed_ms = round(min(measurements), 2)
            max_elapsed_ms = round(max(measurements), 2)

            log.info(
                "Thunk %s profiling - %sms ±%sms (min: %sms, max: %sms)",
                op_id,
                avg_elapsed_ms,
                std_elapsed_ms,
                min_elapsed_ms,
                max_elapsed_ms,
            )

        return exec_func  # type: ignore
