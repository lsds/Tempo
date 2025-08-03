import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Callable, Generic, Tuple

from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.analysis_ctx import AnalysisCtx
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import BackendTensorT
from tempo.core.tensor_op import TensorOp
from tempo.core.thunk import Thunk, ThunkEmissionCtx
from tempo.runtime.inplace_buffer_thunk_wrapper import (
    has_buffer_stored_outputs,
    make_inplace_write_wrapper,
)
from tempo.runtime.lazy_slice_thunk_wrapper import make_lazy_slice_thunk_wrapper
from tempo.utils import logger

log = logger.get_logger(__name__)


class ThunkEmitter(Generic[BackendTensorT], ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def _emit_thunk_for_op(
        self,
        op: TensorOp,
        ctx: ThunkEmissionCtx[BackendTensorT],
    ) -> Thunk[BackendTensorT]:
        pass

    def emit_thunk_for_op(
        self,
        op: TensorOp,
        ctx: ThunkEmissionCtx[BackendTensorT],
    ) -> Thunk[BackendTensorT]:
        # Since both dataflow and UDFs are handled generically, we handle them here
        if type(op) is top.ExecDataflowOp:
            return self._dataflow_op_translation(op, ctx)
        elif type(op) is top.UDFOp:
            return op.desc.thunk_translation(op, ctx)
        else:
            # Otherwise, emit the thunk using the backend-specific method
            return self._emit_thunk_for_op(op, ctx)

    def _dataflow_op_translation(
        self,
        op: top.ExecDataflowOp,
        emit_ctx: ThunkEmissionCtx[BackendTensorT],
    ) -> Thunk[BackendTensorT]:
        assert isinstance(op, top.ExecDataflowOp)

        internal_thunk_map = {
            o.op_id: self.emit_thunk_for_op(
                o,
                dataclasses.replace(emit_ctx, dg=op.dataflow.subgraph),
            )
            for o in op.dataflow.nodes
        }

        all_shapes = list(emit_ctx.dg.get_output_shapes(op).values()) + list(
            emit_ctx.dg.get_input_shapes(op).values()
        )
        is_dynamic = any(s.is_dynamic() for s in all_shapes)
        if not is_dynamic and emit_ctx.exec_cfg.enable_codegen_dataflows:
            lam = lambda op, ins: internal_thunk_map[op.op_id](ins, None)  # type: ignore
            try:
                return self.compile_dataflow_to_thunk(
                    op,
                    lambda xs: op.dataflow.execute(xs, thunk_map=lam),
                    emit_ctx.compile_time_known_symbol_values,
                    emit_ctx.exec_cfg,
                    emit_ctx.analysis_ctx,
                )
            except Exception as e:
                log.error(
                    "Error while codegening dataflow %s",
                    op,
                )
                raise e
        else:
            # NOTE: dynamic dataflows require ctx
            return lambda xs, ctx: op.dataflow.execute(
                xs, thunk_map=lambda op, ins: internal_thunk_map[op.op_id](ins, ctx)
            )

    def compile_dataflow_to_thunk(
        self,
        op: top.ExecDataflowOp,
        interp_exec_func_: Callable[[Tuple[BackendTensorT, ...]], Tuple[BackendTensorT, ...]],
        compile_time_known_symbol_values: Mapping[ie.Symbol, int],
        # TODO: Just pass in the CompilationCtx...
        exec_cfg: ExecutionConfig,
        analysis_ctx: AnalysisCtx,
    ) -> Thunk[BackendTensorT]:
        op_id = op.op_id
        pg = op.dataflow.subgraph.parent_graph
        assert pg is not None

        exec_dev = analysis_ctx.get_op_device(pg.ops_by_id[op_id].op)

        input_shapes = pg.get_input_shapes_list(op)
        input_dtypes = pg.get_input_dtypes_list(op)
        output_types = pg.get_output_dtypes(op)

        input_devs = CompilationCtx(pg, analysis_ctx, exec_cfg).get_input_devices_list(op)

        from tempo.runtime.backends.backend import DLBackend

        backend = DLBackend.get_backend(exec_cfg.backend)
        # interp_exec_func = interp_exec_func_
        # TODO: I don' actually think this helps, and its breaking encapsulation. Remove later.
        #if backend.get_backend_name() == DLBackendName.TORCH:
        #    # interp_exec_func = lambda xs: tuple(
        #    #    o.contiguous()  # type: ignore
        #    #    for o in interp_exec_func_(xs)
        #    # )
        #    interp_exec_func = interp_exec_func_

        #    if len(input_shapes) > 0:
        #        from torch.fx import symbolic_trace

        #        interp_exec_func = symbolic_trace(interp_exec_func).forward
        #    else:
        #        interp_exec_func = interp_exec_func_
        #else:
        #    interp_exec_func = interp_exec_func_

        log.debug(
            "Codegening dataflow %s with example inputs %s",
            op_id,
            tuple(zip(input_shapes, input_dtypes, strict=False)),
        )

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
            for (s, dt, dev) in zip(input_shapes, input_dtypes, input_devs, strict=False)
        )

        donatable_args: list[int] = []
        if analysis_ctx._donatable_args is not None:
            donatable_args = list(analysis_ctx.donatable_args[op_id])

        if (
            exec_cfg.enable_inplace_write
            and exec_cfg.enable_hybrid_tensorstore
            and has_buffer_stored_outputs(pg, op, analysis_ctx)
        ):
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

        if exec_cfg.enable_lazy_slice and exec_cfg.enable_hybrid_tensorstore:
            # NOTE: We apply the lazy slice wrapper after the inplace write wrapper,
            # as the lazy slice wrapper only affects the original inputs.
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

        exec_func = backend.trace_codegen_thunk(
            interp_exec_func_,
            op_id,
            exec_dev,
            exec_cfg,
            example_inputs,
            tuple(donatable_args),
            analysis_ctx,
            pg,
        )

        return exec_func  # type: ignore
