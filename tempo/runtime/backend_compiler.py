import heapq

from tempo.core import tensor_ops as top
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.dl_backend import DLBackend
from tempo.core.external_state_store import ExternalStateStore
from tempo.core.utils import bytes_to_human_readable
from tempo.runtime.executor.executor import Executor, ExecutorCtx
from tempo.runtime.executor.nongen_executor import NonGenInterpreterExecutor
from tempo.runtime.tensor_store.hybrid_tensor_store import HybridTensorStore
from tempo.runtime.tensor_store.point_tensor_store import PointTensorStore
from tempo.utils import logger
from tempo.utils.common import Timer
from tempo.utils.memory_estimator import MemoryEstimator

log = logger.get_logger(__name__)


def compile_backend(
    ctx: CompilationCtx,
    silent: bool = False,
) -> tuple[Executor, CompilationCtx]:
    """Compiles the backend portion of the program.

    Args:
        ctx: The analysis context containing compilation information
        dg: The program dependence graph
        exec_cfg: The execution configuration
        silent: Whether to suppress logging output

    Returns:
        An executor for running the compiled program and the compilation context
    """
    cfg = ctx.exec_cfg
    with Timer() as backend_config_timer:
        backend: DLBackend = DLBackend.get_backend(cfg.backend)()
        backend.configure(cfg)

    if not silent:
        log.info("Starting backend (%s) compilation...", cfg.backend)
        est = MemoryEstimator(ctx)
        all_bytes = [est.estimate_op_size_bytes(op.op_id) or 0 for op in ctx.dg.nodes]
        top_10 = heapq.nlargest(10, all_bytes)
        top_10_human = [bytes_to_human_readable(b) for b in top_10]

        log.info(
            "Top 10 op sizes: %s",
            ", ".join(list(top_10_human)),
        )

    with Timer() as ts_timer:
        if cfg.enable_hybrid_tensorstore:
            ts_class = HybridTensorStore
        else:
            ts_class = PointTensorStore
        ts = ts_class(ctx)
        if cfg.executor_debug_mode:
            # ts.wrap_all_tensors_with_debug_checker()
            ...

        clean_up_dict = {}
        for udf in ctx.dg.stateful_udf_nodes:
            assert isinstance(udf, top.UDFOp)
            clean_up_dict.update(udf.desc.clean_up_state or {})
        external_state_store = ExternalStateStore(clean_up_dict)

    # NOTE: Setup a timer for all loading times.
    ctx.analysis_ctx._loading_time_timer = Timer()

    with Timer() as codegen_timer:
        executor_ctx = ExecutorCtx(external_state_store, ts, ctx, backend)
        executor = NonGenInterpreterExecutor(executor_ctx)

    total_ms = backend_config_timer.elapsed_ms + ts_timer.elapsed_ms + codegen_timer.elapsed_ms

    ctx.analysis_ctx.compilation_profile_ms["Backend"] = {
        "Total": total_ms,
        "DLBackendConfig": backend_config_timer.elapsed_ms,
        "TensorStore": ts_timer.elapsed_ms,
        "Codegen": codegen_timer.elapsed_ms + ctx.analysis_ctx.load_timer.elapsed_ms,
    }

    if not silent:
        log.info(
            "Backend (%s) compilation took %s seconds.",
            cfg.backend,
            round(total_ms / 1000, 2),
        )

    return executor, ctx
