import heapq

from tempo.core import tensor_ops as top
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.external_state_store import ExternalStateStore
from tempo.core.utils import bytes_to_human_readable
from tempo.runtime.backends.backend import DLBackend
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
) -> Executor:
    """Compiles the backend portion of the program.

    Args:
        ctx: The analysis context containing compilation information
        dg: The program dependence graph
        exec_cfg: The execution configuration
        silent: Whether to suppress logging output

    Returns:
        An executor for running the compiled program
    """
    cfg = ctx.exec_cfg
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

    with Timer() as timer:
        if cfg.enable_hybrid_tensorstore:
            ts_class = HybridTensorStore
        else:
            ts_class = PointTensorStore
        ts = ts_class(ctx)
        if cfg.executor_debug_mode:
            # ts.wrap_all_tensors_with_debug_checker()
            pass

        clean_up_dict = {}
        for udf in ctx.dg.stateful_udf_nodes:
            assert isinstance(udf, top.UDFOp)
            clean_up_dict.update(udf.desc.clean_up_state or {})
        external_state_store = ExternalStateStore(clean_up_dict)

        executor_ctx = ExecutorCtx(ctx.dg, external_state_store, ts, cfg, ctx.analysis_ctx, backend)
        executor = NonGenInterpreterExecutor(executor_ctx)

    ctx.analysis_ctx.compilation_time_breakdown["backend_compilation"] = timer.elapsed_ms

    if not silent:
        log.info(
            "Backend (%s) compilation took %s seconds.",
            cfg.backend,
            timer.elapsed_s,
        )

    return executor
