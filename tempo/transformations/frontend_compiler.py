import pprint
from typing import Callable, Optional, Tuple

from tempo.core.analysis_ctx import AnalysisCtx
from tempo.core.configs import ExecutionConfig
from tempo.core.dependence_graph import PDG
from tempo.core.isl_context_factory import get_isl_context
from tempo.transformations.compilation_pass import AbstractCompilationPass, CompilationCtx
from tempo.transformations.iterate_compilation_pass import Pipeline
from tempo.utils import logger
from tempo.utils.common import Timer
from tempo.utils.dg_utils import is_window_access

log = logger.get_logger(__name__)


def compile_frontend(
    dg: PDG,
    exec_cfg: ExecutionConfig,
    custom_pipeline_fnc: Optional[Callable[[CompilationCtx], AbstractCompilationPass]] = None,
    silent: bool = False,
) -> Tuple[CompilationCtx, float]:
    """Compiles the frontend portion of the program.

    Args:
        dg: The program dependence graph
        exec_cfg: The execution configuration
        custom_pipeline_fnc: Optional custom pipeline function
        silent: Whether to suppress logging output

    Returns:
        The compilation context after compilation and the elapsed time in milliseconds
    """

    # NOTE: All parameters must have a bound definition.

    if custom_pipeline_fnc is None:
        pipeline_fnc: Callable[[CompilationCtx], AbstractCompilationPass] = (
            Pipeline.get_default_pipeline(exec_cfg)
        )
    else:
        pipeline_fnc = custom_pipeline_fnc

    if not silent:
        log.info(
            "Starting frontend compilation pipeline with config:\n %s",
            pprint.pformat(exec_cfg.__dict__),
        )

    with Timer() as timer:
        isl_ctx = get_isl_context(exec_cfg)
        compilation_ctx = CompilationCtx(dg, AnalysisCtx(isl_ctx), exec_cfg)
        compilation_ctx.analysis_ctx._is_incremental_algo = any(
            is_window_access(m) for _, _, e in dg.get_all_edges() for m in e.expr.members
        )
        pipeline = pipeline_fnc(compilation_ctx)
        new_ctx, _, _ = pipeline.run()

    new_ctx.analysis_ctx._compilation_time_breakdown = (
        pipeline.elapsed_ms_profile
        if isinstance(pipeline, Pipeline)
        else {"frontend_compilation": timer.elapsed_ms}
    )

    if not silent:
        log.info(
            "Frontend compilation pipeline took %s seconds",
            timer.elapsed_s,
        )
        log.info("Stage times (ms):\n %s", pipeline.elapsed_ms_profile)  # type: ignore

    return new_ctx, timer.elapsed_ms
