from typing import Callable, Optional

from tempo.core import index_expr as ie
from tempo.core.configs import ExecutionConfig
from tempo.core.dependence_graph import PDG
from tempo.runtime.backend_compiler import compile_backend
from tempo.runtime.executor.executor import Executor
from tempo.transformations.compilation_pass import AbstractCompilationPass, CompilationCtx
from tempo.transformations.frontend_compiler import compile_frontend
from tempo.utils import logger

log = logger.get_logger(__name__)


class Compiler:
    def __init__(
        self,
        dg: PDG,
        exec_cfg: ExecutionConfig,
        silent: bool = False,
    ) -> None:
        self.dg = dg
        self.exec_cfg = exec_cfg
        self.silent = silent

    def compile(  # noqa: A001, A003
        self,
        custom_pipeline_fnc: Optional[Callable[[CompilationCtx], AbstractCompilationPass]] = None,
    ) -> Executor:
        with ie.StructEqCheckCtxManager():
            return self._compile(custom_pipeline_fnc)

    def _compile(  # noqa: A001, A003
        self,
        custom_pipeline_fnc: Optional[Callable[[CompilationCtx], AbstractCompilationPass]] = None,
    ) -> Executor:
        new_ctx, _ = compile_frontend(self.dg, self.exec_cfg, custom_pipeline_fnc, self.silent)
        return compile_backend(new_ctx, self.silent)
