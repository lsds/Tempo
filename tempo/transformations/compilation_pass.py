import copy
from abc import ABC, abstractmethod
from typing import Any, Tuple

from tempo.core.analysis_ctx import AnalysisCtx
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.dependence_graph import PDG
from tempo.utils.common import Timer
from tempo.utils.logger import get_logger

log = get_logger(__name__)


class AbstractCompilationPass(ABC):
    """A CompilationPass is any operation done on a CompilationContext.
    This includes analysis, optimization, or transformation.
    """

    def __init__(self, ctx: CompilationCtx):
        self.ctx = ctx

    def name(self) -> str:
        return self.__class__.__name__

    def copy_dg(self) -> PDG:
        return copy.deepcopy(self.ctx.dg)

    def copy_ctx(self) -> CompilationCtx:
        return copy.deepcopy(self.ctx)

    def run(self) -> Tuple[CompilationCtx, bool, float]:
        """Runs the pass on the input context and returns the new context,
        whether it was modified, and the runtime of the transformation.

        Returns:
            Tuple[CompilationCtx, bool, float]: A tuple containing the modified context,
            a boolean indicating whether the context was modified, and the runtime in milliseconds.
        """
        log.debug("----------- STARTING %s ------------", self.name())
        import tempo.core.global_objects as glob

        with Timer() as timer:
            glob.set_active_dg(self.ctx.dg)
            new_ctx, modified = self._create_new_context(self._run())  # type: ignore
            glob.set_active_dg(new_ctx.dg)

        log.debug("----------- ENDED %s ------------", self.name())
        log.debug("%s took %dms to run. Modified? %s", self.name(), timer.elapsed_ms, modified)

        return new_ctx, modified, timer.elapsed_ms

    @abstractmethod
    def _run(self) -> Any:
        """Abstract method to be implemented by subclasses.
        Must return either a new PDG and a boolean indicating modification
        or a new AnalysisCtx depending on the pass type.
        """
        pass

    @abstractmethod
    def _create_new_context(self, result: Any) -> Tuple[CompilationCtx, bool]:
        """Creates a new CompilationCtx from the result of `_run`.

        Args:
            result (Union[Tuple[PDG, bool], AnalysisCtx]): The result of `_run`.

        Returns:
            Tuple[CompilationCtx, bool]: The new CompilationCtx and a boolean modification.
        """
        pass


class CompilationPass(AbstractCompilationPass):
    """Base class for general compilation passes that modify the PDG and AnalysisCtx."""

    def __init__(self, ctx: CompilationCtx):
        super().__init__(ctx)

    @abstractmethod
    def _run(self) -> Tuple[CompilationCtx, bool]:
        """Performs the transformation on the given context by returning a new modified context.

        Returns:
            Tuple[CompilationCtx, bool]: A pair of the new CompilationCtx and a boolean modified.
        """
        pass

    def _create_new_context(
        self, result: Tuple[CompilationCtx, bool]
    ) -> Tuple[CompilationCtx, bool]:
        new_ctx, modified = result
        return new_ctx, modified


class Transformation(AbstractCompilationPass):
    """Base class for transformations modifying the PDG."""

    def __init__(self, ctx: CompilationCtx):
        super().__init__(ctx)

    @abstractmethod
    def _run(self) -> Tuple[PDG, bool]:
        """Performs the transformation on the given context by returning a new modified context.

        Returns:
            Tuple[PDG, bool]: A pair of the new PDG and a boolean was modified.
        """
        pass

    def _create_new_context(self, result: Tuple[PDG, bool]) -> Tuple[CompilationCtx, bool]:
        new_dg, modified = result
        new_ctx = CompilationCtx(new_dg, self.ctx.analysis_ctx, self.ctx.exec_cfg)
        return new_ctx, modified


class Analysis(AbstractCompilationPass):
    """Base class for analyses that modify the AnalysisCtx."""

    def __init__(self, ctx: CompilationCtx):
        super().__init__(ctx)

    @abstractmethod
    def _run(self) -> AnalysisCtx:
        """Performs the analysis on the given context by returning a new AnalysisCtx.

        Returns:
            AnalysisCtx: The updated AnalysisCtx after analysis.
        """
        pass

    def _create_new_context(self, result: AnalysisCtx) -> Tuple[CompilationCtx, bool]:
        new_analysis_ctx = result
        new_ctx = CompilationCtx(self.ctx.dg, new_analysis_ctx, self.ctx.exec_cfg)
        return new_ctx, True
