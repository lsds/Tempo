import time

from tempo.core.dependence_graph import PDG
from tempo.core.dg_renderer import DGRenderer
from tempo.transformations.compilation_pass import CompilationCtx, Transformation


class VisualizeGraph(Transformation):
    def __init__(self, ctx: CompilationCtx, name: str):
        super().__init__(ctx)
        self._name = name

    def name(self) -> str:
        return str(super().name() + "_" + self._name)

    def _run(self) -> tuple[PDG, bool]:
        file_name = "./dgs/" + time.strftime("%Y%m%d-%H%M") + "/" + self._name

        renderer = DGRenderer(self.ctx, file_name)
        renderer.render()

        return self.ctx.dg, False
