# from __future__ import annotations
#
# from typing import Tuple
#
# from tempo.core.dependence_graph import PDG
# from tempo.core.schedule.schedule_post_processor import SchedulePostProcessor
# from tempo.core.schedule.schedule_printer import SchedulePrinter
# from tempo.transformations.compilation_pass import CompilationCtx, Transformation
# from tempo.utils import logger
#
# log = logger.get_logger(__name__)
#
#
# class PostProcessSchedule(Transformation):
#    def __init__(self, ctx: CompilationCtx) -> None:
#        super().__init__(ctx)
#
#    def _run(self) -> Tuple[PDG, bool]:  # noqa: C901
#        tpo_schedule = SchedulePostProcessor(
#            self.ctx.analysis_ctx.execution_schedule, self.ctx.dg, self.ctx.exec_cfg
#        ).start_walk()
#
#        self.ctx.analysis_ctx._execution_schedule = tpo_schedule
#
#        if self.ctx.exec_cfg.render_schedule:
#            try:
#                tpo_schedule.render_to_dot(self.ctx.exec_cfg.path + "/post_proc_tpo_ast.dot")
#                printer = SchedulePrinter(self.ctx)
#                str_schedule = printer.print_schedule()
#                with open(self.ctx.exec_cfg.path + "/post_proc_tpo_schedule.py", "w") as f:
#                    f.write(str_schedule)
#            except Exception as e:
#                log.error("Skipping rendering Tempo AST to dot due to error: %s", e)
#
#        return self.ctx.dg, True
#
