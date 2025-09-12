from __future__ import annotations

from pathlib import Path

import islpy as isl

from tempo.core.dependence_graph import PDG
from tempo.core.schedule.isl_to_tempo_schedule import (
    build_isl_ast_from_schedule,
    get_ast_builder,
    isl_ast_to_tempo_schedule,
)
from tempo.core.schedule.schedule_printer import SchedulePrinter
from tempo.transformations.compilation_pass import CompilationCtx, Transformation
from tempo.transformations.scheduling.isl_schedule_constraint_builder import (
    IslScheduleConstraintsBuilder,
)
from tempo.utils import logger
from tempo.utils.common import Timer

log = logger.get_logger(__name__)


class ScheduleExecution(Transformation):
    def __init__(self, ctx: CompilationCtx, quiet: bool = False) -> None:
        super().__init__(ctx)
        self.quiet = quiet

    def _run(self) -> tuple[PDG, bool]:  # noqa: C901
        new_dg = self.ctx.dg
        isl_ctx = self.ctx.analysis_ctx.isl_ctx

        # 1. Setup the isl schedule constraints
        constraint_builder = IslScheduleConstraintsBuilder(self.ctx)
        # NOTE: ScheduleConstraints will copy the isl_ctx from the domain
        with Timer() as timer:
            sc = constraint_builder.build_schedule_constraints()
        log.debug("Schedule constraint building time: %s seconds", timer.elapsed_s)

        # 2. Run the isl scheduler
        try:
            with Timer() as timer:
                self.isl_schedule = isl.ScheduleConstraints.compute_schedule(sc)
            log.debug("Compute schedule time: %s seconds", timer.elapsed_s)
        except Exception as e:
            # cc = CycleChecker(self.ctx.exec_cfg, self.ctx.dg)
            # cycles = cc.check_cycles()
            # print(f"Found {len(cycles)} impossible cycles.")
            # print(cycles)

            raise ValueError(
                "Unable to find schedule. Check your computation for feasability."
            ) from e

        # def set_to_sequence_node(node: isl.ScheduleNode) -> isl.ScheduleNode:
        #    if (
        #        node.get_type() == isl.schedule_node_type.set
        #    ):
        #        n_children = node.n_children()
        #        children = [node.get_child(i) for i in range(n_children)]

        #        new_node = node.insert_sequence(children)
        #        return new_node
        #    else:
        #        return node

        # self.isl_schedule = self.isl_schedule.map_schedule_node_bottom_up(
        #    set_to_sequence_node
        # )

        if self.ctx.exec_cfg.enable_parallel_block_detection:

            def mark_set_nodes(node: isl.ScheduleNode) -> isl.ScheduleNode:
                # TODO I think this needs to be a "Leaf Set" node, meaning there are no
                # nested Band nodes.
                if node.get_type() == isl.schedule_node_type.set and node.n_children() > 1:
                    # NOTE: can also attach arbitrary python objects to the mark
                    mark = isl.Id("parallel", isl_ctx)
                    return node.insert_mark(mark)
                else:
                    return node

            self.isl_schedule = self.isl_schedule.map_schedule_node_bottom_up(mark_set_nodes)

        if self.ctx.exec_cfg.enable_isolate_loop_conditions:

            def set_loop_types(node: isl.ScheduleNode) -> isl.ScheduleNode:
                if node.get_type() == isl.schedule_node_type.band:
                    for i in range(node.band_n_member()):
                        node = node.band_member_set_ast_loop_type(i, isl.ast_loop_type.separate)
                return node

            self.isl_schedule = self.isl_schedule.map_schedule_node_bottom_up(set_loop_types)

        # log.debug("Schedule in C:\n%s", isl_sched_to_c(self.isl_schedule))

        with Timer() as timer:
            builder = get_ast_builder(isl_ctx)
            isl_ast = build_isl_ast_from_schedule(builder, self.isl_schedule)
            tpo_schedule = isl_ast_to_tempo_schedule(self.ctx, isl_ast, quiet=self.quiet)
        log.debug("Raise to tpo schedule time: %s seconds", timer.elapsed_s)

        # 4. Update the execution schedule in new_dg
        self.ctx.analysis_ctx._execution_schedule = tpo_schedule
        self.ctx.analysis_ctx._isl_execution_schedule = self.isl_schedule

        if not self.quiet and self.ctx.exec_cfg.render_schedule:
            name = "tpo_ast"
            full_path = Path(self.ctx.exec_cfg.path) / f"{name}.dot"
            if full_path.exists():
                name += "_2nd_round"
                full_path = Path(self.ctx.exec_cfg.path) / f"{name}.dot"

            tpo_schedule.render_to_dot(str(full_path))
            printer = SchedulePrinter(self.ctx)
            str_schedule = printer.print_schedule()
            sched_path = Path(self.ctx.exec_cfg.path) / f"{name}.py"
            with open(sched_path, "w") as f:
                f.write(str_schedule)
            log.info("Schedule written to %s", sched_path)

        return new_dg, True
