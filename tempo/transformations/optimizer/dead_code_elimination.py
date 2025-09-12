from tempo.core import tensor_ops as top
from tempo.core.datatypes import OpId
from tempo.core.dependence_graph import PDG
from tempo.transformations.compilation_pass import Transformation
from tempo.utils import isl as isl_utils
from tempo.utils import logger

log = logger.get_logger(__name__)


def _is_dead(
    dg: PDG,
    node: top.TensorOp,
    visited: set[OpId] | None = None,
) -> bool:
    if visited is None:
        visited = {node.op_id}

    all_dependents = [x for x, _ in dg.get_flat_direct_dependents(node)]

    if len(all_dependents) == 0:
        return not node.is_sink

    return all(
        dependent in visited or _is_dead(dg, dependent, visited.union({dependent.op_id}))
        for dependent in all_dependents
    )


class DeadCodeElimination(Transformation):
    def _run(self) -> tuple[PDG, bool]:  # noqa: C901
        # new_dg = self.copy_dg()
        dg = self.ctx.dg

        removed = 0
        while True:
            changed = False
            removed += self.clean_up_ident_ops(dg)

            for node in list(dg.nodes_with_no_dependents):
                if _is_dead(dg, node):
                    dg.remove_op(node)
                    removed += 1
                    changed = True

            for comp in list(dg.weakly_connected_components):
                if not any(op.is_sink for op in comp):
                    # log.info(f"Removing {len(comp)} dead ops from a dead component.")
                    for op in comp:
                        dg.remove_op(op)
                        removed += 1
                        changed = True
            if not changed:
                break

        log.info("Removed %s dead ops", removed)

        return dg, removed > 0

    def clean_up_ident_ops(self, dg: PDG) -> int:
        removed = 0
        for node in list(dg.nodes):
            if isinstance(node, top.IdentOp):
                depy, depy_data = dg.get_flat_direct_dependencies(node)[0]
                for dep, dep_data in dg.get_flat_direct_dependents(node):
                    if isl_utils.can_combine_edges(
                        dep,
                        dep_data,
                        node,
                        depy_data,
                        depy,
                        dg.static_bounds,
                        self.ctx.analysis_ctx.isl_ctx,
                    ):
                        combined_dep_data = isl_utils.combine_edges(
                            dep,
                            dep_data,
                            node,
                            depy_data,
                            depy,
                            dg.static_bounds,
                            self.ctx.analysis_ctx.isl_ctx,
                        )
                        dg.remove_edge(dep, node, dep_data)
                        # print(f"Removing edge from {dep} to {node}: {dep_data}")
                        # print(f"Adding edge from {dep} to {depy}: {combined_dep_data}")
                        dg.add_edge(dep, depy, combined_dep_data)
                        removed += 1
        return removed
