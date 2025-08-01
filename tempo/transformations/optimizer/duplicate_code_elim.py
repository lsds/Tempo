from itertools import combinations
from typing import Tuple

from tempo.core.dependence_graph import PDG
from tempo.core.op_tags import NO_DEDUP_TAG
from tempo.transformations.compilation_pass import Transformation
from tempo.utils import logger

log = logger.get_logger(__name__)


class DuplicateCodeElimination(Transformation):
    # NOTE: The intuition behind this transformation is that we iterate the nodes, until we
    # find a node that has more than one dependent. If two of those dependents do the same
    # operation, on the same data, they are duplicates and we can remove one of them.

    def _run(self) -> Tuple[PDG, bool]:  # noqa: C901
        # new_dg = self.copy_dg()
        new_dg = self.ctx.dg

        removed = 0
        removed_ops = set()
        for n1, n2 in combinations(list(new_dg.nodes_with_no_dependencies), 2):
            if (NO_DEDUP_TAG in n1.tags) or (NO_DEDUP_TAG in n2.tags):
                continue
            if n1.op_id in removed_ops or n2.op_id in removed_ops:
                continue
            if n1.equivalent(n2):
                new_dg.move_dependents(n2, n1)
                new_dg.remove_op(n2)
                removed_ops.add(n2.op_id)
                log.debug("Deduplicating ops %s and %s", n1, n2)
                removed += 1
        changed = True
        while changed:
            changed = False
            for node in list(new_dg.nodes):
                if node.op_id in removed_ops:
                    continue
                deps = new_dg.get_flat_direct_dependents(node)
                if len(deps) > 1:
                    for (dep1_op, dep1_data), (dep2_op, dep2_data) in combinations(deps, 2):
                        if NO_DEDUP_TAG in dep1_op.tags or NO_DEDUP_TAG in dep2_op.tags:
                            continue
                        if dep1_op.op_id in removed_ops or dep2_op.op_id in removed_ops:
                            continue
                        if dep1_op.op_id == dep2_op.op_id:
                            continue
                        if dep1_op.equivalent(dep2_op):
                            dep1_dependencies = {
                                op.op_id for op, _ in new_dg.get_flat_direct_dependencies(dep1_op)
                            }
                            dep2_dependencies = {
                                op.op_id for op, _ in new_dg.get_flat_direct_dependencies(dep2_op)
                            }
                            if dep1_dependencies == dep2_dependencies:
                                if dep1_data == dep2_data:
                                    new_dg.move_dependents(dep2_op, dep1_op)
                                    new_dg.remove_op(dep2_op)
                                    removed_ops.add(dep2_op.op_id)
                                    log.debug(
                                        "Deduplicating ops %s and %s, with deps %s and %s",
                                        dep1_op,
                                        dep2_op,
                                        dep1_data,
                                        dep2_data,
                                    )

                                    removed += 1
                                    changed = True

        log.info("Deduplicated %s duplicate ops", removed)

        return new_dg, removed > 0
