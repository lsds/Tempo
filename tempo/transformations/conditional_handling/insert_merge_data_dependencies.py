import functools

from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.datatypes import OpInId
from tempo.core.dependence_graph import PDG, DependencyData
from tempo.transformations.compilation_pass import Transformation
from tempo.utils import isl as isl_utils
from tempo.utils import logger

log = logger.get_logger(__name__)


def remove_unconditional_merges(ctx: CompilationCtx) -> bool:  # noqa: C901
    dg = ctx.dg
    modified = False
    all_merges = [op for op in dg.nodes if isinstance(op, top.MergeOp)]
    for merge_op in all_merges:
        # NOTE: If the first non-false branch is true, then we can remove the merge
        is_suitable, idx = get_first_non_false_branch(dg, merge_op)

        if is_suitable:
            dependencies = dg.get_flat_direct_dependencies(merge_op)
            dependency_op, dependency_data = dependencies[idx]

            for dependent_op, dependent_op_data in dg.get_flat_direct_dependents(merge_op):
                combined_edge_data = isl_utils.combine_edges(
                    dependent_op,
                    dependent_op_data,
                    merge_op,
                    dependency_data,
                    dependency_op,
                    dg.static_bounds,
                    ctx.analysis_ctx.isl_ctx,
                )
                dg.add_edge(dependent_op, dependency_op, combined_edge_data)
            dg.remove_op(merge_op)
            modified = True

    return modified


def get_first_non_false_branch(dg: PDG, merge_op: top.MergeOp) -> tuple[bool, int]:
    is_suitable = True
    idx = -1
    # NOTE: If the first non-false branch is true, then we can remove the merge
    for i, (_, depy_data) in enumerate(dg.get_flat_direct_dependencies(merge_op)):
        cond = depy_data.cond
        if cond is None:
            cond = ie.ConstBool(True)

        if cond.struct_eq(ie.ConstBool(True)):
            # The first True branch found will always be the first branch to evaluate to true
            # Thus we can remove the merge
            idx = i
            break
        elif cond.struct_eq(ie.ConstBool(False)):
            # We can skip False branches because they never evaluate to true
            ...
        else:
            is_suitable = False
            break
    return is_suitable, idx


class InsertMergeDataDependencies(Transformation):
    def _run(self) -> tuple[PDG, bool]:
        new_dg = self.ctx.dg

        modified = False

        # Then process the remaining merges, adding their dependencies
        for op in new_dg.nodes:
            if isinstance(op, top.MergeOp):
                self._insert_dependency_edges(new_dg, op)
                modified |= True

        modified |= remove_unconditional_merges(self.ctx)
        return new_dg, modified

    def _insert_dependency_edges(self, new_dg: PDG, op: top.MergeOp) -> None:
        branch_conds = new_dg.ops_by_id[op.op_id].uncommitted_branch_conds
        assert len(branch_conds) > 0, "Merge op has no branch conditions"
        conds = [c for c, _, _ in branch_conds]
        for i, (cond, def_tensor, expr) in enumerate(branch_conds):
            new_cond = self._compute_cond(conds, i, cond)

            def_op = new_dg.ops_by_id[def_tensor.op_id].op
            dep_data = DependencyData(expr, def_tensor.output_id, OpInId(i), new_cond)
            new_dg.add_edge(op, def_op, dep_data)
            log.debug("Inserted edge %s -> %s with %s", op, def_op, dep_data)

        branch_conds.clear()

    def _compute_cond(
        self, orig_conds: list[ie.BooleanIndexValue], i: int, cond: ie.BooleanIndexValue
    ) -> ie.BooleanIndexValue:
        if i == 0:
            new_cond = cond
        elif i == 1:
            new_cond = cond & (~orig_conds[0])
        else:
            new_cond = cond & (~functools.reduce(lambda a, b: a | b, orig_conds[:i]))

        return new_cond
