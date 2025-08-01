import functools
from typing import Tuple

from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.dependence_graph import PDG
from tempo.core.domain import Domain
from tempo.transformations.compilation_pass import Transformation
from tempo.utils import logger
from tempo.utils.dg_utils import remove_dim_from_op_domain

log = logger.get_logger(__name__)


def _is_dynamic_exception(dg: PDG, op: top.TensorOp, var_to_rem_from_dom: ie.Symbol) -> bool:
    if isinstance(op, top.RandOp):
        return True
    if isinstance(op, top.UDFOp):
        return True
    if isinstance(op, top.ExpandOp):
        if var_to_rem_from_dom in op.sizes.vars_used():
            return True
    if isinstance(op, top.PadOp):
        if (
            var_to_rem_from_dom in ie.lift_to_int_ie(op.padding[0]).vars_used()
            or var_to_rem_from_dom in ie.lift_to_int_ie(op.padding[1]).vars_used()
        ):
            return True
    if isinstance(op, top.IndexSliceOp):
        if isinstance(op.length, ie.IntIndexValue) and var_to_rem_from_dom in op.length.vars_used():
            return True
    if isinstance(op, top.EvalSymbolOp):
        # if reduction_dim.equivalent(op.symbol):
        return True
    if isinstance(op, top.ConstOp) and var_to_rem_from_dom in op.shape.vars_used():
        return True
    return False


def all_dependents_are_point(dg: PDG, op: top.TensorOp, reduction_dim: ie.Symbol) -> bool:
    reduction_dim_idx = op.domain.find_variable_index(reduction_dim)
    for _, dep_data in dg.get_flat_direct_dependents(op):
        e_idx = dep_data.expr.members[reduction_dim_idx]
        if not e_idx.is_point():
            return False
    return True


class DomainReduction(Transformation):
    def _run(self) -> Tuple[PDG, bool]:  # noqa: C901
        new_dg = self.ctx.dg
        dims_removed_count = 0

        # Start with all nodes to check
        nodes_to_check = set(new_dg.nodes)

        while nodes_to_check:
            # Get the next node to process
            op = nodes_to_check.pop()

            union_depy_domain = self._get_dependency_union_dom(new_dg, op)
            # NOTE: If all depys have a smaller domain than the dependent, then the dependent
            # is presumably varying with an extra dimension for no reason.
            if union_depy_domain.is_contained_in(op.domain) and not union_depy_domain == op.domain:
                diff_dom = Domain.difference(op.domain, union_depy_domain)
                op_to_remove_dim_from = op
                for dim_to_maybe_remove in diff_dom.variables:
                    if (
                        not _is_dynamic_exception(new_dg, op, dim_to_maybe_remove)
                    ) and all_dependents_are_point(new_dg, op, dim_to_maybe_remove):
                        op_to_remove_dim_from = remove_dim_from_op_domain(
                            new_dg, op_to_remove_dim_from, dim_to_maybe_remove
                        )
                        dims_removed_count += 1

                        # Add all flat direct dependents back to nodes_to_check
                        # since they might be affected by the domain reduction
                        for dependent_op, _ in new_dg.get_flat_direct_dependents(
                            op_to_remove_dim_from
                        ):
                            nodes_to_check.add(dependent_op)
                        break

        log.info("Domain reduction count: %d", dims_removed_count)

        return new_dg, dims_removed_count > 0

    def _get_dependency_union_dom(self, new_dg: PDG, op: top.TensorOp) -> Domain:
        return functools.reduce(
            lambda acc, depy_dom: Domain.union(acc, depy_dom),
            [
                Domain.union(
                    depy_op.domain,
                    depy_data.expr.vars_used(),
                    depy_data.cond.vars_used() if depy_data.cond is not None else (),
                )
                for depy_op, depy_data in new_dg.get_flat_direct_dependencies(op)
            ],
            Domain.empty(),
        )
