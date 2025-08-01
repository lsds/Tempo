from typing import Dict, Set, Tuple

import islpy as isl

from tempo.core import isl_types as islt
from tempo.core import tensor_ops as top
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.datatypes import OpInId, OpOutId
from tempo.core.dependence_graph import PDG
from tempo.transformations.compilation_pass import Transformation
from tempo.utils import isl as isl_utils
from tempo.utils import logger

log = logger.get_logger(__name__)


def update_domain(  # noqa: C901
    new_dg: PDG,
    op: top.TensorOp,
    params_set: islt.Set,
    bounds_cache: Dict[top.TensorOp, islt.Set],
    map_cache: Dict[Tuple[top.TensorOp, OpOutId, top.TensorOp, OpInId], islt.Map],
    ctx: CompilationCtx,
) -> bool:
    op = new_dg.ops_by_id[op.op_id].op

    op_e_name = isl_utils.op_id_to_exec_name(op.op_id)
    modified = False

    isl_ctx = ctx.analysis_ctx.isl_ctx
    prev_dom = ctx.analysis_ctx.get_or_make_domain(op)
    new_dom = isl.UnionSet.read_from_str(isl_ctx, "[ ] -> { }")

    if len(new_dg.get_flat_direct_dependents(op)) == 0:
        return False

    # NOTE: Here we update the domain of the op based on the domain, dependence expression and
    # conditions of its dependents. This is done by mapping the dependents' domain to the op's
    # values it depends upon, then unioning the results together.
    for dep_op, dep_data in new_dg.get_flat_direct_dependents(op):
        map_ = map_cache.get((dep_op, dep_data.src_out_idx, op, dep_data.sink_in_idx))
        if map_ is None:
            dep_e_name = isl_utils.op_id_to_exec_name(dep_op.op_id)
            map_ = isl_utils.dependence_to_isl_map(
                dep_data.expr,
                dep_op.domain,
                op.domain,
                dep_e_name,
                op_e_name,
                condition=dep_data.cond,
                ctx=isl_ctx,
            )
            map_cache[(dep_op, dep_data.src_out_idx, op, dep_data.sink_in_idx)] = map_
        dep_dom = ctx.analysis_ctx.get_or_make_domain(dep_op)
        instances_needed_by_dependents = dep_dom.apply(map_)
        new_dom = new_dom.union(instances_needed_by_dependents)

    # Finally, make sure we stay within bounds
    bounds_set = bounds_cache.get(op)
    if bounds_set is None:
        vars_comma_str2 = ",".join([str(v) for v in op.domain.variables])
        parameters_str, bound_conds = isl_utils.get_parameters_and_var_bounds_strs(op.domain)

        full_str = f"[{parameters_str}] -> {{ {op_e_name}[{vars_comma_str2}]: {bound_conds}}}"
        bounds_set = isl.Set.read_from_str(isl_ctx, full_str)
        bounds_cache[op] = bounds_set

    new_dom = new_dom.intersect(bounds_set).coalesce().gist_params(params_set).coalesce()

    modified = new_dom != prev_dom
    if modified:
        ctx.analysis_ctx.isl_domains[op.op_id] = new_dom
        log.debug(
            "Updated domain for %s from %s to %s, based on dependents",
            op,
            prev_dom,
            new_dom,
        )
    return bool(modified)


class PropagateDomainConditions(Transformation):
    """The domain of an operator can (and often must) be reduced to account for conditions
    of its dependents. For example, if a dependent only depents on the op for certain timesteps,
    then the domain of the op should be reduced to only those timesteps. This then requires
    updating the domain of any dependencies of op, which may in turn have reduced domains.
    This process continues until no more domain conditions can be propagated.
    """

    def _run(self) -> Tuple[PDG, bool]:  # noqa: C901
        new_dg = self.ctx.dg
        isl_ctx = self.ctx.analysis_ctx.isl_ctx

        universe_params_str, _ = isl_utils.get_parameters_and_var_bounds_strs(new_dg.universe)
        params_set = isl.Set.read_from_str(isl_ctx, f"[{universe_params_str}] -> {{ : }}")

        modified = False
        bounds_cache: Dict[top.TensorOp, islt.Set] = {}
        map_cache: Dict[Tuple[top.TensorOp, OpOutId, top.TensorOp, OpInId], islt.Map] = {}

        # This loop will keep going until no more domain conditions can be propagated
        trip_count = 0
        modifications = 0
        to_propagate: Set[top.TensorOp] = set(new_dg.nodes)
        while to_propagate:
            new_to_propagate: Set[top.TensorOp] = set()
            for op in to_propagate:
                modded = update_domain(new_dg, op, params_set, bounds_cache, map_cache, self.ctx)
                if modded:
                    modifications += 1
                    new_to_propagate = new_to_propagate.union(
                        {dep_op for dep_op, _ in new_dg.get_flat_direct_dependencies(op)}
                    )
                    modified = True
            trip_count += 1
            to_propagate = new_to_propagate

        log.info(
            "Propagated domain conditions in %d iterations, with %d total modifications",
            trip_count,
            modifications,
        )

        for node in new_dg.nodes:
            if node.op_id not in self.ctx.analysis_ctx.isl_domains:
                log.warning("Node %s has no domain", node)

        return new_dg, modified
