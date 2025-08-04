from typing import Set

from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.datatypes import OpId, OpInId
from tempo.core.dependence_graph import PDG, DependencyData
from tempo.core.op_tags import BACKWARD_REGION_TAG, REGION_TAG
from tempo.core.tensor_ops import TensorOp
from tempo.transformations.vectorization.vec_rules import OP_VEC_RULES
from tempo.utils import logger
from tempo.utils.dg_utils import is_window_access

log = logger.get_logger(__name__)


def filter_ops_to_vectorize(
    ctx: CompilationCtx,
    ops_to_vectorize: Set[TensorOp],
    dim: ie.Symbol,
) -> Set[TensorOp]:
    ops_to_vectorize = _filter_ops_to_vectorize_index_select(ctx, ops_to_vectorize, dim)
    ops_to_vectorize = _filter_small_groups(ctx, ops_to_vectorize, dim)
    ops_to_vectorize = _filter_groups_with_dynamic_dependents(ctx, ops_to_vectorize, dim)
    return ops_to_vectorize


def _filter_ops_to_vectorize_index_select(
    ctx: CompilationCtx,
    ops_to_vectorize: Set[TensorOp],
    dim: ie.Symbol,
) -> Set[TensorOp]:
    """
    Filter out index select ops with symbol index params, where the spatial src dim
    is not equal to the full domain range of the temporal dim, i.e., where vectorization
    could lead to incorrect results.
    """
    dg = ctx.dg
    ops_to_vec_copy = ops_to_vectorize.copy()
    for op in set(ops_to_vectorize):
        if type(op) is top.IndexSelectOp:
            depys = dg.get_flat_direct_dependencies(op)

            INDEX_PARAM_IN_IDX = 1
            index_param_op = depys[INDEX_PARAM_IN_IDX][0]
            index_param_op_vectorized = index_param_op in ops_to_vec_copy

            src_tensor_in_idx = 0
            src_tensor_shape = dg.get_input_shape(op, OpInId(src_tensor_in_idx))

            if (
                index_param_op_vectorized
                and isinstance(index_param_op, top.EvalSymbolOp)
                and ie.struct_eq(index_param_op.symbol, dim)
            ):
                src_shape_at_dim: ie.IntIndexValueLike = ie.lift_to_int_ie(
                    src_tensor_shape.at(op.dim)
                ).partial_eval(dg.static_bounds)
                bound = index_param_op.symbol.as_bound().partial_eval(dg.static_bounds)

                # NOTE: This is the key part: Do not vectorize if the spatial src dim
                # is not equal to the full domain range of the temporal dim.
                if not bound.struct_eq(src_shape_at_dim):
                    # print(f"Filtering index op: {op} on dim {dim}")
                    ops_to_vec_copy.remove(op)
                    ops_to_vec_copy.remove(index_param_op)

    log.info(
        "Filtered out %d ops due to index select with symbol index param for dim %s",
        len(ops_to_vectorize) - len(ops_to_vec_copy),
        dim,
    )

    return ops_to_vec_copy


def _filter_small_groups(
    ctx: CompilationCtx,
    ops_to_vectorize: Set[TensorOp],
    dim: ie.Symbol,
) -> Set[TensorOp]:
    """
    Filter out operations that belong to small weakly connected components.
    If a weakly connected component has fewer than 5 operations, we filter out
    all operations in that component.

    This is sort of a heuristic, but it seems to work well in practice.
    We want to avoid introducing too many index_selects, as they are expensive.
    We also want to avoid creating small dataflows.
    """
    dg = ctx.dg

    min_group_size = ctx.exec_cfg.reject_vec_groups_smaller_than
    allowed_ext_deps_ratio = ctx.exec_cfg.reject_vec_groups_with_external_deps_ratio_greater_than

    # Create a subgraph containing only the ops_to_vectorize
    subgraph = dg.induced_subgraph(OpId(-1), ops_to_vectorize)

    # Get weakly connected components of the subgraph
    wccs = list(subgraph.weakly_connected_components)

    wcc_ext_deps_count = []
    for wcc in wccs:
        num_ext_deps = 0
        for op in wcc:
            num_ext_deps += len([d for d in dg.get_flat_direct_dependents(op) if d[0] not in wcc])
        wcc_ext_deps_count.append(num_ext_deps)

    # Filter out ops in small components (less than 5 ops)
    ops_in_large_components = set()
    for i, wcc in enumerate(wccs):
        # NOTE: Cond 1: Reject groups smaller than min_group_size
        if len(wcc) <= min_group_size:
            continue
        # NOTE: Cond 2: Reject groups with lots of external dependencies in relation to their size
        if wcc_ext_deps_count[i] >= len(wcc) * allowed_ext_deps_ratio:
            continue

        # NOTE: Not rejected yet, so keep the group
        ops_in_large_components.update(wcc)

    log.info(
        "Filtered out %d ops due to small groups for dim %s",
        len(ops_to_vectorize) - len(ops_in_large_components),
        dim,
    )

    return ops_in_large_components


def _filter_groups_with_dynamic_dependents(
    ctx: CompilationCtx,
    ops_to_vectorize: Set[TensorOp],
    dim: ie.Symbol,
) -> Set[TensorOp]:
    """
    Filter out groups containing index_slice ops with dynamic length params.
    This is because support for this still needs to be added in statifying incrementalization.
    """
    dg = ctx.dg

    # Create a subgraph containing only the ops_to_vectorize
    subgraph = dg.induced_subgraph(OpId(-1), ops_to_vectorize)

    # Get weakly connected components of the subgraph
    wccs = list(subgraph.weakly_connected_components)

    wcc_ext_dyn_deps_count = []

    def check_dynamic(snk: TensorOp, src: TensorOp, data: DependencyData) -> bool:
        src_dim_idx = src.domain.find_variable_index(dim)
        e = data.expr.members[src_dim_idx]
        s = e.evaluate_shape(dg.static_bounds)
        if s == ():
            return False
        assert len(s) == 1
        s_ = s[0]
        return type(s_) is not int

    for wcc in wccs:
        num_ext_deps = 0
        for op in wcc:
            num_ext_deps += len(
                [
                    snk_op
                    for snk_op, data in dg.get_flat_direct_dependents(op)
                    if check_dynamic(snk_op, op, data)
                ]
            )
        wcc_ext_dyn_deps_count.append(num_ext_deps)

    # Filter out ops in small components (less than 5 ops)
    ops_with_no_dynamic_dependents = set()
    for i, wcc in enumerate(wccs):
        if wcc_ext_dyn_deps_count[i] == 0:
            ops_with_no_dynamic_dependents.update(wcc)

    log.info(
        "Filtered out %d ops due to dynamic dependents for dim %s",
        len(ops_to_vectorize) - len(ops_with_no_dynamic_dependents),
        dim,
    )

    return ops_with_no_dynamic_dependents


def can_vectorize_basic(  # noqa: C901
    op: TensorOp,
    dim: ie.Symbol,
    dg: PDG,
) -> bool:
    # If the op doesn't index on the dimension we are vectorizing, it can't be vectorized
    if not op.domain.has_dim(dim):
        return False

    # TODO another way to handle this would be to split the dimension for this op into
    # many dimensions, one for each branch or something.
    if isinstance(op, top.MergeOp):
        for _, depy_data in dg.get_flat_direct_dependencies(op):
            cond = depy_data.cond
            assert cond is not None
            vars_used_in_cond = cond.vars_used()
            if dim in vars_used_in_cond:
                return False

    # If the op is a user defined function, and there is no defined vectorization function, we
    # cannot perform basic vectorization on it
    if isinstance(op, top.UDFOp):
        if op.desc.vectorize is None:
            # log.info("UDF %s has no vectorization function", op)
            return False

    ## NOTE: This is here to work around the JAX bug which leads to:
    ## "jax INTERNAL: Failed to allocate 204800000 bytes for new constant"
    #if isinstance(op, top.RandOp):
    #    # NOTE: Vectorize only on trivial dims
    #    if not is_trivial_dim(dg, dim):
    #        return False

    # Attempt to get the vectorization rule for the op - if there is none then we can't do
    # vectorization
    vec_rule = OP_VEC_RULES.get(type(op))
    if vec_rule is None:
        log.info("No vectorization rule for op %s", op)
        return False

    # NOTE: If the op has shapes dynamic on dim, we can't vectorize it
    if any(dim in ie.lift_to_ie(s.prod()).vars_used() for s in dg.get_output_shapes_list(op)):
        return False

    # Same for input shapes
    if any(dim in ie.lift_to_ie(s.prod()).vars_used() for s in dg.get_input_shapes_list(op)):
        return False

    # If none of the above conditions are met, we can vectorize this op
    return True


def can_vectorize(  # noqa: C901
    dg: PDG,
    op: TensorOp,
    dim: ie.Symbol,
    nonvectorizable_ops_due_to_cycles: Set[TensorOp],
) -> bool:
    if op in nonvectorizable_ops_due_to_cycles:
        return False

    # Checks basic cases for vectorization
    if not can_vectorize_basic(op, dim, dg):
        return False

    # Special case for UDFs that access external state
    # NOTE: basically if multiple ops access the same external state, we need to ensure that
    # they are vectorized atomically.
    if (
        isinstance(op, top.UDFOp)
        and op.desc.require_simultaneous_vectorization
        and op.desc.state_store_access_keys is not None
    ):
        # NOTE: because there may be other ops sharing state, we must ensure that all
        # of them will be vectorized atomically.
        for key in op.desc.state_store_access_keys:
            udfs_accessing_key = [
                u
                for u in dg.nodes
                if isinstance(u, top.UDFOp)
                and (
                    u.desc.state_store_access_keys is not None
                    and key in u.desc.state_store_access_keys
                )
            ]

            are_all_vectorizable = all(
                (
                    can_vectorize_basic(u, dim, dg)
                )  # TODO or has_been_vectorized(u, dim) #TODO this was removed in 10/07.
                # If it breaks PPO, need to add it back
                and (u not in nonvectorizable_ops_due_to_cycles)
                for u in udfs_accessing_key
            )

            if not are_all_vectorizable:
                return False

    return True


def is_trivial_edge(snk: TensorOp, src: TensorOp, data: DependencyData, dim: ie.Symbol) -> bool:
    e = data.expr
    idx = src.domain.find_variable_index(dim)
    e_idx = e.members[idx]
    is_trivial_edge_dep = (e_idx.struct_eq(dim)) or (e_idx.is_constant())
    dim_in_cond = data.cond is not None and dim in data.cond.vars_used()
    return is_trivial_edge_dep and not dim_in_cond


def is_trivial_dim(dg: PDG, dim: ie.Symbol) -> bool:
    for snk, src, data in dg.get_all_edges(include_control=False):
        if src.domain.has_dim(dim):
            # idx = src.domain.find_variable_index(dim)
            # e = data.expr[idx]
            # if not (e.is_constant() or e.struct_eq(dim)):
            if not is_trivial_edge(snk, src, data, dim):
                return False
    return True


def get_ops_to_vectorize(
    dg: PDG,
    dim: ie.Symbol,
) -> Set[TensorOp]:
    # Step 1: Identify candidate ops (have dim and are basically vectorizable)
    candidate_ops = set()
    non_candidate_ops = set()

    for op in dg.nodes:
        if can_vectorize_basic(op, dim, dg):
            candidate_ops.add(op)
        else:
            non_candidate_ops.add(op)

    if is_trivial_dim(dg, dim):
        return candidate_ops

    # print(f" Number of candidate ops: {len(candidate_ops)}")
    # print(f" Number of non-candidate ops: {len(non_candidate_ops)}")

    # Short-circuit if no candidates
    if not candidate_ops:
        return set()

    nodes_with_dim = [op for op in dg.nodes if op.domain.has_dim(dim)]
    dim_dg = dg.induced_subgraph(OpId(-1), nodes_with_dim)

    # NOTE:
    # Start by marking the snk and source of non-trivial edges as non-candidate
    # At this point, we have a set of candidate ops which we want to prove are not vectorizable.
    # For each, we can start generating cycles in the "with_dim" subgraph, aiming to prove that
    # the candidate is in cycle with a non-candidate op or in which an edge is non-trivial.
    # No need to actually generate all cycles, since as soon as we find one proof, we can stop.

    # An issue is that for nodes early in the iteration, we may check that all neighbors are okay
    # and they are, but later on we set the neighbor to non-candidate, and we need to re-check.
    # Thus, we iterate until convergence.

    keep_going = True
    iter_count = 0
    while keep_going:
        keep_going = False
        iter_count += 1
        for op in list(candidate_ops):
            for c in dim_dg.op_cycles(op):
                if any(c_op in non_candidate_ops for c_op in c):
                    non_candidate_ops.update(c)
                    candidate_ops.difference_update(c)
                    keep_going = True
                    break

                for c_snk, c_src in zip(c[1:], c[1:], strict=True):
                    for data in dim_dg.get_edges_between(c_snk, c_src):
                        if not is_trivial_edge(c_snk, c_src, data, dim):
                            non_candidate_ops.update(c)
                            candidate_ops.difference_update(c)
                            keep_going = True
                            break
                if keep_going:
                    break
            if keep_going:
                break

    _restore_original_submission_vectorization(dg, dim, candidate_ops, non_candidate_ops)
    _propagate_vectorization(dg, dim, candidate_ops, non_candidate_ops)

    return candidate_ops


def _restore_original_submission_vectorization(
    dg: PDG, dim: ie.Symbol, candidate_ops: Set[TensorOp], non_candidate_ops: Set[TensorOp]
) -> None:
    """
    In the original submission, backpropagation itself would vectorize the backward region.
    Now, we no longer do that for generality. So, instead, we achieve the same effect by
    grabbing all operations in the backward region and vectorizing them.

    This method is also far faster than the original submission, which involved checking all cycles
    in the graph for triviality.

    """
    if len(dg.universe.variables) > 2:
        t = dg.universe.variables[2]

        any_window_access_patterns = any(
            (
                src.domain.has_dim(t)
                and is_window_access(data.expr.members[src.domain.find_variable_index(t)])
            )
            for _, src, data in dg.get_all_edges()
        )

        # NOTE: If there are window access patterns, then we can incrementally backpropagate,
        # which is better than vectorizing.
        if dim.struct_eq(t) and not any_window_access_patterns:
            for op in dg.nodes:
                if BACKWARD_REGION_TAG in op.flat_tags.get(REGION_TAG, ()) and can_vectorize_basic(
                    op, dim, dg
                ):
                    candidate_ops.add(op)
                    non_candidate_ops.remove(op)


def _propagate_vectorization(
    dg: PDG, dim: ie.Symbol, candidate_ops: Set[TensorOp], non_candidate_ops: Set[TensorOp]
) -> None:
    """
    For any set of vec candidates proposed, we can easily propagate vectorization to all
    dependents and dependencies of the candidates if they are all in the candidate set already.
    This is a simple and fast way to propagate vectorization.
    We iterate until convergence.
    """
    added_any = True
    while added_any:
        added_any = False
        for op in list(non_candidate_ops):
            all_depys_vec = all(
                depy in candidate_ops or (not depy.domain.has_dim(dim))
                for depy, _ in dg.get_flat_direct_dependencies(op)
            )
            all_deps_vec = all(
                dep in candidate_ops or (not dep.domain.has_dim(dim))
                for dep, _ in dg.get_flat_direct_dependents(op)
            )
            # NOTE: If for some reason we missed a candidate op, we add it here.
            if can_vectorize_basic(op, dim, dg) and (all_depys_vec or all_deps_vec):
                added_any = True
                candidate_ops.add(op)
                non_candidate_ops.remove(op)


def get_ops_to_vectorize_old(
    dg: PDG,
    dim: ie.Symbol,
) -> Set[TensorOp]:
    # print(f"Getting ops to vectorize for dim {dim} with is_trivial_dim={is_trivial_dim}")

    # NOTE: This version was too eager to vectorize unvectorizable ops, due to using SCCs instead
    # of checking actual cycles.

    # Step 1: Identify candidate ops (have dim and are basically vectorizable)
    candidate_ops = set()
    non_candidate_ops = set()

    for op in dg.nodes:
        if can_vectorize_basic(op, dim, dg):
            candidate_ops.add(op)
        else:
            non_candidate_ops.add(op)

    if is_trivial_dim(dg, dim):
        return candidate_ops

    # Short-circuit if no candidates
    if not candidate_ops:
        return set()

    # Step 2: Compute SCCs for the entire graph
    sccs = list(dg.weakly_connected_components)
    node_to_scc = {}
    for i, scc in enumerate(sccs):
        for node in scc:
            node_to_scc[node] = i

    # Step 3: Identify SCCs with non-trivial edges
    scc_has_non_trivial = [False] * len(sccs)

    # Check all edges in the graph
    for snk, src, data in dg.get_all_edges():
        # Skip edges not between candidate ops
        if snk not in candidate_ops or src not in candidate_ops:
            continue

        scc_id = node_to_scc[snk]
        ## Skip if already marked or different SCC
        # if scc_has_non_trivial[scc_id] or node_to_scc[src] != scc_id:
        #    continue

        if not is_trivial_edge(snk, src, data, dim):
            scc_has_non_trivial[scc_id] = True
            scc_has_non_trivial[node_to_scc[src]] = True

    # Step 4: Collect vectorizable ops
    vectorizable_ops = set()
    for i, scc in enumerate(sccs):
        # Skip SCCs with non-candidate ops
        if any(op in non_candidate_ops for op in scc):
            continue
        # Skip SCCs with non-trivial edges
        if scc_has_non_trivial[i]:
            continue
        # All ops in this SCC are vectorizable
        # vectorizable_ops.update(scc.intersection(candidate_ops))
        vectorizable_ops.update(scc)

    return vectorizable_ops
