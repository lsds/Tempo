import functools

import islpy as isl
import networkx as nx

from tempo.core import index_expr as ie
from tempo.core.analysis_ctx import AnalysisCtx
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import OpId
from tempo.core.dependence_graph import PDG, DependencyData
from tempo.core.domain import Domain
from tempo.core.tensor_ops import (
    CastOp,
    CatOp,
    ConstOp,
    ElementWiseOp,
    EvalSymbolOp,
    ExecDataflowOp,
    IndexSelectOp,
    MergeOp,
    MovementOp,
    RandOp,
    SplitOp,
    TensorOp,
    UDFOp,
)
from tempo.utils import isl as isl_utils
from tempo.utils import logger
from tempo.utils.memory_estimator import MemoryEstimator

log = logger.get_logger(__name__)

INIT_MAX_DEPTH = 5


def get_group_dom(
    group_nodes: set[TensorOp],
) -> Domain:
    doms = [op.domain for op in group_nodes]
    return Domain.union(*doms)


def get_group_isl_dom(
    ctx: CompilationCtx, group_nodes: set[TensorOp], op_id: OpId | None = None
) -> isl.UnionSet:
    len_of_largest_domains = max([len(op.domain) for op in group_nodes])
    doms = [
        isl_utils.rename_union_set_tuples(ctx.analysis_ctx.get_or_make_domain(n), "")
        for n in group_nodes
        if len(n.domain) == len_of_largest_domains
    ]

    group_dom = isl_utils.rename_union_set_tuples(
        functools.reduce(lambda x, y: x.union(y), doms),
        isl_utils.op_id_to_exec_name(op_id) if op_id is not None else "",
    )

    return group_dom


def _is_dynamic_shaped(main_dg: PDG, node: TensorOp) -> bool:
    inputs_static = all(s.is_static() for s in main_dg.get_input_shapes(node).values())
    outputs_static = all(s.is_static() for s in main_dg.get_output_shapes(node).values())

    return not (inputs_static and outputs_static)


def _can_be_part_of_group(
    main_dg: PDG,
    node: TensorOp,
) -> bool:
    # NOTE: we do not include UDFs because they cannot be compiled
    # NOTE: Similarly, we do not include EvalIndexExprValueOps because they are not compiled
    # NOTE: RandOps, on the other hand, should be able to be part of a dataflow
    # but unfortunately, they are not supported by the current code generation tracer
    if isinstance(node, (RandOp, MergeOp, ExecDataflowOp, UDFOp, EvalSymbolOp)):
        return False

    ## NOTE: we do not group dynamic ops as these require access to the ThunkExecutionCtx,
    ## but we can group static ops with dynamic shapes.
    # if node.is_dynamic():
    #    return False

    return True


def get_group_dependents(
    main_dg: PDG,
    group: set[TensorOp],
    include_control: bool = False,
) -> set[tuple[TensorOp, DependencyData]]:
    return {
        (o, d)
        for n in group
        for o, d in main_dg.get_flat_direct_dependents(n, include_control=include_control)
    }


def _can_merge_groups(
    main_dg: PDG,
    group_1_id: int,
    group_2_id: int,
    group_to_nodes_: dict[int, set[TensorOp]],
) -> bool:
    group_1 = group_to_nodes_[group_1_id]
    group_2 = group_to_nodes_[group_2_id]

    for n in group_1:
        for depy, depy_data in main_dg.get_flat_direct_dependencies(n, include_control=False):
            if depy not in group_2:
                continue

            if not _can_add_to_group_basic(
                main_dg,
                n,
                depy,
                depy_data,
                True,
            ):
                return False

    for n in group_2:
        for depy, depy_data in main_dg.get_flat_direct_dependencies(n, include_control=False):
            if depy not in group_1:
                continue

            if not _can_add_to_group_basic(
                main_dg,
                n,
                depy,
                depy_data,
                True,
            ):
                return False

    return True


def _can_add_to_group_basic(  # noqa: C901
    main_dg: PDG,
    sink: TensorOp,
    source: TensorOp,
    dep_data: DependencyData,
    adding_source: bool,
) -> bool:
    node_to_add = source if adding_source else sink
    node_in_group = sink if adding_source else source

    # Check that the node is  not one of the forbidden ops
    if not _can_be_part_of_group(main_dg, node_to_add):
        return False

    # NOTE: prevent dynamic shape ops from being grouped with dynamic (ctx) ops
    if node_in_group.is_static() != node_to_add.is_static():
        return False

    if dep_data.is_control_edge:
        return False

    # If the relation across this edge is not unconditional basis, we cannot add
    if not (dep_data.is_unconditional() and dep_data.expr.struct_eq(source.domain.basis_expr)):
        return False

    # Expensive, should be last
    # NOTE: prevent dynamic shape ops from being grouped with static shape ops
    if _is_dynamic_shaped(main_dg, sink) != _is_dynamic_shaped(main_dg, source):
        return False

    return True


def _can_add_to_group(  # noqa: C901
    main_dg: PDG,
    analysis_ctx: AnalysisCtx,
    sink: TensorOp,
    source: TensorOp,
    dep_data: DependencyData,
    group: int,
    node_to_group: dict[TensorOp, int],
    group_to_nodes_: dict[int, set[TensorOp]],
    adding_source: bool,
) -> bool:
    node_to_add = source if adding_source else sink

    if not _can_add_to_group_basic(main_dg, sink, source, dep_data, adding_source):
        return False

    if sink.domain != source.domain:
        return False

    # We only group nodes with the same domain (with exception of consts)
    snk_dom = isl_utils.rename_union_set_tuples(analysis_ctx.get_or_make_domain(sink), "")
    src_dom = isl_utils.rename_union_set_tuples(analysis_ctx.get_or_make_domain(source), "")

    if snk_dom != src_dom:
        return False

    relatives = (
        main_dg.get_flat_direct_dependents(node_to_add, include_control=True)
        if adding_source
        else main_dg.get_flat_direct_dependencies(node_to_add, include_control=True)
    )

    data_relatives = []
    control_relatives = []
    for r in relatives:
        if r[1].is_control_edge:
            control_relatives.append(r)
        else:
            data_relatives.append(r)

    # Check for any control edges between node_to_add and group nodes.
    # If there are any, we should not group together.
    for n, _ in control_relatives:
        if node_to_group[n] == group:
            return False

    for n, d in data_relatives:
        # all data relatives must be part of group already
        if node_to_group[n] != group:
            return False
        # all data relatives must have unconditional basis exprs
        relative_src = source if adding_source else n
        if not (d.is_unconditional() and d.expr.struct_eq(relative_src.domain.basis_expr)):
            return False

    # TODO: more research
    # my_dependents = get_group_dependents(main_dg, {node_to_add}, include_control=False)
    # conds_on_me = set()

    # for _, d in my_dependents:
    #    if d.cond is not None:
    #        conds_on_me.add(d.cond)

    # if conds_on_me:
    #    group_dependents = get_group_dependents(
    #        main_dg, group_to_nodes_[group], include_control=False
    #    )
    #    conds_on_group = set()
    #    for _, d in group_dependents:
    #        if d.cond is not None:
    #            conds_on_group.add(d.cond)

    #    #TODO: maybe add requirement that dependent matches
    #    product_conds = itertools.product(conds_on_me, conds_on_group)
    #    for cond_on_me, cond_on_group in product_conds:
    #        possible = isl_utils.simplify_boolean_index_expr(
    #            node_to_add.domain, cond_on_me & cond_on_group, main_dg.static_bounds
    #        ).partial_eval(main_dg.static_bounds)
    #        if isinstance(possible, ie.ConstBool) and not possible.const:
    #            return False

    return True


def build_initial_groupings(
    ctx: CompilationCtx,
) -> tuple[int, dict[int, set[TensorOp]]]:
    new_dg: PDG = ctx.dg
    exec_cfg: ExecutionConfig = ctx.exec_cfg

    sorted_valid_nodes = sorted(
        [n for n in new_dg.nodes if _can_be_part_of_group(new_dg, n)],
        key=lambda n: (-int(n.op_id) if isinstance(n, (ConstOp, MovementOp)) else int(n.op_id)),
    )

    group_to_nodes = {idx: {node} for idx, node in enumerate(sorted_valid_nodes)}

    iterations, group_to_nodes_ = propagate_groupings(
        ctx, group_to_nodes, exec_cfg.enable_conservative_grouping
    )

    return iterations, group_to_nodes_


def propagate_groupings(  # noqa: C901
    ctx: CompilationCtx, group_to_nodes: dict[int, set[TensorOp]], conservative: bool = True
) -> tuple[int, dict[int, set[TensorOp]]]:
    dg: PDG = ctx.dg
    analysis_ctx = ctx.analysis_ctx

    iterations = 0
    changes = True
    node_to_group: dict[TensorOp, int] = {
        n: g for g, nodes in group_to_nodes.items() for n in nodes
    }

    # Mark the invalid nodes
    for node in ctx.dg.nodes:
        if node not in node_to_group:
            node_to_group[node] = -1

    # Invert the mapping to get the groups
    group_to_nodes_: dict[int, set[TensorOp]] = {}
    for n, n_group in node_to_group.items():
        if n_group not in group_to_nodes_:
            group_to_nodes_[n_group] = set()
        group_to_nodes_[n_group].add(n)

    while changes:
        iterations += 1
        changes = False
        nodes_sorted_by_id = sorted(dg.nodes, key=lambda x: node_to_group[x], reverse=True)
        for n in nodes_sorted_by_id:
            n_group = node_to_group[n]
            if n_group == -1:
                continue

            dependents = dg.get_flat_direct_dependents(n, include_control=True)
            # NOTE: This makes the graphs way more conservatively sized.
            ## Break graphs when they have more then one dependent
            if conservative:
                # unique_groups = {dep[0] for dep in dependents}
                unique_groups = {node_to_group[dep[0]] for dep in dependents}
                if len(unique_groups) > 1:  # Use 2 to be less conservative
                    continue
            for dep_op, dep_data in dependents:
                if _can_add_to_group(
                    dg,
                    analysis_ctx,
                    dep_op,
                    n,
                    dep_data,
                    n_group,
                    node_to_group,
                    group_to_nodes_,
                    False,
                ):
                    other_group = node_to_group[dep_op]
                    if n_group > other_group:
                        node_to_group[dep_op] = n_group
                        group_to_nodes_[n_group].add(dep_op)
                        group_to_nodes_[other_group].remove(dep_op)
                        changes = True
                    elif n_group < other_group:
                        node_to_group[n] = other_group
                        group_to_nodes_[other_group].add(n)
                        group_to_nodes_[n_group].remove(n)
                        n_group = other_group
                        changes = True

            dependencies = dg.get_flat_direct_dependencies(n, include_control=True)
            for depy_op, depy_data in dependencies:
                if _can_add_to_group(
                    dg,
                    analysis_ctx,
                    n,
                    depy_op,
                    depy_data,
                    n_group,
                    node_to_group,
                    group_to_nodes_,
                    True,
                ):
                    other_group = node_to_group[depy_op]
                    if n_group > other_group:
                        node_to_group[depy_op] = n_group
                        group_to_nodes_[n_group].add(depy_op)
                        group_to_nodes_[other_group].remove(depy_op)
                        changes = True
                    elif n_group < other_group:
                        node_to_group[n] = other_group
                        group_to_nodes_[other_group].add(n)
                        group_to_nodes_[n_group].remove(n)
                        n_group = other_group
                        changes = True

    final_groups = []
    for n_group, nodes in group_to_nodes_.items():
        if n_group != -1:
            induced_subgraph = dg.induced_subgraph(OpId(-1), nodes)
            for wcc in induced_subgraph.weakly_connected_components:
                final_groups.append(wcc)

    group_to_nodes_ = dict(enumerate(final_groups))
    return iterations, group_to_nodes_


def ungroup_nodes(
    group_to_nodes: dict[int, set[TensorOp]],
    group_id: int,
) -> dict[int, set[TensorOp]]:
    group = group_to_nodes[group_id]

    next_group_id = max(group_to_nodes.keys()) + 1
    for node in group:
        group_to_nodes[next_group_id] = {node}
        next_group_id += 1

    del group_to_nodes[group_id]

    return group_to_nodes


def absorb_low_cost_ops_into_groups(  # noqa: C901
    ctx: CompilationCtx,
    group_id_to_nodes: dict[int, set[TensorOp]],
) -> tuple[dict[int, set[TensorOp]], int]:
    dg: PDG = ctx.dg
    node_to_group_id = {n: g for g, nodes in group_id_to_nodes.items() for n in nodes}
    fusions = 0
    mem_estimator = MemoryEstimator(ctx)

    group_ids_with_only_low_cost_ops = set()
    for low_cost_group_id, nodes in group_id_to_nodes.items():
        if all(
            (
                isinstance(n, (ConstOp, MovementOp, CastOp, IndexSelectOp))
                and not isinstance(n, (SplitOp, CatOp))
            )
            or (
                isinstance(n, (ElementWiseOp))
                and mem_estimator.estimate_op_size_bytes(n.op_id) < 0.1 ** (2**20)
            )
            for n in nodes
        ):
            group_ids_with_only_low_cost_ops.add(low_cost_group_id)

    # NOTE: We can only merge low-cost groups
    mergeable_low_cost_and_dependent = set()
    for low_cost_group_id in group_ids_with_only_low_cost_ops:
        low_cost_group_external_dependent_ops = set()
        low_cost_group_nodes = group_id_to_nodes[low_cost_group_id]
        for node in low_cost_group_nodes:
            for dep_op, _ in dg.get_flat_direct_dependents(node):
                if dep_op not in low_cost_group_nodes:
                    low_cost_group_external_dependent_ops.add(dep_op)

        # NOTE: If any dependent is not in a group (i.e. is dynamic), we cannot merge.
        if any(edo not in node_to_group_id.keys() for edo in low_cost_group_external_dependent_ops):
            continue

        group_ids_of_external_dependent_ops = {
            node_to_group_id[op] for op in low_cost_group_external_dependent_ops
        }

        # TODO: or, we can merge all 3 groups together. No?
        # NOTE: We can only merge if all dependents come from a single group.
        if len(group_ids_of_external_dependent_ops) != 1:
            continue

        dep_group_id = list(group_ids_of_external_dependent_ops)[0]
        low_cost_group_dom = get_group_dom(group_id_to_nodes[low_cost_group_id])
        dep_group_dom = get_group_dom(group_id_to_nodes[dep_group_id])
        # low_cost_group_dom = get_group_isl_dom(ctx, group_id_to_nodes[low_cost_group_id])
        # dep_group_dom = get_group_isl_dom(ctx, group_id_to_nodes[dep_group_id])

        # NOTE: we can merge if the low-cost group's domain
        # is a subset of the dependent group domain
        # if low_cost_group_dom.is_subset(dep_group_dom) or low_cost_group_dom == dep_group_dom:
        if low_cost_group_dom.is_contained_in(dep_group_dom) or low_cost_group_dom == dep_group_dom:
            # --- DAG check: simulate the merge and check if the induced subgraph is a DAG ---
            merged_nodes = group_id_to_nodes[low_cost_group_id] | group_id_to_nodes[dep_group_id]
            induced_subgraph = dg.induced_subgraph(OpId(-1), merged_nodes)
            if induced_subgraph.is_dag():
                if _can_merge_groups(
                    dg,
                    low_cost_group_id,
                    dep_group_id,
                    group_id_to_nodes,
                ):
                    mergeable_low_cost_and_dependent.add((low_cost_group_id, dep_group_id))
            else:
                log.warning(
                    "Merging %s and %s would create a cycle. Skipping.",
                    low_cost_group_id,
                    dep_group_id,
                )

    log.debug(
        "Identified %s FUSABLE low-cost groups.",
        len(mergeable_low_cost_and_dependent),
    )

    ## NOTE: chains of groups may exit. In this case, we should follow the chain.
    # while True: #TODO
    #    no_changes = True
    #    mergeable_low_cost_and_dependent_dict = dict(mergeable_low_cost_and_dependent)
    #    new_mergeable_low_cost_and_dependent = set()
    #    for low_cost_group_id, dep_group_id in mergeable_low_cost_and_dependent:
    #        if dep_group_id in mergeable_low_cost_and_dependent_dict:
    #            # Update the dependent group id to point to the end of the chain
    #            new_dep_group_id = mergeable_low_cost_and_dependent_dict[dep_group_id]
    #            new_mergeable_low_cost_and_dependent.add((low_cost_group_id, new_dep_group_id))
    #            no_changes = False
    #        else:
    #            # Keep the original mapping if no chain exists
    #            new_mergeable_low_cost_and_dependent.add((low_cost_group_id, dep_group_id))
    #    mergeable_low_cost_and_dependent = new_mergeable_low_cost_and_dependent
    #    if no_changes:
    #        break
    mergeable_low_cost_and_dependent = dict(mergeable_low_cost_and_dependent)

    group_to_nodes_copy = {**group_id_to_nodes}
    for (
        low_cost_group_id,
        dep_group_id,
    ) in mergeable_low_cost_and_dependent.items():
        # Follow any chains of updates
        while dep_group_id not in group_to_nodes_copy:
            dep_group_id = mergeable_low_cost_and_dependent[dep_group_id]

        group_to_nodes_copy[dep_group_id].update(group_id_to_nodes[low_cost_group_id])
        del group_to_nodes_copy[low_cost_group_id]
        fusions += 1

    return group_to_nodes_copy, fusions


def absorb_consts_into_groups(  # noqa: C901
    ctx: CompilationCtx,
    group_to_nodes: dict[int, set[TensorOp]],
) -> tuple[dict[int, set[TensorOp]], int]:
    new_dg: PDG = ctx.dg
    node_to_group = {n: g for g, nodes in group_to_nodes.items() for n in nodes}

    group_to_nodes_copy = {**group_to_nodes}

    fusions = 0
    for node in node_to_group.keys():
        if isinstance(node, ConstOp):
            deps = new_dg.get_flat_direct_dependents(node)
            assert len(deps) == 1, f"Const should be individualized. Has {len(deps)} dependents."
            snk = deps[0][0]

            const_group = node_to_group[node]
            if snk in node_to_group:
                snk_group = node_to_group[snk]
                if const_group != snk_group and _can_merge_groups(
                    new_dg,
                    const_group,
                    snk_group,
                    group_to_nodes,
                ):
                    group_to_nodes_copy[snk_group].add(node)
                    del group_to_nodes_copy[const_group]
                    fusions += 1
    return group_to_nodes_copy, fusions


def build_group_dependency_graph(
    dg: PDG,
    node_to_group: dict[TensorOp, int],
    group_doms: dict[int, Domain],
    dim: ie.Symbol,
) -> nx.MultiDiGraph:
    group_graph = nx.MultiDiGraph()
    for snk, src, edge_data in dg.get_all_edges():
        if snk in node_to_group and src in node_to_group:
            snk_group = node_to_group[snk]
            src_group = node_to_group[src]

            if (
                snk_group != src_group
                and group_doms[snk_group].has_dim(dim)
                and group_doms[src_group].has_dim(dim)
                and (
                    not src.domain.has_dim(dim)
                    or ie.struct_eq(
                        edge_data.expr.members[src.domain.find_variable_index(dim)], dim
                    )
                )
            ):
                group_graph.add_edge(snk_group, src_group, edge=(snk, src, edge_data))
    return group_graph


def check_groups_are_dags(
    dg: PDG, group_to_nodes: dict[int, set[TensorOp]], stage_name: str
) -> None:
    if not check_groups_are_dags_(dg, group_to_nodes):
        raise ValueError(f"Group is not a DAG after {stage_name}.")


def check_groups_are_dags_(dg: PDG, group_to_nodes: dict[int, set[TensorOp]]) -> bool:
    for group_id, group_nodes in group_to_nodes.items():
        induced_subgraph = dg.induced_subgraph(OpId(group_id), group_nodes)
        if not induced_subgraph.is_dag():
            return False
    return True


def fix_any_cycles_generalized(
    ctx: CompilationCtx,
    group_to_nodes: dict[int, set[TensorOp]],
) -> dict[int, set[TensorOp]]:
    new_dg: PDG = ctx.dg
    dims = new_dg.universe.variables

    iterations = 0
    changes = True
    while changes:
        changes = False
        for dim in dims:
            node_to_group = {n: g for g, nodes in group_to_nodes.items() for n in nodes}
            group_doms = {g: get_group_dom(nodes) for g, nodes in group_to_nodes.items()}
            group_graph = build_group_dependency_graph(new_dg, node_to_group, group_doms, dim)
            cycles = list(nx.simple_cycles(group_graph))

            if len(cycles) > 0:
                log.info(
                    "Found %s cycles with lengths %s for dim %s",
                    len(cycles),
                    [len(c) for c in cycles],
                    dim,
                )
            if not cycles:
                continue

            groups_involved = {gi for cycle in cycles for gi in cycle}
            # Add in all surrounding groups
            for op in group_to_nodes[list(groups_involved)[0]]:
                for dep_op, _ in new_dg.get_flat_direct_dependents(op):
                    if dep_op in node_to_group:
                        groups_involved.add(node_to_group[dep_op])
                for depy_op, _ in new_dg.get_flat_direct_dependencies(op):
                    if depy_op in node_to_group:
                        groups_involved.add(node_to_group[depy_op])

            for group_id in groups_involved:
                group_to_nodes = ungroup_nodes(group_to_nodes, group_id)

            # NOTE: Be more conservative.
            _, group_to_nodes = propagate_groupings(ctx, group_to_nodes, True)
            changes = True
        iterations += 1
    return group_to_nodes


# def fix_any_cycles_generalized_(
#    ctx: CompilationCtx,
#    group_to_nodes: Dict[int, Set[TensorOp]],
# ) -> Dict[int, Set[TensorOp]]:
#    new_dg: PDG = ctx.dg
#    dims = new_dg.universe.variables
#
#    for dim in dims:
#        max_depth_ = INIT_MAX_DEPTH
#        while True:
#            node_to_group = {n: g for g, nodes in group_to_nodes.items() for n in nodes}
#            group_doms = {g: get_group_dom(nodes) for g, nodes in group_to_nodes.items()}
#            group_graph = build_group_dependency_graph(new_dg, node_to_group, group_doms, dim)
#            cycles = list(nx.simple_cycles(group_graph))
#
#            log.info(
#                "Found %s cycles with lengths %s for dim %s",
#                len(cycles),
#                [len(c) for c in cycles],
#                dim,
#            )
#            if not cycles:
#                break
#
#            for cycle in cycles:
#                while max_depth_ <= 1000:
#                    if try_break_cycle_backtrack(
#                        cycle=cycle,
#                        group_graph=group_graph,
#                        group_to_nodes=group_to_nodes,
#                        node_to_group=node_to_group,
#                        new_dg=new_dg,
#                        dim=dim,
#                        depth=0,
#                        max_depth=max_depth_,
#                    ):
#                        print(f"Successfully broke cycle with max depth {max_depth_}")
#                        break
#                    max_depth_ *= 2
#                    print(f"Trying again with max depth {max_depth_}")
#                    break
#                else:
#                    raise ValueError(
#                        f"Could not break some cycles in dim {dim} with max depth {max_depth_}"
#                    )
#
#    # Clean up empty groups
#    for g in list(group_to_nodes):
#        if not group_to_nodes[g]:
#            del group_to_nodes[g]
#    return group_to_nodes
#
#
# def try_break_cycle_backtrack(
#    cycle: List[int],
#    group_graph: nx.MultiDiGraph,
#    group_to_nodes: Dict[int, Set[TensorOp]],
#    node_to_group: Dict[TensorOp, int],
#    new_dg: PDG,
#    dim: ie.Symbol,
#    depth: int,
#    max_depth: int,
# ) -> bool:
#    if depth >= max_depth:
#        return False
#
#    # Try every pair of groups in the cycle
#    for i in range(len(cycle)):
#        g_from = cycle[i]
#        g_to = cycle[(i + 1) % len(cycle)]
#
#        if g_from == g_to:
#            continue
#
#        # For cycles of length 2, first try merging the two groups, and check if
#        # 1. internally they are dags
#        # 2. None of the internal edges are not unconditional basis
#        # if len(cycle) == 2:
#        #    old_group_to_nodes = {k: set(v) for k, v in group_to_nodes.items()}
#        #    old_node_to_group = dict(node_to_group)
#        #    group_to_nodes[g_from].update(group_to_nodes[g_to])
#        #    for n in group_to_nodes[g_to]:
#        #        node_to_group[n] = g_from
#        #    del group_to_nodes[g_to]
#        #    induced_subgraph = new_dg.induced_subgraph(OpId(-1), group_to_nodes[g_from])
#        #    if check_groups_are_dags(new_dg, group_to_nodes, "merging groups") and all(
#        #        ie.struct_eq(e.expr, src_.domain.basis_expr)
#        #        for snk_, src_, e in induced_subgraph.get_all_edges()
#        #    ):
#        #        return True
#        #    else:
#        #        group_to_nodes.clear()
#        #        group_to_nodes.update({k: set(v) for k, v in old_group_to_nodes.items()})
#        #        node_to_group.clear()
#        #        node_to_group.update(old_node_to_group)
#
#        # Try changing every edge in the cycle
#        for edge in group_graph[g_from][g_to].values():
#            snk, src, edge_data = edge["edge"]
#            candidate = src if node_to_group[src] == g_from else snk
#            assert isinstance(candidate, TensorOp)
#            assert isinstance(edge_data, DependencyData)
#
#            # TODO: this should be an assert no?
#            if candidate not in group_to_nodes[g_from]:
#                continue
#
#            # Save state for backtracking
#            old_group_to_nodes = {k: set(v) for k, v in group_to_nodes.items()}
#            old_node_to_group = dict(node_to_group)
#
#            idx = src.domain.find_variable_index(dim)
#            # NOTE: Once we are more desperate, we try creating new groups
#            #edge_is_uncond_basis = not (
#            #    edge_data.is_unconditional() and edge_data.expr.members[idx].struct_eq(dim)
#            #)
#            if max_depth > INIT_MAX_DEPTH * 2:
#                # print(f"Skipping {candidate} move because it is not unconditional basis")
#                # New idea, move the candidate to its own group
#                group_to_nodes[g_from].remove(candidate)
#                new_group_id = max(group_to_nodes.keys()) + 1
#                group_to_nodes[new_group_id] = {candidate}
#                node_to_group[candidate] = new_group_id
#                print(f"Moving {candidate} to new group {new_group_id}")
#
#            else:
#                if not (
#                    edge_data.is_unconditional() and edge_data.expr.members[idx].struct_eq(dim)
#                ):
#                    continue
#
#                # If the domain of candidate is larger than the domain of g_to, continue
#                to_group_dom = get_group_dom(group_to_nodes[g_to])
#                if (
#                    to_group_dom.is_contained_in(candidate.domain)
#                    and to_group_dom != candidate.domain
#                ):
#                    # print(f"Skipping {candidate}
#                    move because it would increase domain of {g_to}")
#                    continue
#
#                # Perform the move
#                group_to_nodes[g_from].remove(candidate)
#                group_to_nodes[g_to].add(candidate)
#                node_to_group[candidate] = g_to
#
#            # Recompute group graph
#            group_doms = {g: get_group_dom(nodes) for g, nodes in group_to_nodes.items()}
#            new_group_graph = build_group_dependency_graph(new_dg, node_to_group, group_doms, dim)
#            new_cycles = list(nx.simple_cycles(new_group_graph))
#
#            # Check if the specific cycle is gone
#            # if not any(set(cycle).issubset(set(c)) for c in new_cycles):
#            #    return True
#            if not new_cycles:
#                return True
#
#            # Else recurse
#            for next_cycle in new_cycles:
#                if try_break_cycle_backtrack(
#                    next_cycle,
#                    new_group_graph,
#                    group_to_nodes,
#                    node_to_group,
#                    new_dg,
#                    dim,
#                    depth + 1,
#                    max_depth,
#                ):
#                    return True
#
#            # Rollback if we reach here
#            group_to_nodes.clear()
#            group_to_nodes.update({k: set(v) for k, v in old_group_to_nodes.items()})
#            node_to_group.clear()
#            node_to_group.update(old_node_to_group)
#
#    return False
#
#
# def seperate_wccs(
#   ctx: CompilationCtx,
#   group_to_nodes: Dict[int, Set[TensorOp]],
# ) -> Tuple[Dict[int, Set[TensorOp]], int]:
#   dg = ctx.dg
#   fissions = 0
#   for group_id, group_nodes in list(group_to_nodes.items()):
#       induced_subgraph = dg.induced_subgraph(OpId(group_id), group_nodes)
#       wccs = list(induced_subgraph.weakly_connected_components)
#       if len(wccs) > 1:
#           print(f"Separating {len(wccs)} WCCs")
#           for wcc in wccs[1:]:
#               new_group_id = max(group_to_nodes.keys()) + 1
#               fissions += 1
#               group_to_nodes[new_group_id] = set(wcc)
#               group_to_nodes[group_id] -= wcc
#
#   return group_to_nodes, fissions


def get_potential_fusions(  # noqa: C901
    ctx: CompilationCtx,
    group_to_nodes: dict[int, set[TensorOp]],
) -> list[tuple[int, int]]:
    dg = ctx.dg

    group_to_node_id = {
        k: group_node.op_id for k, group in group_to_nodes.items() for group_node in group
    }
    node_id_to_group = {v: k for k, v in group_to_node_id.items()}
    node_ids = list(group_to_node_id.values())

    potential_fusions = []

    for node_id in node_ids:
        node = dg.get_op_by_id(node_id)
        for src, _ in dg.get_flat_direct_dependencies(node):
            if src.op_id not in node_ids:
                continue
            if node_id_to_group[node_id] == node_id_to_group[src.op_id]:
                continue

            dom_node = get_group_isl_dom(ctx, group_to_nodes[node_id_to_group[node_id]])
            dom_src = get_group_isl_dom(ctx, group_to_nodes[node_id_to_group[src.op_id]])

            # TODO: Tempo used this
            # if not (dom_src.is_subset(dom_node)):
            if dom_node != dom_src:
                continue

            all_basis = all(d.is_unconditional_basis() for d in dg.get_edges_between(node, src))
            if not all_basis:
                continue
            num_edges_other_way = dg.get_num_edges_between(src, node)
            if num_edges_other_way != 0:
                continue

            potential_fusions.append((node_id_to_group[node_id], node_id_to_group[src.op_id]))

    return list(set(potential_fusions))
