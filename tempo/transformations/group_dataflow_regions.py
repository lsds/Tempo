from dataclasses import replace

from tempo.core import index_expr as ie
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.dataflow_graph import DataflowGraph
from tempo.core.datatypes import OpId, OpInId, OpOutId, TensorId
from tempo.core.dependence_graph import PDG, DependencyData, OpData
from tempo.core.tensor_ops import ExecDataflowOp, TensorOp
from tempo.transformations.compilation_pass import CompilationPass
from tempo.transformations.graph_partitioning_utils import ilp_based_cut
from tempo.transformations.grouping_utils import (
    _can_merge_groups,
    absorb_consts_into_groups,
    absorb_low_cost_ops_into_groups,
    build_initial_groupings,
    fix_any_cycles_generalized,
    get_group_dom,
    get_group_isl_dom,
    get_potential_fusions,
)
from tempo.utils import logger
from tempo.utils.memory_estimator import MemoryEstimator

log = logger.get_logger(__name__)


# NOTE: On how to deal with control edges:
# We do not group together nodes with control edges between them.
# When inserting dataflow ops, we add control edges back to the original graph.


class GroupDataflowRegions(CompilationPass):
    # Given the required parameters to create a ExecuteDataflowSubgraphOp, along
    # with the original graph, this function inserts the
    # ExecuteDataflowSubgraphOp into the graph in place of the subgraph it contains
    def insert_execute_dataflow_subgraph_op(  # noqa: C901
        self,
        dg: PDG,
        subgraph_nodes: set[TensorOp],
        irouter: tuple[tuple[tuple[OpId, OpInId], ...], ...],
        orouter: tuple[tuple[OpId, OpOutId], ...],
    ) -> OpId:
        id_ = dg.get_next_op_id()
        # Create the subgraph from the set of nodes
        subgraph = dg.induced_subgraph(id_, subgraph_nodes)

        # if not nx.is_directed_acyclic_graph(subgraph._G):
        if not subgraph.is_dag():
            raise ValueError("Subgraph is not a DAG")

        # Create the ExecuteDataflowSubgraphOp node to be inserted into the graph
        dataflow_graph: DataflowGraph = DataflowGraph(
            subgraph=subgraph,
            irouter=irouter,
            orouter=orouter,
        )

        domain = get_group_dom(subgraph_nodes).copy()
        # domain._ubound_overrides.clear() #TODO: is this right?

        tags = {}
        for n in subgraph_nodes:
            for k, v in n.tags.items():
                if k not in tags:
                    tags[k] = ()
                tags[k] = tuple({*tags[k], v})
        exec_dataflow_op: ExecDataflowOp = ExecDataflowOp(  # type: ignore
            id_, domain=domain, tags=tags, dataflow=dataflow_graph
        )

        # Get the output shapes for the ExecuteDataflowSubgraphOp
        output_shapes = {
            OpOutId(i): dg.get_output_shapes(dg.ops_by_id[op_id].op)[op_out_id]
            for i, (op_id, op_out_id) in enumerate(orouter)
        }

        # Get the output dtypes for the ExecuteDataflowSubgraphOp
        output_dtypes = {
            OpOutId(i): dg.get_output_dtypes(dg.ops_by_id[op_id].op)[op_out_id]
            for i, (op_id, op_out_id) in enumerate(orouter)
        }

        # Construct an OpData to be inserted into the graph for the
        # ExecuteDataflowSubgraphOp
        dataflow_op_data = OpData(
            op=exec_dataflow_op,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
        )

        # Insert the ExecuteDataflowSubgraphOp into the new dg
        dg.insert_op(dataflow_op_data)

        # Iterate through the irouter and set up incoming (dependencies) connections
        for i, shared_deps in enumerate(irouter):
            op_id, op_in_id = shared_deps[0]

            # Find the op in DG, and find the connection feeding into op_in_id
            irouter_op_data = dg.ops_by_id[op_id]  # --> OpData
            deps = dg.get_flat_direct_dependencies(irouter_op_data.op)
            for src_op, dep_data in deps:
                if dep_data.sink_in_idx == op_in_id:
                    new_dep_data = replace(dep_data, sink_in_idx=OpInId(i))
                    try:
                        dg.add_edge(exec_dataflow_op, src_op, new_dep_data)
                    except Exception as e:
                        log.error("Failed to add edge: %s", e)
                        log.error("Subgraph nodes: %s", subgraph_nodes)
                        raise e

        # print("----------------------")
        # Iterate through the orouter and set up outgoing (dependent) connections
        for i, (op_id, op_out_id) in enumerate(orouter):
            # Find the op in DG, and find the connection feeding into op_out_id
            orouter_op_data = dg.ops_by_id[op_id]  # --> OpData
            route_src = orouter_op_data.op
            deps = dg.get_flat_direct_dependents(route_src)
            for sink_op, dep_data in deps:
                if dep_data.src_out_idx == op_out_id:
                    new_expr = self._get_updated_expr(subgraph_nodes, dep_data, route_src)
                    new_dep_data = replace(dep_data, src_out_idx=OpOutId(i), expr=new_expr)
                    dg.add_edge(sink_op, exec_dataflow_op, new_dep_data)

        # NOTE: now add control edges back
        for op in subgraph_nodes:
            for depy_op, depy_data in dg.get_flat_direct_dependencies(op, include_control=True):
                if depy_data.is_control_edge:
                    assert depy_op not in subgraph_nodes
                    dg.add_edge(exec_dataflow_op, depy_op, depy_data.copy())
            for dep_op, dep_data in dg.get_flat_direct_dependents(op, include_control=True):
                if dep_data.is_control_edge:
                    assert dep_op not in subgraph_nodes
                    dg.add_edge(dep_op, exec_dataflow_op, dep_data.copy())

        # Remove the old operations from the subgraph now contained in the
        # ExecuteDataflowSubgraphOp (also removes old connections/edges)
        for n in subgraph_nodes:
            dg.remove_op(n)

        return id_

    def _get_updated_expr(
        self, subgraph_nodes: set[TensorOp], dep_data: DependencyData, route_src: TensorOp
    ) -> ie.IndexSequence:
        orig_expr = dep_data.expr

        group_dom = get_group_dom(subgraph_nodes)
        new_expr = group_dom.basis_expr
        for i, v in enumerate(group_dom.variables):
            if route_src.domain.has_dim(v):
                src_idx = route_src.domain.find_variable_index(v)
                src_m = orig_expr.members[src_idx]
                new_expr = new_expr.replace_idx(i, src_m)
        return new_expr

    def _build_routers(  # noqa: C901
        self, dg: PDG, subgraph_nodes: set[TensorOp]
    ) -> tuple[
        tuple[tuple[tuple[OpId, OpInId], ...], ...],
        tuple[tuple[OpId, OpOutId], ...],
        dict[tuple[OpId, OpOutId, ie.IndexSequence], int],
    ]:
        # NOTE: sometimes, a fused dataflow may have multiple inner nodes consuming the same
        # tensor. In that case, we want to deduplicate the consumption. We keep track of the
        # consumed tensors here, identified by their TensorId and the index sequence of dependence
        input_index_tracker: dict[tuple[OpId, OpOutId, ie.IndexSequence], int] = {}

        # To build the irouter, we want to find the ops in the subgraph,
        # which have dependencies on ops outside the subgraph
        irouter_: list[tuple[tuple[OpId, OpInId]]] = []
        for n in subgraph_nodes:
            dependencies = dg.get_flat_direct_dependencies(n)
            for dependency_op, dep_data in dependencies:
                if dependency_op not in subgraph_nodes:
                    dep = (dependency_op.op_id, dep_data.src_out_idx, dep_data.expr)

                    if dep in input_index_tracker:
                        dep_idx = input_index_tracker[dep]
                        irouter_[dep_idx] = irouter_[dep_idx] + (  # type: ignore
                            (n.op_id, dep_data.sink_in_idx),
                        )
                    else:
                        dep_idx = len(irouter_)
                        input_index_tracker[dep] = dep_idx
                        irouter_.append(((n.op_id, dep_data.sink_in_idx),))

        irouter = tuple(irouter_)

        # To build the orouter, we want to find the ops in the subgraph,
        # which have dependents outside the subgraph
        orouter_set: set[tuple[OpId, OpOutId]] = set()
        for n in subgraph_nodes:
            dependents = dg.get_flat_direct_dependents(n)
            for dependent_op, dep_data in dependents:
                if dependent_op not in subgraph_nodes:
                    # assert dep_data.cond is None
                    orouter_set.add((n.op_id, dep_data.src_out_idx))

        orouter = tuple(orouter_set)

        return irouter, orouter, input_index_tracker

    def _build_ctx_for_grouping(
        self, ctx: CompilationCtx, group_to_nodes: dict[int, set[TensorOp]]
    ) -> tuple[CompilationCtx, dict[int, OpId]]:
        group_to_node_id = {}

        new_ctx = ctx.shallow_copy()
        modifiable_dg = new_ctx.dg

        for group_key, group_nodes in group_to_nodes.items():
            irouter, orouter, _ = self._build_routers(modifiable_dg, group_nodes)
            try:
                op_id = self.insert_execute_dataflow_subgraph_op(
                    modifiable_dg,
                    group_nodes,
                    irouter,
                    orouter,
                )
            except Exception as e:
                log.error("Failed to insert dataflow subgraph op for group %s", group_key)
                log.error("Group nodes: %s", group_nodes)
                raise e

            group_to_node_id[group_key] = op_id
            group_dom = get_group_isl_dom(new_ctx, group_nodes, op_id)

            isl_doms = new_ctx.analysis_ctx.isl_domains
            dev_assignment = new_ctx.analysis_ctx._device_assignment
            assert dev_assignment is not None
            # TODO make a "upcast device" function in device.py
            group_dev = max(
                (dev_assignment[n.op_id] for n in group_nodes), key=lambda x: x.priority
            )

            # TODO move this up.
            dev_assignment[op_id] = group_dev

            isl_doms[op_id] = group_dom
            for node in group_nodes:
                if node.op_id in isl_doms:
                    del isl_doms[node.op_id]
                if node.op_id in dev_assignment:
                    del dev_assignment[node.op_id]

            if new_ctx.analysis_ctx._tensor_prealloc_value is None:
                new_ctx.analysis_ctx._tensor_prealloc_value = {}

            # NOTE: remap any TIDs to the new dataflow op
            for router_out_idx, (router_op_id, router_op_out_id) in enumerate(orouter):
                tid = TensorId(router_op_id, router_op_out_id)
                if tid in new_ctx.analysis_ctx.tensor_prealloc_value:
                    new_tid = TensorId(op_id, OpOutId(router_out_idx))
                    new_ctx.analysis_ctx._tensor_prealloc_value[new_tid] = (
                        new_ctx.analysis_ctx.tensor_prealloc_value[tid]
                    )
                    del new_ctx.analysis_ctx.tensor_prealloc_value[tid]

        return new_ctx, group_to_node_id

    def _fix_any_grouped_nonbasis_edges(
        self,
        new_dg: PDG,
        group_to_nodes: dict[int, set[TensorOp]],
    ) -> tuple[dict[int, set[TensorOp]], int]:
        next_group_id = max(group_to_nodes.keys()) + 1

        changes = True
        nonbasis_edges_fixed = 0
        while changes:
            changes = False

            for group_id, nodes in list(group_to_nodes.items()):
                induced_subgraph = new_dg.induced_subgraph(OpId(-1), nodes)
                problem_edges = []
                for snk, src, edge_data in induced_subgraph.get_all_edges():
                    if not edge_data.expr.struct_eq(src.domain.basis_expr):
                        problem_edges.append((snk, src, edge_data))
                    elif edge_data.is_control_edge:
                        problem_edges.append((snk, src, edge_data))

                if problem_edges:
                    irouter, orouter, inp_ind_tracker = self._build_routers(self.ctx.dg, nodes)
                    group1, group2, _ = ilp_based_cut(
                        new_dg,
                        nodes,
                        router_info=(irouter, orouter, inp_ind_tracker),
                        bytes_importance=0.25,
                        max_allowed_imbalance_percent=1.0,
                        required_cut_edges=problem_edges,
                        mem_est=MemoryEstimator(self.ctx),
                    )
                    group_to_nodes[group_id] = group1
                    group_to_nodes[next_group_id] = group2
                    next_group_id += 1
                    changes = True
                    nonbasis_edges_fixed += 1
                    break
        return group_to_nodes, nonbasis_edges_fixed

    def _cut_large_groups(  # noqa: C901
        self,
        group_to_nodes: dict[int, set[TensorOp]],
    ) -> tuple[dict[int, set[TensorOp]], int, int]:
        tot_edges_cut = 0
        fissions = 0
        next_group_id = max(group_to_nodes.keys()) + 1

        changes = True
        while changes:
            new_group_to_nodes = {}
            changes = False
            for k, v in group_to_nodes.items():
                if len(v) > self.ctx.exec_cfg.max_dataflow_group_size:
                    changes = True
                    irouter, orouter, inp_ind_tracker = self._build_routers(self.ctx.dg, v)
                    cluster1, cluster2, num_edges_cut = ilp_based_cut(
                        self.ctx.dg,
                        v,
                        (irouter, orouter, inp_ind_tracker),
                        0.5,
                        0.25,
                        mem_est=MemoryEstimator(self.ctx),
                    )
                    tot_edges_cut += num_edges_cut
                    new_group_to_nodes[k] = cluster1
                    # Any number not present in group_to_nodes.keys()
                    new_group_to_nodes[next_group_id] = cluster2
                    next_group_id += 1
                    fissions += 1
                else:
                    new_group_to_nodes[k] = v
            group_to_nodes = new_group_to_nodes

        return new_group_to_nodes, fissions, tot_edges_cut

    def _fuse_groups(  # noqa: C901
        self,
        group_to_nodes: dict[int, set[TensorOp]],
    ) -> tuple[dict[int, set[TensorOp]], int]:
        removed_groups = set()
        potential_fusions = get_potential_fusions(self.ctx, group_to_nodes)

        fusions = 0
        while potential_fusions:
            group_snk, group_src = potential_fusions.pop()
            if group_snk in removed_groups or group_src in removed_groups:
                continue

            group_snk_isl_dom = get_group_isl_dom(self.ctx, group_to_nodes[group_snk])
            group_src_isl_dom = get_group_isl_dom(self.ctx, group_to_nodes[group_src])

            # if not (group_src_isl_dom.is_subset(group_snk_isl_dom)):
            if not (group_src_isl_dom == group_snk_isl_dom):
                continue

            if not _can_merge_groups(
                self.ctx.dg,
                group_snk,
                group_src,
                group_to_nodes,
            ):
                continue

            # group_dom_snk = get_group_dom(group_to_nodes[group_snk])
            # group_dom_src = get_group_dom(group_to_nodes[group_src])
            # if not (group_dom_src == group_dom_snk):
            #    continue

            # Build a graph with the fusion done
            fused_group = group_to_nodes[group_snk].union(group_to_nodes[group_src])
            new_group_to_nodes = group_to_nodes.copy()
            new_group_to_nodes[group_snk] = fused_group
            del new_group_to_nodes[group_src]
            try:
                new_ctx, new_group_to_node_id = self._build_ctx_for_grouping(
                    self.ctx, new_group_to_nodes
                )
            except Exception:
                log.error(
                    "Failed to build context for fusion of groups %s and %s", group_snk, group_src
                )
                continue
            fused_dg = new_ctx.dg

            # Create a copy of the fused graph with regressive edges removed
            # fused_dg_copy = copy.deepcopy(fused_dg)
            for snk, src, d in fused_dg.get_all_edges():
                if d.expr.is_regressive():
                    fused_dg.remove_edge(snk, src, d)

            # If there are any cycles left involving the new op, they are non-regressive,
            # and thus the fusion fails
            cycle_op = fused_dg.get_op_by_id(new_group_to_node_id[group_snk])

            if not fused_dg.is_in_cycle(cycle_op):
                # No cycle found, so the fusion is valid
                group_to_nodes = new_group_to_nodes
                removed_groups.add(group_src)
                fusions += 1

        return group_to_nodes, fusions

    def _run(self) -> tuple[CompilationCtx, bool]:
        # 1. Build a set of conservative groups
        log.debug("Building conservative initial groupings")
        iterations, group_to_nodes = build_initial_groupings(self.ctx)

        if not group_to_nodes:  # NOTE: no groupings were found
            return self.ctx, False

        # check_groups_are_dags(self.ctx.dg, group_to_nodes, "initial grouping")

        group_to_nodes, nonbasis_edges_fixed = self._fix_any_grouped_nonbasis_edges(
            self.ctx.dg, group_to_nodes
        )
        if nonbasis_edges_fixed > 0:
            log.info("Fixed %s grouped non-basis edges", nonbasis_edges_fixed)

        # group_to_nodes, seperated_wccs = seperate_wccs(self.ctx, group_to_nodes)
        # if seperated_wccs > 0:
        #    log.info("Separated %s WCCs", seperated_wccs)

        ## 2. Fix any cycles in that conservative grouping
        group_to_nodes = fix_any_cycles_generalized(self.ctx, group_to_nodes)

        group_to_nodes, nonbasis_edges_fixed = self._fix_any_grouped_nonbasis_edges(
            self.ctx.dg, group_to_nodes
        )
        if nonbasis_edges_fixed > 0:
            log.info("Fixed %s grouped non-basis edges", nonbasis_edges_fixed)

        # group_to_nodes, seperated_wccs = seperate_wccs(self.ctx, group_to_nodes)
        # if seperated_wccs > 0:
        #    log.info("Separated %s WCCs", seperated_wccs)

        ## 2. Fix any cycles in that conservative grouping
        group_to_nodes = fix_any_cycles_generalized(self.ctx, group_to_nodes)

        fusions = 0
        if self.ctx.exec_cfg.enable_group_fusions:
            # 1.1. Absorb low-cost & consts into the groups
            group_to_nodes, fusions_ = absorb_low_cost_ops_into_groups(self.ctx, group_to_nodes)
            if fusions_ > 0:
                log.info("Absorbed %s low-cost groups", fusions_)
            fusions += fusions_

            # check_groups_are_dags(self.ctx.dg, group_to_nodes, "absorbing low-cost ops")

            group_to_nodes, fusions_ = absorb_consts_into_groups(self.ctx, group_to_nodes)
            if fusions_ > 0:
                log.info("Absorbed %s constants into groups", fusions_)
            fusions += fusions_

            # check_groups_are_dags(self.ctx.dg, group_to_nodes, "absorbing constants")

            # 3. Try to fuse the conservative groups
            group_to_nodes, fusions_ = self._fuse_groups(group_to_nodes)
            if fusions_ > 0:
                log.info("Fused %s groups into larger groups", fusions_)
            fusions += fusions_
            # check_groups_are_dags(self.ctx.dg, group_to_nodes, "fusing groups")

            group_to_nodes, nonbasis_edges_fixed = self._fix_any_grouped_nonbasis_edges(
                self.ctx.dg, group_to_nodes
            )
            if nonbasis_edges_fixed > 0:
                log.info("Fixed %s grouped non-basis edges", nonbasis_edges_fixed)

            group_to_nodes = fix_any_cycles_generalized(self.ctx, group_to_nodes)

        fissions = 0
        edges_cut = 0
        if self.ctx.exec_cfg.enable_group_fissions:
            group_to_nodes, fissions, edges_cut = self._cut_large_groups(group_to_nodes)

        # 4. Remove any small groups
        log.debug("Removing small groups")
        for gid in list(group_to_nodes.keys()):
            group_nodes = group_to_nodes[gid]
            if len(group_nodes) < self.ctx.exec_cfg.min_dataflow_group_size:
                del group_to_nodes[gid]

        log.debug("Building the final graph with grouped subgraphs")
        new_ctx, _ = self._build_ctx_for_grouping(self.ctx, group_to_nodes)
        log.info(
            "In %s iterations, did %s fusions, did %s fissions (%s edges cut). \
              Grouped %s subgraphs of sizes: %s",
            iterations,
            fusions,
            fissions,
            edges_cut,
            len(group_to_nodes),
            sorted([len(s) for s in group_to_nodes.values()], reverse=True),
        )
        return new_ctx, True
