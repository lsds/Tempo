# from dataclasses import replace
# from typing import List, Set, Tuple
#
# import islpy as isl
#
# from tempo.core import index_expr as ie
# from tempo.core.configs import ExecutionConfig
# from tempo.core.datatypes import OpId
# from tempo.core.dependence_graph import DependenceGraph
# from tempo.transformations.scheduling.isl_schedule_constraint_builder import (
#    IslScheduleConstraintsBuilder,
# )
# from tempo.utils.isl import get_isl_context
#
#
# class CycleChecker:
#
#    def __init__(self, exec_cfg: ExecutionConfig, dg: DependenceGraph) -> None:
#        self.exec_cfg = replace(
#            exec_cfg,
#            enable_parallel_block_detection=False,
#            enable_gc=False,
#            enable_swap=False,
#        )
#        self.dg = dg
#
#    def check_cycles(  # noqa: C901
#        self,
#    ) -> List[Tuple[Set[OpId], List[List[ie.IndexExpr]]]]:
#
#        # TODO I think this is marking nearly everything as impossible because
#        # it is missing the other conditions of merges.
#        impossible_cycles = []
#        for cycle in self.dg.get_all_cycles_with_edges():
#            cycle_ops = {edge[0] for edge in cycle}
#            cycle_ops.update([edge[1] for edge in cycle])
#
#            subgraph = self.dg.induced_subgraph(cycle_ops)
#
#            isl_ctx = get_isl_context(subgraph, self.exec_cfg)
#
#            sc = IslScheduleConstraintsBuilder(
#                subgraph, self.exec_cfg, isl_ctx
#            ).build_schedule_constraints()
#
#            try:
#                isl.ScheduleConstraints.compute_schedule(sc)
#            except Exception:
#                op_ids = {op.op_id for op in cycle_ops}
#                edge_exprs = []
#                for part in cycle:
#                    edges = []
#                    for edge in part[2]:
#                        if not edge.expr.is_basis():
#                            edges.append(edge.expr)
#                        if edges:
#                            edge_exprs.append(edges)
#                print("Found impossible cycle in computation graph: %s" % edge_exprs)
#                impossible_cycles.append((op_ids, edge_exprs))
#        return impossible_cycles  # type: ignore
#
