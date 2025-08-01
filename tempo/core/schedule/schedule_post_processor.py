# from __future__ import annotations
#
# from collections.abc import Mapping
# from dataclasses import dataclass, field
# from typing import Dict, List, Optional, Set, Tuple, Union
#
# from tempo.core import index_expr as ie
# from tempo.core import tensor_ops as top
# from tempo.core.configs import ExecutionConfig
# from tempo.core.datatypes import OpOutId, TensorId
# from tempo.core.dependence_graph import PDG
# from tempo.core.schedule.execution_schedule import (
#    DeallocInstruction,
#    ExecInstruction,
#    ExecutionSchedule,
#    FetchInstruction,
#    ForLoop,
#    IfGuard,
#    OffloadInstruction,
#    ParallelBlock,
#    ScheduleItem,
#    SequentialBlock,
# )
#
#
# @dataclass(frozen=True, slots=True)
# class PostProcCtx:
#    counters: Mapping[ie.Symbol, ForLoop] = field(default_factory=dict)
#    branches: Set[ie.BooleanIndexValue] = field(default_factory=set)
#    currently_in_mem: Mapping[TensorId, Set[ie.IndexSequence]] = field(default_factory=dict)
#    tids: Set[TensorId] = field(default_factory=set)
#
#
# @dataclass(frozen=True, slots=True)
# class PostProcResults:
#    item: ScheduleItem
#    created_tensors: Dict[TensorId, Set[ie.IndexSequence]] = field(default_factory=dict)
#    accessed_tensors: Dict[
#        TensorId, Set[Tuple[Optional[ie.BooleanIndexValue], ie.IndexSequence]]
#    ] = field(default_factory=dict)
#    deallocated_tensors: Dict[TensorId, Set[ie.IndexSequence]] = field(default_factory=dict)
#
#    def merge(self, item: ScheduleItem, other: Union[PostProcResults, None]) -> PostProcResults:
#        merged_created_tensors = {**self.created_tensors}
#        merged_accessed_tensors = {**self.accessed_tensors}
#        merged_deallocated_tensors = {**self.deallocated_tensors}
#        if other is not None:
#            for k, v in other.created_tensors.items():
#                v = set(v)
#                merged_created_tensors[k] = merged_created_tensors.setdefault(k, v).union(v)
#            for k, v in other.accessed_tensors.items():
#                v = set(v)
#                merged_accessed_tensors[k] = merged_accessed_tensors.setdefault(k, v).union(v)
#            for k, v in other.deallocated_tensors.items():
#                v = set(v)
#                merged_deallocated_tensors[k] =
#                   merged_deallocated_tensors.setdefault(k, v).union(v)
#
#        return PostProcResults(
#            item,
#            created_tensors=merged_created_tensors,
#            accessed_tensors=merged_accessed_tensors,
#            deallocated_tensors=merged_deallocated_tensors,
#        )
#
#    @staticmethod
#    def merge_all(item: ScheduleItem, others: List[PostProcResults]) -> PostProcResults:
#        first = others[0]
#        others = others[1:]
#        merged_created_tensors = {**first.created_tensors}
#        merged_accessed_tensors = {**first.accessed_tensors}
#        merged_deallocated_tensors = {**first.deallocated_tensors}
#        for other in others:
#            for k, v in other.created_tensors.items():
#                v = set(v)
#                merged_created_tensors[k] = merged_created_tensors.setdefault(k, v).union(v)
#            for k, v in other.accessed_tensors.items():
#                v = set(v)
#                merged_accessed_tensors[k] = merged_accessed_tensors.setdefault(k, v).union(v)
#            for k, v in other.deallocated_tensors.items():
#                v = set(v)
#                merged_deallocated_tensors[k] =
#                   merged_deallocated_tensors.setdefault(k, v).union(v)
#
#        return PostProcResults(
#            item,
#            created_tensors=merged_created_tensors,
#            accessed_tensors=merged_accessed_tensors,
#            deallocated_tensors=merged_deallocated_tensors,
#        )
#
#
# @dataclass(frozen=True, slots=True)
# class SchedulePostProcessor:
#    schedule: ExecutionSchedule
#    dg: PDG
#    exec_cfg: ExecutionConfig
#
#    def start_walk(self) -> ExecutionSchedule:
#        if not self.exec_cfg.enable_swap:
#            return self.schedule
#
#        tids = {
#            tid
#            for tid, _, _, _ in self.dg.get_all_tensor_descriptions()
#            if self._requires_swap(tid)
#        }
#        res = self._node_type_dispatcher(self.schedule.schedule, PostProcCtx(tids=set(tids)))
#        return ExecutionSchedule(res.item)
#
#    def _requires_swap(self, tid: TensorId) -> bool:
#        # stor = self.dg.analysis_ctx._tensor_storage_classes
#        # NOTE: The comment below is not correct. The block store be full prealloc and still
#        #  require swapping.
#        # if stor is not None and tid in stor:
#        #    tid_storage = stor[tid]
#        #    # NOTE: If it is a block store that has no block sizes, there is no point in swapping.
#        #    if tid_storage.is_full_prealloc():
#        #        return False
#        return (
#            self.dg.estimate_tensor_size_bytes(
#                tid.op_id,
#                out_idx=tid.output_id,
#                bound_size_estimate=self.exec_cfg.default_dim_upper_bound_size,
#            )
#            > self.exec_cfg.gc_bytes_min
#        )
#
#    def _node_type_dispatcher(self, node: ScheduleItem, ctx: PostProcCtx) -> PostProcResults:
#        if isinstance(node, ExecInstruction):
#            return self._proc_exec_instruction(node, ctx)
#        elif isinstance(node, DeallocInstruction):
#            return self._proc_dealloc_instruction(node, ctx)
#        elif isinstance(node, OffloadInstruction):
#            return self._proc_offload_instruction(node, ctx)
#        elif isinstance(node, FetchInstruction):
#            return self._proc_fetch_instruction(node, ctx)
#        elif isinstance(node, SequentialBlock):
#            return self._proc_sequential_block(node, ctx)
#        elif isinstance(node, ParallelBlock):
#            # TODO will need to wrap the parallel in a sequential block to do swaps
#            # return self._proc_parallel_block(node)
#            raise NotImplementedError("Parallel block not implemented")
#        elif isinstance(node, IfGuard):
#            return self._proc_if_guard(node, ctx)
#        elif isinstance(node, ForLoop):
#            return self._proc_for_loop(node, ctx)
#        else:
#            raise ValueError(f"Unknown node type: {node}")
#
#    def _proc_exec_instruction(self, node: ExecInstruction, ctx: PostProcCtx) -> PostProcResults:
#        op = self.dg.get_op_by_id(node.op_id)
#        num_outs = op.num_outputs
#
#        op_domain_basis_remmapped = op.domain.basis_expr.remap(node.domain_map)
#        created_tensors = {
#            TensorId(node.op_id, OpOutId(i)): {op_domain_basis_remmapped}
#            for i in range(num_outs)
#            if TensorId(node.op_id, OpOutId(i)) in ctx.tids
#        }
#
#        accessed_tensors: Dict[
#            TensorId, Set[Tuple[Optional[ie.BooleanIndexValue], ie.IndexSequence]]
#        ] = {}
#        if isinstance(op, top.MergeOp):
#            ordered_deps = sorted(
#                self.dg.get_flat_direct_dependencies(op),
#                key=lambda x: x[1].src_out_idx,
#            )
#            for dep_, dep_data in ordered_deps:
#                index = dep_data.expr
#                cond = dep_data.cond
#                tid = TensorId(dep_.op_id, dep_data.src_out_idx)
#                if tid in ctx.tids:
#                    remapped_index = index.remap(node.domain_map)
#                    remapped_cond = cond.remap(node.domain_map)
#                    assert isinstance(remapped_cond, ie.BooleanIndexValue)
#                    # TODO it may be the case that given the cond, counters and branches that we
#                    # can simply figure out which branch is taken.
#                    # For now, we just add the condition to the access.
#                    accessed_tensors[tid] = {(remapped_cond, remapped_index)}
#        else:
#            for depy, dep_data in self.dg.get_flat_direct_dependencies(op):
#                tid = TensorId(depy.op_id, dep_data.src_out_idx)
#                if tid in ctx.tids:
#                    access_expr = dep_data.expr.remap(node.domain_map)
#
#                    accessed_tensors[tid] = {(None, access_expr)}
#        return PostProcResults(
#            node, created_tensors=created_tensors, accessed_tensors=accessed_tensors
#        )
#
#    def _proc_dealloc_instruction(
#        self, node: DeallocInstruction, ctx: PostProcCtx
#    ) -> PostProcResults:
#        tid = node.tensor_id
#        # op_isl_domain = self.dg.analysis_ctx.get_or_make_domain(op)
#        # op_isl_domain = isl_utils.rename_union_set_tuples(self.dg.analysis_ctx
#        #       .get_or_make_domain(op), "")
#        # dealloc_set = isl_utils.index_sequence_to_isl_union_set(
#        #    node.index, ctx=self.dg.analysis_ctx.isl_ctx
#        # ).intersect(op_isl_domain)
#        return PostProcResults(
#            node, deallocated_tensors={tid: {node.index}} if tid in ctx.tids else {}
#        )
#
#    def _proc_offload_instruction(
#        self, node: OffloadInstruction, ctx: PostProcCtx
#    ) -> PostProcResults:
#        return PostProcResults(node)
#
#    def _proc_fetch_instruction(self, node: FetchInstruction, ctx: PostProcCtx) -> PostProcResults:
#        return PostProcResults(node)
#
#    def _proc_sequential_block(  # noqa: C901
#        self, node: SequentialBlock, ctx: PostProcCtx
#    ) -> PostProcResults:
#        children = node.children
#
#        in_mem = {**ctx.currently_in_mem}
#        dispatch_results: List[PostProcResults] = []
#        for child in children:
#            if isinstance(child, ExecInstruction):
#                op_domain_basis_remmapped = self.dg.get_op_by_id(
#                    child.op_id
#                ).domain.basis_expr.remap(child.domain_map)
#                for i in range(self.dg.get_op_by_id(child.op_id).num_outputs):
#                    tid = TensorId(child.op_id, OpOutId(i))
#                    if tid in ctx.tids:
#                        in_mem.setdefault(tid, set()).add(op_domain_basis_remmapped)
#            dispatch_results.append(
#                self._node_type_dispatcher(
#                    child, PostProcCtx(ctx.counters, ctx.branches, in_mem, ctx.tids)
#                )
#            )
#
#        large_accesses: Dict[
#            TensorId, Set[Tuple[Optional[ie.BooleanIndexValue], ie.IndexSequence]]
#        ] = {}
#        large_creations: Dict[TensorId, Set[ie.IndexSequence]] = {}
#        large_nested_creations_we_need: Dict[TensorId, Tuple[int, Set[ie.IndexSequence]]] = {}
#        large_deletions: Dict[TensorId, Set[ie.IndexSequence]] = {}
#
#        for child, res in zip(children, dispatch_results, strict=False):
#            if isinstance(child, ExecInstruction):
#                for k, v in res.created_tensors.items():
#                    large_creations[k] = large_creations.get(k, set()).union(v)
#                for k, v in res.accessed_tensors.items():
#                    large_accesses[k] = large_accesses.get(k, set()).union(v)
#
#            if isinstance(child, DeallocInstruction):
#                for k_, v_ in res.deallocated_tensors.items():
#                    large_deletions[k_] = large_deletions.get(k_, set()).union(v_)
#
#        for i, child in enumerate(children):
#            if not isinstance(child, (ExecInstruction, DeallocInstruction)):
#                res = dispatch_results[i]
#                for k_, v_ in res.created_tensors.items():
#                    if len(ctx.currently_in_mem) == 0:  # First sequence
#                        print(f"First seq len of nested created {len(v_)}")
#                    if k_ in large_accesses:
#                        # we only want to fetch the stuff that is accessed.
#                        # TODO is thsi correct?
#                        position_fetch = {x[1] for x in large_accesses[k_]}.intersection(v_)
#                        # initial_fetch = large_accesses[k].difference(position_fetch)
#                        # Record the position at which this is created. Fetches will need to
#                        #  be inserted after this.
#                        large_nested_creations_we_need[k_] = (i, position_fetch)
#                        # if initial_fetch.is_empty():
#                        #    del large_accesses[k]
#                        # else:
#                        #    large_accesses[k] = initial_fetch
#
#        children_copy = [x.item for x in dispatch_results]
#
#        sorted_flattened_large_nested_creations = sorted(
#            large_nested_creations_we_need.items(), key=lambda x: x[1][0], reverse=True
#        )
#        for tid, (pos, creation_set) in sorted_flattened_large_nested_creations:
#            for point in creation_set:
#                children_copy.insert(pos + 1, FetchInstruction(tid, point))
#
#        for tid, access_set in large_accesses.items():
#            # Fetches
#            internal_creation = large_creations.get(tid, set())
#            nested_creation = large_nested_creations_we_need.get(tid, (-1, set()))[1]
#
#            # For each large access done in this sequence, we need to fetch the data,
#            # unless the data is:
#            # - created by an op in this sequence
#            # - created by an op in a nested sequence
#            # - already in memory
#            for cond, point in access_set:
#                if (
#                    point in internal_creation
#                    or point in nested_creation
#                    or point in ctx.currently_in_mem.get(tid, set())
#                ):
#                    continue
#                if cond is None or cond.equivalent(ie.ConstBool(True)):
#                    if len(ctx.currently_in_mem) == 0:  # First sequence
#                        print(f"Fetching {tid} at {point}")
#                        print(f"{internal_creation=}")
#                        print(f"{nested_creation=}")
#                        print(f"full nested_creations={large_nested_creations_we_need}")
#                    children_copy.insert(0, FetchInstruction(tid, point))
#                else:
#                    children_copy.insert(0, IfGuard(cond, FetchInstruction(tid, point), None))
#
#            # Offloads
#            dealloc = large_deletions.get(tid, set())
#
#            for cond, point in access_set:
#                if point in dealloc or point in ctx.currently_in_mem.get(tid, set()):
#                    continue
#                if cond is None or cond.equivalent(ie.ConstBool(True)):
#                    children_copy.append(OffloadInstruction(tid, point))
#                else:
#                    children_copy.append(IfGuard(cond, OffloadInstruction(tid, point), None))
#
#        item = SequentialBlock(children_copy)
#
#        res = PostProcResults.merge_all(item, dispatch_results)
#
#        return res
#
#    # def _proc_parallel_block(self, node: ParallelBlock) -> PostProcResults:
#    #    # TODO parallel block
#    #    return node
#
#    def _proc_if_guard(self, node: IfGuard, ctx: PostProcCtx) -> PostProcResults:
#        # TODO could simplify if_cond
#
#        new_ctx_if = PostProcCtx(
#            {**ctx.counters},
#            {*ctx.branches, node.if_cond},
#            {**ctx.currently_in_mem},
#            ctx.tids,
#        )
#        new_ctx_else = PostProcCtx(
#            {**ctx.counters},
#            {*ctx.branches, ie.Not(node.if_cond)},
#            {**ctx.currently_in_mem},
#            ctx.tids,
#        )
#        simplified_if_ = node.if_cond
#        # simplified_if_ = isl_utils.simplify_boolean_index_expr(self.dg.universe, node.if_cond,
#        #  self.dg.static_bounds)
#        then_res = self._node_type_dispatcher(node.then_inner, new_ctx_if)
#        else_res = (
#            None
#            if node.else_inner is None
#            else self._node_type_dispatcher(node.else_inner, new_ctx_else)
#        )
#
#        if simplified_if_.equivalent(ie.ConstBool(True)):
#            return then_res
#        if simplified_if_.equivalent(ie.ConstBool(False)):
#            assert else_res is not None, "need to implement ability to skip if alltogether"
#            return else_res
#        if_ = IfGuard(
#            simplified_if_,
#            then_res.item,
#            (None if else_res is None else else_res.item),
#        )
#        res = then_res.merge(if_, else_res)
#
#        return res
#
#    def _proc_for_loop(self, node: ForLoop, ctx: PostProcCtx) -> PostProcResults:
#        new_ctx = PostProcCtx(
#            {**ctx.counters, node.counter: node},
#            {*ctx.branches},
#            {**ctx.currently_in_mem},
#            ctx.tids,
#        )
#        inner_res = self._node_type_dispatcher(node.inner, new_ctx)
#        loop = ForLoop(node.counter, node.init, node.cond, node.increment, inner_res.item)
#
#        return PostProcResults(loop, inner_res.created_tensors)
#
