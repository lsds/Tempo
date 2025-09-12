from __future__ import annotations

from collections.abc import Mapping, Sequence

import islpy as isl

from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.datatypes import OpId, OpOutId, TensorId
from tempo.core.schedule.execution_schedule import (
    DeallocInstruction,
    ExecInstruction,
    ExecutionSchedule,
    FetchInstruction,
    ForLoop,
    IfGuard,
    MemManInstr,
    OffloadInstruction,
    ParallelBlock,
    ScheduleItem,
    SequentialBlock,
)
from tempo.core.statement_type import StmtType
from tempo.core.symbol_dict import SymbolDict
from tempo.utils import isl as isl_utils
from tempo.utils import logger

log = logger.get_logger(__name__)

# def _can_join(self, a: ie.IndexAtom, b: ie.IndexAtom) -> bool:
#    if isinstance(a, ie.Slice) and isinstance(b, ie.Slice):
#        # Can join if slices are adjacent (ignoring step, assumed to be 1)
#        return a.stop.equivalent(b.start)
#    elif isinstance(a, ie.IntIndexValue) and isinstance(b, ie.IntIndexValue):
#        # Can join if integers are adjacent
#        return self._are_adjacent(a, b)
#    elif isinstance(a, ie.Slice) and isinstance(b, ie.IntIndexValue):
#        # Can join if integer is adjacent to the end of the slice
#        return a.stop.equivalent(b)
#    elif isinstance(a, ie.IntIndexValue) and isinstance(b, ie.Slice):
#        # Can join if integer is adjacent to the start of the slice
#        return a.equivalent(b.start)
#    else:
#        # Slices and integers may be joined into a bigger slice if adjacent
#        return False
#
##TODO: eventually, use isl_utils to do all of this. Basically create one set per index,
#  then union them all
## and check if the result can be described by a single set or needs to be a union of multiple sets
# def _are_adjacent(self, a: ie.IntIndexValue, b: ie.IntIndexValue) -> bool:
#    # Check if a and b are adjacent
#    diff = ie.Sub(a, b)
#    simplified_diff = self._simplify(diff)
#    if isinstance(simplified_diff, ie.ConstInt) and abs(simplified_diff.const) == 1:
#        return True
#    else:
#        # For symbolic expressions like t+3 and t+4
#        # Extract symbols and offsets
#        sym_a, offset_a = self._get_symbol_and_offset(a)
#        sym_b, offset_b = self._get_symbol_and_offset(b)
#        if sym_a and sym_b and sym_a.equivalent(sym_b):
#            return abs(offset_a - offset_b) == 1
#        else:
#            return False
#
# def _get_symbol_and_offset(self, expr: ie.IntIndexValue) -> Tuple[Optional[ie.Symbol], int]:
#    if isinstance(expr, ie.Symbol):
#        return expr, 0
#    elif isinstance(expr, ie.Add):
#        # Handle expressions like Symbol + ConstInt
#        left, right = expr.left_operand, expr.right_operand
#        if isinstance(left, ie.Symbol) and isinstance(right, ie.ConstInt):
#            return left, right.const
#        elif isinstance(right, ie.Symbol) and isinstance(left, ie.ConstInt):
#            return right, left.const
#    elif isinstance(expr, ie.ConstInt):
#        return None, expr.const
#    return None, 0  # Default case
#
# def _simplify(self, expr: ie.IntIndexValue) -> ie.IntIndexValue:
#    # Simplify the expression for basic arithmetic cases
#    if isinstance(expr, ie.Sub):
#        left = self._simplify(expr.left_operand)
#        right = self._simplify(expr.right_operand)
#        if isinstance(left, ie.ConstInt) and isinstance(right, ie.ConstInt):
#            return ie.ConstInt(left.const - right.const)
#        elif left.equivalent(right):
#            return ie.ConstInt(0)
#        else:
#            return ie.Sub(left, right)
#    elif isinstance(expr, ie.Add):
#        left = self._simplify(expr.left_operand)
#        right = self._simplify(expr.right_operand)
#        if isinstance(left, ie.ConstInt) and isinstance(right, ie.ConstInt):
#            return ie.ConstInt(left.const + right.const)
#        else:
#            return ie.Add(left, right)
#    else:
#        return expr  # Return as is if no simplification is possible
#
# def _join(self, a: ie.IndexAtom, b: ie.IndexAtom) -> ie.IndexAtom:
#    if isinstance(a, ie.Slice) and isinstance(b, ie.Slice):
#        # Join adjacent slices
#        new_start = a.start
#        new_stop = b.stop
#        return ie.Slice(new_start, new_stop)
#    elif isinstance(a, ie.IntIndexValue) and isinstance(b, ie.IntIndexValue):
#        # Create a slice that includes both indices
#        # Since they are adjacent, the slice includes both
#        new_start = ie.Min((a, b))
#        new_stop = ie.Add(ie.Max((a, b)), ie.ConstInt(1))
#        return ie.Slice(new_start, new_stop)
#    elif isinstance(a, ie.Slice) and isinstance(b, ie.IntIndexValue):
#        # Join slice and adjacent integer index
#        new_start = a.start
#        new_stop = ie.Add(ie.Max((a.stop, b)), ie.ConstInt(1))
#        return ie.Slice(new_start, new_stop)
#    elif isinstance(a, ie.IntIndexValue) and isinstance(b, ie.Slice):
#        # Join integer index and adjacent slice
#        new_start = ie.Min((a, b.start))
#        new_stop = b.stop
#        return ie.Slice(new_start, new_stop)
#    else:
#        raise ValueError("Cannot join the given IndexAtoms")


class IslASTToTempoScheduleWalker:
    def __init__(self, comp_ctx: CompilationCtx, quiet: bool = False) -> None:
        self.comp_ctx = comp_ctx
        self.dg = comp_ctx.dg
        self.isl_ctx = comp_ctx.analysis_ctx.isl_ctx
        self.exec_cfg = comp_ctx.exec_cfg
        self.next_is_parallel = False
        self.removed_swap_ops = 0
        self.seen_swap_ops = 0
        self.seen_mem_man_ops = 0
        self.promoted_ops = 0

        self.visited_first_block = False
        self.quiet = quiet

    def start_walk(self, root: isl.AstNode) -> ExecutionSchedule:
        sched_inner = self._node_type_dispatcher(root)
        assert sched_inner is not None
        res = ExecutionSchedule(sched_inner)

        removed_percent = (
            round(self.removed_swap_ops / self.seen_swap_ops * 100, 2)
            if self.seen_swap_ops > 0
            else 0
        )
        promoted_percent = (
            round(self.promoted_ops / self.seen_mem_man_ops * 100, 2)
            if self.seen_mem_man_ops > 0
            else 0
        )
        if not self.quiet:
            log.info(
                "Removed %s%% of ops because redundant and promoted %s%%.",
                removed_percent,
                promoted_percent,
            )

        return res

    def _node_type_dispatcher(self, node: isl.AstNode) -> ScheduleItem | None:
        if node.get_type() == isl.ast_node_type.error:
            return self._visit_error(node)
        if node.get_type() == isl.ast_node_type.for_:
            return self._visit_for(node)
        if node.get_type() == isl.ast_node_type.if_:
            return self._visit_if(node)
        if node.get_type() == isl.ast_node_type.block:
            return self._visit_block(node)
        if node.get_type() == isl.ast_node_type.user:
            return self._visit_user(node)
        return self._visit_mark(node)

    def _visit_mark(self, node: isl.AstNode) -> ScheduleItem | None:
        mark = node.mark_get_id().to_str()
        node_ = node.mark_get_node()
        marked = self._node_type_dispatcher(node_)
        if marked is not None and "parallel" in mark:
            if node_.get_type() == isl.ast_node_type.block:
                self.next_is_parallel = True
                # Otherwise, we ignore it...
        # assert node.mark_get_node().get_type() == isl.ast_node_type.block,
        # f"Expected block, got {node.mark_get_node().get_type()}"
        return marked

    def _promote_block_ops(
        self,
        flat_children: list[ScheduleItem],
        for_loop: ForLoop,
    ) -> ScheduleItem:
        block_instrs = []
        mem_man_only = [n for n in flat_children if isinstance(n, MemManInstr)]
        for m in mem_man_only:
            new_expr = self._get_promoted_index_expr(for_loop, m.index)
            block_instrs.append(m.__class__(m.tensor_id, new_expr))
            self.promoted_ops += 1

        if len(block_instrs) == 1:
            return block_instrs[0]
        return SequentialBlock(block_instrs)

    def _get_promoted_index_expr(
        self, for_loop: ForLoop, index: ie.IndexSequence
    ) -> ie.IndexSequence:
        counter = for_loop.counter
        init = for_loop.init
        cond = for_loop.cond
        # inc = for_loop.inc

        new_expr = []
        assert sum(1 for i in index if counter in i.vars_used()) == 1, (
            f"Counter can only appear once in {index}"
        )

        for member in index:
            if counter in member.vars_used():  # member.equivalent(counter):
                # These asserts are just to shut up mypy
                # TODO add inc/step when supported
                mapped_member: ie.IntIndexValue = member.remap({counter: init})  # type: ignore
                # assert isinstance(mapped_member, ie.BooleanBinaryExpr)
                mapped_cond_right: ie.IntIndexValue = member.remap(  # type: ignore
                    {counter: cond.right_operand}  # type: ignore
                )

                # TODO this is going to be problematic for non-constant stuff..
                mapped_member_eval_tried = mapped_member.try_eval(self.dg.static_bounds)
                mapped_member_right_eval_tried = mapped_cond_right.try_eval(self.dg.static_bounds)
                if isinstance(mapped_member_eval_tried, int) and isinstance(
                    mapped_member_right_eval_tried, int
                ):
                    if mapped_member_eval_tried > mapped_member_right_eval_tried:
                        mapped_member, mapped_cond_right = (
                            ie.ConstInt(mapped_member_right_eval_tried),
                            ie.ConstInt(mapped_member_eval_tried),
                        )
                    else:
                        mapped_member, mapped_cond_right = (
                            ie.ConstInt(mapped_member_eval_tried),
                            ie.ConstInt(mapped_member_right_eval_tried),
                        )

                    # print(
                    #    f"Promoting member {member} of index ({m.index}) to {mapped_member}
                    #  and {mapped_cond_right} with {counter=} {init=} {cond.right_operand=}"
                    # )
                new_expr.append(
                    # NOTE: +1 is needed because slice is exclusive
                    ie.slice_(start=mapped_member, stop=mapped_cond_right + 1)
                )
            else:
                new_expr.append(member)
        return ie.IndexSequence(tuple(new_expr))

    def _visit_for(self, node: isl.AstNode) -> ScheduleItem | None:
        counter = isl_utils.isl_expr_to_tempo_expr(node.for_get_iterator())
        init = isl_utils.isl_expr_to_tempo_expr(node.for_get_init())
        cond = isl_utils.isl_expr_to_tempo_expr(node.for_get_cond())
        inc = isl_utils.isl_expr_to_tempo_expr(node.for_get_inc())

        body = self._node_type_dispatcher(node.for_get_body())

        if body is None:
            return None

        # NOTE: Try to promote the body to a Sequential of Block MemManInstrs
        cond_valid = isinstance(cond, ie.LessThanOrEqual) and cond.left_operand.struct_eq(counter)
        flat_children = body.flat_recursive_tree
        all_mem_man = all(
            isinstance(m, (MemManInstr, SequentialBlock, ParallelBlock)) for m in flat_children
        )
        # TODO: more fine-grained. Some mem man ops are not eligible for promotion, but we can just
        # surround each such with a copy of the loop.
        inc_is_1 = isinstance(inc, ie.ConstInt) and inc.const == 1
        for_loop = ForLoop(counter, init, cond, inc, body)  # type: ignore
        counters_used_once = all(
            sum(1 for i in m.index if counter in i.vars_used()) == 1
            for m in flat_children
            if isinstance(m, MemManInstr)
        )
        validity_conds = all_mem_man and cond_valid and inc_is_1 and counters_used_once

        if self.exec_cfg.enable_ast_promo and validity_conds:
            return self._promote_block_ops(flat_children, for_loop)  # type: ignore
        # else:
        #    print(f"Could not promote for loop {for_loop} because
        #  {cond_valid=} {all_mem_man=} {inc_is_1=}")

        return for_loop

    def _visit_if(self, node: isl.AstNode) -> ScheduleItem | None:
        if_cond = isl_utils.isl_expr_to_tempo_boolean_expr(node.if_get_cond())
        then_node = self._node_type_dispatcher(node.if_get_then_node())
        else_node = None
        if node.if_has_else_node():
            else_node = self._node_type_dispatcher(node.if_get_else_node())

        if then_node is None and else_node is None:
            return None

        if then_node is None and else_node is not None:
            if_cond = ie.Not(if_cond)
            return IfGuard(if_cond, else_node, then_node)

        return IfGuard(if_cond, then_node, else_node)  # type: ignore

    def _is_distributable(self, item: ScheduleItem) -> bool:
        distributable = True
        for s in item.flat_recursive_tree:
            if isinstance(s, ExecInstruction):
                op = self.dg.ops_by_id[s.op_id].op
                if op.is_stateful:
                    distributable = False
                    break
        return distributable

    def _in_loop_redundant_swap_removals(
        self, instrs: list[ScheduleItem]
    ) -> tuple[bool, list[ScheduleItem]]:
        changes = False
        processed_instrs: list[ScheduleItem] = []
        skip_next = False
        for instr, next_instr in zip(instrs[:-1], instrs[1:], strict=True):
            if skip_next:
                skip_next = False
                continue

            if isinstance(instr, ForLoop) and isinstance(next_instr, FetchInstruction):
                for_body = instr.inner
                if isinstance(for_body, SequentialBlock):
                    for_body_instrs = for_body.inner_block
                    # Find any offload matching the fetch tensor id
                    for_body_offload_idx = next(
                        (
                            i
                            for i, x in enumerate(for_body_instrs)
                            if isinstance(x, OffloadInstruction)
                            and x.tensor_id == next_instr.tensor_id
                        ),
                        None,
                    )
                    if for_body_offload_idx is not None:
                        # Check if the offload expanded to the for loop range equals the fetch
                        # index. If so, remove the offload.
                        for_body_offload = for_body_instrs[for_body_offload_idx]
                        assert isinstance(for_body_offload, OffloadInstruction)
                        for_body_offload_index = for_body_offload.index
                        for_body_offload_index_promoted = self._get_promoted_index_expr(
                            instr, for_body_offload_index
                        )
                        if for_body_offload_index_promoted.struct_eq(next_instr.index):
                            changes = True
                            skip_next = True
                            self.removed_swap_ops += 2
                            # Remove the offload from the for body and reappend, skipping the fetch
                            new_block_instrs = list(for_body_instrs)
                            new_block_instrs.pop(for_body_offload_idx)
                            new_for_loop_body = (
                                SequentialBlock(new_block_instrs)
                                if len(new_block_instrs) > 1
                                else new_block_instrs[0]
                            )
                            processed_instrs.append(
                                ForLoop(
                                    instr.counter,
                                    instr.init,
                                    instr.cond,
                                    instr.increment,
                                    new_for_loop_body,
                                )
                            )
                        else:
                            processed_instrs.append(instr)
                    else:
                        processed_instrs.append(instr)
                if (
                    isinstance(for_body, OffloadInstruction)
                    and for_body.tensor_id == next_instr.tensor_id
                ):
                    for_body_offload_index = for_body.index
                    for_body_offload_index_promoted = self._get_promoted_index_expr(
                        instr, for_body_offload_index
                    )
                    if for_body_offload_index_promoted.struct_eq(next_instr.index):
                        changes = True
                        skip_next = True
                        self.removed_swap_ops += 2
                        # skip this for loop and the next fetch instruction
                    else:
                        processed_instrs.append(instr)
            else:
                processed_instrs.append(instr)

        # Add the last instruction
        processed_instrs.append(instrs[-1])

        return changes, processed_instrs

    def _intra_block_redundant_swap_removals(  # noqa: C901
        self, instrs: list[ScheduleItem]
    ) -> tuple[bool, list[ScheduleItem]]:
        # This method removes redundant swap instructions within a block. These are:
        # 1. Offload -> ... -> Fetch pairs where the tensor and index are the same
        # 2. Offload  -> ... -> Deallocate pairs where the tensor and index are the same
        # 3. Execute op -> ... -> Fetch tensorid(op_id, any) pairs
        # 4. Duplicate fetches or offloads where the tensor and index are the same
        changes = False

        def exec_domain_map_to_index_seq(
            op: top.TensorOp, domain_map: Mapping[ie.Symbol, ie.IntIndexValue]
        ) -> ie.IndexSequence:
            domain_list: list[ie.IndexValue] = []
            for var in op.domain.variables:
                domain_list.append(domain_map[var])  # type: ignore
            return ie.IndexSequence(tuple(domain_list))

        processed_instrs: list[ScheduleItem] = []

        # Maps (tensor_id, index) to position in processed_instrs
        offload_map: dict[tuple[TensorId, ie.IndexSequence], int] = {}

        # Maps (tensor_id, index) to position in processed_instrs
        fetch_map: dict[tuple[TensorId, ie.IndexSequence], int] = {}

        # Maps (tensor_id,) to position in processed_instrs
        fetch_map_no_user: dict[TensorId, tuple[int, ie.IndexSequence]] = {}

        # Maps (tensor_id, index) to position in processed_instrs
        execute_prod_map: dict[tuple[TensorId, ie.IndexSequence], int] = {}

        def adjust_maps(pos: int) -> None:
            for key, idx in offload_map.items():
                if idx > pos:
                    offload_map[key] = idx - 1
            for key, idx in fetch_map.items():
                if idx > pos:
                    fetch_map[key] = idx - 1
            for key, idx in fetch_map_no_user.items():
                if idx[0] > pos:
                    fetch_map_no_user[key] = (idx[0] - 1, idx[1])
            for key, idx in execute_prod_map.items():
                if idx > pos:
                    execute_prod_map[key] = idx - 1

        for instr in instrs:
            if isinstance(instr, OffloadInstruction):
                tensor_index = (instr.tensor_id, instr.index)
                if tensor_index in offload_map:
                    # Duplicate offload, remove the duplicate
                    self.removed_swap_ops += 1
                    # skip this OffloadInstruction
                    changes = True
                else:
                    offload_map[tensor_index] = len(processed_instrs)
                    processed_instrs.append(instr)

            elif isinstance(instr, FetchInstruction):
                # TODO if no clears have happened and fetch not in prod map,
                # then move to start of block
                tensor_index = (instr.tensor_id, instr.index)
                if tensor_index in offload_map:
                    # NOTE: A change was introduced here. We keep fetches even though they may look
                    # redundant because they may be needed outside the block
                    self.removed_swap_ops += 1
                    # Remove the OffloadInstruction
                    pos = offload_map.pop(tensor_index)
                    processed_instrs.pop(pos)
                    changes = True
                    # Adjust positions in offload_map, fetch_map, execute_map
                    adjust_maps(pos)

                if tensor_index in execute_prod_map:
                    # Fetch after Execute, redundant
                    self.removed_swap_ops += 1
                    # Do not append this FetchInstruction
                    changes = True

                elif tensor_index in fetch_map:
                    # Duplicate fetch, remove the duplicate
                    self.removed_swap_ops += 1
                    # Do not append this FetchInstruction
                    changes = True
                else:
                    fetch_map[tensor_index] = len(processed_instrs)
                    fetch_map_no_user[tensor_index[0]] = (
                        len(processed_instrs),
                        instr.index,
                    )
                    processed_instrs.append(instr)

            elif isinstance(instr, DeallocInstruction):
                tensor_index = (instr.tensor_id, instr.index)
                if tensor_index in offload_map:
                    self.removed_swap_ops += 1
                    # Remove the OffloadInstruction
                    changes = True
                    pos = offload_map.pop(tensor_index)
                    processed_instrs.pop(pos)
                    adjust_maps(pos)
                if tensor_index[0] in fetch_map_no_user:
                    # Deallocate after Fetch, possibly redundant
                    pos, index = fetch_map_no_user.pop(tensor_index[0])
                    is_superset = self.check_a_is_superset_of_b(instr.index, index)
                    if is_superset or index == instr.index:
                        # The dealloc is a superset of the fetch, so we can remove the fetch
                        changes = True
                        self.removed_swap_ops += 1
                        processed_instrs.pop(pos)
                        adjust_maps(pos)

                # Always append the DeallocInstruction
                processed_instrs.append(instr)

            elif isinstance(instr, ExecInstruction):
                op = self.dg.ops_by_id[instr.op_id].op
                domain_map = instr.domain_map
                index_seq = exec_domain_map_to_index_seq(op, domain_map)
                for o in range(op.num_outputs):
                    tensor_id = TensorId(instr.op_id, OpOutId(o))
                    tensor_index = (tensor_id, index_seq)
                    execute_prod_map[tensor_index] = len(processed_instrs)

                for depy, depy_data in self.dg.get_flat_direct_dependencies(op):
                    tid_ = TensorId(depy.op_id, depy_data.src_out_idx)

                    if tid_ in fetch_map_no_user:
                        # There is a user of this fetch, (at least conservatively)
                        fetch_map_no_user.pop(tid_)

                processed_instrs.append(instr)

            else:
                # So then it's some nested access. In this case, we must reset the maps
                # because we can't guarantee that the nest won't modify the on-device
                #  state of the tensor.
                # TODO there is another issue: if there is offload-nested-offload-fetch,
                #  then we will remove the offload-fetch pair expecting the tensor to be on device
                # Though we have removed that offload fetch dedupe already...
                offload_map = {}
                fetch_map = {}
                fetch_map_no_user = {}
                execute_prod_map = {}
                # TODO use this to be less conservative. pop only keys that are used in
                # offload/fetch in the nested block
                # instr.flat_recursive_tree
                processed_instrs.append(instr)
        return changes, processed_instrs

    def check_a_is_superset_of_b(self, a: ie.IndexSequence, b: ie.IndexSequence) -> bool:
        is_superset = False
        a_start, a_stop = (
            a.as_lower_bound_access(),
            a.as_upper_bound_access(),
        )
        b_start, b_stop = (
            b.as_lower_bound_access(),
            b.as_upper_bound_access(),
        )
        if (
            a_start.is_constant()
            and b_start.is_constant()
            and a_stop.is_constant()
            and b_stop.is_constant()
        ):
            # dealloc_start <= fetch_start and dealloc_stop >= fetch_stop
            # NOTE: At this point, a_ and b_ are always indexvalues, not slices.
            # thus, the type ignores, because mypy does not get it.
            is_superset = all(
                (a_ <= b_).evaluate(self.dg.static_bounds)  # type: ignore
                for a_, b_ in zip(a_start, b_start, strict=False)
            ) and all(
                (a_ >= b_).evaluate(self.dg.static_bounds)  # type: ignore
                for a_, b_ in zip(a_stop, b_stop, strict=False)
            )

        return is_superset

    def _flatten_side_by_side(self, instrs: list[ScheduleItem]) -> tuple[bool, list[ScheduleItem]]:
        # This method flattens side-by-side memory management instructions that end up covering
        # a single contiguous memory region. This is useful for reducing the number of memory
        # management instructions that are executed.
        changes = False
        new_instrs: list[ScheduleItem] = []
        skip_indices = set()
        i = 0
        while i < len(instrs):
            if i in skip_indices:
                i += 1
                continue
            instr = instrs[i]
            if isinstance(instr, MemManInstr):
                # Collect subsequent MemManInstrs to try to merge
                same_type_instrs = [instr]
                skip_indices.add(i)
                j = i + 1
                while j < len(instrs):
                    if j in skip_indices:
                        j += 1
                        continue
                    instr_j = instrs[j]
                    if (
                        isinstance(instr_j, instr.__class__)
                        and instr_j.tensor_id == instr.tensor_id
                    ):
                        # Collect it
                        same_type_instrs.append(instr_j)
                        skip_indices.add(j)
                        j += 1
                    elif (
                        isinstance(instr_j, MemManInstr)
                        and instr_j.tensor_id == instr.tensor_id
                        and instr_j.__class__ != instr.__class__
                    ):
                        # Stop collecting
                        break
                    elif not isinstance(instr_j, MemManInstr):
                        # Stop collecting
                        break
                    else:
                        # MemManInstr with different tensor_id, skip over it
                        j += 1
                num_index_components = len(instr.index)
                if len(same_type_instrs) > 1:
                    expr_list = []
                    compatible = True
                    for idx in range(num_index_components):
                        # Collect the idx-th index from each instruction
                        indices = [instr2.index[idx] for instr2 in same_type_instrs]
                        if all(x.struct_eq(indices[0]) for x in indices):
                            # All indices are the same, no need to join
                            expr_list.append(indices[0])
                        else:
                            # Try to join all indices in 'indices'
                            joined = isl_utils.try_join_atoms(tuple(indices))
                            if joined is not None:
                                expr_list.append(joined)
                            else:
                                compatible = False
                                break
                    if compatible:
                        # Create a new instruction with the joined indices
                        joined_expr = ie.IndexSequence(tuple(expr_list))
                        new_instrs.append(instr.__class__(instr.tensor_id, joined_expr))
                        changes = True
                    else:
                        # Cannot merge, add the original instruction
                        new_instrs.append(instr)
                else:
                    # Only one instr, add the original instruction
                    new_instrs.append(instr)
            else:
                new_instrs.append(instr)
            i += 1
        return changes, new_instrs

    def _flatten_side_by_side2(self, instrs: list[ScheduleItem]) -> tuple[bool, list[ScheduleItem]]:
        # This method flattens side-by-side memory management instructions that end up covering
        # a single contiguous memory region. This is useful for reducing the number of memory
        # management instructions that are executed.
        changes = False
        new_instrs: list[ScheduleItem] = []
        i = 0
        while i < len(instrs):
            instr = instrs[i]
            if isinstance(instr, MemManInstr):
                # Collect subsequent MemManInstrs of the same type
                same_type_instrs = [instr]
                j = i + 1
                jth_instr = instrs[j] if j < len(instrs) else None
                while (
                    j < len(instrs)
                    and isinstance(jth_instr, MemManInstr)
                    and isinstance(jth_instr, instr.__class__)
                    and jth_instr.tensor_id == instr.tensor_id  # type: ignore
                ):
                    same_type_instrs.append(jth_instr)
                    j += 1

                num_index_components = len(instr.index)
                if len(same_type_instrs) > 1:
                    expr_list = []
                    compatible = True
                    for idx in range(num_index_components):
                        # Collect the idx-th index from each instruction
                        indices = [instr2.index[idx] for instr2 in same_type_instrs]
                        if all(x.struct_eq(indices[0]) for x in indices):
                            # All indices are the same, no need to join
                            expr_list.append(indices[0])
                        else:
                            # Try to join all indices in 'indices'
                            joined = isl_utils.try_join_atoms(tuple(indices))
                            if joined is not None:
                                expr_list.append(joined)
                            else:
                                compatible = False
                                break
                    if compatible:
                        # Create a new instruction with the joined indices
                        joined_expr = ie.IndexSequence(tuple(expr_list))
                        new_instrs.append(instr.__class__(instr.tensor_id, joined_expr))
                        changes = True
                        i = j  # Skip over the processed instructions
                        continue
                    else:
                        # Cannot merge, add the original instruction
                        new_instrs.append(instr)
                        i += 1
                else:
                    # Only one instr, add the original instruction
                    new_instrs.append(instr)
                    i += 1
            else:
                new_instrs.append(instr)
                i += 1
        return changes, new_instrs

    def _visit_block(self, node: isl.AstNode) -> ScheduleItem | None:
        is_first_block = not self.visited_first_block

        if not self.visited_first_block:
            self.visited_first_block = True

        node_list = node.block_get_children()
        n_children = node_list.n_ast_node()
        children = [node_list.get_at(j) for j in range(n_children)]

        sched_items = [self._node_type_dispatcher(child_node) for child_node in children]
        sched_items = [x for x in sched_items if x is not None]
        if len(sched_items) == 0:
            return None
        sched_items = self._flatten_sched_items(sched_items)

        # sched_items = self._flatten_side_by_side(sched_items)

        changes = True
        while changes and self.exec_cfg.enable_ast_promo:
            changes = False
            red_changes, sched_items = self._intra_block_redundant_swap_removals(sched_items)
            changes |= red_changes

            in_loop_red_changes, sched_items = self._in_loop_redundant_swap_removals(sched_items)
            changes |= in_loop_red_changes

            flat_changes, sched_items = self._flatten_side_by_side(sched_items)
            changes |= flat_changes

        if is_first_block:
            # Remove every memman operation after the last non-memman operation
            for j in range(len(sched_items) - 1, -1, -1):
                if any(isinstance(x, ExecInstruction) for x in sched_items[j].flat_recursive_tree):
                    break
                else:
                    if isinstance(sched_items[j], (OffloadInstruction, FetchInstruction)):
                        self.removed_swap_ops += 1
                    sched_items.pop(j)

        # TODO: Move all fetches to start of block if possible. This may not be possible if they
        # are created by a previous (nested) operation in the block. But we could try to check
        # the recursive flat tree of all instrs behind and if not present, move it.

        if len(sched_items) == 1:
            return sched_items[0]

        if self.next_is_parallel:
            self.next_is_parallel = False
            main_thread_only = []
            distributable = []
            for i in sched_items:
                if self._is_distributable(i):
                    distributable.append(i)
                else:
                    main_thread_only.append(i)

            return ParallelBlock(sched_items, distributable, main_thread_only)
        else:
            return SequentialBlock(sched_items)

    def _flatten_side_by_side_parallel(  # noqa: C901
        self, sched_items: Sequence[ScheduleItem]
    ) -> Sequence[ScheduleItem]:
        new_sched_items: list[ScheduleItem] = []
        i = 0
        while i < len(sched_items):
            item = sched_items[i]
            next_item = sched_items[i + 1] if i + 1 < len(sched_items) else None
            if isinstance(item, ParallelBlock) and isinstance(next_item, ParallelBlock):
                # Flatten the parallel block
                all_item_mem_man = all(
                    isinstance(m, (ParallelBlock, DeallocInstruction))
                    for m in item.flat_recursive_tree
                )
                all_next_item_mem_man = all(
                    isinstance(m, (ParallelBlock, DeallocInstruction))
                    for m in next_item.flat_recursive_tree
                )
                if all_item_mem_man and all_next_item_mem_man:
                    new_sched_items.append(
                        ParallelBlock(
                            list(item.inner_block) + list(next_item.inner_block),
                            list(item.distributable) + list(next_item.distributable),
                            list(item.main_thread_only) + list(next_item.main_thread_only),
                        )
                    )
                    i += 2
                else:
                    new_sched_items.append(item)
                    i += 1
            else:
                new_sched_items.append(item)
                i += 1
        return new_sched_items

    def _flatten_sched_items(  # noqa: C901
        self, sched_items: list[ScheduleItem]
    ) -> list[ScheduleItem]:
        new_sched_items: list[ScheduleItem] = []
        if self.next_is_parallel:
            # flatten any child parallel blocks
            for item in sched_items:
                if isinstance(item, ParallelBlock):
                    new_sched_items.extend(item.inner_block)
                else:
                    new_sched_items.append(item)
        else:
            # flatten any child sequence blocks
            for item in sched_items:
                if isinstance(item, SequentialBlock):
                    new_sched_items.extend(item.inner_block)
                else:
                    new_sched_items.append(item)
        return new_sched_items

    def _visit_user(self, node: isl.AstNode) -> ScheduleItem | None:  # noqa: C901
        expr = node.user_get_expr()
        if expr.get_op_type() == isl.ast_expr_op_type.call:
            stmt_name = str(expr.get_op_arg(0).to_C_str())
            tokens = stmt_name.split("_")

            stmt_type = tokens[0]
            op_id = OpId(int(tokens[1]))

            op = self.dg.ops_by_id[op_id].op
            if isinstance(op, top.EvalSymbolOp) and self.exec_cfg.enable_symbol_prealloc_store:
                return None
            op_domain_vars = op.domain.variables
            num_args = expr.get_op_n_arg()

            from tempo.core.global_objects import get_active_dg

            active_dg = get_active_dg()
            capacity = len(active_dg.universe.variables)
            domain_mapping = SymbolDict(capacity * 2)
            domain_mapping.load_keys(list(op_domain_vars))

            domain_list: list[ie.IndexValue] = []
            for i in range(num_args - 1):
                arg = expr.get_op_arg(i + 1)  # NOTE: first arg is id
                arg_expr = isl_utils.isl_expr_to_tempo_expr(arg)
                if arg_expr is None:
                    raise ValueError(f"Unsupported expression type {arg.get_op_type()}: {str(arg)}")

                # TODO if we ever have mem man instructions with more domain than the producer
                #      we will need to handle this. The domain of said mem man instruction will
                #      not be able to just be the domain of the producer.
                # if i >= len(op_domain_vars):
                #    print(
                #        f"WARNING: {stmt_name} has more args than
                #        domain vars: {len(op_domain_vars)}. Skipping arg {i}={arg_expr}."
                #    )
                #    break
                try:
                    op_domain_var = op_domain_vars[i]
                except Exception as e:
                    print(
                        f"Error: getting idx={i} of {op_domain_vars} failed. \
                          Arg expr: {arg_expr}. Op: {op}, domain: {op.domain}, \
                          domain_list: {domain_list}, stmt_type: {stmt_type}, \
                          num_args: {num_args}, \
                          len(op_domain_vars): {len(op_domain_vars)}, \
                          len(domain_list): {len(domain_list)}"
                    )
                    raise e
                try:
                    domain_mapping[op_domain_var] = arg_expr  # type: ignore
                except Exception as e:
                    print(f"Error: setting {op_domain_var}, idx={op_domain_var.idx} to {arg_expr}")
                    raise e
                domain_list.append(arg_expr)
            index_seq = ie.IndexSequence(tuple(domain_list))

            if stmt_type == StmtType.EXECUTE.value:
                return ExecInstruction(op_id, domain_mapping)  # type: ignore
            if stmt_type == StmtType.DEALLOCATE.value:
                op_out_id = OpOutId(int(tokens[2]))
                self.seen_mem_man_ops += 1
                return DeallocInstruction(TensorId(op_id, op_out_id), index_seq)
            if stmt_type == StmtType.OFFLOAD.value:
                self.seen_mem_man_ops += 1
                self.seen_swap_ops += 1
                op_out_id = OpOutId(int(tokens[2]))
                return OffloadInstruction(TensorId(op_id, op_out_id), index_seq)
            if stmt_type == StmtType.FETCH.value:
                self.seen_mem_man_ops += 1
                self.seen_swap_ops += 1
                op_out_id = OpOutId(int(tokens[2]))
                return FetchInstruction(TensorId(op_id, op_out_id), index_seq)
            raise ValueError(f"Unsupported statement type {stmt_type}")

        else:
            raise ValueError(f"Unsupported expression type {expr.get_op_type()}")
        # return None, None, None

    def _visit_error(self, node: isl.AstNode) -> ScheduleItem:
        raise NotImplementedError


def get_ast_builder(ctx: isl.Context) -> isl.AstBuild:
    ast_build = isl.AstBuild.alloc(ctx)
    # NOTE here, we may want to set callbacks, like before_each_for and such
    return ast_build


def build_isl_ast_from_schedule(ast_build: isl.AstBuild, sched: isl.Schedule) -> isl.AstNode:
    root = ast_build.node_from_schedule(sched)
    return root


def build_isl_ast_from_schedule_map(ast_build: isl.AstBuild, sched: isl.UnionMap) -> isl.AstNode:
    root = ast_build.node_from_schedule_map(sched)
    return root


def isl_ast_to_tempo_schedule(
    comp_ctx: CompilationCtx, root: isl.AstNode, quiet: bool = False
) -> ExecutionSchedule:
    walker = IslASTToTempoScheduleWalker(comp_ctx, quiet)
    sched = walker.start_walk(root)
    return sched
