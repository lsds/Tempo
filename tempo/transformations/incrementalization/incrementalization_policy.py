# type: ignore
# TODO: remove this
from functools import partial
from math import ceil, floor
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.datatypes import OpInId
from tempo.core.dependence_graph import PDG, DependencyData
from tempo.core.dim_utils import normalize_negative_dim
from tempo.core.utils import bytes_to_human_readable
from tempo.transformations.compilation_pass import CompilationCtx
from tempo.transformations.incrementalization.incrementalization_common import (
    ALLOWED_START_OPS,
    IncKind,
    IncRoundCtx,
    IncStartOp,
    _can_incrementalize_base,
    create_inc_symbol_and_block_idxs,
)
from tempo.utils import logger
from tempo.utils.dg_utils import (
    is_matmul_batch_dim,
    is_matmul_contracting_dim,
    is_window_access,
    recursively_follow_op_in_dim_through_dependencies,
)
from tempo.utils.memory_estimator import MemoryEstimator

log = logger.get_logger(__name__)


def closest_divisor(x: int, y: float) -> int:
    # Start by looking for a larger divisor
    candidate = ceil(y)
    while candidate < x:
        if x % candidate == 0:
            return candidate
        candidate += 1

    # If no larger divisor is found, look for a smaller one
    candidate = floor(y)
    while candidate > 0:
        if x % candidate == 0:
            return candidate
        candidate -= 1

    return 1


def get_divisor_at_percentile(n: int, percentile: int = 75) -> int:
    """Get the divisor at the given percentile of the sorted divisors of n.
    NOTE: If used as a block size, larger percentiles will result in larger block sizes,
    meaning fewer block iterations.

    """
    divisors = sorted({d for i in range(1, int(n**0.5) + 1) if n % i == 0 for d in (i, n // i)})

    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")

    if not divisors:
        return 1

    # Calculate index based on percentile
    index = (len(divisors) - 1) * percentile // 100
    return divisors[int(round(index))]


def _needs_incrementalize(
    op: top.TensorOp,
    comp_ctx: CompilationCtx,
    initial: bool = False,
    mem_est: Optional[MemoryEstimator] = None,
    start_ops: Optional[Sequence[top.TensorOp]] = None,
) -> bool:
    if mem_est is None:
        mem_est = MemoryEstimator(comp_ctx)

    exec_cfg = comp_ctx.exec_cfg

    max_memory_allowed = exec_cfg.incrementalize_mem_threshold

    # if start_ops is not None and len(start_ops) > 0:
    #    print(f"Received Start ops: {start_ops}")
    #    if len(start_ops) == 1:
    #        import networkx as nx

    #        G = comp_ctx.dg._G
    #        distance = nx.shortest_path_length(G, source=op, target=start_ops[0])
    #        print(f"Distance from {op} to {start_ops[0]}: {distance}")

    #        if distance > 5:
    #            return False

    op_memory = mem_est.estimate_op_size_bytes(op.op_id)
    assert op_memory is not None, f"Op {op} has no memory estimate"
    op_large: bool = op_memory > max_memory_allowed // 2
    # if op_large:
    #    print(f"Op {op} is too large: {op_memory} > {max_memory_allowed}")

    return op_large


def dimension_originates_from_temporal_indexing(
    dg: PDG,
    op: top.TensorOp,
    in_id: OpInId,
    in_dim: int,
    in_dim_size: int,
    max_depth_to_check: int = 50,
) -> Tuple[bool, Optional[ie.Symbol]]:
    """Check if a dimension originates from temporal indexing.

    A dimension originates from temporal indexing if it was created through symbolic indexing.
    This is detected when in_dim < num_slices in the dependency traversal.

    Args:
        dg: The dependence graph
        op: The operation to start from
        in_id: The input ID for the operation
        in_dim: The dimension to check
        max_depth_to_check: Maximum depth to traverse in the dependency graph

    Returns:
        True if the dimension originates from temporal indexing, False otherwise
    """
    has_temporal_indexing = False
    temporal_indexing_var = None

    def should_recurr(
        dg: PDG,
        op: top.TensorOp,
        depy_data: DependencyData,
        depy_op: top.TensorOp,
        op_out_dim: int,
        depy_out_dim: int,
        ctx: int,
    ) -> bool:
        nonlocal has_temporal_indexing
        if has_temporal_indexing:
            return False  # We are done already

        # Check if we've exceeded the maximum depth
        if ctx >= max_depth_to_check:
            return False

        # Continue recursing until we hit a temporal indexing op or the dimension is eliminated
        return True

    def effect(
        dg: PDG,
        op: top.TensorOp,
        op_out_dim: int,
        op_in_dims: Dict[OpInId, int],
        ctx: int,
    ) -> Tuple[PDG, Any]:
        nonlocal has_temporal_indexing
        nonlocal temporal_indexing_var
        ctx += 1

        # Check if this operation creates temporal indexing by examining its dependencies
        for depy_op, depy_data in dg.get_flat_direct_dependencies(op):
            dim_eliminated = depy_data.sink_in_idx not in op_in_dims
            if not dim_eliminated:
                in_dim = op_in_dims[depy_data.sink_in_idx]
                num_slices = sum(not m.is_point() for m in depy_data.expr.members)

                # If in_dim < num_slices, then the dim of interest was just created
                # by symbolic indexing (temporal indexing)
                dim_created_by_symb_index = in_dim < num_slices and num_slices > 0

                if dim_created_by_symb_index:
                    slices_count = 0

                    for v, e in zip(depy_op.domain.variables, depy_data.expr.members, strict=True):
                        if not e.is_point():
                            if slices_count == in_dim:
                                temporal_indexing_var = v
                                break
                            slices_count += 1

                    has_temporal_indexing = True
                    break
        return dg, ctx

    try:
        # NOTE: Because recursively_follow_op_in_dim_through_dependencies does not invoke effect
        # on op itself, we need to do it first.
        # TODO: Update recursively_follow_op_in_dim_through_dependencies to
        #     invoke effect on op itself.
        # NOTE: None is because the dim is presumabely eliminated.
        effect(dg, op, None, {in_id: in_dim}, 0)  # type: ignore
        if has_temporal_indexing:
            assert temporal_indexing_var is not None
            return True, temporal_indexing_var
        recursively_follow_op_in_dim_through_dependencies(
            dg, op, in_id, in_dim, should_recurr, effect, 1
        )
    except Exception:
        # NOTE: If we hit a reshape, that throws an exception.
        # TODO: Resolve this by breaking down reshapes into merge and split dims ops.
        return False, None

    return has_temporal_indexing, temporal_indexing_var


class IncrementalizationPolicy:
    """An incrementalization policy determines the incrementalization parameters for a given
    compilation context.

    This involves choices such as:
    - Which ops to incrementalize
    - Which spatial dimensions of those ops to incrementalize on
    - How to compute the block/tile size
    - How to compute the number of blocks/tiles
    - How to derive the block/tile symbol and block idxs
    """

    def __init__(self):
        pass

    def get_round_info(
        self,
        ctx: CompilationCtx,
        inc_round: int,
    ) -> Optional[IncRoundCtx]:
        """Get the incrementalization round context based on the policy."""
        raise NotImplementedError("Subclasses must implement get_round_info")


# TODO: This file is a mess, due to last minute rush. Come back and clean up.
class PreferTemporalDimsFirstBatchSecond(IncrementalizationPolicy):
    """Default incrementalization policy implementation.

    This policy prefers to incrementalize on dimensions that originate from temporal indexing.
    This is good because it allows for better downstream scheduling.
    """

    def __init__(
        self,
        max_inc_rounds: int = 2,
    ):
        super().__init__()
        self.max_inc_rounds = max_inc_rounds

        self.mem_est: Optional[MemoryEstimator] = None
        self.ctx: Optional[CompilationCtx] = None

    def _compute_block_size_and_num_blocks(
        self,
        ub: Union[int, ie.IntIndexValue],
        inc_start_ops_with_reduced_ub: List[IncStartOp],
    ) -> Tuple[int, ie.IntIndexValueLike]:
        assert self.ctx is not None
        assert self.mem_est is not None

        block_size: int = self._determine_block_size(ub, inc_start_ops_with_reduced_ub)

        # TODO I wonder if this will work with dynamic bounds
        num_blocks: ie.IntIndexValueLike

        if isinstance(ub, int):
            num_blocks = ub // block_size
            assert ub % block_size == 0, f"dim_ub {ub} is not divisible by block_size {block_size}"
        else:
            try:
                num_blocks = (ub // block_size).evaluate(self.ctx.dg.static_bounds)
            except Exception:
                num_blocks = ub // block_size
        return block_size, num_blocks

    def _determine_block_size(
        self,
        ub: ie.IntIndexValueLike,
        inc_start_ops: List[IncStartOp],
    ) -> int:
        if self.fully_incrementalize:
            return 1

        percentile = self.ctx.exec_cfg.incrementalization_percentile
        # NOTE: 50th percentile is the middle of the divisor range.
        # This is a good compromise between larger block sizes and more blocks.
        # if DLBackendName.str_to_enum(self.ctx.exec_cfg.backend) == DLBackendName.TORCH:
        #    # NOTE: Torch uses more memory
        #    percentile -= 15

        block_size = get_divisor_at_percentile(ub, percentile=percentile)

        log.info("Using divisor at percentile %s, which is block_size=%s", percentile, block_size)

        block_size = max(1, block_size)
        assert isinstance(ub, int)
        block_size = min(block_size, ub)

        return block_size

    def _determine_block_size_from_mem_usage(
        self,
        ub: ie.IntIndexValueLike,
        inc_start_ops: List[IncStartOp],
    ) -> int:
        assert self.ctx is not None
        assert self.mem_est is not None

        # NOTE: This heuristic is not working well.
        # Disabling it and favoring original fixed percentile strategy.
        # TODO: Fix this.

        max_mem_allowed = self.ctx.exec_cfg.incrementalize_mem_threshold
        mem_used = max([self.mem_est.estimate_op_size_bytes(op.op_id) for op in inc_start_ops])
        overuse_ratio = mem_used / max_mem_allowed

        # Map overuse_ratio to appropriate percentile
        # When overuse_ratio is close to 1, use high percentile for larger block sizes
        # When overuse_ratio is higher than 1, use low percentile for smaller block sizes
        if overuse_ratio <= 1.0:
            # When memory usage is within limits, prefer larger block sizes (high percentile)
            percentile = int(75 + (1.0 - overuse_ratio) * 25)  # Maps [0,1] to [100,75]
        else:
            # When memory usage exceeds limits, prefer smaller block sizes (low percentile)
            # Cap the overuse_ratio to prevent extreme values
            capped_overuse = min(overuse_ratio, 10.0)  # Cap at 10x overuse
            percentile = int(75 * (1.0 / capped_overuse))  # Maps [1,10] to [75,7.5]
            percentile = max(25, percentile)  # Ensure minimum percentile of 25

        mem_used_gb = bytes_to_human_readable(mem_used)
        max_mem_allowed_gb = bytes_to_human_readable(max_mem_allowed)
        log.info(
            "Picking percentile: Mem used: %s, Max op mem: %s, overuse_ratio: %s, percentile: %s",
            mem_used_gb,
            max_mem_allowed_gb,
            overuse_ratio,
            percentile,
        )
        block_size = get_divisor_at_percentile(ub, percentile=percentile)

        block_size = max(1, block_size)
        assert isinstance(ub, int)
        block_size = min(block_size, ub)
        return block_size

    def pick_inc_start_dims(
        self, inc_start_ops: Sequence[IncStartOp], inc_round: int
    ) -> Tuple[Optional[int], List[Tuple[IncStartOp, int, ie.IntIndexValueLike]]]:
        assert self.ctx is not None
        dg = self.ctx.dg

        # NOTE: 2.0 Finds all dims contracted by matmul ops and reduced by reduce ops
        potential_reduce_dim_sizes: Set[Tuple[IncStartOp, int, ie.IntIndexValueLike]] = set()
        for op in inc_start_ops:
            s = dg.get_input_shape(op, OpInId(0))
            if isinstance(op, top.MatMulOp):
                last_dim = normalize_negative_dim(-1, s)
                contracting_dim_size = s._shape[last_dim]
                potential_reduce_dim_sizes.add((op, last_dim, contracting_dim_size))
            else:
                for d in op.dims:
                    reduced_dim_size = s._shape[d]
                    potential_reduce_dim_sizes.add((op, d, reduced_dim_size))

        # NOTE: 2.5 Pick a specific dimension (by size) to incrementalize on
        potential_reduce_dim_sizes = [
            s for s in potential_reduce_dim_sizes if isinstance(s[2], int) and s[2] > 1
        ]

        if inc_round == 0:
            preferred_dims = []
            for op, d, size in potential_reduce_dim_sizes:
                created_by_temporal, t_var = dimension_originates_from_temporal_indexing(
                    dg, op, OpInId(0), d, size
                )
                if created_by_temporal:
                    preferred_dims.append((op, d, size, t_var))

            if preferred_dims:
                max_tup = max(preferred_dims, key=lambda x: ie.evaluate_int(x[2], dg.static_bounds))
                picked_var = max_tup[3]
                ub = ie.evaluate_int(max_tup[2], dg.static_bounds)
                self.inc_is_temporal = True
                self.temporal_indexing_var = picked_var
                preferred_dims = [(op, d, size) for op, d, size, _ in preferred_dims if size == ub]
                return ub, preferred_dims

        # Otherwise, pick the batch dim
        if any(s[2] == self.batch_dim_size for s in potential_reduce_dim_sizes):
            preferred_dims = [
                (op, d, size)
                for op, d, size in potential_reduce_dim_sizes
                if size == self.batch_dim_size
            ]
            return self.batch_dim_size, preferred_dims

        # Pick largest
        if potential_reduce_dim_sizes:
            ub = (
                max([ie.evaluate_int(d[2], dg.static_bounds) for d in potential_reduce_dim_sizes])
                if potential_reduce_dim_sizes
                else None
            )
            preferred_dims = [
                (op, d, size) for op, d, size in potential_reduce_dim_sizes if size == ub
            ]
            return ub, preferred_dims

        return None, []

    def _search_dependents(self, dg: PDG, op: top.TensorOp, max_depth: int) -> Set[top.TensorOp]:
        to_explore = {op}
        for _ in range(max_depth):
            if not to_explore:
                return set()
            op_to_explore = to_explore.pop()
            for dep, _ in dg.get_flat_direct_dependents(op_to_explore):
                if isinstance(dep, top.SinkOp):
                    return set()
                if _can_incrementalize_base(dg, dep) and isinstance(dep, ALLOWED_START_OPS):
                    return {dep}

        return set()

    def _get_start_ops(
        self, dg: PDG, ops_needing_inc: Sequence[top.TensorOp]
    ) -> Sequence[top.TensorOp]:
        can_start_inc = {
            op
            for op in ops_needing_inc
            if _can_incrementalize_base(dg, op) and isinstance(op, ALLOWED_START_OPS)
        }
        expanded_ops_needing_inc = set(ops_needing_inc) - can_start_inc

        for op in expanded_ops_needing_inc:
            new_can_start_inc = self._search_dependents(dg, op, max_depth=10)
            can_start_inc.update(new_can_start_inc)

        return list(can_start_inc)

    def _get_valid_start_ops(self, inc_round: int) -> Tuple[Optional[int], Sequence[IncStartOp]]:
        assert self.ctx is not None
        assert self.mem_est is not None

        dg = self.ctx.dg

        ops_needing_inc = [
            op
            for op in dg.nodes
            if _needs_incrementalize(op, self.ctx, initial=True, mem_est=self.mem_est)
        ]

        # NOTE: 1. Find ops needing incrementalization
        inc_start_ops = self._get_start_ops(dg, ops_needing_inc)

        if len(inc_start_ops) == 0:
            return None, []

        # NOTE: 2. Pick spatial dimension to incrementalize on
        picked_dim_size, ops_and_input_0_spatial_dim_and_sizes = self.pick_inc_start_dims(
            inc_start_ops, inc_round
        )
        if picked_dim_size is None:
            return None, []

        # NOTE: add all others with 0:T access
        if self.fully_incrementalize:
            inc_candidates = set(inc_start_ops)

            var = self.temporal_indexing_var
            for snk, src, edge_data in dg.get_all_edges():
                if (
                    any(
                        e.partial_eval(dg.static_bounds).struct_eq(
                            ie.slice_(0, var.as_bound()).partial_eval(dg.static_bounds)
                        )
                        for e in edge_data.expr.members
                    )
                    and self.mem_est.estimate_tensor_point_size_bytes(
                        src.op_id, edge_data.src_out_idx
                    )
                    > 1 * 2**20  # 1MB
                ):
                    inc_candidates.add(snk)

            inc_start_ops = self._get_start_ops(dg, list(inc_candidates))
            picked_dim_size, ops_and_input_0_spatial_dim_and_sizes = self.pick_inc_start_dims(
                inc_start_ops, inc_round
            )

        # NOTE: 3. Filter ops to only those that have the chosen dimension
        chosen_inc_start_ops: List[IncStartOp] = []
        for op, _, size in ops_and_input_0_spatial_dim_and_sizes:
            if size == picked_dim_size:
                # TODO: should be appending (op, d, in_id) to simplify the code below
                chosen_inc_start_ops.append(op)

        if len(chosen_inc_start_ops) == 0:
            return None, []

        return picked_dim_size, chosen_inc_start_ops

    def get_round_info(  # noqa: C901
        self,
        ctx: CompilationCtx,
        inc_round: int,
    ) -> Optional[IncRoundCtx]:
        self.ctx = ctx
        self.mem_est = MemoryEstimator(ctx)
        self.fully_incrementalize = False
        self.inc_is_temporal = False
        self.temporal_indexing_var = None

        dg = ctx.dg

        # Batch is the temporal dim not in the domain of any op
        options = set(dg.universe.variables)
        for op in dg.nodes:
            for d in op.domain.variables:
                if d in options:
                    options.remove(d)
            if len(options) == 1:
                break
        assert len(options) == 1
        self.batch_dim_size = dg.static_bounds[options.pop().as_bound()]

        if inc_round >= self.max_inc_rounds:
            return None

        any_window_access_in_dg = any(
            is_window_access(m) for _, _, e in dg.get_all_edges() for m in e.expr.members
        )
        if any_window_access_in_dg and inc_round == 0:
            log.info("Detected window access patterns, fully incrementalizing.")
            self.fully_incrementalize = True

        ub, inc_start_ops_with_reduced_ub = self._get_valid_start_ops(inc_round)
        if ub is None:
            return None

        # NOTE: 4. Compute block size and number of blocks
        block_size, num_blocks = self._compute_block_size_and_num_blocks(
            ub, inc_start_ops_with_reduced_ub
        )

        # NOTE: 6. Create the start op and input side dims mapping
        # NOTE: If many dims have same size, we always pick the first one, which is arbitrary
        def get_start_op_input_and_dim(op: top.TensorOp) -> Sequence[Tuple[OpInId, int]]:
            shape0 = dg.get_input_shape(op, OpInId(0))
            if isinstance(op, top.MatMulOp):
                # matmul, contracting dim -> both sides inc
                # matmul, non-contracting and batch dim -> both sides inc
                # matmul, non-contracting and non-batch dim -> one side inc only
                for d in range(len(shape0)):
                    if shape0._shape[d] == ub:
                        if is_matmul_contracting_dim(dg, op, OpInId(0), d):
                            return [(OpInId(0), d), (OpInId(1), d - 1)]
                        elif is_matmul_batch_dim(dg, op, OpInId(0), d):
                            return [(OpInId(0), d), (OpInId(1), d)]
                        else:  # non-contracting and non-batch dim case for 0'th input
                            return [(OpInId(0), d)]

                # NOTE: If we get here,
                # then we are in a non-contracting, non-batch dim case for 1st input
                shape1 = dg.get_input_shape(op, OpInId(1))
                for d in range(len(shape1)):
                    if shape1._shape[d] == ub:
                        return [(OpInId(1), d)]

            elif isinstance(op, (top.SumOp, top.MaxOp)):
                # sum, max, etc -> just the input and the dim over which we are summing/maxing
                for d in op.dims:
                    if shape0._shape[d] == ub:
                        return [(OpInId(0), d)]
            raise ValueError(f"Failed to find inc dim for start op: {op}")

        start_op_and_input_side_dims = {
            op: get_start_op_input_and_dim(op) for op in inc_start_ops_with_reduced_ub
        }

        if self.inc_is_temporal and self.fully_incrementalize:
            # TODO: need to expand to all ops that do this temporal indexing.
            inc_var = self.temporal_indexing_var
            block_idxs = None  # create_block_idxs(1, inc_var)
        else:
            # NOTE: 5. Create the incrementalization symbol and block idxs
            inc_var, block_idxs = create_inc_symbol_and_block_idxs(
                dg, inc_round, block_size, num_blocks, allow_reuse_symbol=False
            )

        needs_inc_fn = partial(_needs_incrementalize, mem_est=self.mem_est)

        # For full incrementalization, we don't want to stop after memory usage goes down.
        if self.fully_incrementalize:
            needs_inc_fn = lambda op, comp_ctx, is_initial: True

        return IncRoundCtx(
            IncKind.MEMORY,
            set(inc_start_ops_with_reduced_ub),
            start_op_and_input_side_dims,
            inc_var,
            ub,
            block_size,
            block_idxs,
            ie.lift_to_int_ie(num_blocks),
            ctx,
            needs_incrementalization=needs_inc_fn,
            finalize_incremental=True,  # self.fully_incrementalize,
            max_depth=None if self.fully_incrementalize else 8,
        )


class IncTemporalOnce(IncrementalizationPolicy):
    """Simplified incrementalization policy that only does temporal fully incrementalized case.

    This policy only runs once (inc_round == 0) and only if temporal indexing is detected.
    It always fully incrementalizes when temporal indexing is found.
    """

    def __init__(self):
        super().__init__()
        self.mem_est: Optional[MemoryEstimator] = None
        self.ctx: Optional[CompilationCtx] = None

    def _get_start_ops(
        self, dg: PDG, ops_needing_inc: Sequence[top.TensorOp]
    ) -> Sequence[top.TensorOp]:
        """Find ops that can start incrementalization."""
        can_start_inc = {
            op
            for op in ops_needing_inc
            if _can_incrementalize_base(dg, op) and isinstance(op, ALLOWED_START_OPS)
        }
        expanded_ops_needing_inc = set(ops_needing_inc) - can_start_inc

        for op in expanded_ops_needing_inc:
            new_can_start_inc = self._search_dependents(dg, op, max_depth=10)
            can_start_inc.update(new_can_start_inc)

        return list(can_start_inc)

    def _search_dependents(self, dg: PDG, op: top.TensorOp, max_depth: int) -> Set[top.TensorOp]:
        """Search for dependent ops that can start incrementalization."""
        to_explore = {op}
        for _ in range(max_depth):
            if not to_explore:
                return set()
            op_to_explore = to_explore.pop()
            for dep, _ in dg.get_flat_direct_dependents(op_to_explore):
                if isinstance(dep, top.SinkOp):
                    return set()
                if _can_incrementalize_base(dg, dep) and isinstance(dep, ALLOWED_START_OPS):
                    return {dep}
        return set()

    def _find_temporal_dims(
        self, dg: PDG, inc_start_ops: Sequence[top.TensorOp]
    ) -> Tuple[Optional[int], Optional[ie.Symbol], List[top.TensorOp]]:
        """Find dimensions that originate from temporal indexing."""
        # Find all potential reduce dimensions
        potential_reduce_dim_sizes: Set[Tuple[top.TensorOp, int, ie.IntIndexValueLike]] = set()
        for op in inc_start_ops:
            s = dg.get_input_shape(op, OpInId(0))
            if isinstance(op, top.MatMulOp):
                last_dim = normalize_negative_dim(-1, s)
                contracting_dim_size = s._shape[last_dim]
                potential_reduce_dim_sizes.add((op, last_dim, contracting_dim_size))
            else:
                for d in op.dims:
                    reduced_dim_size = s._shape[d]
                    potential_reduce_dim_sizes.add((op, d, reduced_dim_size))

        # Filter to dimensions > 1
        potential_reduce_dim_sizes = [
            s for s in potential_reduce_dim_sizes if isinstance(s[2], int) and s[2] > 1
        ]

        # Find temporal dimensions
        temporal_dims = []
        for op, d, size in potential_reduce_dim_sizes:
            created_by_temporal, t_var = dimension_originates_from_temporal_indexing(
                dg, op, OpInId(0), d, size
            )
            if created_by_temporal:
                temporal_dims.append((op, d, size, t_var))

        if not temporal_dims:
            return None, None, []

        # Pick the largest temporal dimension
        max_tup = max(temporal_dims, key=lambda x: ie.evaluate_int(x[2], dg.static_bounds))
        picked_var = max_tup[3]
        ub = ie.evaluate_int(max_tup[2], dg.static_bounds)

        # Get all ops with this dimension size
        chosen_ops = [op for op, d, size, _ in temporal_dims if size == ub]

        return ub, picked_var, chosen_ops

    def _expand_to_temporal_access_ops(
        self, dg: PDG, inc_start_ops: List[top.TensorOp], temporal_var: ie.Symbol
    ) -> List[top.TensorOp]:
        """Expand to include all ops that access the temporal dimension."""
        inc_candidates = set(inc_start_ops)

        for snk, src, edge_data in dg.get_all_edges():
            if (
                any(
                    e.partial_eval(dg.static_bounds).struct_eq(
                        ie.slice_(0, temporal_var.as_bound()).partial_eval(dg.static_bounds)
                    )
                    for e in edge_data.expr.members
                )
                and self.mem_est.estimate_tensor_point_size_bytes(src.op_id, edge_data.src_out_idx)
                > 1 * 2**20  # 1MB
            ):
                inc_candidates.add(snk)

        return self._get_start_ops(dg, list(inc_candidates))

    def _get_start_op_input_and_dim(
        self, op: top.TensorOp, ub: int
    ) -> Sequence[Tuple[OpInId, int]]:
        """Get input dimensions for a start op."""
        dg = self.ctx.dg
        shape0 = dg.get_input_shape(op, OpInId(0))

        if isinstance(op, top.MatMulOp):
            # matmul, contracting dim -> both sides inc
            # matmul, non-contracting and batch dim -> both sides inc
            # matmul, non-contracting and non-batch dim -> one side inc only
            for d in range(len(shape0)):
                if shape0._shape[d] == ub:
                    if is_matmul_contracting_dim(dg, op, OpInId(0), d):
                        return [(OpInId(0), d), (OpInId(1), d - 1)]
                    elif is_matmul_batch_dim(dg, op, OpInId(0), d):
                        return [(OpInId(0), d), (OpInId(1), d)]
                    else:  # non-contracting and non-batch dim case for 0'th input
                        return [(OpInId(0), d)]

            # NOTE: If we get here, then we are in a non-contracting,
            #  non-batch dim case for 1st input
            shape1 = dg.get_input_shape(op, OpInId(1))
            for d in range(len(shape1)):
                if shape1._shape[d] == ub:
                    return [(OpInId(1), d)]

        elif isinstance(op, (top.SumOp, top.MaxOp)):
            # sum, max, etc -> just the input and the dim over which we are summing/maxing
            for d in op.dims:
                if shape0._shape[d] == ub:
                    return [(OpInId(0), d)]

        raise ValueError(f"Failed to find inc dim for start op: {op}")

    def get_round_info(
        self,
        ctx: CompilationCtx,
        inc_round: int,
    ) -> Optional[IncRoundCtx]:
        """Get incrementalization round info - only runs once for temporal indexing."""
        if inc_round != 0:
            return None

        self.ctx = ctx
        self.mem_est = MemoryEstimator(ctx)
        dg = ctx.dg
        # self.any_window_access_in_dg = any(
        #    is_window_access(m) for _, _, e in dg.get_all_edges() for m in e.expr.members
        # )

        self.any_window_access_in_dg = ctx.analysis_ctx._is_incremental_algo

        # Find ops needing incrementalization
        ops_over_memory_threshold = [
            op
            for op in dg.nodes
            if _needs_incrementalize(op, ctx, initial=True, mem_est=self.mem_est)
        ]
        if not ops_over_memory_threshold:
            return None

        # Get start ops
        inc_start_ops = self._get_start_ops(dg, ops_over_memory_threshold)
        if not inc_start_ops:
            return None

        # Find temporal dimensions
        ub, temporal_var, chosen_ops = self._find_temporal_dims(dg, inc_start_ops)
        if ub is None or temporal_var is None:
            return None

        # Expand to include all temporal access ops
        final_inc_start_ops = self._expand_to_temporal_access_ops(dg, chosen_ops, temporal_var)
        if not final_inc_start_ops:
            return None

        if not self.any_window_access_in_dg:
            final_inc_start_ops = [o for o in final_inc_start_ops if o in ops_over_memory_threshold]

        # Create start op and input side dims mapping
        start_op_and_input_side_dims = {
            op: self._get_start_op_input_and_dim(op, ub) for op in final_inc_start_ops
        }

        percentile = self.ctx.exec_cfg.incrementalization_percentile

        block_size = (
            1 if self.any_window_access_in_dg else get_divisor_at_percentile(ub, percentile)
        )

        num_blocks = ub // block_size
        inc_var, block_idxs = (
            (temporal_var, None)
            if block_size == 1
            else create_inc_symbol_and_block_idxs(
                dg, inc_round, block_size, num_blocks, allow_reuse_symbol=False
            )
        )

        if self.any_window_access_in_dg:
            needs_inc_fn = lambda op, comp_ctx, is_initial: True
        else:
            needs_inc_fn = partial(_needs_incrementalize, mem_est=self.mem_est)

        return IncRoundCtx(
            IncKind.MEMORY,
            set(final_inc_start_ops),
            start_op_and_input_side_dims,
            inc_var,
            ub,
            block_size,
            block_idxs,
            ie.lift_to_int_ie(num_blocks),
            ctx,
            needs_incrementalization=needs_inc_fn,
            finalize_incremental=True,
            max_depth=None,
        )
