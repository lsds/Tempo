import dataclasses
from collections.abc import Sequence

from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.dependence_graph import DependencyData
from tempo.core.symbolic_tensor import SymbolicTensor, get_symbolic_tensor_for_op_output
from tempo.transformations.optimizer.algebraic.match_replacer import MatchReplacer
from tempo.utils.logger import get_logger

log = get_logger(__name__)


# x[a:b].sum() ⟶ x[0:T].cumsum().index(b - 1) - (x[0:T].cumsum().index(a - 1) if a > 0 else 0)
# NOTE: The above pattern is general enough to handle:
# x[0:t].sum() ⟶ x[0:T].cumsum().index(t - 1)
# x[0:t+1].sum() ⟶ x[0:T].cumsum().index(t)
# x[t:T].sum() ⟶ x[0:T].cumsum().index(T - 1) - x[0:T].cumsum().index(t - 1)


"""Patterns:

    #TODO: Add the discount sum and discount cum sum patterns, if needed
    Discount sum pattern of the form:
    ```
    x = ... # some tensor with domain (..., t, ...)
    discount_sum_x = RecurrentTensor.like(x)

    discount_factor = ... # scalar discount factor (e.g., gamma < 1)

    discount_sum_x[..., 0, ...] = x[..., 0, ...]
    discount_sum_x[..., t, ...] = discount_factor * discount_sum_x[..., t-1, ...] + x[..., t, ...]

    y = f(discount_sum_x[..., T-1, ...])
    ```

    Discounted cumulative sum pattern of the form:
    ```
    x = ... # some tensor with domain (..., t, ...)
    discount_sum_x = RecurrentTensor.like(x)

    discount_factor = ... # scalar discount factor (e.g., gamma < 1)

    discount_sum_x[..., 0, ...] = x[..., 0, ...]
    discount_sum_x[..., t, ...] = discount_factor * discount_sum_x[..., t-1, ...] + x[..., t, ...]

    y = f(discount_sum_x[..., t, ...])
    """


@dataclasses.dataclass
class MergeAddPatternInfo:
    """Information extracted from a merge/add recurrence pattern."""

    accum_symbol: ie.Symbol
    idx: int
    src_depy_op: top.TensorOp
    src_depy_data: DependencyData
    carry_depy_op: top.TensorOp
    carry_depy_data: DependencyData
    dependent_exprs: Sequence[ie.IndexSequence]


def _match_recurrent_sum_base(
    ctx: CompilationCtx, merge_op: top.TensorOp
) -> MergeAddPatternInfo | None:
    """
    Shared utility: Checks for the basic merge/add recurrence structure and returns
    MergeAddPatternInfo if it's a candidate, None otherwise.

    This matches on the following pattern ():
    ```
    x = ... # some tensor with domain (..., t, ...)
    merge_op = RecurrentTensor.like(x)
    merge_op[..., 0, ...] = x[..., 0, ...]                               (src)
    merge_op[..., t, ...] = merge_op[..., t-1, ...] + x[..., t, ...]     (carry)
    ```
    """
    if not isinstance(merge_op, top.MergeOp):
        return None

    if merge_op.num_inputs != 2:
        return None

    dg = ctx.dg
    (src_depy_op, src_depy_data), (carry_depy_op, carry_depy_data) = (
        dg.get_flat_direct_dependencies(merge_op)
    )

    for idx, accum_symbol in enumerate(merge_op.domain.variables):
        # NOTE: Skip dynamic bound symbols (TODO: this was done to avoid statify_inc dims)
        if accum_symbol.as_bound() in dg.dynamic_bounds:
            continue

        if not (
            isinstance(carry_depy_op, top.AddOp)
            and carry_depy_data.expr == carry_depy_op.domain.basis_expr
            and (
                carry_depy_data.cond == ie.Not(accum_symbol.symb_eq(0))
                or carry_depy_data.cond is None
                or carry_depy_data.cond == ie.ConstBool(True)
            )
            and src_depy_data.expr == src_depy_op.domain.basis_expr
            and src_depy_data.cond == accum_symbol.symb_eq(0)
        ):
            continue

        (carry_depy_op0, carry_depy_data0), (carry_depy_op1, carry_depy_data1) = (
            dg.get_flat_direct_dependencies(carry_depy_op)
        )

        # NOTE: We are +'ing the merge and src results
        if not (carry_depy_op0 == merge_op or carry_depy_op1 == merge_op):
            continue
        if not (carry_depy_op0 == src_depy_op or carry_depy_op1 == src_depy_op):
            continue

        # NOTE: figure out which side is the merge and which is the src
        carry_merge_depy_data = carry_depy_data0 if carry_depy_op0 == merge_op else carry_depy_data1
        carry_src_depy_data = (
            carry_depy_data0 if carry_depy_op0 == src_depy_op else carry_depy_data1
        )

        if not (
            carry_src_depy_data.is_unconditional_basis()
            and carry_merge_depy_data.cond is None
            and carry_merge_depy_data.expr
            == merge_op.domain.basis_expr.replace_idx(idx, accum_symbol - 1)
        ):
            continue
        if not (src_depy_data.src_out_idx == carry_src_depy_data.src_out_idx):
            continue

        # TODO: needed? I suppose so, so that we can manually remove the recurrence,
        # as otherwise, the deadcodeelim will not catch these.
        if not (len(dg.get_flat_direct_dependents(carry_depy_op)) == 1):
            continue

        dependent_exprs = [
            data.expr
            for dep_op, data in dg.get_flat_direct_dependents(merge_op)
            if dep_op != carry_depy_op
        ]
        if not dependent_exprs:
            continue
        if not all(e.struct_eq(dependent_exprs[0]) for e in dependent_exprs):
            continue
        # If we get here, it's a candidate
        return MergeAddPatternInfo(
            accum_symbol=accum_symbol,
            idx=idx,
            src_depy_op=src_depy_op,
            src_depy_data=src_depy_data,
            carry_depy_op=carry_depy_op,
            carry_depy_data=carry_depy_data,
            dependent_exprs=dependent_exprs,
        )
    return None


class RecurrentSumLift(MatchReplacer):
    """
    Match and replace sum patterns in MergeOp as vectorization patterns.
    Sum pattern of the form:
    ```
    x = ... # some tensor with domain (..., t, ...)
    sum_x = RecurrentTensor.like(x)
    sum_x[..., 0, ...] = x[..., 0, ...]
    sum_x[..., t, ...] = sum_x[..., t-1, ...] + x[..., t, ...]
    y = f(sum_x[..., C, ...]) # where C is a constant point, e.g., T-1

    ```

    """

    @dataclasses.dataclass
    class Result:
        accum_symbol: ie.Symbol
        idx: int
        src_depy_op: top.TensorOp
        src_depy_data: DependencyData
        src_symb_t: SymbolicTensor
        expr_list: Sequence[ie.IndexExpr]
        cons_e: ie.IntIndexValue
        src_idx: int
        carry_depy_op: top.TensorOp

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        info = _match_recurrent_sum_base(ctx, op)
        if info is None:
            return None
        idx = info.idx
        dependent_exprs = info.dependent_exprs
        cons_e = dependent_exprs[0][idx]
        # Only match sum pattern: y = f(sum_x[..., C, ...]) where C is a constant point
        if cons_e.is_constant() and cons_e.is_point():
            src_symb_t = get_symbolic_tensor_for_op_output(
                ctx.dg, info.src_depy_op, info.src_depy_data.src_out_idx
            )
            expr_list = list(info.src_depy_op.domain.basis_expr)
            src_idx = info.src_depy_op.domain.find_variable_index(info.accum_symbol)
            return RecurrentSumLift.Result(
                accum_symbol=info.accum_symbol,
                idx=idx,
                src_depy_op=info.src_depy_op,
                src_depy_data=info.src_depy_data,
                src_symb_t=src_symb_t,
                expr_list=expr_list,
                cons_e=cons_e,  # type: ignore
                src_idx=src_idx,
                carry_depy_op=info.carry_depy_op,
            )
        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        dg = ctx.dg

        expr_list = list(mr.expr_list)
        expr_list[mr.idx] = ie.slice_(ie.ConstInt(0), mr.cons_e + ie.lift_to_ie(1))
        expr = ie.IndexSequence(tuple(expr_list))  # type: ignore

        lifted = mr.src_symb_t.symbolic_index(expr).sum((0,), keepdim=False)
        merge_dom_idx = op.domain.find_variable_index(mr.accum_symbol)
        for dep_op, dep_data in dg.get_flat_direct_dependents(op):
            fixed_expr = dep_data.expr.skip_idx(merge_dom_idx)
            fixed_dep_data = dataclasses.replace(dep_data, expr=fixed_expr)
            dg.add_edge(dep_op, lifted.op, fixed_dep_data)

        dg.remove_op(op)
        dg.remove_op(mr.carry_depy_op)


class RecurrentCumSumLift(MatchReplacer):
    """
    Match and replace cumsum patterns in MergeOp as vectorization patterns.
    ```
    x = ... # some tensor with domain (..., t, ...)
    sum_x = RecurrentTensor.like(x)
    sum_x[..., 0, ...] = x[..., 0, ...]
    sum_x[..., t, ...] = sum_x[..., t-1, ...] + x[..., t, ...]
    y = f(sum_x[..., t, ...])
    ```
    """

    @dataclasses.dataclass
    class Result:
        accum_symbol: ie.Symbol
        idx: int
        src_depy_op: top.TensorOp
        src_depy_data: DependencyData
        src_symb_t: SymbolicTensor
        expr_list: Sequence[ie.IndexExpr]
        cons_e: ie.IntIndexValue
        src_idx: int
        carry_depy_op: top.TensorOp

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        info = _match_recurrent_sum_base(ctx, op)
        if info is None:
            return None
        idx = info.idx
        dependent_exprs = info.dependent_exprs
        cons_e = dependent_exprs[0][idx]
        # Only match cumsum pattern: y = f(pref_sum_x[..., t, ...])
        # where t is a variable (not constant)
        if cons_e.is_point() and not cons_e.is_constant():
            src_symb_t = get_symbolic_tensor_for_op_output(
                ctx.dg, info.src_depy_op, info.src_depy_data.src_out_idx
            )
            expr_list = list(info.src_depy_op.domain.basis_expr)
            src_idx = info.src_depy_op.domain.find_variable_index(info.accum_symbol)
            match_ = RecurrentCumSumLift.Result(
                accum_symbol=info.accum_symbol,
                idx=idx,
                src_depy_op=info.src_depy_op,
                src_depy_data=info.src_depy_data,
                src_symb_t=src_symb_t,
                expr_list=expr_list,
                cons_e=cons_e,  # type: ignore
                src_idx=src_idx,
                carry_depy_op=info.carry_depy_op,
            )
            log.info("RecurrentCumSumLift match!: %s", match_)
            return match_
        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        dg = ctx.dg

        expr_list = list(mr.expr_list)
        expr_list[mr.idx] = ie.slice_(ie.ConstInt(0), mr.cons_e + ie.lift_to_int_ie(1))
        expr = ie.IndexSequence(expr_list)  # type: ignore

        lifted = mr.src_symb_t.symbolic_index(expr).cumsum(0)
        for dep_op, dep_data in dg.get_flat_direct_dependents(op):
            fixed_expr = dep_data.expr
            fixed_dep_data = dataclasses.replace(dep_data, expr=fixed_expr)
            dg.add_edge(dep_op, lifted.op, fixed_dep_data)

        dg.remove_op(op)
        dg.remove_op(mr.carry_depy_op)


class SlidingSumLift(MatchReplacer):
    """
    Match and replace cumsum patterns in MergeOp as vectorization patterns.

    x[a:b].sum() ⟶ x[0:T].cumsum().index(b - 1) - (x[0:T].cumsum().index(a - 1) if a > 0 else 0)
    """

    @dataclasses.dataclass
    class Result:
        sum_dim: int
        v: ie.Symbol
        e: ie.Slice
        idx: int
        src_op: top.TensorOp
        src_data: DependencyData
        num_slices_so_far: int

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        dg = ctx.dg

        if not isinstance(op, top.SumOp):
            return None

        ((src_op, src_data),) = dg.get_flat_direct_dependencies(op)

        sum_dims = op.dims

        if len(sum_dims) != 1:
            return None

        sum_dim = sum_dims[0]

        num_slices_so_far = 0
        for idx, v in enumerate(src_op.domain.variables):
            e = src_data.expr.members[idx]
            if not isinstance(e, ie.Slice):
                continue
            else:
                num_slices_so_far += 1
            if e.is_constant():
                continue
            if (num_slices_so_far - 1) != sum_dim:
                continue
            return SlidingSumLift.Result(
                sum_dim=sum_dim,
                v=v,
                e=e,
                idx=idx,
                src_op=src_op,
                src_data=src_data,
                num_slices_so_far=num_slices_so_far,
            )

        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        dg = ctx.dg
        a, b = mr.e.start, mr.e.stop

        src_symb_t = get_symbolic_tensor_for_op_output(
            ctx.dg, mr.src_op, mr.src_data.src_out_idx
        ).symbolic_index(mr.src_data.expr.replace_idx(mr.idx, ie.slice_(0, mr.v.as_bound())))

        cumsum_src = src_symb_t.cumsum(mr.num_slices_so_far)

        last_ = cumsum_src.spatial_index_select(mr.num_slices_so_far, b - 1)

        if ie.struct_eq(a, 0):
            res = last_
        else:
            first_ = cumsum_src.spatial_index_select(mr.num_slices_so_far, a - 1)
            res = last_ - first_

        dg.move_dependents(op, res.op)
        dg.remove_op(op)
