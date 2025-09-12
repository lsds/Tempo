from __future__ import annotations

import typing
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import IntEnum
from math import prod

import numpy as np

from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.datatypes import OpInId, OpOutId
from tempo.core.dependence_graph import DependencyData, OpData
from tempo.core.domain import Domain
from tempo.core.dtype import dtypes
from tempo.core.shape import Shape
from tempo.core.symbolic_tensor import (
    SymbolicTensor,
    get_symbolic_tensor_for_op_output,
    lift_to_symbolic_tensor,
)
from tempo.transformations.optimizer.algebraic.match_replacer import MatchReplacer
from tempo.utils import isl as isl_utils
from tempo.utils import logger
from tempo.utils.dg_utils import (
    is_expanded_dim,
    move_dependents_two_steps,
    move_select_dependents_two_steps,
)

log = logger.get_logger(__name__)


def is_const_uniform(op: top.TensorOp, value: int | None = None) -> bool:
    return (
        isinstance(op, top.ConstOp)
        and op.is_uniform
        and (value is None or op.uniform_value == value)
    )


class ZeroAddOptimization(MatchReplacer):
    """Optimization: x + 0 -> x"""

    @dataclass
    class Result:
        zero_op: top.TensorOp
        zero_data: DependencyData
        other_op: top.TensorOp
        other_data: DependencyData

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.AddOp):
            return None

        dg = ctx.dg
        dependencies = list(dg.get_flat_direct_dependencies(op))
        assert len(dependencies) == 2, f"Expected 2 dependencies, got {len(dependencies)}"

        op_a, op_a_data = dependencies[0]
        op_b, op_b_data = dependencies[1]

        if isinstance(op_a, top.ConstOp) and op_a.is_uniform and op_a.uniform_value == 0:
            return ZeroAddOptimization.Result(
                zero_op=op_a,
                zero_data=op_a_data,
                other_op=op_b,
                other_data=op_b_data,
            )

        if isinstance(op_b, top.ConstOp) and op_b.is_uniform and op_b.uniform_value == 0:
            return ZeroAddOptimization.Result(
                zero_op=op_b,
                zero_data=op_b_data,
                other_op=op_a,
                other_data=op_a_data,
            )
        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.AddOp), f"Expected AddOp, got {type(op)}"
        dg = ctx.dg

        # Get the symbolic tensor for the other operand
        other_symb_t = get_symbolic_tensor_for_op_output(dg, mr.other_op, mr.other_data.src_out_idx)
        orig_dtype = dg.get_output_dtypes(op)[OpOutId(0)]
        other_symb_t = other_symb_t.to_dtype(orig_dtype)

        # Move dependents from the add op to the other operand
        move_dependents_two_steps(ctx, op, other_symb_t.op, mr.other_data)


class ZeroMulOptimization(MatchReplacer):
    """Optimization: x * 0 -> 0"""

    @dataclass
    class Result:
        zero_op: top.TensorOp
        zero_data: DependencyData

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.MulOp):
            return None

        dg = ctx.dg
        dependencies = list(dg.get_flat_direct_dependencies(op))
        assert len(dependencies) == 2, f"Expected 2 dependencies, got {len(dependencies)}"

        # Check if one of the inputs is a zero constant
        for dep_op, dep_data in dependencies:
            if isinstance(dep_op, top.ConstOp) and dep_op.is_uniform and dep_op.uniform_value == 0:
                return ZeroMulOptimization.Result(
                    zero_op=dep_op,
                    zero_data=dep_data,
                )
        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.MulOp), f"Expected MulOp, got {type(op)}"
        dg = ctx.dg

        # Get the symbolic tensor for the zero constant
        zero_symb_t = get_symbolic_tensor_for_op_output(dg, mr.zero_op, mr.zero_data.src_out_idx)
        orig_dtype = dg.get_output_dtypes(op)[OpOutId(0)]
        zero_symb_t = zero_symb_t.to_dtype(orig_dtype)

        # Move dependents from the mul op to the zero constant
        move_dependents_two_steps(ctx, op, zero_symb_t.op, mr.zero_data)


class ZeroDivOptimization(MatchReplacer):
    """Optimization: 0 / x -> 0"""

    @dataclass
    class Result:
        zero_op: top.TensorOp
        zero_data: DependencyData

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.DivOp):
            return None

        dg = ctx.dg
        dependencies = list(dg.get_flat_direct_dependencies(op))
        assert len(dependencies) == 2, f"Expected 2 dependencies, got {len(dependencies)}"

        op_a, op_a_data = dependencies[0]
        op_b, op_b_data = dependencies[1]

        if isinstance(op_b, top.ConstOp) and op_b.is_uniform and op_b.uniform_value == 0:
            raise ValueError(f"Found division by zero for {op}, with lhs {op_a} and rhs {op_b}")

        # Check if the first input (lhs) is a zero constant
        if isinstance(op_a, top.ConstOp) and op_a.is_uniform and op_a.uniform_value == 0:
            return ZeroDivOptimization.Result(
                zero_op=op_a,
                zero_data=op_a_data,
            )

        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.DivOp), f"Expected DivOp, got {type(op)}"
        dg = ctx.dg

        # Get the symbolic tensor for the zero constant
        zero_symb_t = get_symbolic_tensor_for_op_output(dg, mr.zero_op, mr.zero_data.src_out_idx)
        orig_dtype = dg.get_output_dtypes(op)[OpOutId(0)]
        zero_symb_t = zero_symb_t.to_dtype(orig_dtype)

        # Move dependents from the div op to the zero constant
        move_dependents_two_steps(ctx, op, zero_symb_t.op, mr.zero_data)


class NegNegOptimization(MatchReplacer):
    """Optimization: -(-x) -> x"""

    @dataclass
    class Result:
        inner_neg_op: top.TensorOp
        inner_neg_data: DependencyData
        inner_op: top.TensorOp
        inner_data: DependencyData

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.NegOp):
            return None

        dg = ctx.dg
        dependencies = list(dg.get_flat_direct_dependencies(op))
        assert len(dependencies) == 1, f"Expected 1 dependency, got {len(dependencies)}"

        depy_op, depy_data = dependencies[0]

        if not isinstance(depy_op, top.NegOp):
            return None

        # NOTE: I removed this check because even if the dependency has other dependents,
        # all we are doing is moving the dependents of the outer neg op to the src op.
        # NOTE: We do not remove the inner neg op, so those dependents stay as is.
        ## Check if the dependency has only one dependent (this op)
        # if len(dg.get_flat_direct_dependents(depy_op)) != 1:
        #    return None

        # Get the dependency of the inner neg op
        inner_depys = list(dg.get_flat_direct_dependencies(depy_op))
        if len(inner_depys) != 1:
            return None

        inner_op, inner_data = inner_depys[0]
        return NegNegOptimization.Result(
            inner_neg_op=depy_op, inner_neg_data=depy_data, inner_op=inner_op, inner_data=inner_data
        )

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.NegOp), f"Expected NegOp, got {type(op)}"
        dg = ctx.dg

        # Move all dependents from the outer neg op to the inner op
        for dept, dept_data in dg.get_flat_direct_dependents(op):
            # Combine the edges through the two neg ops

            ops_ordered = [dept, op, mr.inner_neg_op, mr.inner_op]
            edges_ordered = [dept_data, mr.inner_neg_data, mr.inner_data]
            new_dep_data = isl_utils.combine_many_edges(
                ops_ordered, edges_ordered, dg.static_bounds, ctx.analysis_ctx.isl_ctx
            )

            dg.add_edge(dept, mr.inner_op, new_dep_data)
            dg.remove_edge(dept, op, dept_data)


class UnifConstAlgebraOptimization(MatchReplacer):
    """Optimization: algebra of uniform consts -> uniform const"""

    @dataclass
    class Result:
        unif_depys: list[tuple[top.ConstOp, DependencyData]]

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        valid_move_ops = (
            top.SqueezeOp,
            top.UnsqueezeOp,
            top.ReshapeOp,
            top.ExpandOp,
            top.PadOp,
            top.FlipOp,
            top.PermuteOp,
        )
        if not (
            isinstance(op, (top.ElementWiseOp, top.ReduceOp)) or isinstance(op, valid_move_ops)
        ):
            return None

        dg = ctx.dg
        depys = list(dg.get_flat_direct_dependencies(op))

        all_depys_unif_const = all(is_const_uniform(d) for d, _ in depys)
        if not all_depys_unif_const:
            return None

        all_basis_edges = all(
            depy_data.is_unconditional() and depy_data.expr.struct_eq(depy.domain.basis_expr)
            for depy, depy_data in depys
        )
        if not all_basis_edges:
            return None

        if isinstance(op, top.ReduceOp):
            # NOTE: Otherwise we risk not being able to compute the reduction.
            all_input_shapes_static = dg.get_input_shape(op, OpInId(0)).is_static()
            if not all_input_shapes_static:
                return None

        if isinstance(op, top.PadOp):
            const_depy = typing.cast(top.ConstOp, depys[0][0])
            unif_const_0 = const_depy.uniform_value
            pad_val_matches_const = op.mode == top.PadMode.CONSTANT and op.value == unif_const_0
            pad_mode_any = op.mode == top.PadMode.ANY
            if not (pad_val_matches_const or pad_mode_any):
                return None

        return UnifConstAlgebraOptimization.Result(
            unif_depys=typing.cast(list[tuple[top.ConstOp, DependencyData]], depys),
        )

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        dg = ctx.dg

        @dataclass
        class FoldCtx:
            unif_values: list[int | float | bool]
            shapes: list[Shape]
            op: top.TensorOp

        top_to_lambda = {
            top.MaxOp: lambda ctx: ctx.unif_values[0],
            top.SumOp: lambda ctx: ctx.unif_values[0]
            * prod(ctx.shapes[0].at(dim) for dim in ctx.op.dims),
            top.NegOp: lambda ctx: -ctx.unif_values[0],
            top.AddOp: lambda ctx: ctx.unif_values[0] + ctx.unif_values[1],
            top.SubOp: lambda ctx: ctx.unif_values[0] - ctx.unif_values[1],
            top.MulOp: lambda ctx: ctx.unif_values[0] * ctx.unif_values[1],
            top.DivOp: lambda ctx: ctx.unif_values[0] / ctx.unif_values[1],
            top.CastOp: lambda ctx: ctx.unif_values[0],  # TODO: needs a proper implementation no?
            top.PowOp: lambda ctx: ctx.unif_values[0] ** ctx.unif_values[1],
            top.AndOp: lambda ctx: ctx.unif_values[0] & ctx.unif_values[1],
            top.OrOp: lambda ctx: ctx.unif_values[0] | ctx.unif_values[1],
            top.NotOp: lambda ctx: ~ctx.unif_values[0],
            top.ExpOp: lambda ctx: np.exp(ctx.unif_values[0]),
            top.LnOp: lambda ctx: np.log(ctx.unif_values[0]),
            top.SinOp: lambda ctx: np.sin(ctx.unif_values[0]),
            top.SqrtOp: lambda ctx: np.sqrt(ctx.unif_values[0]),
            top.WhereOp: lambda ctx: (
                ctx.unif_values[1] if ctx.unif_values[0] else ctx.unif_values[2]
            ),
            # NOTE: For all the movement ops, we just return the first unif const.
            top.SqueezeOp: lambda ctx: ctx.unif_values[0],
            top.UnsqueezeOp: lambda ctx: ctx.unif_values[0],
            top.ReshapeOp: lambda ctx: ctx.unif_values[0],
            top.ExpandOp: lambda ctx: ctx.unif_values[0],
            top.PadOp: lambda ctx: ctx.unif_values[0],
            top.FlipOp: lambda ctx: ctx.unif_values[0],
            top.PermuteOp: lambda ctx: ctx.unif_values[0],
        }

        unif_values = [depy.uniform_value for depy, _ in mr.unif_depys]
        shapes = [depy.shape for depy, _ in mr.unif_depys]
        fold_ctx = FoldCtx(unif_values=unif_values, shapes=shapes, op=op)

        res = top_to_lambda[type(op)](fold_ctx)  # type: ignore
        op_data = dg.ops_by_id[op.op_id]
        np_dtype = dtypes.to_np(op_data.output_dtypes[OpOutId(0)])
        new_op = top.ConstOp(
            op_id=dg.get_next_op_id(),
            domain=op.domain.copy(),
            tags=dict(op.tags),
            shape=op_data.output_shapes[OpOutId(0)],
            dtype=op_data.output_dtypes[OpOutId(0)],
            value=np.asarray(res, dtype=np_dtype),
            is_uniform=True,
        )
        new_op_data = OpData(
            op=new_op,
            output_shapes=op_data.output_shapes,
            output_dtypes=op_data.output_dtypes,
        )
        dg.insert_op(new_op_data)
        dg.move_dependents(op, new_op)

        for depy, depy_data in mr.unif_depys:
            dg.remove_edge(op, depy, depy_data)
        dg.remove_op(op)


class ZeroSubOptimization(MatchReplacer):
    """Optimization: x - 0 -> x"""

    @dataclass
    class Result:
        zero_op: top.TensorOp
        zero_data: DependencyData
        other_op: top.TensorOp
        other_data: DependencyData

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.SubOp):
            return None

        dg = ctx.dg
        dependencies = list(dg.get_flat_direct_dependencies(op))
        assert len(dependencies) == 2, f"Expected 2 dependencies, got {len(dependencies)}"

        op_a, op_a_data = dependencies[0]
        op_b, op_b_data = dependencies[1]

        # Check if the second input (rhs) is a zero constant
        if is_const_uniform(op_b, 0):
            return ZeroSubOptimization.Result(
                zero_op=op_b,
                zero_data=op_b_data,
                other_op=op_a,
                other_data=op_a_data,
            )
        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.SubOp), f"Expected SubOp, got {type(op)}"
        dg = ctx.dg

        # Get the symbolic tensor for the other operand
        other_symb_t = get_symbolic_tensor_for_op_output(dg, mr.other_op, mr.other_data.src_out_idx)
        orig_dtype = dg.get_output_dtypes(op)[OpOutId(0)]
        other_symb_t = other_symb_t.to_dtype(orig_dtype)

        # Move dependents from the sub op to the other operand
        move_dependents_two_steps(ctx, op, other_symb_t.op, mr.other_data)


class OneMulOptimization(MatchReplacer):
    """Optimization: x * 1 -> x"""

    @dataclass
    class Result:
        one_op: top.TensorOp
        one_data: DependencyData
        other_op: top.TensorOp
        other_data: DependencyData

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.MulOp):
            return None

        dg = ctx.dg
        dependencies = list(dg.get_flat_direct_dependencies(op))
        assert len(dependencies) == 2, f"Expected 2 dependencies, got {len(dependencies)}"

        op_a, op_a_data = dependencies[0]
        op_b, op_b_data = dependencies[1]

        if is_const_uniform(op_a, 1):
            return OneMulOptimization.Result(
                one_op=op_a,
                one_data=op_a_data,
                other_op=op_b,
                other_data=op_b_data,
            )

        if is_const_uniform(op_b, 1):
            return OneMulOptimization.Result(
                one_op=op_b,
                one_data=op_b_data,
                other_op=op_a,
                other_data=op_a_data,
            )
        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.MulOp), f"Expected MulOp, got {type(op)}"
        dg = ctx.dg

        # Get the symbolic tensor for the other operand
        other_symb_t = get_symbolic_tensor_for_op_output(dg, mr.other_op, mr.other_data.src_out_idx)
        orig_dtype = dg.get_output_dtypes(op)[OpOutId(0)]
        other_symb_t = other_symb_t.to_dtype(orig_dtype)

        # Move dependents from the mul op to the other operand
        move_dependents_two_steps(ctx, op, other_symb_t.op, mr.other_data)


class OneDivOptimization(MatchReplacer):
    """Optimization: x / 1 -> x"""

    @dataclass
    class Result:
        one_op: top.TensorOp
        one_data: DependencyData
        other_op: top.TensorOp
        other_data: DependencyData

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.DivOp):
            return None

        dg = ctx.dg
        dependencies = list(dg.get_flat_direct_dependencies(op))
        assert len(dependencies) == 2, f"Expected 2 dependencies, got {len(dependencies)}"

        op_a, op_a_data = dependencies[0]
        op_b, op_b_data = dependencies[1]

        # Check if the second input (rhs) is a one constant
        if is_const_uniform(op_b, 1):
            return OneDivOptimization.Result(
                one_op=op_b,
                one_data=op_b_data,
                other_op=op_a,
                other_data=op_a_data,
            )
        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.DivOp), f"Expected DivOp, got {type(op)}"
        dg = ctx.dg

        # Get the symbolic tensor for the other operand
        other_symb_t = get_symbolic_tensor_for_op_output(dg, mr.other_op, mr.other_data.src_out_idx)
        orig_dtype = dg.get_output_dtypes(op)[OpOutId(0)]
        other_symb_t = other_symb_t.to_dtype(orig_dtype)

        # Move dependents from the div op to the other operand
        move_dependents_two_steps(ctx, op, other_symb_t.op, mr.other_data)


class ExpLnOptimization(MatchReplacer):
    """Optimization: exp(ln(x)) -> x and ln(exp(x)) -> x"""

    @dataclass
    class Result:
        outer_op: top.TensorOp
        inner_op: top.TensorOp
        inner_data: DependencyData
        base_op: top.TensorOp
        base_data: DependencyData

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, (top.ExpOp, top.LnOp)):
            return None

        dg = ctx.dg
        dependencies = list(dg.get_flat_direct_dependencies(op))
        assert len(dependencies) == 1, f"Expected 1 dependency, got {len(dependencies)}"

        dep_op, dep_data = dependencies[0]

        # Case 1: exp(ln(x)) -> x
        if isinstance(op, top.ExpOp) and isinstance(dep_op, top.LnOp):
            # Get the dependency of the ln op
            inner_deps = list(dg.get_flat_direct_dependencies(dep_op))
            assert len(inner_deps) == 1, f"Expected 1 dependency, got {len(inner_deps)}"

            inner_op, inner_data = inner_deps[0]
            return ExpLnOptimization.Result(
                outer_op=op,
                inner_op=dep_op,
                inner_data=dep_data,
                base_op=inner_op,
                base_data=inner_data,
            )

        # Case 2: ln(exp(x)) -> x
        elif isinstance(op, top.LnOp) and isinstance(dep_op, top.ExpOp):
            # Get the dependency of the exp op
            inner_deps = list(dg.get_flat_direct_dependencies(dep_op))
            assert len(inner_deps) == 1, f"Expected 1 dependency, got {len(inner_deps)}"

            inner_op, inner_data = inner_deps[0]
            return ExpLnOptimization.Result(
                outer_op=op,
                inner_op=dep_op,
                inner_data=dep_data,
                base_op=inner_op,
                base_data=inner_data,
            )

        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, (top.ExpOp, top.LnOp)), f"Expected ExpOp or LnOp, got {type(op)}"
        dg = ctx.dg

        # Move all dependents from the outer op to the base op
        for dept, dept_data in dg.get_flat_direct_dependents(mr.outer_op):
            # Combine the edges through the outer and inner ops
            ops_ordered = [dept, mr.outer_op, mr.inner_op, mr.base_op]
            edges_ordered = [dept_data, mr.inner_data, mr.base_data]

            new_dep_data = isl_utils.combine_many_edges(
                ops_ordered, edges_ordered, dg.static_bounds, ctx.analysis_ctx.isl_ctx
            )

            dg.add_edge(dept, mr.base_op, new_dep_data)
            dg.remove_edge(dept, mr.outer_op, dept_data)


class SqueezeUnsqueezeOptimization(MatchReplacer):
    """Optimization:
    squeeze(unsqueeze(x, dim), dim) -> x
    unsqueeze(squeeze(x, dim), dim) -> x
    """

    @dataclass
    class Result:
        outer_op: top.TensorOp
        inner_op: top.TensorOp
        inner_data: DependencyData
        base_op: top.TensorOp
        base_data: DependencyData

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        assert isinstance(op, (top.SqueezeOp, top.UnsqueezeOp)), (
            f"Expected SqueezeOp or UnsqueezeOp, got {type(op)}"
        )

        dg = ctx.dg
        dep_op, dep_data = dg.get_flat_direct_dependencies(op)[0]

        # Case 1: squeeze(unsqueeze(x, dim), dim) -> x
        if isinstance(op, top.SqueezeOp) and isinstance(dep_op, top.UnsqueezeOp):
            # Check if the dimensions match
            if dep_op.dim != op.dim:
                return None

            # Get the dependency of the unsqueeze op
            inner_op, inner_data = dg.get_flat_direct_dependencies(dep_op)[0]

            return SqueezeUnsqueezeOptimization.Result(
                outer_op=op,
                inner_op=dep_op,
                inner_data=dep_data,
                base_op=inner_op,
                base_data=inner_data,
            )

        # Case 2: unsqueeze(squeeze(x, dim), dim) -> x
        elif isinstance(op, top.UnsqueezeOp) and isinstance(dep_op, top.SqueezeOp):
            # Check if the dimensions match
            if dep_op.dim != op.dim:
                return None

            # Get the dependency of the squeeze op
            inner_op, inner_data = dg.get_flat_direct_dependencies(dep_op)[0]

            return SqueezeUnsqueezeOptimization.Result(
                outer_op=op,
                inner_op=dep_op,
                inner_data=dep_data,
                base_op=inner_op,
                base_data=inner_data,
            )

        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, (top.SqueezeOp, top.UnsqueezeOp)), (
            f"Expected SqueezeOp or UnsqueezeOp, got {type(op)}"
        )
        dg = ctx.dg

        # Move all dependents from the outer op to the base op
        for dept, dept_data in dg.get_flat_direct_dependents(mr.outer_op):
            # Combine the edges through the outer and inner ops
            ops_ordered = [dept, mr.outer_op, mr.inner_op, mr.base_op]
            edges_ordered = [dept_data, mr.inner_data, mr.base_data]

            new_dep_data = isl_utils.combine_many_edges(
                ops_ordered, edges_ordered, dg.static_bounds, ctx.analysis_ctx.isl_ctx
            )

            dg.add_edge(dept, mr.base_op, new_dep_data)
            dg.remove_edge(dept, mr.outer_op, dept_data)


class SliceSliceOptimization(MatchReplacer):
    """Optimization: slice(slice(x, dim), dim) -> slice(x, dim) with combined indices"""

    @dataclass
    class Result:
        outer_src_op: top.TensorOp
        outer_src_data: DependencyData
        outer_start_idx_op: top.TensorOp
        outer_start_idx_data: DependencyData
        inner_src_op: top.TensorOp
        inner_src_data: DependencyData
        inner_start_idx_op: top.TensorOp
        inner_start_idx_data: DependencyData

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.IndexSliceOp):
            return None

        dg = ctx.dg

        dependencies = dg.get_flat_direct_dependencies(op)
        outer_src_op, outer_src_data = dependencies[0]
        outer_start_idx_op, outer_start_idx_data = dependencies[1]

        # Check if the dependency is also an IndexSliceOp on the same dimension
        if isinstance(outer_src_op, top.IndexSliceOp) and outer_src_op.dim == op.dim:
            # Get the dependencies of the inner slice op
            inner_deps = dg.get_flat_direct_dependencies(outer_src_op)
            inner_src_op, inner_src_data = inner_deps[0]
            inner_start_idx_op, inner_start_idx_data = inner_deps[1]

            return SliceSliceOptimization.Result(
                outer_src_op=outer_src_op,
                outer_src_data=outer_src_data,
                outer_start_idx_op=outer_start_idx_op,
                outer_start_idx_data=outer_start_idx_data,
                inner_src_op=inner_src_op,
                inner_src_data=inner_src_data,
                inner_start_idx_op=inner_start_idx_op,
                inner_start_idx_data=inner_start_idx_data,
            )

        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.IndexSliceOp), f"Expected IndexSliceOp, got {type(op)}"
        dg = ctx.dg
        # Combine the edges through the two slice ops
        new_dep_data = isl_utils.combine_edges(
            op,
            mr.outer_src_data,
            mr.outer_src_op,
            mr.inner_src_data,
            mr.inner_src_op,
            dg.static_bounds,
            ctx.analysis_ctx.isl_ctx,
        )

        # Get symbolic tensors for the start indices
        outer_start_idx_symb_t = get_symbolic_tensor_for_op_output(
            dg, mr.outer_start_idx_op, mr.outer_start_idx_data.src_out_idx
        ).symbolic_index(mr.outer_start_idx_data.expr)
        inner_start_idx_symb_t = get_symbolic_tensor_for_op_output(
            dg, mr.inner_start_idx_op, mr.inner_start_idx_data.src_out_idx
        ).symbolic_index(mr.inner_start_idx_data.expr)

        # Combine the start indices
        start_combined = outer_start_idx_symb_t + inner_start_idx_symb_t

        # Create the new slice operation
        inner_symb_t = get_symbolic_tensor_for_op_output(
            dg, mr.inner_src_op, new_dep_data.src_out_idx
        ).symbolic_index(new_dep_data.expr)

        # NOTE: create the combined slice op. The start index is combined from the two slice ops,
        # while the length is the same as the outer slice op.
        sliced = inner_symb_t.index_slice(op.dim, start_combined, op.length)

        # Move dependents to the new slice operation
        dg.move_dependents(op, sliced.op)
        dg.remove_op(op)


class PermutePermuteOptimization(MatchReplacer):
    """Optimization: permute(permute(x, dims1), dims2) -> permute(x, combined_dims)"""

    @dataclass
    class Result:
        inner_permute_op: top.PermuteOp
        inner_permute_data: DependencyData
        inner_depy_op: top.TensorOp
        inner_depy_data: DependencyData

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.PermuteOp):
            return None

        dg = ctx.dg
        dependencies = list(dg.get_flat_direct_dependencies(op))
        if len(dependencies) != 1:
            return None

        inner_perm_op, inner_perm_data = dependencies[0]
        if not isinstance(inner_perm_op, top.PermuteOp):
            return None

        # Get the dependency of the inner permute op
        inner_depy_op, inner_depy_data = dg.get_flat_direct_dependencies(inner_perm_op)[0]

        # Check domain containment and unconditional basis
        # NOTE: This is necessary because otherwise, the number of dims of the
        # final permute != number of dims of perm_op
        # TODO: I think we can just remove this restriction. Below we handle this with dims, no?
        if (
            not inner_perm_op.domain.is_contained_in(inner_depy_op.domain)
            or not op.domain.is_contained_in(inner_perm_op.domain)
            or not inner_perm_data.is_unconditional_basis()
        ):
            return None

        return PermutePermuteOptimization.Result(
            inner_permute_op=inner_perm_op,
            inner_permute_data=inner_perm_data,
            inner_depy_op=inner_depy_op,
            inner_depy_data=inner_depy_data,
        )

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.PermuteOp), f"Expected PermuteOp, got {type(op)}"
        dg = ctx.dg

        num_slices = mr.inner_permute_data.expr.num_slices()
        dims = list(range(num_slices)) + [d + num_slices for d in mr.inner_permute_op.dims]
        new_dims = tuple(dims[d] for d in op.dims)

        # Combine the edges through the two permute ops
        ops_ordered = [op, mr.inner_permute_op, mr.inner_depy_op]
        edges_ordered = [mr.inner_permute_data, mr.inner_depy_data]
        new_dep_data = isl_utils.combine_many_edges(
            ops_ordered, edges_ordered, dg.static_bounds, ctx.analysis_ctx.isl_ctx
        )

        # Create the new permute operation
        symb_t = get_symbolic_tensor_for_op_output(
            dg, mr.inner_depy_op, new_dep_data.src_out_idx
        ).symbolic_index(new_dep_data.expr)
        new_permute_op = symb_t.permute(new_dims).op

        # Move dependents to the new permute operation
        dg.move_dependents(op, new_permute_op)


class ReshapeOpType(IntEnum):
    UNSQUEEZE = 1
    SQUEEZE = 2


class PushDir(IntEnum):
    DOWN = 1
    UP = 2


class MatMulReassocDir(IntEnum):
    LEFT = 1
    RIGHT = 2


class ReshapeOptimization(MatchReplacer):
    """Optimization: reshape(x, shape) -> unsqueeze/squeeze when possible"""

    @dataclass
    class Result:
        dep_op: top.TensorOp
        dep_data: DependencyData
        operation_type: ReshapeOpType
        dim: int

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.ReshapeOp):
            return None
        dg = ctx.dg
        dependencies = list(dg.get_flat_direct_dependencies(op))
        if len(dependencies) != 1:
            return None
        dep_op, dep_data = dependencies[0]
        input_shape = dg.get_input_shape(op, OpInId(0))
        new_shape = op.shape

        # Check if new_shape is input_shape with an extra 1 (unsqueeze case)
        is_extra_one, dim = input_shape.is_shape_without_extra_dim(new_shape)
        if is_extra_one:
            return ReshapeOptimization.Result(
                dep_op=dep_op,
                dep_data=dep_data,
                operation_type=ReshapeOpType.UNSQUEEZE,
                dim=dim,
            )

        # Check if new_shape is input_shape with one less 1 (squeeze case)
        is_less_one, dim = new_shape.is_shape_without_extra_dim(input_shape)
        if is_less_one:
            return ReshapeOptimization.Result(
                dep_op=dep_op,
                dep_data=dep_data,
                operation_type=ReshapeOpType.SQUEEZE,
                dim=dim,
            )

        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.ReshapeOp), f"Expected ReshapeOp, got {type(op)}"
        dg = ctx.dg

        # Get the symbolic tensor for the dependency
        dep_symb_t = get_symbolic_tensor_for_op_output(
            dg, mr.dep_op, mr.dep_data.src_out_idx
        ).symbolic_index(mr.dep_data.expr)
        if mr.operation_type == ReshapeOpType.UNSQUEEZE:
            result = dep_symb_t.unsqueeze(mr.dim)
        else:  # squeeze
            result = dep_symb_t.squeeze(mr.dim)

        # Move dependents to the new operation
        dg.move_dependents(op, result.op)


class SumDimSizeOneToSqueezeOptimization(MatchReplacer):
    """Optimization: sum(x, dim) -> squeeze when dim size is 1"""

    @dataclass
    class Result:
        dep_op: top.TensorOp
        dep_data: DependencyData
        dim: int

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.SumOp):
            return None

        dg = ctx.dg
        dep_op, dep_data = dg.get_flat_direct_dependencies(op)[0]
        input_shape = dg.get_input_shape(op, OpInId(0))

        # Check if we're summing over a dimension of size 1
        if len(op.dims) == 1 and input_shape.at(op.dims[0]) == 1:
            return SumDimSizeOneToSqueezeOptimization.Result(
                dep_op=dep_op,
                dep_data=dep_data,
                dim=op.dims[0],
            )

        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.SumOp), f"Expected SumOp, got {type(op)}"
        dg = ctx.dg

        # Get the symbolic tensor for the dependency
        dep_symb_t = get_symbolic_tensor_for_op_output(
            dg, mr.dep_op, mr.dep_data.src_out_idx
        ).symbolic_index(mr.dep_data.expr)

        # Summing over a dimension of size 1 is equivalent to squeezing
        result = dep_symb_t.squeeze(mr.dim)

        # Move dependents to the new operation
        dg.move_dependents(op, result.op)


class RedundantGatherOptimization(MatchReplacer):
    """Optimization: gather(x, const_index) -> x when index is an arange which covers dimension"""

    @dataclass
    class Result:
        src_op: top.TensorOp
        src_data: DependencyData

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.GatherOp):
            return None

        dg = ctx.dg
        sb = dg.static_bounds

        (src_op, src_op_data), (index_op, index_op_data) = dg.get_flat_direct_dependencies(op)

        # Check if both dependencies have unconditional basis
        if isinstance(index_op, top.ConstOp) and index_op.is_int_arange:
            len_of_index: int = index_op.value.shape[0].partial_eval(sb)  # type: ignore
            len_of_src_dim = ie.lift_to_int_ie(dg.get_input_shape(op, OpInId(0)).at(op.dim))

            # If the index covers the full dimension, we can remove the gather
            if ie.struct_eq(len_of_index, len_of_src_dim.partial_eval(sb)):
                return RedundantGatherOptimization.Result(
                    src_op=src_op,
                    src_data=src_op_data,
                )

        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.GatherOp), f"Expected GatherOp, got {type(op)}"
        # Move dependents from the gather op to the source op
        move_dependents_two_steps(ctx, op, mr.src_op, mr.src_data)


class RedundantConstIndexSelectOptimization(MatchReplacer):
    """
    Optimization: index_select(x, dim, [0,1,2,..., S-1]) => x,
    where S is the size of the dimension.
    """

    @dataclass
    class Result:
        dependents_to_move: list[tuple[top.TensorOp, DependencyData]]
        src_op: top.TensorOp
        src_data: DependencyData
        index_op: top.TensorOp
        index_data: DependencyData

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.IndexSelectOp):
            return None

        dg = ctx.dg
        sb = dg.static_bounds

        (src_op, src_data), (index_op, index_op_data) = dg.get_flat_direct_dependencies(op)

        # Get the source tensor shape at the index dimension
        src_tensor_shape = dg.get_input_shape(op, OpInId(0))  # NOTE 0 cause src is the first input
        src_shape_at_dim = ie.lift_to_int_ie(src_tensor_shape.at(op.dim))

        # Check if the index operation is a ConstOp with int_arange
        if isinstance(index_op, top.ConstOp) and index_op.is_int_arange:
            len_of_index = ie.lift_to_int_ie(index_op.value.shape[0])

            # If the index covers the full dimension, we can remove the index select
            if ie.struct_eq(len_of_index.partial_eval(sb), src_shape_at_dim.partial_eval(sb)):
                return RedundantConstIndexSelectOptimization.Result(
                    dependents_to_move=list(dg.get_flat_direct_dependents(op)),
                    src_op=src_op,
                    src_data=src_data,
                    index_op=index_op,
                    index_data=index_op_data,
                )

        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.IndexSelectOp), f"Expected IndexSelectOp, got {type(op)}"
        # Move dependents from the index select op to the source op
        move_select_dependents_two_steps(ctx, op, mr.src_op, mr.src_data, mr.dependents_to_move)


class RedundantSymbolConstIndexSelectOptimization(MatchReplacer):
    """
    Optimization: y --> index_select(dim, t) --> Const(x, shape=(T,), is_uniform=True)
    => y ---> x[0]
    """

    @dataclass
    class Result:
        dependents_to_move: list[tuple[top.TensorOp, DependencyData]]
        src_op: top.ConstOp
        src_data: DependencyData
        index_op: top.TensorOp
        index_data: DependencyData
        const_value: int | float

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.IndexSelectOp):
            return None

        dg = ctx.dg

        (src_op, src_data), (index_op, index_op_data) = dg.get_flat_direct_dependencies(op)

        # Check if dependents undo indexing by temporal indexing.
        if (
            is_const_uniform(src_op)
            and len(src_op.shape) == 1  # type: ignore # consts have shape # TODO: generalize
            and isinstance(index_op, top.EvalSymbolOp)
            and index_op.symbol.is_variable()
            and src_data.is_unconditional_basis()
        ):
            assert isinstance(src_op, top.ConstOp)

            assert op.dim == 0, "TODO: generalize"
            dependents_for_case_1 = []

            assert isinstance(src_op, top.ConstOp)
            const_value = src_op.uniform_value
            expected_e = op.domain.basis_expr
            for dep, dep_data in dg.get_flat_direct_dependents(op):
                e = dep_data.expr.partial_eval(dg.static_bounds)
                if ie.struct_eq(e, expected_e):
                    dependents_for_case_1.append((dep, dep_data))

            if len(dependents_for_case_1) > 0:
                return RedundantSymbolConstIndexSelectOptimization.Result(
                    dependents_to_move=dependents_for_case_1,
                    src_op=src_op,
                    src_data=src_data,
                    index_op=index_op,
                    index_data=index_op_data,
                    const_value=const_value,
                )

        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.IndexSelectOp), f"Expected IndexSelectOp, got {type(op)}"
        dg = ctx.dg

        new_op_symb_t = SymbolicTensor.full(mr.const_value, shape=(), dtype=mr.src_op.dtype)

        # NOTE: manually move the dependents, because combining the edges is wrong due to the
        # temporal indexing we matched on.
        for dep, dep_data in mr.dependents_to_move:
            new_dep_data = replace(
                dep_data, expr=new_op_symb_t.domain.basis_expr, src_out_idx=OpOutId(0)
            )
            dg.remove_edge(dep, op, dep_data)
            dg.add_edge(dep, new_op_symb_t.op, new_dep_data)


class RedundantSymbolIndexSelectOptimization(MatchReplacer):
    """
    Optimization: y --0:T--> index_select(dim, t) --> x => y ---> x
    when dependents undo indexing by temporal indexing.
    """

    @dataclass
    class Result:
        dependents_to_move: list[tuple[top.TensorOp, DependencyData]]
        src_op: top.TensorOp
        src_data: DependencyData
        index_op: top.TensorOp
        index_data: DependencyData

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.IndexSelectOp):
            return None

        dg = ctx.dg

        (src_op, src_data), (index_op, index_op_data) = dg.get_flat_direct_dependencies(op)

        # Check if dependents undo indexing by temporal indexing.
        if (
            isinstance(index_op, top.EvalSymbolOp)
            and index_op.symbol.is_variable()
            and index_op_data.is_unconditional_basis()
        ):
            dependents_for_case_1 = []
            s = index_op.symbol
            try:
                idx = op.domain.find_variable_index(s)
            except ValueError as e:
                print(f"Variable {s} not found in domain {op.domain} of op {op}")
                raise ValueError(f"Variable {s} not found in domain {op.domain} of op {op}") from e

            # TODO: can be generalized, as this is not the only valid expected_e.
            expected_e = op.domain.basis_expr.replace_idx(
                idx, ie.slice_(0, s.as_bound())
            ).partial_eval(dg.static_bounds)
            for dep, dep_data in dg.get_flat_direct_dependents(op):
                e = dep_data.expr.partial_eval(dg.static_bounds)
                if ie.struct_eq(e, expected_e):
                    dependents_for_case_1.append((dep, dep_data))

            if len(dependents_for_case_1) > 0:
                return RedundantSymbolIndexSelectOptimization.Result(
                    dependents_to_move=dependents_for_case_1,
                    src_op=src_op,
                    src_data=src_data,
                    index_op=index_op,
                    index_data=index_op_data,
                )

        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.IndexSelectOp), f"Expected IndexSelectOp, got {type(op)}"
        dg = ctx.dg

        # NOTE: manually move the dependents, because combining the edges is wrong due to the
        # temporal indexing we matched on.
        for dep, dep_data in mr.dependents_to_move:
            src_symb_t = get_symbolic_tensor_for_op_output(
                dg, mr.src_op, mr.src_data.src_out_idx
            ).symbolic_index(mr.src_data.expr)
            if op.dim != 0:
                dims = list(range(len(src_symb_t.shape)))
                dims.pop(op.dim)
                dims.insert(0, op.dim)
                src_symb_t = src_symb_t.permute(dims)
            new_e = src_symb_t.domain.basis_expr
            new_dep_data = replace(dep_data, expr=new_e, src_out_idx=mr.src_data.src_out_idx)
            dg.remove_edge(dep, op, dep_data)
            dg.add_edge(dep, src_symb_t.op, new_dep_data)


class ExpandSelectOptimization(MatchReplacer):
    """Optimization:
    x.expand(sizes).index_select(x, dim, index) -> x.expand(remaining_sizes_possibly_none),
    when dim is one of the expanded sizes.


    NOTE: We substantially changed this, in order to support temporal slices. If it is failing,
    add checks for basis expressions on expand_op_data and index_op_data.
    """

    @dataclass
    class Result:
        index_op: top.TensorOp
        index_op_data: DependencyData
        expand_op: top.ExpandOp
        expand_op_data: DependencyData
        expanded_src_op: top.TensorOp
        expanded_src_op_data: DependencyData
        num_slices_expand_op_data: int

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.IndexSelectOp):
            return None

        dg = ctx.dg
        (expand_op, expand_op_data), (index_op, index_op_data) = dg.get_flat_direct_dependencies(op)

        if not isinstance(expand_op, top.ExpandOp):
            return None

        num_slices_expand_op_data = sum(
            1 for member in expand_op_data.expr.members if not member.is_point()
        )

        expanded_src_op, expanded_src_op_data = dg.get_flat_direct_dependencies(expand_op)[0]
        expand_op_input_shape = dg.get_input_shape(expand_op, OpInId(0))

        expanded_dims = expand_op.dims_affected((expand_op_input_shape,))

        # dim is moved forward by however many slices the index expr has
        if (op.dim - num_slices_expand_op_data) not in expanded_dims:
            return None

        # TODO I suppose technically, we have to check whether the index is not a 1D index,
        # with repeated indices.
        index_shape = dg.get_input_shape(op, OpInId(1))
        if not index_shape.is_scalar():
            return None

        return ExpandSelectOptimization.Result(
            index_op=index_op,
            index_op_data=index_op_data,
            expand_op=expand_op,
            expand_op_data=expand_op_data,
            expanded_src_op=expanded_src_op,
            expanded_src_op_data=expanded_src_op_data,
            num_slices_expand_op_data=num_slices_expand_op_data,
        )

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.IndexSelectOp), f"Expected IndexSelectOp, got {type(op)}"
        dg = ctx.dg
        dim = op.dim

        # Get the symbolic tensor for the expanded operation
        expanded_src_t = get_symbolic_tensor_for_op_output(
            dg, mr.expanded_src_op, mr.expanded_src_op_data.src_out_idx
        ).symbolic_index(mr.expanded_src_op_data.expr)

        index_t = get_symbolic_tensor_for_op_output(
            dg, mr.index_op, mr.index_op_data.src_out_idx
        ).symbolic_index(mr.index_op_data.expr)
        index_is_scalar = index_t.shape.is_scalar()

        if index_is_scalar:
            # Squeeze the extra dimension that was expanded
            expanded_src_t = expanded_src_t.squeeze(dim - mr.num_slices_expand_op_data)

        assert isinstance(mr.expand_op, top.ExpandOp)
        new_expand_sizes = list(mr.expand_op.sizes)
        new_expand_sizes.pop(dim - mr.num_slices_expand_op_data)

        expanded_t = expanded_src_t.expand(Shape.from_(new_expand_sizes))

        expanded_t = expanded_t.symbolic_index(mr.expand_op_data.expr).ident()

        for dep, dep_data in list(ctx.dg.get_flat_direct_dependents(op)):
            op_out_id = expanded_t.tensor_id.output_id
            edge_data = DependencyData(
                expanded_t.op.domain.basis_expr, op_out_id, dep_data.sink_in_idx, dep_data.cond
            )
            ctx.dg.add_edge(dep, expanded_t.op, edge_data)
            ctx.dg.remove_edge(dep, op, dep_data)


class IndexSelectTemporalSliceOptimization(MatchReplacer):
    """Optimization:
    x.index(dim, t)[0:T] -> x.permute(dim_to_front)[skip_idx] when index is variable"""

    @dataclass
    class Result:
        tensor_op: top.TensorOp
        tensor_data: DependencyData
        dependent: top.TensorOp
        dependent_data: DependencyData
        symbol_idx: int
        dim: int

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.IndexSelectOp):
            return None

        dg = ctx.dg
        (tensor_op, tensor_op_data), (index_op, index_op_data) = dg.get_flat_direct_dependencies(op)

        # TODO: work to remove this restriction
        # Check if both dependencies have unconditional basis
        basis_exprs = (
            tensor_op_data.is_unconditional_basis() and index_op_data.is_unconditional_basis()
        )

        if basis_exprs and isinstance(index_op, top.EvalSymbolOp) and index_op.symbol.is_variable():
            symbol = index_op.symbol
            symbol_idx = op.domain.find_variable_index(symbol)

            # Check if any dependents have slice expressions that match the full range
            for dependent, dependent_data in dg.get_flat_direct_dependents(op):
                expr = dependent_data.expr
                if len(expr.members) < symbol_idx + 1:
                    continue

                member_at_idx = expr.members[symbol_idx].partial_eval(dg.static_bounds)
                full_expr = ie.slice_(ie.ConstInt(0), symbol.as_bound()).partial_eval(
                    dg.static_bounds
                )

                if member_at_idx.struct_eq(full_expr):
                    return IndexSelectTemporalSliceOptimization.Result(
                        tensor_op=tensor_op,
                        tensor_data=tensor_op_data,
                        dependent=dependent,
                        dependent_data=dependent_data,
                        symbol_idx=symbol_idx,
                        dim=op.dim,
                    )

        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.IndexSelectOp), f"Expected IndexSelectOp, got {type(op)}"
        dg = ctx.dg

        # Get the symbolic tensor for the tensor operation
        tensor = get_symbolic_tensor_for_op_output(dg, mr.tensor_op, mr.tensor_data.src_out_idx)

        # Create permutation to move the index dimension to the front
        perm_dims = list(range(len(tensor.shape)))
        perm_dims.remove(mr.dim)
        perm_dims.insert(0, mr.dim)
        permuted_tensor = tensor.permute(tuple(perm_dims))

        # Create new expression by skipping the symbol index
        new_expr = mr.dependent_data.expr.skip_idx(mr.symbol_idx)

        # Create new dependency data
        new_dep_data = DependencyData(
            new_expr,
            mr.tensor_data.src_out_idx,
            mr.dependent_data.sink_in_idx,
            mr.dependent_data.cond,
        )

        # Connect the dependent to the permuted tensor
        dg.add_edge(mr.dependent, permuted_tensor.op, new_dep_data)
        dg.remove_edge(mr.dependent, op, mr.dependent_data)


class PadExpandElideOptimization(MatchReplacer):
    """Optimization:
    pad(expand(x, sizes), dim, padding) -> expand(x, new_sizes),
    where new_sizes[dim] = padding[0] + old_sizes[dim] + padding[1]

    This only works for padding with ANY value. If padding with a specific value is needed,
    we cannot perform this optimization unless we could confirm that the value we are replicating
    with the expand is that same value.
    """

    @dataclass
    class Result:
        expand_op: top.ExpandOp
        expand_data: DependencyData
        expand_src_op: top.TensorOp
        expand_src_data: DependencyData
        dim: int

    def match(self, ctx: CompilationCtx, pad_op: top.TensorOp) -> Result | None:
        if not isinstance(pad_op, top.PadOp):
            return None

        dg = ctx.dg
        depy_op, depy_data = dg.get_flat_direct_dependencies(pad_op)[0]

        # Check if the pad is not static and the dependency is an expand op
        if pad_op.is_dynamic() and isinstance(depy_op, top.ExpandOp) and pad_op.is_any_pad():
            dim = pad_op.dim
            num_slices_in_depy_expr = depy_data.expr.num_slices()
            new_dim = dim - num_slices_in_depy_expr
            in_shapes_depy_op = dg.get_input_shapes_list(depy_op)

            expand_src_op, expand_src_data = dg.get_flat_direct_dependencies(depy_op)[0]

            # Check if the pad dimension is affected by the expand
            if new_dim in depy_op.dims_affected(in_shapes_depy_op):
                return PadExpandElideOptimization.Result(
                    expand_op=depy_op,
                    expand_data=depy_data,
                    expand_src_op=expand_src_op,
                    expand_src_data=expand_src_data,
                    dim=dim,
                )

        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.PadOp), f"Expected PadOp, got {type(op)}"
        dg = ctx.dg

        dep_out_shape = dg.get_output_shape(mr.expand_op, OpOutId(0))
        combined_size = op.padding[0] + dep_out_shape.at(mr.dim) + op.padding[1]
        combined_size = isl_utils.simplify_int_index_value(combined_size)

        # Create new expand shape with the combined size
        new_expand_shape = mr.expand_op.sizes.resize_dim(mr.dim, combined_size)

        # Get the symbolic tensor for the expand source operation
        expand_src_t = get_symbolic_tensor_for_op_output(
            dg, mr.expand_src_op, mr.expand_src_data.src_out_idx
        ).symbolic_index(mr.expand_src_data.expr)

        new_expand_t = (
            expand_src_t.expand(Shape.from_(new_expand_shape))
            .symbolic_index(mr.expand_data.expr)
            .ident()
        )

        # Move dependents to new expand op
        dg.move_dependents(op, new_expand_t.op)


class PadPushdownOptimization(MatchReplacer):
    """Optimization: pad(elementwise_op(x, ...), dim, padding) -> elementwise_op(pad(x, ...), ...)
    Supports unary, binary, and ternary (WhereOp) elementwise ops.
    Handles non-unconditional_basis edges by combining edges and shifting pad_dim as needed.
    """

    @dataclass
    class Result:
        pad_op: top.PadOp
        pad_data: DependencyData
        elem_op: top.TensorOp

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.PadOp):
            return None
        dg = ctx.dg
        # Only support pad on elementwise ops (unary, binary, ternary)
        dep_op, dep_data = dg.get_flat_direct_dependencies(op)[0]

        # TODO: add support for other movement ops?
        # Use propagate methods to propagate the pad_dim through the depy
        if not (isinstance(dep_op, (top.UnaryElementWiseOp, top.BinaryElementWiseOp, top.WhereOp))):
            return None

        # NOTE: reject if dim is created by depy or depy edge
        for dep_op, dep_data in dg.get_flat_direct_dependencies(op):
            num_slices = dep_data.expr.num_slices()
            shifted_dim = op.dim - num_slices
            if shifted_dim < 0:
                return None
            if is_expanded_dim(dg, dep_op, shifted_dim):
                return None

        # Gather all inputs to the elementwise op
        return PadPushdownOptimization.Result(
            pad_op=op,
            pad_data=dep_data,
            elem_op=dep_op,
        )

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.PadOp), f"Expected PadOp, got {type(op)}"
        dg = ctx.dg
        pad_op = mr.pad_op
        pad_data = mr.pad_data
        pad_dim = pad_op.dim
        elem_op = mr.elem_op
        elem_inputs = tuple(dg.get_flat_direct_dependencies(elem_op))
        # For each input to the elementwise op, create a PadOp if the shape matches
        new_inputs = []
        for in_op, in_data in elem_inputs:
            # Determine the correct pad_dim for this input
            # Shift pad_dim if the input has fewer leading dims due to broadcasting or indexing
            # Combine edges: pad_op --pad_data--> elem_op --in_data--> in_op
            # The number of slices in pad_data.expr tells us how many dims are introduced
            num_slices = pad_data.expr.num_slices()
            shifted_dim = pad_dim - num_slices
            # Combine edges for correct dependency
            combined_edge = isl_utils.combine_edges(
                pad_op,
                pad_data,
                elem_op,
                in_data,
                in_op,
                dg.static_bounds,
                ctx.analysis_ctx.isl_ctx,
            )
            # Create new PadOp for this input
            orig_shape = dg.get_output_shape(in_op, combined_edge.src_out_idx)
            # Only pad if the input has the dimension
            assert 0 <= shifted_dim < len(orig_shape), f"{shifted_dim=} {orig_shape=}"
            new_pad_op = replace(
                pad_op,
                op_id=dg.get_next_op_id(),
                dim=shifted_dim,
            )
            old_pad_op_data = dg.ops_by_id[pad_op.op_id]
            new_pad_op_data = replace(
                old_pad_op_data,
                op=new_pad_op,
                # NOTE: no change to shape if elementwise and pad takes all slices to its input expr
                # output_shapes={OpOutId(0): new_pad_op.infer_output_shapes((orig_shape,))[0]},
                output_dtypes={OpOutId(0): dg.get_output_dtypes(in_op)[combined_edge.src_out_idx]},
            )
            dg.insert_op(new_pad_op_data)
            # Connect new PadOp to the input
            dg.add_edge(
                new_pad_op,
                in_op,
                DependencyData(
                    combined_edge.expr,
                    combined_edge.src_out_idx,
                    OpInId(0),
                    combined_edge.cond,
                ),
            )
            new_inputs.append((new_pad_op, OpOutId(0)))
        # Create new elementwise op with new inputs
        # Get input shapes and dtypes
        new_input_shapes = [dg.get_output_shape(op_, out_id) for op_, out_id in new_inputs]
        new_input_dtypes = [dg.get_output_dtypes(op_)[out_id] for op_, out_id in new_inputs]
        new_elem_op = replace(elem_op, op_id=dg.get_next_op_id())
        new_elem_op_data = OpData(
            op=new_elem_op,
            output_shapes={OpOutId(0): new_elem_op.infer_output_shapes(tuple(new_input_shapes))[0]},
            output_dtypes={OpOutId(0): new_elem_op.infer_output_dtypes(tuple(new_input_dtypes))[0]},
        )
        dg.insert_op(new_elem_op_data)
        # Connect new elementwise op to its inputs
        for idx, (op_, out_id) in enumerate(new_inputs):
            dg.add_edge(
                new_elem_op,
                op_,
                DependencyData(
                    new_elem_op.domain.basis_expr,
                    out_id,
                    OpInId(idx),
                ),
            )
        # Move dependents from the original PadOp to the new elementwise op
        dg.move_dependents(op, new_elem_op)


class MatMulToMulOptimization(MatchReplacer):
    """Optimization: matmul(x, y) -> mul(x, y) when x is (*, N, 1) and y is (*, 1, M)"""

    @dataclass
    class Result:
        dep1_op: top.TensorOp
        dep1_data: DependencyData
        dep2_op: top.TensorOp
        dep2_data: DependencyData

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.MatMulOp):
            return None

        dg = ctx.dg
        dependencies = list(dg.get_flat_direct_dependencies(op))
        if len(dependencies) != 2:
            return None

        dep1_op, dep1_data = dependencies[0]
        dep2_op, dep2_data = dependencies[1]

        input_shape_left = dg.get_input_shape(op, OpInId(0))
        input_shape_right = dg.get_input_shape(op, OpInId(1))

        # Check if the matmul is (*, N, 1) @ (*, 1, M)
        # which can be optimized to (*, N, 1) * (*, 1, M)
        if ie.struct_eq(
            ie.lift_to_int_ie(input_shape_left.at(-1)).partial_eval(dg.static_bounds),
            ie.lift_to_int_ie(input_shape_right.at(-2)).partial_eval(dg.static_bounds),
            1,
        ):
            return MatMulToMulOptimization.Result(
                dep1_op=dep1_op,
                dep1_data=dep1_data,
                dep2_op=dep2_op,
                dep2_data=dep2_data,
            )

        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.MatMulOp), f"Expected MatMulOp, got {type(op)}"
        dg = ctx.dg

        # Get the symbolic tensors for both operands
        dep1_symb_t = get_symbolic_tensor_for_op_output(
            dg, mr.dep1_op, mr.dep1_data.src_out_idx
        ).symbolic_index(mr.dep1_data.expr)
        dep2_symb_t = get_symbolic_tensor_for_op_output(
            dg, mr.dep2_op, mr.dep2_data.src_out_idx
        ).symbolic_index(mr.dep2_data.expr)

        # Replace matmul with element-wise multiplication
        bcast_shape = Shape.broadcast(dep1_symb_t.shape, dep2_symb_t.shape)
        dep1_symb_t = dep1_symb_t.broadcast_to_shape(bcast_shape)
        dep2_symb_t = dep2_symb_t.broadcast_to_shape(bcast_shape)
        result = dep1_symb_t.mul(dep2_symb_t)

        # Move dependents to the new operation
        dg.move_dependents(op, result.op)


# TODO: Is this not the same as the pushdown optimization?
class UnaryElementwiseMovementOptimization(MatchReplacer):
    """Optimization: unary_op(movement_op(x)) -> movement_op(unary_op(x)) when possible"""

    @dataclass
    class Result:
        mov_op: top.TensorOp
        mov_data: DependencyData
        mov_dep_op: top.TensorOp
        mov_dep_data: DependencyData

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        assert isinstance(op, top.UnaryElementWiseOp), (
            f"Expected UnaryElementWiseOp, got {type(op)}"
        )

        # Skip certain operations that shouldn't be pushed down
        if isinstance(op, top.CastOp):
            return None

        dg = ctx.dg
        dependencies = list(dg.get_flat_direct_dependencies(op))
        if len(dependencies) != 1:
            return None

        dep_op, dep_data = dependencies[0]

        # Check if the dependency is a movement operation (but not CatOp)
        if isinstance(dep_op, top.MovementOp) and not isinstance(dep_op, top.CatOp):
            # Get the dependency of the movement op
            mov_deps = list(dg.get_flat_direct_dependencies(dep_op))
            if len(mov_deps) == 1:
                mov_dep_op, mov_dep_data = mov_deps[0]

                return UnaryElementwiseMovementOptimization.Result(
                    mov_op=dep_op,
                    mov_data=dep_data,
                    mov_dep_op=mov_dep_op,
                    mov_dep_data=mov_dep_data,
                )

        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.UnaryElementWiseOp), (
            f"Expected UnaryElementWiseOp, got {type(op)}"
        )
        dg = ctx.dg

        # Combine the edges through the unary op and movement op
        ops_ordered = [op, mr.mov_op, mr.mov_dep_op]
        edges_ordered = [mr.mov_data, mr.mov_dep_data]
        new_dep_data = isl_utils.combine_many_edges(
            ops_ordered, edges_ordered, dg.static_bounds, ctx.analysis_ctx.isl_ctx
        )

        # Create new unary operation
        new_unary_op = replace(op, op_id=dg.get_next_op_id())

        # Create the operation data
        new_op_data = OpData(
            op=new_unary_op,
            output_shapes={
                OpOutId(o): s.try_resolve(dg.static_bounds)
                for o, s in enumerate(
                    new_unary_op.infer_output_shapes(
                        (dg.get_output_shape(mr.mov_dep_op, OpOutId(0)),)
                    )
                )
            },
            output_dtypes={**dg.ops_by_id[op.op_id].output_dtypes},
        )
        dg.insert_op(new_op_data)

        # Connect the new unary op to the movement dependency
        dg.add_edge(
            new_unary_op,
            mr.mov_dep_op,
            DependencyData(new_dep_data.expr, new_dep_data.src_out_idx, OpInId(0), None),
        )

        # Move dependents from the original op to the movement op
        dg.move_dependents(op, mr.mov_op)

        # Connect the movement op to the new unary op
        dg.add_edge(
            mr.mov_op,
            new_unary_op,
            DependencyData(new_unary_op.domain.basis_expr, OpOutId(0), OpInId(0), None),
        )

        # Update the movement op's output shapes and dtypes
        dg.ops_by_id[mr.mov_op.op_id] = OpData(
            mr.mov_op,
            {
                OpOutId(i): s
                for i, s in enumerate(
                    mr.mov_op.infer_output_shapes(tuple(dg.get_input_shapes_list(mr.mov_op)))
                )
            },
            {
                OpOutId(i): d
                for i, d in enumerate(
                    mr.mov_op.infer_output_dtypes(
                        (dg.ops_by_id[new_unary_op.op_id].output_dtypes[OpOutId(0)],)
                    )
                )
            },
        )

        # Update the original op's output shapes and dtypes
        dg.ops_by_id[op.op_id] = OpData(
            op,
            {
                OpOutId(i): s
                for i, s in enumerate(op.infer_output_shapes(tuple(dg.get_input_shapes_list(op))))
            },
            {
                OpOutId(i): d
                for i, d in enumerate(
                    op.infer_output_dtypes(
                        (dg.ops_by_id[mr.mov_op.op_id].output_dtypes[OpOutId(0)],)
                    )
                )
            },
        )


class UnaryPushdownOptimization(MatchReplacer):
    """Optimization:
    unary_op(x) -> unary_op(x) with symbolic pushdown when possible
    y --e--> unary --basis--> x => y --basis--> unary --e--> x

    Push-up would be the opposite:
    y --basis--> unary --e--> x => y --e--> unary --basis--> x
    #TODO: make a wrapper, which checks for match, but then only allows the match
    if it leads to better grouping results?

    """

    @dataclass
    class Result:
        un_depy_op: top.TensorOp
        un_depy_data: DependencyData
        un_dept: top.TensorOp
        un_dept_data: DependencyData
        push_dir: PushDir

    def match(
        self,
        ctx: CompilationCtx,
        un_op: top.TensorOp,
        match_only_dirs: Iterable[PushDir] | None = None,
    ) -> Result | None:
        if match_only_dirs is None:
            match_only_dirs = [PushDir.DOWN]  # , PushDir.UP

        assert isinstance(un_op, (top.UnaryElementWiseOp)), (
            f"Expected UnaryElementWiseOp, got {type(un_op)}"
        )

        # Skip certain operations that shouldn't be pushed down
        if isinstance(un_op, top.CastOp):
            return None

        dg = ctx.dg
        un_depy_op, un_depy_data = dg.get_flat_direct_dependencies(un_op)[0]

        ## Skip ValToValOp as it's used for masking
        # if isinstance(un_depy_op, top.ValToValOp):
        #    return None
        assert not isinstance(un_depy_op, top.ValToValOp), (
            "Should we just remove ValToValOp? It's really a where op."
        )

        # Check if the dependency has unconditional basis
        if un_depy_data.is_unconditional_basis():
            # Check if any dependents have non-unconditional basis
            for un_dept, un_dept_data in dg.get_flat_direct_dependents(un_op):
                if not un_dept_data.is_unconditional_basis() and len(un_dept.domain) < len(
                    un_op.domain
                ):
                    if PushDir.DOWN in match_only_dirs:
                        return UnaryPushdownOptimization.Result(
                            un_depy_op=un_depy_op,
                            un_depy_data=un_depy_data,
                            un_dept=un_dept,
                            un_dept_data=un_dept_data,
                            push_dir=PushDir.DOWN,
                        )

        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.UnaryElementWiseOp), (
            f"Expected UnaryElementWiseOp, got {type(op)}"
        )
        dg = ctx.dg

        # TODO: this also needs to be adapted for push-up

        # Get the symbolic tensor for the dependency
        un_depy_t = get_symbolic_tensor_for_op_output(
            dg, mr.un_depy_op, mr.un_depy_data.src_out_idx
        ).symbolic_index(mr.un_dept_data.expr)

        # Create a new unary operation with the same type as the original
        new_un_op = replace(op, domain=un_depy_t.domain.copy(), op_id=dg.get_next_op_id())

        # Create the operation data
        new_op_data = OpData(
            op=new_un_op,
            output_shapes={
                OpOutId(o): s.try_resolve(dg.static_bounds)
                for o, s in enumerate(new_un_op.infer_output_shapes((un_depy_t.shape,)))
            },
            output_dtypes={**dg.ops_by_id[op.op_id].output_dtypes},
        )
        dg.insert_op(new_op_data)

        # Connect the new op to its dependency
        dg.add_edge(
            new_un_op,
            mr.un_depy_op,
            DependencyData(mr.un_dept_data.expr, mr.un_dept_data.src_out_idx, OpInId(0), None),
        )

        # Connect the dependent to the new op
        dg.add_edge(
            mr.un_dept,
            new_un_op,
            DependencyData(
                new_un_op.domain.basis_expr,
                OpOutId(0),
                mr.un_dept_data.sink_in_idx,
                mr.un_dept_data.cond,
            ),
        )
        dg.remove_edge(mr.un_dept, op, mr.un_dept_data)


class BinEWMovPushdownOptimization(MatchReplacer):
    """Optimization: binary_op(mov_op(x), mov_op(y)) -> mov_op(binary_op(x, y)) when possible"""

    @dataclass
    class Result:
        bin_op: top.TensorOp
        dep1_op: top.TensorOp
        dep1_data: DependencyData
        dep2_op: top.TensorOp
        dep2_data: DependencyData
        mov1_depy_op: top.TensorOp
        mov1_depy_data: DependencyData
        mov2_depy_op: top.TensorOp
        mov2_depy_data: DependencyData

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        assert isinstance(op, (top.BinaryElementWiseOp, top.MatMulOp)), (
            f"Expected BinaryElementWiseOp or MatMulOp, got {type(op)}"
        )

        dg = ctx.dg
        (depy1_op, depy1_data), (depy2_op, depy2_data) = dg.get_flat_direct_dependencies(op)

        def depy_is_okay(op: top.TensorOp, data: DependencyData) -> bool:
            return (
                isinstance(op, top.MovementOp)
                and not isinstance(op, (top.CatOp, top.IndexSliceOp))
                and data.is_unconditional_basis()
                and op.is_static()
            )

        # Check if both dependencies are the same movement operation
        if not (
            depy1_op.equivalent(depy2_op)
            and depy_is_okay(depy1_op, depy1_data)
            and depy_is_okay(depy2_op, depy2_data)
        ):
            return None

        # For MatMulOp, check that movement only affects batch dims
        if isinstance(op, top.MatMulOp):
            input_shape_list = dg.get_input_shapes_list(op)
            input_shape = input_shape_list[0]
            len_input_shape = len(input_shape)
            assert isinstance(depy1_op, top.MovementOp)  # NOTE: mypy
            dims_affected = depy1_op.dims_affected(input_shapes=tuple(input_shape_list))
            if not all(d < len_input_shape - 2 for d in dims_affected):
                return None

        # Get the dependencies of the movement operations
        mov1_depys = list(dg.get_flat_direct_dependencies(depy1_op))
        mov2_depys = list(dg.get_flat_direct_dependencies(depy2_op))

        if not len(mov1_depys) == 1 and len(mov2_depys) == 1:
            return None

        mov1_depy_op, mov1_depy_data = mov1_depys[0]
        mov2_depy_op, mov2_depy_data = mov2_depys[0]

        # NOTE: if doing the movement would break broadcasting assumptions, we cant do it.
        if isinstance(op, top.ElementWiseOp):
            mov1_depy_symb_t = get_symbolic_tensor_for_op_output(
                dg, mov1_depy_op, mov1_depy_data.src_out_idx
            ).symbolic_index(mov1_depy_data.expr)
            mov2_depy_symb_t = get_symbolic_tensor_for_op_output(
                dg, mov2_depy_op, mov2_depy_data.src_out_idx
            ).symbolic_index(mov2_depy_data.expr)
            if not mov1_depy_symb_t.shape == mov2_depy_symb_t.shape:
                return None

        return BinEWMovPushdownOptimization.Result(
            bin_op=op,
            dep1_op=depy1_op,
            dep1_data=depy1_data,
            dep2_op=depy2_op,
            dep2_data=depy2_data,
            mov1_depy_op=mov1_depy_op,
            mov1_depy_data=mov1_depy_data,
            mov2_depy_op=mov2_depy_op,
            mov2_depy_data=mov2_depy_data,
        )

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.BinaryElementWiseOp), (
            f"Expected BinaryElementWiseOp, got {type(op)}"
        )
        dg = ctx.dg

        # Combine the edges through the movement operations
        combined_edge_1 = isl_utils.combine_edges(
            op,
            mr.dep1_data,
            mr.dep1_op,
            mr.mov1_depy_data,
            mr.mov1_depy_op,
            dg.static_bounds,
            ctx.analysis_ctx.isl_ctx,
        )
        combined_edge_2 = isl_utils.combine_edges(
            op,
            mr.dep2_data,
            mr.dep2_op,
            mr.mov2_depy_data,
            mr.mov2_depy_op,
            dg.static_bounds,
            ctx.analysis_ctx.isl_ctx,
        )

        # Get symbolic tensors for the inner dependencies
        mov1_depy_symb_t = get_symbolic_tensor_for_op_output(
            dg, mr.mov1_depy_op, combined_edge_1.src_out_idx
        ).symbolic_index(combined_edge_1.expr)
        mov2_depy_symb_t = get_symbolic_tensor_for_op_output(
            dg, mr.mov2_depy_op, combined_edge_2.src_out_idx
        ).symbolic_index(combined_edge_2.expr)

        if isinstance(op, top.ElementWiseOp) and mov1_depy_symb_t.shape != mov2_depy_symb_t.shape:
            return  # NOTE: Hack bailout for now

        # Create new input shapes
        new_in_shapes = (
            mov1_depy_symb_t.shape.try_resolve(dg.static_bounds),
            mov2_depy_symb_t.shape.try_resolve(dg.static_bounds),
        )
        prev_op_data = dg.ops_by_id[op.op_id]

        # Create new binary operation
        new_bin_op = replace(
            op,
            op_id=dg.get_next_op_id(),
            domain=Domain.union(mov1_depy_symb_t.domain, mov2_depy_symb_t.domain),
        )
        new_op_data = OpData(
            op=new_bin_op,
            output_shapes={
                OpOutId(o): s.try_resolve(dg.static_bounds)
                for o, s in enumerate(new_bin_op.infer_output_shapes(new_in_shapes))
            },
            output_dtypes={**prev_op_data.output_dtypes},
        )
        dg.insert_op(new_op_data)

        # Create new movement operation (copy of the first movement op)
        new_mov_op_domain = Domain.union(new_bin_op.domain, mr.dep1_op.vars_used())
        new_mov_op = replace(mr.dep1_op, op_id=dg.get_next_op_id(), domain=new_mov_op_domain)
        new_mov_op_data = OpData(
            op=new_mov_op,
            output_shapes={
                OpOutId(o): s.try_resolve(dg.static_bounds)
                for o, s in enumerate(
                    new_mov_op.infer_output_shapes((new_op_data.output_shapes[OpOutId(0)],))
                )
            },
            output_dtypes={**prev_op_data.output_dtypes},
        )
        dg.insert_op(new_mov_op_data)

        # Move old op dependents to new movement op
        dg.move_dependents(op, new_mov_op)

        # Connect new movement op to new binary op
        dg.add_edge(
            new_mov_op,
            new_bin_op,
            DependencyData(new_bin_op.domain.basis_expr, OpOutId(0), OpInId(0), None),
        )

        # Connect new binary op to inner dependencies
        dg.add_edge(
            new_bin_op,
            mr.mov1_depy_op,
            DependencyData(combined_edge_1.expr, combined_edge_1.src_out_idx, OpInId(0), None),
        )
        dg.add_edge(
            new_bin_op,
            mr.mov2_depy_op,
            DependencyData(combined_edge_2.expr, combined_edge_2.src_out_idx, OpInId(1), None),
        )


class SumExpandToMulOptimization(MatchReplacer):
    """Optimization: sum(expand(x, sizes), dim) -> x * size when summing over expanded dimension"""

    @dataclass
    class Result:
        expand_op: top.TensorOp
        expand_data: DependencyData
        expand_dep_op: top.TensorOp
        expand_dep_data: DependencyData
        sum_dim: int
        new_size: ie.IndexExpr

    def match(self, ctx: CompilationCtx, red_op: top.TensorOp) -> Result | None:
        if not isinstance(red_op, top.SumOp):
            return None

        dg = ctx.dg
        ((expand_op, expand_data),) = dg.get_flat_direct_dependencies(red_op)

        # Check if the dependency is an ExpandOp and we're summing over one dimension
        if not (
            isinstance(expand_op, top.ExpandOp)
            and len(red_op.dims)
            == 1  # TODO: this is not needed. We could reinsert a sum for remaining dims.
            and expand_data.is_unconditional_basis()  # TODO: neither is this one.
        ):
            return None

        sum_dim = red_op.dims[0]
        expanded_shape = expand_op.sizes

        # Check if we are summing an expanded dimension
        expanded_dims = expand_op.dims_affected((dg.get_input_shape(expand_op, OpInId(0)),))
        if sum_dim not in expanded_dims:
            return None

        # TODO: to support this we need to reinsert the expand for remaining dims.
        if len(expanded_dims) != 1:
            return None

        # Get the dependency of the expand op
        ((expand_dep_op, expand_dep_data),) = dg.get_flat_direct_dependencies(expand_op)

        return SumExpandToMulOptimization.Result(
            expand_op=expand_op,
            expand_data=expand_data,
            expand_dep_op=expand_dep_op,
            expand_dep_data=expand_dep_data,
            sum_dim=sum_dim,
            new_size=ie.lift_to_int_ie(expanded_shape.at(sum_dim)),
        )

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.SumOp), f"Expected SumOp, got {type(op)}"
        dg = ctx.dg

        # Get the symbolic tensor for the expand dependency
        symb_t = get_symbolic_tensor_for_op_output(
            dg, mr.expand_dep_op, mr.expand_dep_data.src_out_idx
        ).symbolic_index(mr.expand_dep_data.expr)

        # Multiply by the expanded size
        symb_t = symb_t.mul(lift_to_symbolic_tensor(mr.new_size))

        # Move dependents to the result
        dg.move_dependents(op, symb_t.op)


class SumSumOptimization(MatchReplacer):
    """Optimization: sum(sum(x, dims1), dims2) -> sum(x, combined_dims) when possible."""

    @dataclass
    class Result:
        outer_sum_op: top.SumOp
        outer_sum_data: DependencyData
        inner_sum_op: top.SumOp
        sum_sum_depy_data: DependencyData
        sum_sum_depy_op: top.TensorOp

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.SumOp):
            return None

        dg = ctx.dg

        ((depy_op, dep_data),) = dg.get_flat_direct_dependencies(op)

        # NOTE: we may want to avoid undoing the work of incrementalization.
        is_inc_related = any("di" in v.name for v in op.domain.variables) or any(
            "di" in v.name for v in depy_op.domain.variables
        )
        # Check if the dependency is a SumOp with only one dependent
        if (
            isinstance(depy_op, top.SumOp) and not is_inc_related
        ):  # and dep_op.keepdim == op.keepdim:
            # Get the dependency of the inner sum op
            ((sum_sum_depy_op, sum_sum_depy_data),) = dg.get_flat_direct_dependencies(depy_op)

            return SumSumOptimization.Result(
                outer_sum_op=op,
                outer_sum_data=dep_data,
                inner_sum_op=depy_op,
                sum_sum_depy_data=sum_sum_depy_data,
                sum_sum_depy_op=sum_sum_depy_op,
            )

        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.SumOp), f"Expected SumOp, got {type(op)}"
        dg = ctx.dg

        # # ASCII art for the replacement
        # print("[SumSumOptimization] Match and Replacement:")
        # print(f"Sum({op.dims}) --{mr.outer_sum_data.expr}--> Sum({mr.inner_sum_op.dims}) \
        #       --{mr.inner_sum_data.expr}--> {mr.inner_op}")

        # Build the combined expression from the dependencies
        combined_data = isl_utils.combine_edges(
            op,
            mr.outer_sum_data,
            mr.inner_sum_op,
            mr.sum_sum_depy_data,
            mr.sum_sum_depy_op,
            dg.static_bounds,
            ctx.analysis_ctx.isl_ctx,
        )

        # Get the symbolic tensor for the inner operation with the combined expression
        inner_symb_t = get_symbolic_tensor_for_op_output(
            dg, mr.sum_sum_depy_op, combined_data.src_out_idx
        ).symbolic_index(combined_data.expr)
        # combined_shape = inner_symb_t.shape

        # The key insight is that we need to map the dimensions from the original sum operations
        # to the dimensions in the combined tensor. The combined tensor's shape includes:
        # 1. Spatial dimensions introduced by the combined indexing expression
        # 2. The original spatial shape of the inner tensor

        # inner_num_temporal_dims_introduced = inner_sum_data.expr.num_slices()
        outer_num_temporal_dims_introduced = mr.outer_sum_data.expr.num_slices()

        # For each spatialized dimension,
        # determine if it come from the inner or outer sum op symbolic indexing.
        inner_temporal_dims_reduced: list[int] = []
        outer_temporal_dims_reduced: list[int] = []
        dim_idx = 0
        for i, var in enumerate(mr.sum_sum_depy_op.domain.variables):
            if not mr.sum_sum_depy_data.expr.members[i].is_point():
                inner_temporal_dims_reduced.append(dim_idx)
                dim_idx += 1
            elif not mr.outer_sum_data.expr.members[
                mr.inner_sum_op.domain.find_variable_index(var)
            ].is_point():
                outer_temporal_dims_reduced.append(dim_idx)
                dim_idx += 1

        inner_spatial_dims_reduced = {
            d + outer_num_temporal_dims_introduced
            for d in sorted(mr.inner_sum_op.dims)[len(inner_temporal_dims_reduced) :]
        } - set(inner_temporal_dims_reduced)
        inner_dims_mapped = set(inner_temporal_dims_reduced) | inner_spatial_dims_reduced

        # TODO: I'm fairly sure everything above is correct. If anything is wrong it is the
        # function below, which is supposed to return the offset number of inner spatial dims
        # reduced before the outer spatial dim in the combined sum.

        def num_before(outer_spatial_dim: int) -> int:
            shift: int = 0
            for inner_spatial_dim_reduced in inner_spatial_dims_reduced:
                # Current target index of `dim` in the combined tensor,
                # taking into account the inner temporal dims *and* any
                # already-counted inner spatial dims.
                target: int = len(inner_temporal_dims_reduced) + outer_spatial_dim + shift
                if inner_spatial_dim_reduced <= target:
                    shift += 1
                else:
                    # remaining inner-removed dims are to the right – we're done
                    break
            return shift

        outer_spatial_dims_reduced = {
            d + len(inner_temporal_dims_reduced) + num_before(d)
            for d in sorted(mr.outer_sum_op.dims)[len(outer_temporal_dims_reduced) :]
        } - set(outer_temporal_dims_reduced)
        outer_dims_mapped = set(outer_temporal_dims_reduced) | outer_spatial_dims_reduced

        combined_dims = tuple(sorted(inner_dims_mapped | outer_dims_mapped))

        assert len(combined_dims) == len(set(combined_dims)), "Duplicate dimensions in sum_sum"

        # Create the single sum operation
        if mr.outer_sum_op.keepdim == mr.inner_sum_op.keepdim:
            # NOTE: if both agree on keepdim, we can just sum the combined dims and
            # use the consensus keepdim
            summed_t = inner_symb_t.sum(combined_dims, keepdim=mr.outer_sum_op.keepdim)
            # print(f"Sum({summed_t.op.dims}) --{inner_symb_t.index_expr}--> {mr.inner_op}\n")
        else:
            # NOTE: if they disagree on keepdim, we need to sum the combined dims, keeping
            # all dims, then squeeze out the dims reduced by one of the sums.
            summed_t = inner_symb_t.sum(combined_dims, keepdim=True)
            # print(f"Sum({summed_t.op.dims}) --{inner_symb_t.index_expr}--> {mr.inner_op}\n")
            if not mr.outer_sum_op.keepdim:  # NOTE: Do not keep outer dims
                for d in sorted(outer_dims_mapped, reverse=True):
                    summed_t = summed_t.squeeze(d)
            else:  # NOTE: Do not keep inner dims
                for d in sorted(inner_dims_mapped, reverse=True):
                    summed_t = summed_t.squeeze(d)

        # Move dependents to the new combined sum operation
        dg.move_dependents(op, summed_t.op)


class MatMulReassocOptimization(MatchReplacer):
    """
    Optimization: (A @ B) @ C => A @ (B @ C) when FLOP cost is lower
    """

    @dataclass
    class Result:
        direction: MatMulReassocDir
        A_op: top.TensorOp
        A_data: DependencyData
        B_op: top.TensorOp
        B_data: DependencyData
        C_op: top.TensorOp
        C_data: DependencyData

    def match(
        self, ctx: CompilationCtx, op: top.TensorOp
    ) -> MatMulReassocOptimization.Result | None:
        if not isinstance(op, top.MatMulOp):
            return None

        dg = ctx.dg
        outer_depys = list(dg.get_flat_direct_dependencies(op))

        # NOTE: In case both depys are other matmuls, we test both cases.
        for direction, inner_idx in ((MatMulReassocDir.LEFT, 0), (MatMulReassocDir.RIGHT, 1)):
            inner_op, inner_data = outer_depys[inner_idx]
            other_op, other_data = outer_depys[1 - inner_idx]

            # structural sanity
            if not (
                isinstance(inner_op, top.MatMulOp)
                # NOTE: I'm leaving this here because unlike other optims,
                # if there is more than one dependent,
                # we definately do not want to duplicate the matmul work.
                # Though it is possible that in some cases, both dependents would end-up
                # doing the optimization.
                and len(dg.get_flat_direct_dependents(inner_op)) == 1
            ):
                continue

            inner_depys = list(dg.get_flat_direct_dependencies(inner_op))
            # ------------------------ role assignment ------------------------
            if direction == MatMulReassocDir.LEFT:  # (A @ B) @ C
                (A_op, A_data), (B_op, B_data) = inner_depys
                C_op, C_data = other_op, other_data

                shape_A = dg.get_input_shape(inner_op, OpInId(0))  # m × n
                shape_B = dg.get_input_shape(inner_op, OpInId(1))  # n × p
                shape_C = dg.get_input_shape(op, OpInId(1))  # p × q

                orig_cost = lambda m, n, p, q: m * n * p + m * p * q
                new_cost = lambda m, n, p, q: n * p * q + m * n * q
            else:  # A @ (B @ C)
                A_op, A_data = other_op, other_data
                (B_op, B_data), (C_op, C_data) = inner_depys

                shape_A = dg.get_input_shape(op, OpInId(0))  # m × n
                shape_B = dg.get_input_shape(inner_op, OpInId(0))  # n × p
                shape_C = dg.get_input_shape(inner_op, OpInId(1))  # p × q

                orig_cost = lambda m, n, p, q: m * n * q + n * p * q
                new_cost = lambda m, n, p, q: m * n * p + m * p * q
            # -----------------------------------------------------------------

            m, n = shape_A.at(-2), shape_A.at(-1)
            p, q = shape_B.at(-1), shape_C.at(-1)

            original_cost = orig_cost(m, n, p, q)  # type: ignore
            reassoc_cost = new_cost(m, n, p, q)  # type: ignore
            cheaper = reassoc_cost < original_cost

            dom = A_op.domain.union(B_op.domain).union(C_op.domain)
            known = dict(dg.static_bounds)
            if not isinstance(cheaper, bool):
                # TODO: Give match replacers access to the isl ctx
                cheaper = isl_utils.simplify_boolean_index_expr(dom, cheaper, known_symbols=known)
                try:
                    cheaper_bool = cheaper.evaluate(known)
                    # log.info("Evaluated %s to %s", cheaper, cheaper_bool)
                except Exception:
                    log.error("Failed to evaluate %s to a bool", cheaper)
                    cheaper_bool = False
            else:
                cheaper_bool = cheaper

            if cheaper_bool:
                return MatMulReassocOptimization.Result(
                    direction=direction,
                    A_op=A_op,
                    A_data=A_data,
                    B_op=B_op,
                    B_data=B_data,
                    C_op=C_op,
                    C_data=C_data,
                )

        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.MatMulOp), f"Expected MatMulOp, got {type(op)}"
        dg = ctx.dg

        # Get symbolic tensors for A, B, and C
        A_symb = get_symbolic_tensor_for_op_output(
            dg, mr.A_op, mr.A_data.src_out_idx
        ).symbolic_index(mr.A_data.expr)
        B_symb = get_symbolic_tensor_for_op_output(
            dg, mr.B_op, mr.B_data.src_out_idx
        ).symbolic_index(mr.B_data.expr)
        C_symb = get_symbolic_tensor_for_op_output(
            dg, mr.C_op, mr.C_data.src_out_idx
        ).symbolic_index(mr.C_data.expr)

        if mr.direction == MatMulReassocDir.LEFT:
            # (A @ B) @ C -> A @ (B @ C)
            new_inner = B_symb.matmul(C_symb)
            new_outer = A_symb.matmul(new_inner)
        else:
            # A @ (B @ C) -> (A @ B) @ C
            new_inner = A_symb.matmul(B_symb)
            new_outer = new_inner.matmul(C_symb)

        # Move dependents to the new matmul operation
        dg.move_dependents(op, new_outer.op)


class MatMulPermuteOptimization(MatchReplacer):
    """Optimization: matmul(permute(x, dims), permute(y, dims)) -> permute(matmul(x, y), dims)"""

    @dataclass
    class Result:
        perm1_op: top.PermuteOp
        perm1_data: DependencyData
        perm2_op: top.PermuteOp
        perm2_data: DependencyData
        inner1_op: top.TensorOp
        inner1_data: DependencyData
        inner2_op: top.TensorOp
        inner2_data: DependencyData

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.MatMulOp):
            return None

        dg = ctx.dg
        dependencies = list(dg.get_flat_direct_dependencies(op))
        if len(dependencies) != 2:
            return None

        dep1_op, dep1_data = dependencies[0]
        dep2_op, dep2_data = dependencies[1]

        # Check if both dependencies are PermuteOps with the same dimensions
        if (
            isinstance(dep1_op, top.PermuteOp)
            and isinstance(dep2_op, top.PermuteOp)
            and dep1_data.is_unconditional_basis()  # TODO: is this needed?
            and dep2_data.is_unconditional_basis()  # TODO: is this needed?
            and dep1_op.dims == dep2_op.dims
        ):
            # Get the dependencies of the permute ops
            ((inner1_op, inner1_data),) = list(dg.get_flat_direct_dependencies(dep1_op))
            ((inner2_op, inner2_data),) = list(dg.get_flat_direct_dependencies(dep2_op))

            return MatMulPermuteOptimization.Result(
                perm1_op=dep1_op,
                perm1_data=dep1_data,
                perm2_op=dep2_op,
                perm2_data=dep2_data,
                inner1_op=inner1_op,
                inner1_data=inner1_data,
                inner2_op=inner2_op,
                inner2_data=inner2_data,
            )

        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.MatMulOp), f"Expected MatMulOp, got {type(op)}"
        dg = ctx.dg

        # Get symbolic tensors for the inner operations
        inner1_symb_t = get_symbolic_tensor_for_op_output(
            dg, mr.inner1_op, mr.inner1_data.src_out_idx
        ).symbolic_index(mr.inner1_data.expr)
        inner2_symb_t = get_symbolic_tensor_for_op_output(
            dg, mr.inner2_op, mr.inner2_data.src_out_idx
        ).symbolic_index(mr.inner2_data.expr)

        # Do the matmul first
        matmul = inner1_symb_t.matmul(inner2_symb_t)
        # Then permute the result
        permuted = matmul.permute(mr.perm1_op.dims)

        # Move dependents to the permuted result
        dg.move_dependents(op, permuted.op)


class MatMulConstOptimization(MatchReplacer):
    """Optimization: matmul(const, x) -> sum(x, dim) * const when const is uniform"""

    @dataclass
    class Result:
        const_op: top.ConstOp
        const_data: DependencyData
        other_op: top.TensorOp
        other_data: DependencyData
        const_is_lhs: bool

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.MatMulOp):
            return None

        dg = ctx.dg
        ((dep1_op, dep1_data), (dep2_op, dep2_data)) = dg.get_flat_direct_dependencies(op)

        # Check if one of the inputs is a uniform constant
        if isinstance(dep1_op, top.ConstOp) and dep1_op.is_uniform:
            return MatMulConstOptimization.Result(
                const_op=dep1_op,
                const_data=dep1_data,
                other_op=dep2_op,
                other_data=dep2_data,
                const_is_lhs=True,
            )
        elif isinstance(dep2_op, top.ConstOp) and dep2_op.is_uniform:
            return MatMulConstOptimization.Result(
                const_op=dep2_op,
                const_data=dep2_data,
                other_op=dep1_op,
                other_data=dep1_data,
                const_is_lhs=False,
            )

        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.MatMulOp), f"Expected MatMulOp, got {type(op)}"
        dg = ctx.dg

        # Get symbolic tensors
        const_symb_t = get_symbolic_tensor_for_op_output(
            dg, mr.const_op, mr.const_data.src_out_idx
        ).symbolic_index(mr.const_data.expr)
        other_symb_t = get_symbolic_tensor_for_op_output(
            dg, mr.other_op, mr.other_data.src_out_idx
        ).symbolic_index(mr.other_data.expr)
        matmul_symb_t = get_symbolic_tensor_for_op_output(dg, op, OpOutId(0))

        # Determine which dimension to sum
        if mr.const_is_lhs:
            dim_to_sum = -2
            dim_to_sum_normalized = len(other_symb_t.shape) + dim_to_sum
        else:
            dim_to_sum = -1
            dim_to_sum_normalized = len(other_symb_t.shape) + dim_to_sum

        # Sum the appropriate dimension
        summed = other_symb_t.sum((dim_to_sum_normalized,))

        # Multiply by the constant

        mulled = summed.mul(
            SymbolicTensor.full(mr.const_op.uniform_value, summed.shape, const_symb_t.dtype)
        )

        # Expand the result
        expanded = mulled.expand(matmul_symb_t.shape)

        # Move dependents to the expanded result
        dg.move_dependents(op, expanded.op)


class SumPermuteOptimization(MatchReplacer):
    """
    Optimization:
    sum(permute(x, dims), sum_dims) -> permute(sum(x, adjusted_dims), remaining_dims)
    """

    @dataclass
    class Result:
        perm_op: top.TensorOp
        perm_data: DependencyData
        perm_dep_op: top.TensorOp
        perm_dep_data: DependencyData
        skip_permute: bool
        new_sum_dims: tuple[int, ...] | None = None

    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Result | None:
        if not isinstance(op, top.SumOp):
            return None

        dg = ctx.dg
        ((dep_op, dep_data),) = dg.get_flat_direct_dependencies(op)

        # Check if the dependency is a PermuteOp
        if isinstance(dep_op, top.PermuteOp):
            # Get the dependency of the permute op
            perm_deps = list(dg.get_flat_direct_dependencies(dep_op))
            if len(perm_deps) == 1:
                perm_dep_op, perm_dep_data = perm_deps[0]

                # Check if the permute affects dimensions that are being summed
                dims_aff = dep_op.dims_affected(input_shapes=(dg.get_input_shape(op, OpInId(0)),))

                if set(dims_aff).issubset(set(op.dims)):
                    # All affected dimensions are being summed, we can skip the permute
                    return SumPermuteOptimization.Result(
                        perm_op=dep_op,
                        perm_data=dep_data,
                        perm_dep_op=perm_dep_op,
                        perm_dep_data=perm_dep_data,
                        skip_permute=True,
                    )
                else:
                    # Some affected dimensions are not being summed
                    perm_dims = list(dep_op.dims)
                    # Remove from perm_dims the dims that are summed out
                    summed_dims_ordered = sorted(op.dims, reverse=True)
                    for dim in summed_dims_ordered:
                        perm_dims.pop(dim)
                    # If the result is ordered, we can remove the permute
                    if tuple(perm_dims) == tuple(sorted(perm_dims)):
                        return SumPermuteOptimization.Result(
                            perm_op=dep_op,
                            perm_data=dep_data,
                            perm_dep_op=perm_dep_op,
                            perm_dep_data=perm_dep_data,
                            skip_permute=False,
                            new_sum_dims=tuple(dep_op.dims.index(d) for d in op.dims),
                        )

        return None

    def replace(self, ctx: CompilationCtx, op: top.TensorOp, mr: Result) -> None:
        assert isinstance(op, top.SumOp), f"Expected SumOp, got {type(op)}"
        dg = ctx.dg

        if mr.skip_permute:
            # Combine the edges and remove the permute
            combined_edge = isl_utils.combine_edges(
                op,
                mr.perm_data,
                mr.perm_op,
                mr.perm_dep_data,
                mr.perm_dep_op,
                dg.static_bounds,
                ctx.analysis_ctx.isl_ctx,
            )
            dg.remove_edge(op, mr.perm_op, mr.perm_data)
            dg.add_edge(op, mr.perm_dep_op, combined_edge)
        else:
            # Combine the edges and adjust the sum dimensions
            combined_edge = isl_utils.combine_edges(
                op,
                mr.perm_data,
                mr.perm_op,
                mr.perm_dep_data,
                mr.perm_dep_op,
                dg.static_bounds,
                ctx.analysis_ctx.isl_ctx,
            )

            # Create new sum operation with adjusted dimensions
            perm_dep_symb_t = get_symbolic_tensor_for_op_output(
                dg, mr.perm_dep_op, combined_edge.src_out_idx
            ).symbolic_index(combined_edge.expr)

            assert mr.new_sum_dims is not None
            new_sum_op = perm_dep_symb_t.sum(mr.new_sum_dims).op
            dg.move_dependents(op, new_sum_op)
