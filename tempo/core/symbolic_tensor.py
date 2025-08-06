from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Type, Union

import numpy as np

from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.datatypes import DIM_TYPE, NestedList, OpInId, OpOutId, TensorId
from tempo.core.dependence_graph import PDG, DependencyData, OpData
from tempo.core.dim_utils import normalize_negative_dim
from tempo.core.domain import Domain, DomainLike
from tempo.core.dtype import DataType, DataTypeLike, dtypes
from tempo.core.global_objects import get_active_exec_cfg
from tempo.core.shape import Shape, ShapeLike
from tempo.utils import logger
from tempo.utils.common import as_seq

_logger = logger.get_logger(__name__)


def _get_symbolic_tensor_for_op_output(
    dg: PDG,
    op: top.TensorOp,
    out_idx: OpOutId,
) -> SymbolicTensor:
    shape = dg.ops_by_id[op.op_id].output_shapes[out_idx]
    dtype = dg.ops_by_id[op.op_id].output_dtypes[out_idx]
    symb_t = SymbolicTensor(
        op,
        TensorId(op.op_id, out_idx),
        shape,
        dtype,
        None,
    )

    return symb_t


def translate_ie_to_st(expr: ie.IndexExpr) -> SymbolicTensor:  # noqa: C901
    """Translates an index expression to a symbolic tensor."""
    if isinstance(expr, ie.ConstInt):
        return SymbolicTensor.full(expr.const, dtype=dtypes.default_int)
    elif isinstance(expr, ie.ConstBool):
        return SymbolicTensor.full(expr.const, dtype=dtypes.bool_)
    elif isinstance(expr, ie.Symbol):
        return SymbolicTensor.eval_symbol(expr)
    elif isinstance(expr, ie.Add):
        left, right = translate_ie_to_st(expr.left_operand), translate_ie_to_st(expr.right_operand)
        broadcast_shape = Shape.broadcast(left.shape, right.shape)
        return left.broadcast_to_shape(broadcast_shape) + right.broadcast_to_shape(broadcast_shape)
    elif isinstance(expr, ie.Sub):
        left, right = translate_ie_to_st(expr.left_operand), translate_ie_to_st(expr.right_operand)
        broadcast_shape = Shape.broadcast(left.shape, right.shape)
        return left.broadcast_to_shape(broadcast_shape) - right.broadcast_to_shape(broadcast_shape)
    elif isinstance(expr, ie.Mul):
        left, right = translate_ie_to_st(expr.left_operand), translate_ie_to_st(expr.right_operand)
        broadcast_shape = Shape.broadcast(left.shape, right.shape)
        return left.broadcast_to_shape(broadcast_shape) * right.broadcast_to_shape(broadcast_shape)
    elif isinstance(expr, ie.FloorDivision):
        left, right = translate_ie_to_st(expr.left_operand), translate_ie_to_st(expr.right_operand)
        broadcast_shape = Shape.broadcast(left.shape, right.shape)
        return left.broadcast_to_shape(broadcast_shape) // right.broadcast_to_shape(broadcast_shape)
    elif isinstance(expr, ie.Neg):
        return -translate_ie_to_st(expr.operand)
    elif isinstance(expr, ie.Equal):
        left, right = translate_ie_to_st(expr.left_operand), translate_ie_to_st(expr.right_operand)
        broadcast_shape = Shape.broadcast(left.shape, right.shape)
        bcast_left = left.broadcast_to_shape(broadcast_shape)
        bcast_right = right.broadcast_to_shape(broadcast_shape)
        return bcast_left == bcast_right  # type: ignore
    elif isinstance(expr, ie.LessThan):
        left, right = translate_ie_to_st(expr.left_operand), translate_ie_to_st(expr.right_operand)
        broadcast_shape = Shape.broadcast(left.shape, right.shape)
        return left.broadcast_to_shape(broadcast_shape) < right.broadcast_to_shape(broadcast_shape)
    elif isinstance(expr, ie.GreaterThan):
        left, right = translate_ie_to_st(expr.left_operand), translate_ie_to_st(expr.right_operand)
        broadcast_shape = Shape.broadcast(left.shape, right.shape)
        return left.broadcast_to_shape(broadcast_shape) > right.broadcast_to_shape(broadcast_shape)
    elif isinstance(expr, ie.GreaterThanOrEqual):
        left, right = translate_ie_to_st(expr.left_operand), translate_ie_to_st(expr.right_operand)
        broadcast_shape = Shape.broadcast(left.shape, right.shape)
        return left.broadcast_to_shape(broadcast_shape) >= right.broadcast_to_shape(broadcast_shape)
    elif isinstance(expr, ie.And):
        left, right = translate_ie_to_st(expr.left_operand), translate_ie_to_st(expr.right_operand)
        broadcast_shape = Shape.broadcast(left.shape, right.shape)
        return left.broadcast_to_shape(broadcast_shape) & right.broadcast_to_shape(broadcast_shape)
    elif isinstance(expr, ie.Or):
        left, right = translate_ie_to_st(expr.left_operand), translate_ie_to_st(expr.right_operand)
        broadcast_shape = Shape.broadcast(left.shape, right.shape)
        return left.broadcast_to_shape(broadcast_shape) | right.broadcast_to_shape(broadcast_shape)
    elif isinstance(expr, ie.Not):
        return ~translate_ie_to_st(expr.operand)
    elif isinstance(expr, ie.Modulos):
        left, right = translate_ie_to_st(expr.left_operand), translate_ie_to_st(expr.right_operand)
        broadcast_shape = Shape.broadcast(left.shape, right.shape)
        return left.broadcast_to_shape(broadcast_shape) % right.broadcast_to_shape(broadcast_shape)
    elif isinstance(expr, ie.Piecewise):
        conds_and_vals = list(expr.enumerate_all_cond_branches())
        else_expr = conds_and_vals[-1][1]
        assert else_expr is not None
        result = translate_ie_to_st(else_expr)
        for cond, val in reversed(conds_and_vals[:-1]):
            assert cond is not None, f"Non-final condition is None for {expr}"
            cond_t = translate_ie_to_st(cond)
            val_t = translate_ie_to_st(val)
            broadcast_shape = Shape.broadcast(cond_t.shape, val_t.shape, result.shape)
            result = SymbolicTensor.where(
                cond_t.broadcast_to_shape(broadcast_shape),
                val_t.broadcast_to_shape(broadcast_shape),
                result.broadcast_to_shape(broadcast_shape),
            )
        return result

    elif isinstance(expr, ie.Ceil):
        return translate_ie_to_st(expr.operand).ceil()
    elif isinstance(expr, ie.Floor):
        return translate_ie_to_st(expr.operand).floor()
    elif isinstance(expr, ie.Max):
        translated_operands = [translate_ie_to_st(o) for o in expr.operands]
        if not Shape.can_broadcast(*[o.shape for o in translated_operands]):
            raise ValueError(
                f"Cannot broadcast shapes {[(o.shape, o.op.op_id) for o in translated_operands]}"
            )
        broadcast_shape = Shape.broadcast(*[o.shape for o in translated_operands])
        broadcast_operands = [o.broadcast_to_shape(broadcast_shape) for o in translated_operands]
        stacked = stack(*broadcast_operands)
        return stacked.max()[0]
    elif isinstance(expr, ie.Min):
        translated_operands = [-translate_ie_to_st(o) for o in expr.operands]
        if not Shape.can_broadcast(*[o.shape for o in translated_operands]):
            raise ValueError(
                f"Cannot broadcast shapes {[(o.shape, o.op.op_id) for o in translated_operands]}"
            )
        broadcast_shape = Shape.broadcast(*[o.shape for o in translated_operands])
        broadcast_operands = [o.broadcast_to_shape(broadcast_shape) for o in translated_operands]
        stacked = stack(*broadcast_operands)
        return -stacked.max()[0]
    elif isinstance(expr, ie.IndexSequence):
        translated_members = [translate_ie_to_st(i) for i in expr.members]
        if not Shape.can_broadcast(*[o.shape for o in translated_members]):
            raise ValueError(
                f"Cannot broadcast shapes {[(o.shape, o.op.op_id) for o in translated_members]}"
            )
        broadcast_shape = Shape.broadcast(*[o.shape for o in translated_members])
        broadcast_members = [o.broadcast_to_shape(broadcast_shape) for o in translated_members]
        return stack(*broadcast_members)
    elif isinstance(expr, ie.Slice):
        # return translate_ie_to_st(expr.start) + translate_ie_to_st(expr.stop)
        return arange(expr.stop, expr.start)
    else:
        raise ValueError(f"Unsupported index expression type: {type(expr)}")


def lift_to_symbolic_tensor(inp: Any) -> SymbolicTensor:
    if isinstance(inp, (float, int, bool)):
        dtype = dtypes.implied(inp)
        return SymbolicTensor.full(inp, dtype=dtype)
    if isinstance(inp, ie.IndexExpr):
        return translate_ie_to_st(inp)
    if not isinstance(inp, SymbolicTensor):
        raise TypeError(f"Expected SymbolicTensor, got {type(inp)}")
    return inp


def lift_to_symbolic_tensors(
    inps: ManyMaybeSymbolicTensors,
) -> ManySymbolicTensors:
    return list(map(lift_to_symbolic_tensor, inps))


def _register_zero_out(
    op_builder: Type[top.TensorOp],
    inputs: Optional[OneOrManyMaybeSymbolicTensors],
    *op_args: Any,
    **op_kwargs: Any,
) -> None:
    _register_n_outs(op_builder, inputs, None, 0, *op_args, **op_kwargs)


def _register_source(
    op_builder: Type[top.TensorOp],
    domain: DomainLike,
    *op_args: Any,
    **op_kwargs: Any,
) -> SymbolicTensor:
    domain = Domain.from_(domain)
    return _register_n_outs(op_builder, None, domain, 1, *op_args, **op_kwargs)[0]


def _register_one_out(
    op_builder: Type[top.TensorOp],
    inputs: Optional[OneOrManyMaybeSymbolicTensors],
    *op_args: Any,
    **op_kwargs: Any,
) -> SymbolicTensor:
    domain: Optional[Domain] = None
    if "domain" in op_kwargs:
        domain = op_kwargs["domain"]
        del op_kwargs["domain"]
    return _register_n_outs(op_builder, inputs, domain, 1, *op_args, **op_kwargs)[0]


def _register_two_outs(
    op_builder: Type[top.TensorOp],
    inputs: Optional[OneOrManyMaybeSymbolicTensors],
    *op_args: Any,
    **op_kwargs: Any,
) -> Tuple[SymbolicTensor, SymbolicTensor]:
    domain: Optional[Domain] = None
    if "domain" in op_kwargs:
        domain = op_kwargs["domain"]
        del op_kwargs["domain"]
    r = _register_n_outs(op_builder, inputs, domain, 2, *op_args, **op_kwargs)
    return r[0], r[1]


def _register_n_outs(
    op_builder: Type[top.TensorOp],
    inputs: Optional[OneOrManyMaybeSymbolicTensors],
    domain: DomainLike,
    n_outs: int,
    *op_args: Any,
    **op_kwargs: Any,
) -> Sequence[SymbolicTensor]:
    from tempo.core.global_objects import get_active_dg

    dg = get_active_dg()

    inputs = lift_to_symbolic_tensors(as_seq(inputs))

    op, out_shapes, out_dtypes = _insert_op_in_dg(
        op_builder, inputs, domain, n_outs, *op_args, **op_kwargs
    )

    outs = tuple(
        SymbolicTensor(op, TensorId(op.op_id, OpOutId(i)), out_shapes[i], out_dtypes[i], None)
        for i in range(n_outs)
    )

    _logger.debug(
        "Registering new op %s with inputs %s, and outputs %s",
        op,
        inputs,
        outs,
    )

    # Add all edges from inputs and control
    dependencies = list(inputs)
    for i, inp in enumerate(dependencies):
        assert inp.index_expr is not None
        dep_data = DependencyData(
            expr=inp.index_expr,
            src_out_idx=inp.tensor_id.output_id,
            sink_in_idx=OpInId(i),
        )
        dg.add_edge(sink=op, src=inp.op, dependency_data=dep_data)

    return outs


def _insert_op_in_dg(
    op_builder: Type[top.TensorOp],
    inputs: Sequence[SymbolicTensor],
    domain: DomainLike,
    n_outs: int,
    *op_args: Any,
    **op_kwargs: Any,
) -> Tuple[top.TensorOp, Sequence[Shape], Sequence[DataType]]:
    from tempo.core.global_objects import get_active_dg, get_active_tags

    dg = get_active_dg()

    id_ = dg.get_next_op_id()
    # if id_ == OpId(321):
    #   raise Exception(f"inputs: {inputs}")

    if inputs is not None and len(inputs) > 0:
        domain = Domain.union(
            *[inp.domain for inp in inputs], Domain.from_(domain, none_is_empty=True)
        )
    domain = Domain.from_(domain).copy()
    assert domain is not None, f"Domain must be provided if inputs are None. {domain=}, {inputs=}"

    tags = dict(get_active_tags())
    op = op_builder(id_, domain, tags, *op_args, **op_kwargs)

    # TODO currently, this is empty at registration time, because bounds are only given when
    # user calls compile.
    sbounds = dg.static_bounds

    input_shapes = tuple(inp.shape.try_resolve(sbounds) for inp in inputs)

    # if isinstance(op, top.ElementWiseOp) and not all(
    #    is_ == input_shapes[0] for is_ in input_shapes
    # ):
    #    raise ValueError(
    #        f"Expected all input shapes to be equal, got {input_shapes}, when registering {op}"
    #    )

    out_shapes = op.infer_output_shapes(input_shapes)
    assert len(out_shapes) == n_outs, f"Expected {n_outs} output shapes, got {len(out_shapes)}"

    input_dtypes = tuple(inp.dtype for inp in inputs)
    out_dtypes = op.infer_output_dtypes(input_dtypes)
    assert len(out_dtypes) == n_outs, f"Expected {n_outs} output dtypes, got {len(out_dtypes)}"

    output_shapes = {OpOutId(o): out_shapes[o].try_resolve(sbounds) for o in range(n_outs)}
    output_dtypes = {OpOutId(o): out_dtypes[o] for o in range(n_outs)}

    op_data = OpData(op, output_shapes, output_dtypes)
    dg.insert_op(op_data)
    return op, out_shapes, out_dtypes


def marker(marker_name: Optional[str] = None) -> top.TensorOp:
    if marker_name is None:
        marker_name = "anonymous marker"
    op, _, _ = _insert_op_in_dg(top.MarkerOp, (), Domain.empty(), 0, marker_name=marker_name)
    return op


def barrier(marker_name: Optional[str] = None) -> None:
    from tempo.core.global_objects import get_active_dg

    dg = get_active_dg()

    marker_op = marker(marker_name)

    # TODO: also consider uncommitted branch conditions from merge ops
    dependend_by_merge_op_ids = set()
    for op_data in dg.ops_by_id.values():
        for _, def_tensor, _ in op_data.uncommitted_branch_conds:
            dependend_by_merge_op_ids.add(def_tensor.op_id)

    # Insert control-edges to every op in the graph
    for pre_barrier_op in dg.nodes:
        # NOTE: don't add control-edges to the marker op itself
        if pre_barrier_op.op_id == marker_op.op_id:
            continue

        dependents = dg.get_flat_direct_dependents(pre_barrier_op)
        if len(dependents) > 0:
            continue  # NOTE: we only need to add control-edges to the "frontier" ops

        if pre_barrier_op.op_id in dependend_by_merge_op_ids:
            continue  # NOTE: we only need to add control-edges to the "frontier" ops

        # NOTE: We want all domain timesteps of other ops to be executed before the marker op.
        dg.add_edge(
            sink=marker_op,
            src=pre_barrier_op,
            dependency_data=DependencyData.make_control(pre_barrier_op.domain.full_range_expr),
        )


def udf(
    udf_desc: top.UserDefinedThunkDesc,
    inputs: Sequence[MaybeSymbolicTensor],
    domain: DomainLike = None,
) -> Sequence[SymbolicTensor]:
    return _register_n_outs(top.UDFOp, inputs, domain, udf_desc.num_outputs, udf_desc)


# def reset_env(env: EnvDesc, seed: Optional[int] = None) -> SymbolicTensor:
#    if seed is None:
#        seed = random.randint(0, 2**31 - 1)
#
#    return _register_source(
#        top.ResetEnvironmentOp,
#        None,
#        env.observation_shape,
#        env.observation_dtype,
#        env,
#        seed,
#    )
#
#
# def step_env(
#    env_ref: EnvDesc, actions: MaybeSymbolicTensor
# ) -> Tuple[SymbolicTensor, SymbolicTensor, SymbolicTensor, SymbolicTensor]:
#    return _register_four_outs(top.StepEnvironmentOp, [actions], env_ref)
#


def stack(*xs: MaybeSymbolicTensor, dim: int = 0) -> SymbolicTensor:
    xs_ = lift_to_symbolic_tensors(xs)  # type: ignore
    return cat(*[x.unsqueeze(dim) for x in xs_], dim=dim)


# =============== STATIC METHODS ===============
def rand(
    shape: ShapeLike = None,
    dtype: DataTypeLike = None,
    domain: DomainLike = None,
) -> SymbolicTensor:
    shape = Shape.from_(shape)
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    if shape.vars_used():
        domain = Domain.from_(shape.vars_used())
    return _register_source(top.RandOp, domain, shape, dtype)


def full(
    val: Union[float, int, bool, np.ndarray, NestedList],
    shape: ShapeLike = None,
    dtype: DataTypeLike = None,
) -> SymbolicTensor:
    x: SymbolicTensor

    domain = Domain.empty()

    if isinstance(val, ie.IntIndexValue):
        shape = Shape.from_(shape)
        dtype = dtypes.from_(dtype, none_dtype=dtypes.default_int)
        x = translate_ie_to_st(val)
    else:
        if not isinstance(val, np.ndarray):
            # NOTE: includes nested lists, bools, ints, floats, etc
            # NOTE: We turn it into a numpy array to reuse the logic below
            val = np.array(val)

        dtype = dtypes.from_(dtype, none_dtype=dtypes.from_np(val.dtype))

        static_shape = Shape.from_(val.shape)
        if shape is None:
            shape = static_shape
        first_val = val.flat[0]

        is_small_numpy_array = val.size < 1000
        if is_small_numpy_array:
            is_uniform = np.all(val == first_val)
        else:
            is_uniform = False  # NOTE: assume not uniform if large array

        x = _register_source(top.ConstOp, domain, static_shape, dtype, val, is_uniform)

    x = x.to_dtype(dtype)

    shape = Shape.from_(shape)
    if x.shape != shape:
        assert Shape.can_broadcast(x.shape, shape), f"Can't broadcast {x.shape=}&{shape=}"
        dims_needed = len(shape) - len(x.shape)
        for _ in range(dims_needed):
            x = x.unsqueeze(0)
        x = x.expand(shape)
        assert x.shape == shape, f"Expected shapes equal after broadcasting: {x.shape=}&{shape=}"
    return x


def full_like_self(
    self: SymbolicTensor, value: Union[float, int, bool, np.ndarray, NestedList]
) -> SymbolicTensor:
    return full(value, self.shape, self.dtype)


def ones(
    shape: ShapeLike = None,
    dtype: DataTypeLike = None,
) -> SymbolicTensor:
    shape = Shape.from_(shape)
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    return full(1.0, shape, dtype)


def zeros(
    shape: ShapeLike = None,
    dtype: DataTypeLike = None,
) -> SymbolicTensor:
    shape = Shape.from_(shape)
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    return full(0.0, shape, dtype)


def int_arange(
    stop: int,
    start: int = 0,
    step: int = 1,
    dtype: DataTypeLike = None,
) -> SymbolicTensor:
    assert isinstance(stop, int)
    assert isinstance(start, int)
    assert isinstance(step, int)

    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_int)

    if step == 1 and stop - start == 1:
        return full(start, (), dtype)
    size = math.ceil((stop - start) / step)
    a = SymbolicTensor.full(step, Shape((size,)), dtype).cumsum()  # , domain
    return a + a.full_like_self(start - step)


def arange(
    stop: ie.IntIndexValueLike,
    start: ie.IntIndexValueLike = 0,
    step: ie.IntIndexValueLike = 1,
    dtype: DataTypeLike = None,
) -> SymbolicTensor:
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_int)

    if isinstance(stop, int) and isinstance(start, int) and isinstance(step, int):
        return int_arange(stop, start, step, dtype)

    # if isinstance(step, int):
    #    M = get_active_exec_cfg().M
    #    static_arange = int_arange(M, 0, step, dtype)

    #    if step == 1:
    #        size = stop - start
    #    else:
    #        size = ie.Ceil((stop - start) / step)
    #    size = size.try_eval({}) or size

    #    return static_arange.index_slice(0, start=start, length=size)

    size = ie.lift_to_int_ie(stop - start)
    if step != 1:
        size = ie.Ceil(size / step)
    size = size.try_eval({}) or size

    a = SymbolicTensor.full(step, Shape((size,)), dtype).cumsum()
    return a + a.full_like_self(start - step)


def eval_symbol(
    symbol: ie.Symbol,
) -> SymbolicTensor:
    domain: Tuple[ie.Symbol, ...] = () if symbol.is_bound else (symbol,)
    from tempo.core.global_objects import get_dynamic_bounds_or_empty

    dyn_bounds = get_dynamic_bounds_or_empty()
    if symbol.as_bound() in dyn_bounds:
        bound = dyn_bounds[symbol.as_bound()]
        vars_used = bound.vars_used()
        domain = domain + tuple(vars_used)

    return _register_source(top.EvalSymbolOp, domain, symbol=symbol)


def split(
    tensor: MaybeSymbolicTensor,
    dim: int,
    num_splits: int,
) -> Sequence[SymbolicTensor]:
    tensor = lift_to_symbolic_tensor(tensor)
    domain = tensor.domain
    return _register_n_outs(top.SplitOp, [tensor], domain, num_splits, dim, num_splits)


def index_slice(
    tensor: MaybeSymbolicTensor,
    dim: int,
    start: Union[ie.IntIndexValueLike, MaybeSymbolicTensor],
    length: ie.IntIndexValueLike,
) -> SymbolicTensor:
    tensor = lift_to_symbolic_tensor(tensor)

    start_t = lift_to_symbolic_tensor(start)

    dom = Domain.union(tensor.domain, start_t.domain)
    if isinstance(length, ie.IntIndexValue):
        dom = dom.union(Domain.from_(length.vars_used()))

    return _register_one_out(top.IndexSliceOp, [tensor, start_t], dim, length, domain=dom)


def index_select(
    tensor: MaybeSymbolicTensor,
    dim: int,
    index: MaybeSymbolicTensor,
    keepdim: bool = False,
) -> SymbolicTensor:
    tensor = lift_to_symbolic_tensor(tensor)

    if get_active_exec_cfg().enable_index_ops:
        res = _index_based_index_select(tensor, dim, index)
    else:
        res = _gather_based_index_select(tensor, dim, index)

    if keepdim and len(res.shape) < len(tensor.shape):
        res = res.unsqueeze(dim)
    return res


def matmul(
    left: MaybeSymbolicTensor,
    right: MaybeSymbolicTensor,
) -> SymbolicTensor:
    if get_active_exec_cfg().enable_matmul_ops:
        return _register_one_out(top.MatMulOp, [left, right])
    else:
        raise NotImplementedError("SymbolicTensor-level non-prim Matmul not implemented")


def _index_based_index_select(
    tensor: MaybeSymbolicTensor,
    dim: int,
    index: MaybeSymbolicTensor,
) -> SymbolicTensor:
    return _register_one_out(top.IndexSelectOp, [tensor, index], dim)


def _gather_based_index_select(
    tensor: MaybeSymbolicTensor,
    dim: int,
    index: MaybeSymbolicTensor,
) -> SymbolicTensor:
    tensor = lift_to_symbolic_tensor(tensor)
    index = lift_to_symbolic_tensor(index)
    original = index

    if original.shape.is_scalar():
        index = index.unsqueeze(0)

    gather_index = index
    for i in range(len(tensor.shape)):
        if i != dim:
            gather_index = gather_index.unsqueeze(i)

    final_shape = list(tensor.shape)
    final_shape[dim] = index.shape.at(0)
    gather_index = gather_index.expand(Shape.from_(tuple(final_shape)))

    result = gather(tensor, dim, gather_index)

    if original.shape.is_scalar():
        result = result.squeeze(dim)

    return result


def index_add(
    sink: MaybeSymbolicTensor,
    dim: int,
    index: MaybeSymbolicTensor,
    src: MaybeSymbolicTensor,
    alpha: float = 1.0,
) -> SymbolicTensor:
    if get_active_exec_cfg().enable_index_ops:
        return _index_add_based_index_add(sink, dim, index, src, alpha)
    else:
        return _scatter_add_based_index_add(sink, dim, index, multiply(src, alpha))


def _scatter_add_based_index_add(
    sink: MaybeSymbolicTensor,
    dim: int,
    index: MaybeSymbolicTensor,
    src: MaybeSymbolicTensor,
    # alpha: float = 1.0,
) -> SymbolicTensor:
    sink = lift_to_symbolic_tensor(sink)
    index = lift_to_symbolic_tensor(index)
    src = lift_to_symbolic_tensor(src)

    original = index
    if original.shape.is_scalar():
        index = index.unsqueeze(0)

    scatter_index = index
    for i in range(len(sink.shape)):
        if i != dim:
            scatter_index = scatter_index.unsqueeze(i)

    final_index_shape = list(sink.shape)
    final_index_shape[dim] = index.shape.at(0)
    scatter_index = scatter_index.expand(Shape.from_(tuple(final_index_shape)))

    scatter_index = scatter_index.unsqueeze(-1)
    src = src.unsqueeze(-1)

    result = scatter_add(sink, dim, scatter_index, src)  # , alpha

    if original.shape.is_scalar():
        result = result.squeeze(dim)

    return result


def _index_add_based_index_add(
    sink: MaybeSymbolicTensor,
    dim: int,
    index: MaybeSymbolicTensor,
    src: MaybeSymbolicTensor,
    alpha: float = 1.0,
) -> SymbolicTensor:
    return _register_one_out(top.IndexAddOp, [sink, index, src], dim, alpha)


# =============== UNARY METHODS ===============


def to_dtype(tensor: MaybeSymbolicTensor, dtype: DataType) -> SymbolicTensor:
    t = lift_to_symbolic_tensor(tensor)
    if t.dtype == dtype:
        return t

    return _register_one_out(top.CastOp, [t], dtype)


def negate(inp: MaybeSymbolicTensor) -> SymbolicTensor:
    return _register_one_out(top.NegOp, [inp])


def sin(inp: MaybeSymbolicTensor) -> SymbolicTensor:
    return _register_one_out(top.SinOp, [inp])


def not_(inp: MaybeSymbolicTensor) -> SymbolicTensor:
    return _register_one_out(top.NotOp, [inp])


def broadcast_to_shape(inp: MaybeSymbolicTensor, shape: Shape) -> SymbolicTensor:
    inp_ = lift_to_symbolic_tensor(inp)

    # NOTE: this is here to replicate torch behaviour safely.
    if len(inp_.shape) < len(shape):
        diff = len(shape) - len(inp_.shape)
        for _ in range(diff):
            inp_ = inp_.unsqueeze(0)
    return inp_.expand(shape)


def expand(inp: MaybeSymbolicTensor, sizes: Shape) -> SymbolicTensor:
    inp_ = lift_to_symbolic_tensor(inp)

    from tempo.core.global_objects import get_active_dg

    sizes = sizes.try_resolve(get_active_dg().static_bounds)
    comparison = sizes == inp_.shape.try_resolve(get_active_dg().static_bounds)
    if comparison:
        return inp_  # TODO should ident?

    domain = Domain.union(inp_.op.domain, Domain.from_vars(sizes.vars_used()))
    return _register_one_out(top.ExpandOp, [inp_], sizes, domain=domain)


def expand_dim(inp: MaybeSymbolicTensor, dim: int, size: ie.IntIndexValueLike) -> SymbolicTensor:
    inp_ = lift_to_symbolic_tensor(inp)

    s = list(inp_.shape._shape)
    s[dim] = size
    return inp_.expand(Shape.from_(s))


def flip(inp: MaybeSymbolicTensor, dim: int) -> SymbolicTensor:
    return _register_one_out(top.FlipOp, [inp], dim)


def reshape(inp: MaybeSymbolicTensor, shape: Shape) -> SymbolicTensor:
    inp_ = lift_to_symbolic_tensor(inp)
    from tempo.core.global_objects import get_active_dg

    tensor_shape = inp_.shape.try_resolve(get_active_dg().static_bounds)
    shape = shape.try_resolve(get_active_dg().static_bounds)
    if shape == tensor_shape:
        return inp_  # TODO should ident?

    is_unsq, unsq_dim = tensor_shape.is_shape_without_extra_dim(shape)
    if is_unsq:
        return inp_.unsqueeze(unsq_dim)
    is_sq, sq_dim = shape.is_shape_without_extra_dim(tensor_shape)
    if is_sq:
        return inp_.squeeze(sq_dim)

    # domain = Domain.union(inp_.op.domain, Domain.from_vars(shape.get_vars_used()))
    assert Domain.from_(shape.vars_used()).is_contained_in(inp_.domain)
    return _register_one_out(top.ReshapeOp, [inp_], shape)  # , domain=domain)


def permute(inp: MaybeSymbolicTensor, dims: Sequence[int]) -> SymbolicTensor:
    inp_ = lift_to_symbolic_tensor(inp)
    if dims == tuple(range(len(inp_.shape))):
        return inp_.ident()  # TODO should ident?
    return _register_one_out(top.PermuteOp, [inp_], dims)


def transpose(t: MaybeSymbolicTensor, dim0: int = 1, dim1: int = 0) -> SymbolicTensor:
    t = lift_to_symbolic_tensor(t)
    dim0 = normalize_negative_dim(dim0, t.shape)
    dim1 = normalize_negative_dim(dim1, t.shape)
    assert len(t.shape) >= 2, "Transpose requires a tensor of at least 2 dimensions"
    order = list(range(len(t.shape)))
    order[dim0], order[dim1] = order[dim1], order[dim0]
    return t.permute(tuple(order))


def squeeze(inp: MaybeSymbolicTensor, dim: int) -> SymbolicTensor:
    inp_ = lift_to_symbolic_tensor(inp)

    if len(inp_.shape) < dim + 1:
        raise ValueError(f"Can't squeeze dimension {dim} of shape {inp_.shape}")
    if inp_.shape.at(dim) != 1:
        raise ValueError(f"Can't squeeze dimension {dim} of shape {inp_.shape} because it's not 1")

    return _register_one_out(top.SqueezeOp, [inp], dim)


def unsqueeze(inp: MaybeSymbolicTensor, dim: int) -> SymbolicTensor:
    return _register_one_out(top.UnsqueezeOp, [inp], dim)


def cat(*inputs: MaybeSymbolicTensor, dim: int = 0) -> SymbolicTensor:
    if len(inputs) == 1:
        return lift_to_symbolic_tensor(inputs[0])
    assert len(inputs) > 0
    return _register_one_out(top.CatOp, [*inputs], dim, len(inputs))


def cumsum(inp: MaybeSymbolicTensor, dim: int = 0) -> SymbolicTensor:
    return _register_one_out(top.CumSumOp, [inp], dim)


def sum(  # noqa: A001
    inp: MaybeSymbolicTensor,
    dims: Union[int, Sequence[int]],
    keepdim: bool = False,
) -> SymbolicTensor:
    if isinstance(dims, int):
        dims = (dims,)
    res = _register_one_out(top.SumOp, [inp], dims, keepdim)
    return res


def ln(antilogarithm: MaybeSymbolicTensor) -> SymbolicTensor:
    return _register_one_out(top.LnOp, [antilogarithm])


def exp(exponent: MaybeSymbolicTensor) -> SymbolicTensor:
    return _register_one_out(top.ExpOp, [exponent])


def sqrt(x: MaybeSymbolicTensor) -> SymbolicTensor:
    return _register_one_out(top.SqrtOp, [x])


# TODO(alan)
# def insert_into_buffer


def ident(x: MaybeSymbolicTensor) -> SymbolicTensor:
    x_ = lift_to_symbolic_tensor(x)
    if x_.index_expr is None or x_.index_expr.struct_eq(x_.op.domain.basis_expr):
        # x_.index_expr = None
        # return x_
        return dataclasses.replace(x_, _index_expr=None)

    return _register_one_out(top.IdentOp, [x])


# =============== BINARY METHODS ===============


def add(left: MaybeSymbolicTensor, right: MaybeSymbolicTensor) -> SymbolicTensor:
    return _register_one_out(top.AddOp, [left, right])


def subtract(left: MaybeSymbolicTensor, right: MaybeSymbolicTensor) -> SymbolicTensor:
    return _register_one_out(top.SubOp, [left, right])


def multiply(left: MaybeSymbolicTensor, right: MaybeSymbolicTensor) -> SymbolicTensor:
    return _register_one_out(top.MulOp, [left, right])


def divide(left: MaybeSymbolicTensor, right: MaybeSymbolicTensor) -> SymbolicTensor:
    return _register_one_out(top.DivOp, [left, right])


def pow_(left: MaybeSymbolicTensor, right: MaybeSymbolicTensor) -> SymbolicTensor:
    return _register_one_out(top.PowOp, [left, right])


def conv(
    input_: MaybeSymbolicTensor,
    weight: MaybeSymbolicTensor,
    stride: Sequence[int],
    # padding: Tuple[int, ...],
    # dilation: Tuple[int, ...],
    transposed: bool,
    # output_padding: Tuple[int, ...],
    # groups: int,
    n_dims: int,
) -> SymbolicTensor:
    inputs = [input_, weight]

    raise ValueError("TODO: convolutions are currently broken.")
    # assert groups == 1, "TODO: add support for groups != 1"
    return _register_one_out(
        top.ConvOp,
        inputs,
        stride,
        # padding,
        # dilation,
        transposed,
        # output_padding,
        # groups,
        n_dims,
    )


def dilate(x: MaybeSymbolicTensor, dilations: Sequence[int]) -> SymbolicTensor:
    x = lift_to_symbolic_tensor(x)

    n_dims = len(dilations)

    num_non_spatial_dims = x.ndim - n_dims
    non_spatial_size = (1,) * num_non_spatial_dims

    kernel_size = (1,) * n_dims

    weight = SymbolicTensor.ones(Shape((*non_spatial_size, *kernel_size)), dtype=x.dtype)

    stride = tuple(d + 1 for d in dilations)
    return x.conv(weight, stride=stride, transposed=True, n_dims=n_dims)


# ----------------- BINARY COMPARISON METHODS -----------------
def less_than(left: MaybeSymbolicTensor, right: MaybeSymbolicTensor) -> SymbolicTensor:
    left_ = lift_to_symbolic_tensor(left)
    right_ = lift_to_symbolic_tensor(right)
    return _register_one_out(top.LessThanOp, [left_, right_])


def logical_and(left: MaybeSymbolicTensor, right: MaybeSymbolicTensor) -> SymbolicTensor:
    left_ = lift_to_symbolic_tensor(left)
    right_ = lift_to_symbolic_tensor(right)
    if type(left) is bool:
        if left:
            return right_
        else:
            return SymbolicTensor.zeros(right_.shape, dtype=dtypes.bool_)
    if type(right) is bool:
        if right:
            return left_
        else:
            return SymbolicTensor.zeros(left_.shape, dtype=dtypes.bool_)

    return _register_one_out(top.AndOp, [left_, right_])


def logical_or(left: MaybeSymbolicTensor, right: MaybeSymbolicTensor) -> SymbolicTensor:
    left_ = lift_to_symbolic_tensor(left)
    right_ = lift_to_symbolic_tensor(right)
    if type(left) is bool:
        if left:
            return SymbolicTensor.ones(left_.shape, dtype=dtypes.bool_)
        else:
            return right_

    if type(right) is bool:
        if right:
            return SymbolicTensor.ones(right_.shape, dtype=dtypes.bool_)
        else:
            return left_

    return _register_one_out(top.OrOp, [left_, right_])


def equal(left: MaybeSymbolicTensor, right: MaybeSymbolicTensor) -> SymbolicTensor:
    left_ = lift_to_symbolic_tensor(left)
    right_ = lift_to_symbolic_tensor(right)
    return _register_one_out(top.EqualOp, [left_, right_])


def less_than_or_equal(left: MaybeSymbolicTensor, right: MaybeSymbolicTensor) -> SymbolicTensor:
    left_ = lift_to_symbolic_tensor(left)
    right_ = lift_to_symbolic_tensor(right)
    return (left_ < right_) | (left_ == right_)  # type: ignore


# These we implement in terms of lt, and and or
def greater_than(left: MaybeSymbolicTensor, right: MaybeSymbolicTensor) -> SymbolicTensor:
    left_ = lift_to_symbolic_tensor(left)
    right_ = lift_to_symbolic_tensor(right)
    return (right_ < left_) & ~(left_ == right_)  # type: ignore


def greater_than_or_equal(left: MaybeSymbolicTensor, right: MaybeSymbolicTensor) -> SymbolicTensor:
    left_ = lift_to_symbolic_tensor(left)
    right_ = lift_to_symbolic_tensor(right)
    return ~(left_ < right_)  # type: ignore


# =========== N-ARY METHODS ===========


def max_single_tensor(
    x: MaybeSymbolicTensor, dim: int = 0, keepdim: bool = False
) -> Tuple[SymbolicTensor, SymbolicTensor]:
    assert isinstance(dim, int)
    return _register_two_outs(top.MaxOp, [x], (dim,), keepdim)


# ================= TERNARY METHODS =================
def where(
    cond: MaybeSymbolicTensor,
    x: MaybeSymbolicTensor,
    y: MaybeSymbolicTensor,
) -> SymbolicTensor:
    return _register_one_out(top.WhereOp, [cond, x, y])


def isnan(x: MaybeSymbolicTensor) -> SymbolicTensor:
    x = lift_to_symbolic_tensor(x)

    nan = float("nan")
    nan = x.full_like_self(nan)

    raise ValueError("TODO: isnan does not work, right? Because nan does not equal nan.")

    return SymbolicTensor.equals(x, nan)


def val_to_val(x: MaybeSymbolicTensor, in_val: float, out_val: float) -> SymbolicTensor:
    return _register_one_out(top.ValToValOp, [x], in_val, out_val)


def nan_to_num(x: MaybeSymbolicTensor, num: float = 0.0) -> SymbolicTensor:
    x = lift_to_symbolic_tensor(x)
    return val_to_val(x, float("nan"), num)


def nan_to_zero(x: MaybeSymbolicTensor) -> SymbolicTensor:
    return nan_to_num(x, 0.0)


def nan_to_neg_inf(x: MaybeSymbolicTensor) -> SymbolicTensor:
    return nan_to_num(x, -float("inf"))


def nan_to_pos_inf(x: MaybeSymbolicTensor) -> SymbolicTensor:
    return nan_to_num(x, float("inf"))


def gather(
    src: MaybeSymbolicTensor,
    dim: int,
    index: MaybeSymbolicTensor,
) -> SymbolicTensor:
    src = lift_to_symbolic_tensor(src)
    index = lift_to_symbolic_tensor(index)
    assert len(src.shape) == len(index.shape), (
        f"src/idx shape mismatch. {src.shape=}, {index.shape=}"
    )
    return _register_one_out(top.GatherOp, [src, index], dim)


def scatter_add(
    sink: MaybeSymbolicTensor,
    dim: int,
    index: MaybeSymbolicTensor,
    src: MaybeSymbolicTensor,
) -> SymbolicTensor:
    return _register_one_out(top.ScatterAddOp, [sink, index, src], dim)


def merge(
    shape: ShapeLike = None,
    dtype: DataTypeLike = None,
    domain: DomainLike = None,
) -> SymbolicTensor:
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)
    shape = Shape.from_(shape)

    return _register_source(top.MergeOp, domain, shape, dtype)


def merge_like(x: MaybeSymbolicTensor) -> SymbolicTensor:
    x = lift_to_symbolic_tensor(x)
    return SymbolicTensor.merge(x.shape, x.dtype, x.domain)


def add_merge_branch(
    merge: SymbolicTensor, condition: ie.BooleanIndexValueLike, branch: SymbolicTensor
) -> None:
    from tempo.core.global_objects import get_active_dg

    cond = ie.lift_to_bool_ie(condition)
    active_dg = get_active_dg()
    num_branches_already = len(active_dg.get_input_shapes_list(merge.op))
    dep_data = DependencyData(
        branch.index_expr, branch.tensor_id.output_id, OpInId(num_branches_already), cond
    )
    merge_op = merge.op
    assert isinstance(merge_op, top.MergeOp)
    merge_op.increment_num_inputs()
    active_dg.add_edge(merge_op, branch.op, dep_data)


def trunc(x: MaybeSymbolicTensor) -> SymbolicTensor:
    x = lift_to_symbolic_tensor(x)
    return x.to_dtype(dtypes.least_upper_signed_int(x.dtype)).to_dtype(x.dtype)


def floor(x: MaybeSymbolicTensor) -> SymbolicTensor:
    x = lift_to_symbolic_tensor(x)
    b = x.trunc()
    return SymbolicTensor.where((x < b), b - 1, b)


def ceil(x: MaybeSymbolicTensor) -> SymbolicTensor:
    x = lift_to_symbolic_tensor(x)
    b = x.trunc()
    return SymbolicTensor.where((x > b), b + 1, b)


def floor_divide(x: MaybeSymbolicTensor, y: MaybeSymbolicTensor) -> SymbolicTensor:
    x = lift_to_symbolic_tensor(x)
    y = lift_to_symbolic_tensor(y)
    return SymbolicTensor.floor((x / y))


def remainder(x: MaybeSymbolicTensor, y: MaybeSymbolicTensor) -> SymbolicTensor:
    x = lift_to_symbolic_tensor(x)
    y = lift_to_symbolic_tensor(y)
    return x - (x // y) * y


def pad(
    inp: MaybeSymbolicTensor,
    padding: Sequence[Tuple[ie.IntIndexValueLike, ie.IntIndexValueLike]],
    mode: str = "constant",
    value: Optional[float] = None,
) -> SymbolicTensor:
    """Pads the input tensor with a constant value."""
    inp = lift_to_symbolic_tensor(inp)
    for dim, dim_padding in enumerate(padding):
        inp = pad_dim(inp, dim_padding, dim, mode, value)
    return inp


def pad_dim(
    inp: MaybeSymbolicTensor,
    padding: Tuple[ie.IntIndexValueLike, ie.IntIndexValueLike],
    dim: int,
    mode: str = "constant",
    value: Optional[float] = None,
) -> SymbolicTensor:
    """Pads the input tensor with a constant value.

    Args:
        inp: Input tensor to pad
        padding: Tuple of (before, after) padding sizes for the dimension
        dim: Dimension to pad
        mode: Padding mode (default: "constant")
        value: Value to pad with (default: 0.0)

    Returns:
        Padded tensor
    """
    inp = lift_to_symbolic_tensor(inp)
    assert isinstance(padding, tuple), (
        f"Padding must be a tuple of (before, after) for each dimension, got {padding}"
    )
    if len(padding) != 2:
        raise ValueError("Padding must be a tuple of (before, after) for each dimension")
    # TODO: these checks should also be done for constints
    if (isinstance(padding[0], int) and padding[0] < 0) or (
        isinstance(padding[1], int) and padding[1] < 0
    ):
        raise ValueError("Padding must be non-negative")
    if (isinstance(padding[0], int) and padding[0] == 0) and (
        isinstance(padding[1], int) and padding[1] == 0
    ):
        return inp.ident()
    mode_str_to_enum = {
        "constant": top.PadMode.CONSTANT,
        "reflect": top.PadMode.REFLECT,
        "replicate": top.PadMode.REPLICATE,
        "any": top.PadMode.ANY,
    }
    mode_mapped = mode_str_to_enum[mode]
    ret = _register_one_out(
        top.PadOp, [inp], padding=padding, dim=dim, value=value, mode=mode_mapped
    )
    return ret


def symbolic_index(tensor: SymbolicTensor, index_expr: ie.IndexSequence) -> SymbolicTensor:
    curr_expr = tensor._index_expr if tensor._index_expr else ie.IndexSequence(())
    expr = ie.IndexSequence((*curr_expr.members, *index_expr.members))
    if len(expr) > len(tensor.unindexed_domain):
        raise Exception(
            "Can't index a SymbolicTensor with more symbolic dims than it has. "
            + f"{curr_expr=}, {expr=}, {tensor.unindexed_domain=}"
        )
    return SymbolicTensor(tensor.op, tensor.tensor_id, tensor.spatial_shape, tensor.dtype, expr)


@dataclass(frozen=False, unsafe_hash=True)
class SymbolicTensor:
    __slots__ = "op", "tensor_id", "spatial_shape", "dtype", "_index_expr"
    # The op that produces this tensor
    op: top.TensorOp
    tensor_id: TensorId

    spatial_shape: Shape
    dtype: DataType
    _index_expr: Optional[ie.IndexSequence]

    def __str__(self) -> str:
        return f"SymbolicTensor({self.op.__class__}(id={self.tensor_id}, domain={self.op.domain}), \
        {self.spatial_shape=}, {self.dtype=})"

    @property
    def index_expr(self) -> ie.IndexSequence:
        expr = self._index_expr
        if not expr:
            expr = ie.IndexSequence(tuple(self.unindexed_domain.variables))
        elif len(expr) < len(self.unindexed_domain):
            expr = ie.IndexSequence((*expr.members, *self.unindexed_domain.variables[len(expr) :]))
        return expr

    @property
    def shape(self) -> Shape:
        symbolic_shape = self.index_expr.evaluate_shape({})
        result_shape = Shape((*symbolic_shape, *self.spatial_shape._shape))
        return result_shape

    @property
    def unindexed_domain(self) -> Domain:
        return self.op.domain

    @property
    def domain(self) -> Domain:
        return self.op.domain.indexed_real_domain(self.index_expr)

    # TODO this probably should not exist
    def copy_with_no_index(self) -> SymbolicTensor:
        return SymbolicTensor(self.op, self.tensor_id, self.spatial_shape, self.dtype, None)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    # def copy_with_index(self, index_expr: Optional[ie.IndexSequence]) -> SymbolicTensor:
    #    """Returns a copy of this tensor with the given index expression.

    #    Args:
    #        _index_expr (IndexExpr): The index expression to use

    #    Returns:
    #        SymbolicTensor: A copy of this tensor with the given index expression
    #    """
    #    # if self._index_expr:
    #    #    raise Exception("Can't index a SymbolicTensor twice")
    #    return SymbolicTensor(
    #        self.op, self.tensor_id, self.spatial_shape, self.dtype, index_expr
    #    )

    @property
    def creation_traceback(self) -> str:
        return self.op.creation_traceback

    def size(
        self, dim: DIM_TYPE = None
    ) -> Union[Union[int, ie.IntIndexValue], Sequence[Union[int, ie.IntIndexValue]]]:
        if dim is None:
            return self.shape._shape  # type: ignore
        elif isinstance(dim, int):
            return self.shape.at(dim)  # type: ignore
        else:
            return tuple(self.shape.at(d) for d in dim)

    symbolic_index = temporal_index = symbolic_index
    add = __add__ = add
    __radd__ = lambda x, y: add(y, x)

    sub = subtract = __sub__ = subtract
    __rsub__ = lambda x, y: subtract(y, x)

    mul = multiply = __mul__ = multiply
    __rmul__ = lambda x, y: multiply(y, x)

    div = divide = __truediv__ = __div__ = divide
    __rtruediv__ = __rdiv__ = lambda x, y: divide(y, x)
    trunc = trunc
    __floordiv__ = floor_divide = floor_divide

    pow_ = pow_  # noqa: A001, A002, A003

    exp = exp
    ln = ln

    __neg__ = negate

    # NOTE: the type ignore annotation is because these methods are "supposed to" return bools, but
    # we want to return symbolic tensors
    __lt__ = less_than = less_than  # type: ignore[assignment]
    __eq__ = equals = equal  # type: ignore[assignment]
    __and__ = logical_and = logical_and  # type: ignore[assignment]
    __or__ = logical_or = logical_or  # type: ignore[assignment]

    __le__ = less_than_or_equal = less_than_or_equal  # type: ignore[assignment]
    __gt__ = greater_than = greater_than  # type: ignore[assignment]
    __ge__ = greater_than_or_equal = greater_than_or_equal  # type: ignore[assignment]
    __mod__ = mod = modulos = remainder

    __ne__ = not_equals = lambda x, y: not_(x == y)  # type: ignore[assignment]

    __invert__ = logical_not = not_

    sqrt = sqrt
    sin = sin

    sum = sum  # noqa: A001, A002, A003
    # discounted_sum = discounted_sum

    ident = ident
    max = max_single_tensor  # noqa: A001, A002, A003
    where = where

    expand = expand
    expand_dim = expand_dim
    broadcast_to_shape = broadcast_to_shape

    flip = flip
    squeeze = squeeze
    unsqueeze = unsqueeze
    cat = cat
    stack = stack
    split = split
    reshape = reshape
    permute = permute
    transpose = transpose

    cumsum = cumsum

    conv = conv

    full_like_self = full_like_self

    full = staticmethod(full)
    ones = staticmethod(ones)
    zeros = staticmethod(zeros)
    rand = staticmethod(rand)
    arange = staticmethod(arange)
    eval_symbol = staticmethod(eval_symbol)
    merge = staticmethod(merge)
    merge_like = staticmethod(merge_like)
    add_merge_branch = add_merge_branch
    udf = staticmethod(udf)

    to_dtype = to_dtype
    gather = gather
    scatter_add = scatter_add

    index_slice = index_slice
    index_select = spatial_index_select = index_select
    index_add = index_add

    _gather_based_index_select = _gather_based_index_select
    _scatter_based_index_add = _scatter_add_based_index_add

    matmul = matmul
    pad_dim = pad_dim
    pad = pad
    dilate = dilate

    isnan = isnan
    nan_to_num = nan_to_num
    nan_to_zero = nan_to_zero
    nan_to_neg_inf = nan_to_neg_inf
    nan_to_pos_inf = nan_to_pos_inf
    ceil = ceil
    floor = floor
    # index_add = index_add

    barrier = barrier

    lift = staticmethod(lift_to_symbolic_tensor)
    lift_many = staticmethod(lift_to_symbolic_tensors)


ManySymbolicTensors = Sequence[SymbolicTensor]
OneOrManySymbolicTensors = Union[SymbolicTensor, ManySymbolicTensors]

MaybeSymbolicTensor = Union[SymbolicTensor, float, int, bool, ie.IndexExpr]
ManyMaybeSymbolicTensors = Sequence[MaybeSymbolicTensor]
OneOrManyMaybeSymbolicTensors = Union[MaybeSymbolicTensor, ManyMaybeSymbolicTensors]
