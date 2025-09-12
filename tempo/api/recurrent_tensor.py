from __future__ import annotations

import builtins
import functools
import math
from collections.abc import Callable, Mapping, Sequence
from typing import (
    Any,
    Union,
    overload,
)

import numpy as np

from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.datatypes import DIM_TYPE, BackendTensorT, TensorId
from tempo.core.device import device
from tempo.core.dim_utils import (
    normalize_dims,
    normalize_indexes,
    normalize_neg_1s_in_shape_expand,
    normalize_neg_1s_in_shape_reshape,
    normalize_negative_dim,
    normalize_negative_dim_tuple,
    normalize_slice_indexes,
)
from tempo.core.domain import Domain, DomainLike
from tempo.core.dtype import DataType, DataTypeLike, dtypes
from tempo.core.global_objects import get_active_exec_cfg
from tempo.core.shape import (
    Shape,
    ShapeLike,
    unsq_align_shapes_1_pad_left,
    unsq_align_shapes_1_pad_right,
)
from tempo.core.symbolic_tensor import SymbolicTensor
from tempo.core.thunk import (
    Thunk,
    ThunkExecutionCtx,
)
from tempo.core.thunk_emitter import ThunkEmissionCtx
from tempo.core.thunk_udf import UserDefinedThunkDesc
from tempo.utils.common import as_seq
from tempo.utils.logger import get_logger

_logger = get_logger(__name__)


# TODO Can we find a better method to do domain expansion?
def _expand_domain(tensor: RecurrentTensor, domain: Domain) -> RecurrentTensor:
    if tensor.domain.is_contained_in(domain) and not tensor.domain == domain:
        new_tensor = RecurrentTensor.placeholder(tensor.shape, tensor.dtype, domain)
        new_tensor[True] = tensor
        tensor = new_tensor
    return tensor


class AutodiffFn:
    def __init__(self, *inputs: RecurrentTensor) -> None:
        self.needs_input_grad = [i.requires_grad for i in inputs]
        self.requires_grad = (
            True if any(self.needs_input_grad) else None if None in self.needs_input_grad else False
        )
        if self.requires_grad:
            self.parents = inputs

    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        raise NotImplementedError(f"forward not implemented for {type(self)}")

    def backward(
        self,
        grad_output: SymbolicTensor,  # TODO for split needs to be sequence
    ) -> Sequence[SymbolicTensor | None]:
        raise NotImplementedError(f"backward not implemented for {type(self)}")

    @classmethod
    def apply(cls: type[AutodiffFn], *x: RecurrentTensor, **kwargs: Any) -> ManyRecurrentTensors:
        assert all(t._underlying is not None for t in x), (
            "Can't apply autodiff function to uninitialized tensor"
        )

        ctx = cls(*x)
        outs = ctx.forward(*[t._underlying for t in x], **kwargs)
        rets = [
            RecurrentTensor(
                out,
                requires_grad=ctx.requires_grad,
                ctx=ctx if ctx.requires_grad else None,
            )
            for out in outs
        ]
        return tuple(rets)


import tempo.api.autodiff as ad  # noqa: E402


def from_symbol(s: ie.Symbol) -> RecurrentTensor:
    # from tempo.core import global_objects as glob

    # uni = glob.active_dg.universe

    # TODO this wont work for dynamic symbols...
    st = SymbolicTensor.eval_symbol(s)
    return RecurrentTensor(st)


def from_expr(expr: ie.IndexExpr) -> RecurrentTensor:
    from tempo.api.expr_to_recurrent_tensor import translate_ie_to_rt

    return translate_ie_to_rt(expr)


def _lift_if_index_value(
    inp: MaybeRecurrentTensor,
) -> MaybeRecurrentTensor:
    if isinstance(inp, ie.IndexExpr):
        return from_expr(inp)
    else:
        return inp


def lift(inp: MaybeRecurrentTensor) -> RecurrentTensor:
    if isinstance(inp, (bool, int, float, list, np.ndarray)):
        return RecurrentTensor.const(inp)
    if isinstance(inp, ie.IndexExpr):
        return from_expr(inp)
    if not isinstance(inp, RecurrentTensor):
        raise TypeError(f"Expected RecurrentTensor, got {type(inp)}: {inp}")
    return inp


def lift_all_to_rt(*inps: MaybeRecurrentTensor) -> ManyRecurrentTensors:
    return tuple(map(lift, inps))


# TODO I wish it was possible to do all broadcasting and upcasting at the level of symbolic
# tensors, as this would simplify things greatly.
def broadcast_tensors(*tensors: MaybeRecurrentTensor) -> ManyRecurrentTensors:
    tensors_raised = lift_all_to_rt(*tensors)
    broadcasted_shape = Shape.broadcast(*[t.shape for t in tensors_raised])
    return tuple(t.expand(broadcasted_shape) for t in tensors_raised)


def upcast_tensors(*tensors: MaybeRecurrentTensor) -> ManyRecurrentTensors:
    tensors_raised = lift_all_to_rt(*tensors)
    dtype = dtypes.upcast(*[t.dtype for t in tensors_raised])
    return tuple(t.cast(dtype) for t in tensors_raised)


# TODO: all of this UDF stuff will require some refactorings over time
# TODO: especially to remove all the type ignores. For now, its a start


def map_udf(
    x: MaybeRecurrentTensor,
    fun: Callable[[Any], None],
    shape: ShapeLike = None,
    dtype: DataTypeLike = None,
) -> RecurrentTensor:
    def translation_fn(op: top.TensorOp, ctx: ThunkEmissionCtx) -> Thunk[Any]:  # type: ignore
        return lambda inputs, thunk_exec_ctx: fun(inputs[0])  # type: ignore

    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    desc = UserDefinedThunkDesc(
        thunk_translation=translation_fn,
        num_inputs=1,
        num_outputs=1,
        infer_output_shapes=lambda input_shapes: (  # type:ignore
            Shape.from_(shape) if shape is not None else input_shapes[0],
        ),
        infer_output_dtypes=lambda input_dtypes: (  # type:ignore
            dtype if dtype is not None else input_dtypes[0],
        ),
        needs_symbol_setter=False,
    )
    x = lift(x)
    return RecurrentTensor(SymbolicTensor.udf(desc, [x._underlying])[0])


def map_with_ts_udf(
    x: MaybeRecurrentTensor,
    fun: Callable[[Any, Mapping[ie.Symbol, int]], Any],
    shape: ShapeLike = None,
    dtype: DataTypeLike = None,
) -> RecurrentTensor:
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    def translation_fn(op: top.TensorOp, ctx: ThunkEmissionCtx):  # type: ignore
        def thunk(  # type: ignore
            inputs: tuple[Any, ...], thunk_exec_ctx: ThunkExecutionCtx
        ) -> tuple[Any, ...]:
            return (
                fun(
                    inputs[0],
                    {k: thunk_exec_ctx.symbol_values[k] for k in op.domain.variables},
                ),
            )  # type: ignore

        return thunk  # type: ignore

    desc = UserDefinedThunkDesc(
        thunk_translation=translation_fn,  # type:ignore
        num_inputs=1,
        num_outputs=1,
        infer_output_shapes=lambda input_shapes: (  # type:ignore
            Shape.from_(shape) if shape is not None else input_shapes[0],
        ),
        infer_output_dtypes=lambda input_dtypes: (  # type:ignore
            dtype if dtype is not None else input_dtypes[0],
        ),
    )
    x = lift(x)
    return RecurrentTensor(SymbolicTensor.udf(desc, [x._underlying])[0])


def source_udf(
    fun: Callable[[], BackendTensorT],
    shape: ShapeLike,
    dtype: DataTypeLike,
    domain: DomainLike,
    requires_grad: bool = False,
) -> RecurrentTensor:
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    def translation_fn(op: top.TensorOp, ctx: ThunkEmissionCtx):  # type: ignore
        return lambda inputs, thunk_exec_ctx: (fun(),)  # type: ignore

    shape = Shape.from_(shape)
    desc = UserDefinedThunkDesc(
        thunk_translation=translation_fn,
        num_inputs=0,
        num_outputs=1,
        infer_output_shapes=lambda input_shapes: (shape,),  # type:ignore
        infer_output_dtypes=lambda input_dtypes: (dtype,),  # type:ignore
    )
    return RecurrentTensor(SymbolicTensor.udf(desc, [], domain)[0], requires_grad=requires_grad)


def source_with_ts_udf(
    fun: Callable[[Mapping[ie.Symbol, int]], Any],
    shape: ShapeLike,
    dtype: DataTypeLike,
    domain: DomainLike,
    lazy_vectorized_version: Callable[[Mapping[ie.Symbol, int]], Any] | None = None,
    requires_grad: bool = False,
) -> RecurrentTensor:
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    def translation_fn(op: top.TensorOp, ctx: ThunkEmissionCtx):  # type: ignore
        return lambda inputs, thunk_exec_ctx: (fun(dict(thunk_exec_ctx.symbol_values.items())),)  # type: ignore

    if lazy_vectorized_version is None:
        vectorize_fn = None
    else:
        from tempo.core.thunk_udf import UDFVectorizationCtx

        def vectorize_fn(vec_ctx: UDFVectorizationCtx) -> UserDefinedThunkDesc:
            vec_shape = Shape.from_(shape).prepend_dim(vec_ctx.vec_size)

            def vec_translation_fn(op: top.TensorOp, ctx: ThunkEmissionCtx):  # type: ignore
                return lambda inputs, thunk_exec_ctx: (
                    lazy_vectorized_version(dict(thunk_exec_ctx.symbol_values.items())),
                )  # type: ignore

            return UserDefinedThunkDesc(
                thunk_translation=vec_translation_fn,
                num_inputs=0,
                num_outputs=1,
                infer_output_shapes=lambda input_shapes: (vec_shape,),  # type:ignore
                infer_output_dtypes=lambda input_dtypes: (dtype,),  # type:ignore
                needs_symbol_setter=True,
                vectorize=None,
            )

    shape = Shape.from_(shape)
    desc = UserDefinedThunkDesc(
        thunk_translation=translation_fn,
        num_inputs=0,
        num_outputs=1,
        infer_output_shapes=lambda input_shapes: (shape,),  # type:ignore
        infer_output_dtypes=lambda input_dtypes: (dtype,),  # type:ignore
        needs_symbol_setter=True,
        vectorize=vectorize_fn,
    )
    return RecurrentTensor(SymbolicTensor.udf(desc, [], domain)[0], requires_grad=requires_grad)


def sink_many_udf(
    rts_to_sink: Sequence[MaybeRecurrentTensor],
    fun: Callable[[Sequence[Any]], None],
) -> None:
    def translation_fn(op: top.TensorOp, ctx: ThunkEmissionCtx):  # type: ignore
        def thunk(  # type: ignore
            inputs: tuple[Any, ...], thunk_exec_ctx: ThunkExecutionCtx
        ) -> tuple[Any, ...]:
            fun(
                # inputs[0],
                inputs,
            )  # type: ignore
            return ()

        return thunk  # type: ignore

    desc = UserDefinedThunkDesc(
        thunk_translation=translation_fn,
        num_inputs=len(rts_to_sink),
        num_outputs=0,
        infer_output_shapes=lambda input_shapes: (),  # type:ignore
        infer_output_dtypes=lambda input_dtypes: (),  # type:ignore
        needs_symbol_setter=False,
    )
    xs: list[RecurrentTensor] = [lift(x) for x in rts_to_sink]
    SymbolicTensor.udf(desc, [x._underlying for x in xs])


def sink_many_with_ts_udf(
    rts_to_sink: Sequence[MaybeRecurrentTensor],
    fun: Callable[[Sequence[Any], Mapping[ie.Symbol, int]], None],
) -> None:
    # threadpool = ThreadPoolExecutor(max_workers=1)

    def translation_fn(op: top.TensorOp, ctx: ThunkEmissionCtx):  # type: ignore
        def thunk(  # type: ignore
            inputs: tuple[Any, ...], thunk_exec_ctx: ThunkExecutionCtx
        ) -> tuple[Any, ...]:
            fun(
                inputs,
                thunk_exec_ctx.symbol_values,
            )
            # threadpool.submit(
            #    fun,
            #    inputs,
            #    thunk_exec_ctx.symbol_values,
            # )
            return ()

        return thunk  # type: ignore

    desc = UserDefinedThunkDesc(
        thunk_translation=translation_fn,
        num_inputs=len(rts_to_sink),
        num_outputs=0,
        infer_output_shapes=lambda input_shapes: (),  # type:ignore
        infer_output_dtypes=lambda input_dtypes: (),  # type:ignore
    )
    xs = [lift(x) for x in rts_to_sink]
    SymbolicTensor.udf(desc, [x._underlying for x in xs])


def sink_with_ts_udf(
    x: MaybeRecurrentTensor, fun: Callable[[Any, Mapping[ie.Symbol, int]], None]
) -> None:
    def translation_fn(op: top.TensorOp, ctx: ThunkEmissionCtx):  # type: ignore
        def thunk(  # type: ignore
            inputs: tuple[Any, ...], thunk_exec_ctx: ThunkExecutionCtx
        ) -> tuple[Any, ...]:
            fun(
                inputs[0],
                {k: thunk_exec_ctx.symbol_values[k] for k in op.domain.variables},
            )  # type: ignore
            return ()

        return thunk  # type: ignore

    desc = UserDefinedThunkDesc(
        thunk_translation=translation_fn,
        num_inputs=1,
        num_outputs=0,
        infer_output_shapes=lambda input_shapes: (),  # type:ignore
        infer_output_dtypes=lambda input_dtypes: (),  # type:ignore
    )
    x = lift(x)
    SymbolicTensor.udf(desc, [x._underlying])


# TODO: move remaining udf methdos to symbolic tensor
def sink_udf(x: MaybeRecurrentTensor, fun: Callable[[Any], None]) -> None:
    x = lift(x)
    SymbolicTensor.sink_udf(fun, x._underlying)


def add(x: MaybeRecurrentTensor, y: MaybeRecurrentTensor) -> RecurrentTensor:
    x, y = lift_all_to_rt(x, y)
    x, y = broadcast_tensors(x, y)
    x, y = upcast_tensors(x, y)
    ret: RecurrentTensor = ad.Add.apply(x, y)[0]
    return ret


def sub(x: MaybeRecurrentTensor, y: MaybeRecurrentTensor) -> RecurrentTensor:
    x, y = lift_all_to_rt(x, y)
    x, y = broadcast_tensors(x, y)
    x, y = upcast_tensors(x, y)
    ret: RecurrentTensor = ad.Sub.apply(x, y)[0]
    return ret


def mul(  # noqa: C901
    x: MaybeRecurrentTensor, y: MaybeRecurrentTensor
) -> RecurrentTensor:
    # Special cases
    if isinstance(x, (int, float, bool)):
        if float(x) == 0.0:
            ret1: RecurrentTensor = ad.Zero.apply(lift(y))[0]
            return ret1
        if float(x) == 1.0:
            return lift(y)
        if float(x) == -1.0:
            return -lift(y)
    if isinstance(y, (int, float, bool)):
        if float(y) == 0.0:
            ret2: RecurrentTensor = ad.Zero.apply(lift(x))[0]
            return ret2
        if float(y) == 1.0:
            return lift(x)
        if float(y) == -1.0:
            return -lift(x)

    x, y = lift_all_to_rt(x, y)
    x, y = broadcast_tensors(x, y)
    x, y = upcast_tensors(x, y)
    ret: RecurrentTensor = ad.Mul.apply(x, y)[0]
    return ret


def div(x: MaybeRecurrentTensor, y: MaybeRecurrentTensor) -> RecurrentTensor:
    x, y = lift_all_to_rt(x, y)
    x, y = broadcast_tensors(x, y)
    x, y = upcast_tensors(x, y)
    ret: RecurrentTensor = ad.Div.apply(x, y)[0]
    return ret


def floor_div(x: MaybeRecurrentTensor, y: MaybeRecurrentTensor) -> RecurrentTensor:
    # x, y = lift_all_to_rt(x, y)
    return div(x, y).floor()


def trunc(x: MaybeRecurrentTensor) -> RecurrentTensor:
    x = lift(x)
    return x.cast(dtypes.least_upper_signed_int(x.dtype)).cast(x.dtype)


def floor(x: MaybeRecurrentTensor) -> RecurrentTensor:
    x = lift(x)
    b = x.trunc()
    return (x < b).where(b - 1, b)


def ceil(x: MaybeRecurrentTensor) -> RecurrentTensor:
    x = lift(x)
    b = x.trunc()
    return (x > b).where(b + 1, b)


def round(x: MaybeRecurrentTensor) -> RecurrentTensor:  # noqa: A001, A002, A003
    x = lift(x)
    int_dtype = dtypes.least_upper_signed_int(x.dtype)
    b = x.cast(int_dtype) / 2.0
    cond: RecurrentTensor = (x > 0) == (b.cast(int_dtype) == b)  # type: ignore
    return cond.where((x - 0.5).ceil(), (x + 0.5).floor())


def remainder(x: MaybeRecurrentTensor, y: MaybeRecurrentTensor) -> RecurrentTensor:
    x, y = lift_all_to_rt(x, y)
    x, y = broadcast_tensors(x, y)

    return x - (x // y) * y


def exp(exponent: MaybeRecurrentTensor) -> RecurrentTensor:
    exponent = lift(exponent)
    exponent = exponent.cast(dtypes.least_upper_float(exponent.dtype))
    ret: RecurrentTensor = ad.Exp.apply(exponent)[0]
    return ret


def exp2(exponent: MaybeRecurrentTensor) -> RecurrentTensor:
    exponent = lift(exponent)
    return exp(exponent * math.log(2.0))


def exp10(exponent: MaybeRecurrentTensor) -> RecurrentTensor:
    exponent = lift(exponent)
    return exp(exponent * math.log(10))


def reciprocal(x: MaybeRecurrentTensor) -> RecurrentTensor:
    ret: RecurrentTensor = 1.0 / lift(x)
    return ret


def pow_(  # noqa: C901
    base: MaybeRecurrentTensor, exponent: MaybeRecurrentTensor
) -> RecurrentTensor:
    base, exponent_ = lift_all_to_rt(base, exponent)
    base, exponent_ = broadcast_tensors(base, exponent_)

    base, exponent_ = upcast_tensors(base, exponent_)

    if isinstance(exponent, (int, float, bool)):
        exp = float(exponent)
        # if exponent < 0.0:
        #    return reciprocal(base) ** -exponent
        if exp == 0.0:
            early_ret: RecurrentTensor = ad.Zero.apply(base)[0] + 1
            return early_ret
        if exp == 1.0:
            return base
        if exp == 2.0:
            return base * base
        # if exp== 3.0:
        #    return base * base * base
        if exp == 0.5:
            return base.sqrt()
    ret: RecurrentTensor = ad.Pow.apply(base, exponent_)[0]
    return ret

    ## When base is 0, exponent is 0, the result is 1
    ## When base is 0, exponent is not 0, the result is 0
    ## When base is not 0, exponent is 0, the result is 1
    # res = exp(exponent_ * ln(base))
    # res = (base == 0.0).where(0, res)  # type: ignore
    # res = (exponent_ == 0.0).where(1, res)  # type: ignore

    # return res.cast(base.dtype)  # type: ignore


def square(x: MaybeRecurrentTensor) -> RecurrentTensor:
    x = lift(x)
    return x * x


def log_base(base: MaybeRecurrentTensor, antilogarithm: MaybeRecurrentTensor) -> RecurrentTensor:
    base, antilogarithm = lift_all_to_rt(base, antilogarithm)
    base, antilogarithm = broadcast_tensors(base, antilogarithm)
    base = base.cast(dtypes.least_upper_float(base.dtype))
    antilogarithm = antilogarithm.cast(dtypes.least_upper_float(antilogarithm.dtype))
    return ln(antilogarithm) / ln(base)


def ln(antilogarithm: MaybeRecurrentTensor) -> RecurrentTensor:
    antilogarithm = lift(antilogarithm)
    antilogarithm = antilogarithm.cast(dtypes.least_upper_float(antilogarithm.dtype))

    ret: RecurrentTensor = ad.Ln.apply(antilogarithm)[0]
    return ret


def log2(antilogarithm: MaybeRecurrentTensor) -> RecurrentTensor:
    return ln(antilogarithm) / math.log(2.0)


def log10(antilogarithm: MaybeRecurrentTensor) -> RecurrentTensor:
    return ln(antilogarithm) / math.log(10.0)


# def pow_(x: MaybeSymbolicTensor, y: MaybeSymbolicTensor) -> SymbolicTensor:
#    #Change of basis with expe
#    x =lift_to_symbolic_tensor(x)
#    y = lift_to_symbolic_tensor(y)
#    return expe(y * loge(x))


# def log(base: MaybeSymbolicTensor, antilogarithm: MaybeSymbolicTensor) -> SymbolicTensor:
#    base = lift_to_symbolic_tensor(base)
#    antilogarithm = lift_to_symbolic_tensor(antilogarithm)
#    return loge(antilogarithm) / loge(base)


def dot(  # noqa: C901
    x: MaybeRecurrentTensor, y: MaybeRecurrentTensor
) -> RecurrentTensor:
    """Computes the dot product of two tensors.
    If both are 1D, computes the inner product (scalar).
    If both are 2D, computes the matrix multiplication.
    If greater than 2D, computes the batched matrix multiplication
    focusing only on the last two dims.

    Args:
        x (MaybeRecurrentTensor): first tensor
        y (MaybeRecurrentTensor): second tensor

    Returns:
        RecurrentTensor: result of the dot product

    """
    x = lift(x)
    y = lift(y)
    x, y = upcast_tensors(x, y)

    x_dims, y_dims = x.ndim, y.ndim

    assert x_dims != 0 and y_dims != 0, (
        f"Tensors for dot must be at least 1D, but are {x_dims}D and {y_dims}D"
    )

    from tempo.api.tempo_context_manager import get_active_exec_cfg

    if get_active_exec_cfg().enable_matmul_ops:
        # To simplify things, turn every case into a batched matrix-matrix multiply
        # NOTE: this is what allows vectorize to treat this as an elementwise op
        if x_dims == 1 and y_dims == 1:
            # Turn vectors into (1, n) and (n, 1)
            x = x.unsqueeze(0)
            y = y.unsqueeze(1)
            mm = ad.MatMul.apply(x, y)[0]  # (1, 1)
            return mm.squeeze(0).squeeze(0)  # (,)

        # Case 2: x is 1D and y is at least 2D.
        elif x_dims == 1 and y_dims >= 2:
            # First, treat x as a row vector.
            x = x.unsqueeze(0)  # becomes (1, n)
            # Now, align the batch dimensions.
            # Batch dims = total dims - 2 (last two dims are the matrix part).
            x_batch: int = x.ndim - 2  # initially 0
            y_batch: int = y.ndim - 2
            for _ in range(y_batch - x_batch):
                x = x.unsqueeze(0)  # add batch dims at the front
            # After these unsqueezes, the extra (vector promotion) dimension is always
            # at index -2 (the row dimension of the promoted matrix).
            mm = ad.MatMul.apply(x, y)[0]
            # Squeeze the row dimension we added when promoting x.
            return mm.squeeze(-2)

        # Case 3: y is 1D and x is at least 2D.
        elif y_dims == 1 and x_dims >= 2:
            # Promote y to a column vector.
            y = y.unsqueeze(-1)  # becomes (n, 1)
            # Align batch dimensions.
            x_batch = x.ndim - 2
            y_batch = y.ndim - 2  # initially 0
            for _ in range(x_batch - y_batch):
                y = y.unsqueeze(0)
            # The extra (vector promotion) dimension is now at the end (i.e. -1).
            mm = ad.MatMul.apply(x, y)[0]
            return mm.squeeze(-1)

        # Case 4: Both tensors have at least 2 dimensions.
        else:
            # Align the batch dimensions (all dims except the last two).
            x_batch = x.ndim - 2
            y_batch = y.ndim - 2
            if x_batch < y_batch:
                for _ in range(y_batch - x_batch):
                    x = x.unsqueeze(0)
            elif y_batch < x_batch:
                for _ in range(x_batch - y_batch):
                    y = y.unsqueeze(0)
            mm = ad.MatMul.apply(x, y)[0]

            # Identify singleton dims in the contraction axes
            squeeze_dims = []
            if x.shape[-2] == 1:  # If x had a singleton "row" before matmul
                squeeze_dims.append(-2)
            if y.shape[-1] == 1:  # If y had a singleton "column" before matmul
                squeeze_dims.append(-1)

            # Apply the necessary squeezes
            for dim in reversed(squeeze_dims):
                mm = mm.squeeze(dim)
            return mm

    else:
        # Special case for 1D x 1D dot product
        if x_dims == 1 and y_dims == 1:
            x_dim, y_dim = x.shape.at(-1), y.shape.at(-1)
            assert ie.lift_to_int_ie(x_dim).struct_eq(ie.lift_to_int_ie(y_dim)), (
                f"Input Tensor shapes {x.shape} and {y.shape} cannot be dotted ({x_dim} != {y_dim})"
            )
            return (x * y).sum(-1)

        # Handle case where x is 1D
        if x_dims == 1:
            x = x.unsqueeze(0)  # Make it 2D
            result = dot(x, y)  # Recurse with 2D x
            return result.squeeze(-2)  # Remove the added dimension

        # Handle case where y is 1D
        if y_dims == 1:
            y = y.unsqueeze(-1)  # Make it 2D
            result = dot(x, y)  # Recurse with 2D y
            return result.squeeze(-1)  # Remove the added dimension

        # Both tensors are at least 2D at this point
        x_dim, y_dim = x.shape.at(-1), y.shape.at(-2)

        assert ie.lift_to_int_ie(x_dim).struct_eq(ie.lift_to_int_ie(y_dim)), (
            f"Input Tensor shapes {x.shape} and {y.shape} cannot be dotted ({x_dim} != {y_dim})"
        )

        # We may need to unsqueeze
        # num_unsq = builtins.min(x_dims - 1, y_dims - 1, 1)
        x_unsq = builtins.min(x_dims - 1, y_dims - 1, 1)
        y_unsq = builtins.min(x_dims - 1, y_dims - 1, 1)

        x_ = x.reshape(
            (
                *x.shape[0:-1],
                *[1] * x_unsq,
                x.shape.at(-1),
            )
        )
        y_ = y.reshape(
            (
                *y.shape[0:-2],
                *[1] * y_unsq,
                *y.shape[-2:],
            )
        )
        y_ = y_.transpose(-1, -2)
        out = (x_ * y_).sum(-1)

        return out


def _validate_conv(
    input_: RecurrentTensor,
    weight: RecurrentTensor,
    # bias: Optional[RecurrentTensor],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    transposed: bool,
    output_padding: tuple[int, ...],
    groups: int,
    N_dims: int,
) -> None:
    # NOTE: it's sort of a coincidence that they must match
    input_and_weight_match = len(input_.shape) == len(weight.shape) == N_dims + 2

    if (
        (not input_and_weight_match)
        or N_dims != len(stride)
        or N_dims != len(padding)
        or N_dims != len(dilation)
        or N_dims != len(output_padding)
    ):
        raise ValueError(
            f"All input dimensions must match in length. {N_dims}D given,\
             but {len(input_.shape)=}, {len(weight.shape)=}, {len(stride)=}, {len(padding)=},\
             {len(dilation)=}"
        )

    if transposed and len(output_padding) != N_dims:
        raise ValueError("Output padding must match the number of spatial dimensions.")

    if not transposed:
        if weight.shape.int_at(1) * groups != input_.shape.int_at(1):
            print(f"weight.shape: {weight.shape}")
            print(f"input_.shape: {input_.shape}")
            print(f"groups: {groups}")
            raise ValueError(
                f"Number of input channels must be divisible by the number of groups.\
                      {weight.shape[1]} * {groups} != {input_.shape[1]}"
            )


def conv_general(
    input_: MaybeRecurrentTensor,
    weight: MaybeRecurrentTensor,
    n_dims: int,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    transposed: bool = False,
    output_padding: int | tuple[int, ...] = 0,
    groups: int = 1,
) -> RecurrentTensor:
    """This Op performs a N-D Convolution using 2 input tensors: input and weight.
    input.shape: (B?, C_in, *N_dims_in)
    weight.shape: (C_out, C_in/groups, *N_dims_kernel)

    output.shape: (B?, C_out, *N_dims_out)

    If N_dims is None, it will be inferred from the weight shape (by subtracting 2),
    or the stride, padding, dilation, or output_padding. Whichever is the largest.
    Stride, padding, dilation, and output_padding can be either a single int or a tuple of ints of
    length n_dims. If a single int is given, it will be expanded to a tuple of N_dims ints.
    """
    input_ = lift(input_)
    weight = lift(weight)
    # bias = lift_to_rt(bias) if bias is not None else None

    if n_dims > 3 or n_dims <= 0:
        raise ValueError(f"Conv only supports 1, 2 or 3D convolutions. {n_dims}D given.")

    out_channels_dim_added_to_weight = False
    if len(weight.shape) == n_dims + 1:
        out_channels_dim_added_to_weight = True
        weight = weight.unsqueeze(0)
    assert len(weight.shape) == n_dims + 2, (
        f"Weight must be be of shape\
          (?C_out, C_in/groups, *{n_dims}_dims) but got {weight.shape}"
    )

    batch_dim_added_to_input = False
    if len(input_.shape) == n_dims + 1:
        batch_dim_added_to_input = True
        input_ = input_.unsqueeze(0)
    assert len(input_.shape) == n_dims + 2, (
        f"Input must be of shape\
          (?Batch, C_in, *{n_dims}_dims) but got {input_.shape}"
    )

    # Make all int inputs into tuples of size N_dims
    if isinstance(stride, int):
        stride = (stride,) * n_dims
    if isinstance(padding, int):
        padding = (padding,) * n_dims
    if isinstance(dilation, int):
        dilation = (dilation,) * n_dims
    if isinstance(output_padding, int):
        output_padding = (output_padding,) * n_dims

    _validate_conv(
        input_,
        weight,
        # bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        n_dims,
    )

    # NOTE: to simplify AD, we do the padding and dilation ourselves
    if builtins.sum(dilation) > n_dims:
        input_ = input_.dilate(dilations=dilation, n_dims=n_dims)
    if builtins.sum(padding) > 0:
        input_ = input_.pad(padding=padding, mode="constant", value=0)

    ret: RecurrentTensor = ad.Conv.apply(
        input_,
        weight,
        # bias=bias,
        stride=stride,
        transposed=transposed,
        groups=groups,
        n_dims=n_dims,
    )[0]

    if builtins.sum(output_padding) > 0:
        ret = ret.pad(padding=output_padding, mode="constant", value=0)

    # Remove the added dimensions
    # If batch dimension added to input, remove it
    if batch_dim_added_to_input:
        ret = ret.squeeze(0)
    # If out_channels dimension added to weight, remove it
    if out_channels_dim_added_to_weight:
        ret = ret.squeeze(0)

    return ret


def conv1d(
    input_: MaybeRecurrentTensor,
    weight: MaybeRecurrentTensor,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    transposed: bool = False,
    output_padding: int | tuple[int, ...] = 0,
    groups: int = 1,
) -> RecurrentTensor:
    return conv_general(
        input_,
        weight,
        n_dims=1,
        # bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        transposed=transposed,
        output_padding=output_padding,
        groups=groups,
    )


def conv2d(
    input_: MaybeRecurrentTensor,
    weight: MaybeRecurrentTensor,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    transposed: bool = False,
    output_padding: int | tuple[int, ...] = 0,
    groups: int = 1,
) -> RecurrentTensor:
    return conv_general(
        input_,
        weight,
        n_dims=2,
        stride=stride,
        padding=padding,
        dilation=dilation,
        transposed=transposed,
        output_padding=output_padding,
        groups=groups,
    )


def conv3d(
    input_: MaybeRecurrentTensor,
    weight: MaybeRecurrentTensor,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    transposed: bool = False,
    output_padding: int | tuple[int, ...] = 0,
    groups: int = 1,
) -> RecurrentTensor:
    return conv_general(
        input_,
        weight,
        n_dims=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        transposed=transposed,
        output_padding=output_padding,
        groups=groups,
    )


def transpose(tensor: MaybeRecurrentTensor, dim0: int = 1, dim1: int = 0) -> RecurrentTensor:
    t = lift(tensor)
    dim0 = normalize_negative_dim(dim0, t.shape)
    dim1 = normalize_negative_dim(dim1, t.shape)
    assert len(t.shape) >= 2, "Transpose requires a tensor of at least 2 dimensions"
    order = list(range(len(t.shape)))
    order[dim0], order[dim1] = order[dim1], order[dim0]
    return t.permute(tuple(order))


def reshape(
    tensor: MaybeRecurrentTensor,
    shape: ShapeLike,
) -> RecurrentTensor:
    t = lift(tensor)
    shape = Shape.from_(shape)

    if shape.has_negative_dim():
        shape = normalize_neg_1s_in_shape_reshape(t.shape, shape)

    if t.shape == shape:
        return t

    ret: RecurrentTensor = ad.Reshape.apply(t, shape=shape)[0]
    return ret


def flatten(tensor: MaybeRecurrentTensor, start_dim: int = 0, end_dim: int = -1) -> RecurrentTensor:
    t = lift(tensor)

    start_dim = normalize_negative_dim(start_dim, t.shape)
    end_dim = normalize_negative_dim(end_dim, t.shape)

    if start_dim > end_dim:
        raise ValueError(f"start_dim must be less than end_dim, got {start_dim} and {end_dim}")

    flattened_shape = (
        t.shape._shape[:start_dim]
        + (t.shape[start_dim : end_dim + 1].prod(),)
        + t.shape._shape[end_dim + 1 :]
    )
    return t.reshape(flattened_shape)


def permute(tensor: MaybeRecurrentTensor, dims: tuple[int, ...]) -> RecurrentTensor:
    t = lift(tensor)
    dims = normalize_negative_dim_tuple(dims, t.shape)
    assert len(t.shape) == len(dims), (
        f"Tensor shape {t.shape} and dims {dims} must have the same length for permute"
    )
    ret: RecurrentTensor = ad.Permute.apply(t, dims=dims)[0]
    return ret


def neg(x: MaybeRecurrentTensor) -> RecurrentTensor:
    x = lift(x)

    # NOTE: unsure if this is needed, or we should handle it in tensorops and backend op emitters.
    # Convert unsigned types to signed before neg since neg can produce negative values
    if dtypes.is_unsigned_int(x.dtype):
        x = x.cast(dtypes.least_upper_signed_int(x.dtype))
    ret: RecurrentTensor = ad.Negate.apply(x)[0]
    return ret


def sin(x: MaybeRecurrentTensor) -> RecurrentTensor:
    x = lift(x)
    x = x.cast(dtypes.least_upper_float(x.dtype))
    ret: RecurrentTensor = ad.Sin.apply(x)[0]
    return ret


def cos(x: MaybeRecurrentTensor) -> RecurrentTensor:
    x = lift(x)
    # NOTE: this cast ensures enough precision in our representation of pi
    half_pi = const(math.pi / 2.0, dtype=dtypes.float64)
    # half_pi = const(math.pi / 2.0, dtype=dtypes.upcast(x.dtype, dtypes.float32))
    return (half_pi - x).sin().cast(dtypes.least_upper_float(x.dtype))


def tan(x: MaybeRecurrentTensor) -> RecurrentTensor:
    x = lift(x)
    return x.sin() / x.cos()


def relu(x: MaybeRecurrentTensor) -> RecurrentTensor:
    x = lift(x)
    ret: RecurrentTensor = ad.Relu.apply(x)[0]
    return ret


def sqrt(x: MaybeRecurrentTensor) -> RecurrentTensor:
    x = lift(x)
    x = x.cast(dtypes.least_upper_float(x.dtype))
    ret: RecurrentTensor = ad.Sqrt.apply(x)[0]
    return ret


def sigmoid(x: MaybeRecurrentTensor) -> RecurrentTensor:
    x = lift(x)
    x = x.cast(dtypes.least_upper_float(x.dtype))
    ret: RecurrentTensor = ad.Sigmoid.apply(x)[0]
    return ret


def swish(x: MaybeRecurrentTensor) -> RecurrentTensor:
    x = lift(x)
    return x * sigmoid(x)


silu = swish


def softmax(x: MaybeRecurrentTensor, dim: int = -1, stable: bool = True) -> RecurrentTensor:
    x = lift(x)
    dim = normalize_negative_dim(dim, x.shape)
    x_dtype = x.dtype

    # NOTE: Always use at least float32 for softmax.
    sm_dtype = dtypes.least_upper_float(dtypes.upcast(x_dtype, dtypes.float32))

    # TODO: Ideally, this should eventually be automated, by recognizing that
    # a sum of exp has the potential to overflow. The way we recognize this should be general.
    if stable:
        max_val = x.max(dim=dim, keepdim=True)[0]
        x = x - max_val

    exp_ = x.cast(sm_dtype).exp()
    res = exp_ / exp_.sum(dims=dim, keepdim=True)
    return res.cast(x_dtype)


def cross_entropy(
    pred: MaybeRecurrentTensor, target: MaybeRecurrentTensor, class_dim: int = -1
) -> RecurrentTensor:
    """
    Computes the cross-entropy loss along the specified dimension.

    Args:
        pred (MaybeRecurrentTensor): The predicted probabilities (e.g. from softmax).
        target (MaybeRecurrentTensor): The target probabilities or one-hot labels.
        class_dim (int): The dimension representing the features (e.g., classes).

    Returns:
        RecurrentTensor: The computed cross-entropy loss.
    """
    pred = lift(pred)
    target = lift(target)

    per_class = target * pred.log()
    # Sum over the class dimension
    summed = per_class.sum(dims=class_dim)
    return -summed


def tanh(x: MaybeRecurrentTensor) -> RecurrentTensor:
    x = lift(x)

    # This version is cheap but not very stable
    # e = x.exp()
    # inv_e = const(1.0) / e
    # return (e - inv_e) / (e + inv_e)

    # This version is more stable
    # exp_2x = (2 * x).exp()
    # return (exp_2x - 1) / (exp_2x + 1)

    # This version is the most stable (tinygrad)
    return const(2.0) * ((const(2.0) * x).sigmoid()) - const(1.0)


def elu(x: MaybeRecurrentTensor, alpha: float = 1.0) -> RecurrentTensor:
    x = lift(x)
    return x.relu() - const(alpha) * (const(1) - x.exp()).relu()


def leakyrelu(x: MaybeRecurrentTensor, neg_slope: float = 0.01) -> RecurrentTensor:
    x = lift(x)
    return x.relu() - (const(-neg_slope) * x).relu()


def softplus(x: MaybeRecurrentTensor, beta: int = 1) -> RecurrentTensor:
    x = lift(x)
    return const(1.0 / beta) * ((x * const(beta)).exp() + const(1)).ln()


def mish(x: MaybeRecurrentTensor) -> RecurrentTensor:
    x = lift(x)
    return x * x.softplus().tanh()


def sum(  # noqa: A001
    x: MaybeRecurrentTensor, dims: DIM_TYPE = None, keepdim: bool = False
) -> RecurrentTensor:
    x = lift(x)

    if isinstance(dims, tuple) and len(dims) == 0:
        return x

    # TODO similar optimizations can be done for other functions
    if x.shape == ():
        return x
    if x.shape == (1,):
        if keepdim:
            return x
        else:
            return x.squeeze(0)

    dims = normalize_dims(dims, x.shape)

    # Validate that dims is within bounds of shape
    for dim in dims:
        if dim < 0 or dim >= len(x.shape):
            raise ValueError(f"Dimension {dim} is out of bounds for tensor of shape {x.shape}")

    ret: RecurrentTensor = ad.Sum.apply(x, reduce_dims=dims, keepdim=keepdim)[0]
    return ret


def cumsum(x: MaybeRecurrentTensor, dim: int = -1) -> RecurrentTensor:  # noqa: A001
    x = lift(x)
    dim = normalize_negative_dim(dim, x.shape)
    ret: RecurrentTensor = ad.CumSum.apply(x, dim=dim)[0]
    return ret


def cumprod(x: MaybeRecurrentTensor, dim: int = -1, positive: bool = False) -> RecurrentTensor:  # noqa: A001
    x = lift(x)
    dim = normalize_negative_dim(dim, x.shape)

    int_dtype = dtypes.least_upper_signed_int(x.dtype)

    if positive:
        # User contract that all values are positive, no need to handle edge cases
        return x.log().cumsum(dim).exp()

    # Identify zeros and negatives
    is_zero: RecurrentTensor = x == 0  # type: ignore
    is_negative: RecurrentTensor = x < 0  # type: ignore

    # Compute cumulative sum of negatives to determine sign changes
    cumulative_negatives = is_negative.cast(int_dtype).cumsum(dim)

    # Create a mask where cumulative zeros have occurred
    cumulative_zeros: RecurrentTensor = is_zero.cumsum(dim) > 0  # type: ignore

    # Compute the cumulative sign: (-1)^(number of negatives encountered)
    cumprod_sign = (-1) ** cumulative_negatives

    # Make all values positive (we only care about magnitude for now)
    abs_x = x.abs()

    # Compute the cumulative sum of the logarithms
    cumprod_abs = abs_x.log().cumsum(dim).exp()

    # Restore the sign
    result = cumprod_sign * cumprod_abs

    # Any 0s will cause the cumulative product to be 0 from that point on
    result = cumulative_zeros.where(0, result)

    return result

    # sign_ = x.sign()
    # return x.log().cumsum(dim).exp()


def discounted_sum(
    x: MaybeRecurrentTensor,
    gamma: MaybeRecurrentTensor,
    dim: int = 0,
) -> RecurrentTensor:
    """Computes the discounted sum of a tensor along a dimension.

    This is given by the formula:
    \\sum_{t=0}^{T-1} \\gamma^t x_t = x_0^{0} + \\gamma^{1} x_1 + \\ldots + \\gamma^{T-1} x_{T-1}



    Args:
        x (MaybeRecurrentTensor): _description_
        gamma (MaybeRecurrentTensor): _description_
        dim (int, optional): _description_. Defaults to 0.

    Returns:
        RecurrentTensor: _description_

    """
    x, gamma = lift_all_to_rt(x, gamma)
    x_dtype = x.dtype
    dim = normalize_negative_dim(dim, x.shape)

    # Setup the discount factors
    gamma = gamma.expand(x.shape).cast(dtypes.float64)
    discount_factors = gamma.cumprod(dim=dim, positive=True)

    # Multiply by the discount factors
    discounted_x = x * discount_factors

    # cumulative sum
    return discounted_x.cast(x_dtype).sum(dims=dim)


def handle_symbolic_dim(
    orig_tensor: RecurrentTensor, dim: int, keep_symbolic_dim: bool, new_tensor: RecurrentTensor
) -> RecurrentTensor:
    """Helper to restore the symbolic dimension if `keep_symbolic_dim` is True.

    Args:
        orig_tensor: The input tensor with index expressions.
        dim: Target dimension to identify symbolic slice.
        keep_symbolic_dim: Whether to keep the symbolic dimension.
        new_tensor: The tensor after cumulative operations.

    Returns:
        The tensor with the symbolic dimension restored if required.

    """
    if not keep_symbolic_dim:
        return new_tensor

    x_orig_dim_var = None
    num_slices = 0
    for e, v in zip(
        orig_tensor._underlying.index_expr, orig_tensor.unindexed_domain.variables, strict=False
    ):
        if isinstance(e, ie.Slice):
            if num_slices == dim:
                x_orig_dim_var = v
            num_slices += 1

    assert x_orig_dim_var is not None, (
        "Could not find the symbolic dimension to restore in the discounted cumulative sum"
    )
    return new_tensor.index(dim, x_orig_dim_var)


def discounted_cum_sum(
    x: MaybeRecurrentTensor,
    gamma: MaybeRecurrentTensor,
    dim: int = 0,
    keep_symbolic_dim: bool = True,
) -> RecurrentTensor:
    x, gamma = lift_all_to_rt(x, gamma)
    x_ori = x
    x_dtype = x.dtype
    accum_dtype = dtypes.least_upper_float(dtypes.upcast(x_dtype, dtypes.float64))

    # x, gamma = x.cast(dtypes.float64), gamma.cast(dtypes.float64)
    dim = normalize_negative_dim(dim, x.shape)

    # Setup the discount factors
    gamma_expanded = gamma.expand(x.shape)
    gamma_powers = gamma_expanded.cumprod(dim, positive=True) / gamma

    discounted_cumsum = (x * gamma_powers).flip(dim).cast(accum_dtype).cumsum(dim=dim).flip(dim)
    returns = discounted_cumsum / gamma_powers
    returns = returns.cast(x_dtype)

    # Restore symbolic dimension if necessary
    return handle_symbolic_dim(x_ori, dim, keep_symbolic_dim, returns)


def expand(x: MaybeRecurrentTensor, shape: ShapeLike) -> RecurrentTensor:
    shape = Shape.from_(shape)
    x = lift(x)
    # NOTE: this is here to replicate torch behaviour safely.
    # TODO: I think we have a utility for this somewhere...
    if len(x.shape) < len(shape):
        diff = len(shape) - len(x.shape)
        for _ in range(diff):
            x = x.unsqueeze(0)
    shape = normalize_neg_1s_in_shape_expand(x.shape, shape)
    if x.shape == shape:
        return x
    ret: RecurrentTensor = ad.Expand.apply(x, shape=shape)[0]
    return ret


def repeat(x: MaybeRecurrentTensor, num_repeats: int | Sequence[int]) -> RecurrentTensor:
    x = lift(x)
    num_repeats = as_seq(num_repeats)

    num_repeats, _ = unsq_align_shapes_1_pad_right((num_repeats, x.shape))
    base_shape, _ = unsq_align_shapes_1_pad_left((x.shape, num_repeats))

    if all(r == 1 for r in num_repeats):
        return x

    def flatten(l: Sequence[Sequence[ie.IntIndexValueLike]]) -> list[ie.IntIndexValueLike]:
        return [item for sublist in l for item in sublist]

    # NOTE: Unsqueeze behind each dimension in the base shape to allow for expanding that dimension
    unsq_shape = flatten([[1, s] for s in base_shape])

    expanded_shape = flatten([[n, s] for n, s in zip(num_repeats, base_shape, strict=False)])

    final_shape = tuple(r * s for r, s in zip(num_repeats, base_shape, strict=False))

    return x.reshape(unsq_shape).expand(expanded_shape).reshape(final_shape)


def repeat_interleave(x: MaybeRecurrentTensor, num_repeats: int, dim: int) -> RecurrentTensor:
    x = lift(x)
    if num_repeats == 1:
        return x
    dim = normalize_negative_dim(dim, x.shape)
    s = x.shape
    return (
        x.reshape(Shape.from_(s[: dim + 1] + (1,) + s[dim + 1 :]))
        .expand(Shape.from_(s[: dim + 1] + (num_repeats,) + s[dim + 1 :]))
        .reshape(Shape.from_(s[:dim] + (num_repeats * s.at(dim),) + s[dim + 1 :]))
    )


def _flip_one_dim(x: RecurrentTensor, dim: int) -> RecurrentTensor:
    dim = normalize_negative_dim(dim, x.shape)
    ret: RecurrentTensor = ad.Flip.apply(x, dim=dim)[0]
    return ret


def flip(x: MaybeRecurrentTensor, dim: int | Sequence[int]) -> RecurrentTensor:
    x = lift(x)
    dims = as_seq(dim)
    # No duplicates
    assert len(dims) == len(set(dims)), "Duplicate dimensions in flip: " + str(dims)
    ret = x
    for d in dims:
        ret = _flip_one_dim(ret, d)
    return ret


def where(
    condition: MaybeRecurrentTensor,
    true_data: MaybeRecurrentTensor,
    false_data: MaybeRecurrentTensor,
) -> RecurrentTensor:
    condition, true_data, false_data = lift_all_to_rt(condition, true_data, false_data)
    condition, true_data, false_data = broadcast_tensors(condition, true_data, false_data)
    true_data, false_data = upcast_tensors(true_data, false_data)
    ret: RecurrentTensor = ad.Where.apply(condition, true_data, false_data)[0]
    return ret


def gather(
    src: MaybeRecurrentTensor,
    dim: int,
    index: MaybeRecurrentTensor,
) -> RecurrentTensor:
    src, index = lift_all_to_rt(src, index)

    if len(src.shape) < 2:
        return RecurrentTensor.index(src, dim, index)

    # if len(index.shape) == 0:
    #    index = index.unsqueeze(0)
    # src, index = broadcast_tensors(src, index)
    dim = normalize_negative_dim(dim, src.shape)

    index = index.cast(dtypes.default_int)
    ret: RecurrentTensor = ad.Gather.apply(src, index, dim=dim)[0]
    return ret


def scatter_add(
    sink: MaybeRecurrentTensor,
    dim: int,
    index: MaybeRecurrentTensor,
    src: MaybeRecurrentTensor,
) -> RecurrentTensor:
    sink, index, src = lift_all_to_rt(sink, index, src)
    index, src = broadcast_tensors(index, src)
    # TODO which tensors do I need to broadcast ?
    sink, src = upcast_tensors(sink, src)

    dim = normalize_negative_dim(dim, sink.shape)  # TODO make sure not src
    index = index.cast(dtypes.default_int)
    ret: RecurrentTensor = ad.ScatterAdd.apply(sink, index, src, dim=dim)[0]
    return ret


def scatter(
    dim: int,
    index: MaybeRecurrentTensor,
    src: MaybeRecurrentTensor,
) -> RecurrentTensor:
    # TODO broadcast ?
    index, src = lift_all_to_rt(index, src)
    sink: RecurrentTensor = zeros_like(src)
    dim = normalize_negative_dim(dim, sink.shape)  # TODO sink or src?
    index = index.cast(dtypes.default_int)
    return scatter_add(sink, dim, index, src)


def one_hot(class_int: MaybeRecurrentTensor, num_classes: int) -> RecurrentTensor:
    class_int = lift(class_int)
    # one_hot = zeros((num_classes,))
    # return one_hot.scatter_add(-1, class_int, lift(1))

    # Create a tensor of indices [0, 1, 2, ..., num_classes-1]
    indices = RecurrentTensor.arange(num_classes)
    # Compare each index with class_int to create a boolean mask
    # This will be 1 where index matches class_int, 0 elsewhere
    # The equality comparison is automatically broadcasted
    one_hot = (indices.equals(class_int)).cast(class_int.dtype)
    return one_hot


def squeeze(x: MaybeRecurrentTensor, dims: DIM_TYPE = None) -> RecurrentTensor:
    x = lift(x)
    if dims is None:
        dims = tuple(i for i, d in enumerate(x.shape) if isinstance(d, int) and d == 1)
    if isinstance(dims, int):
        dims = (dims,)
    dims = normalize_negative_dim_tuple(dims, x.shape)
    ret = x
    for dim in sorted(dims, reverse=True):
        ret = ad.Squeeze.apply(ret, dim=dim)[0]
    return ret


def unsqueeze(x: MaybeRecurrentTensor, dim: int = 0) -> RecurrentTensor:
    x = lift(x)
    dim = normalize_negative_dim(dim, x.shape, allow_end=True)
    ret: RecurrentTensor = ad.Unsqueeze.apply(x, dim=dim)[0]
    return ret


def split(x: MaybeRecurrentTensor, dim: int, num_splits: int) -> Sequence[RecurrentTensor]:
    x = lift(x)
    dim = normalize_negative_dim(dim, x.shape)
    ret: Sequence[RecurrentTensor] = ad.Split.apply(x, dim=dim, num_splits=num_splits)
    return ret


def cat(
    xs: Sequence[MaybeRecurrentTensor] | MaybeRecurrentTensor,
    *args: MaybeRecurrentTensor,
    dim: int = 0,
) -> RecurrentTensor:
    if not isinstance(xs, (list, tuple, Sequence)):
        xs = [xs, *args]  # type: ignore

    xs_rt = lift_all_to_rt(*xs)
    assert len(xs_rt) >= 2, "Must have at least two tensors to concatenate"
    assert xs_rt[0].shape.equal_in_all_but_dim(*tuple(x_.shape for x_ in xs_rt[1:]), dim=dim), (
        f"Tensor shapes must have same dim size except in cat dim, \
        but got {tuple(x_.shape for x_ in xs_rt)}"
    )
    dim = normalize_negative_dim(dim, xs_rt[0].shape)
    catted: RecurrentTensor = ad.Cat.apply(*xs_rt, dim=dim)[0]
    return catted


def stack(
    xs: Sequence[MaybeRecurrentTensor] | MaybeRecurrentTensor,
    *args: MaybeRecurrentTensor,
    dim: int = 0,
) -> RecurrentTensor:
    if not isinstance(xs, (list, tuple, Sequence)):
        xs = [xs, *args]  # type: ignore

    xs_rt = lift_all_to_rt(*xs)

    assert all(x.ndim == xs_rt[0].ndim for x in xs_rt), (
        "All tensors must have the same number of dimensions"
    )

    dim = normalize_negative_dim(dim, xs_rt[0].shape, allow_end=True)

    return cat(*[x.unsqueeze(dim) for x in xs_rt], dim=dim)


def const(
    val: Any,
    shape: ShapeLike = None,
    dtype: DataTypeLike = None,
    domain: DomainLike = None,
    # TODO: this is silly, consts have no domain, so no updates
    requires_grad: None | bool = None,
) -> RecurrentTensor:
    dtype = dtypes.from_(dtype, none_dtype=dtypes.implied(val))

    if not isinstance(val, (int, float, bool, np.ndarray)):
        val = np.array(val, dtype=dtypes.to_np(dtype))

    if isinstance(val, np.ndarray):
        assert shape is None, "Cannot specify shape when val is an np.ndarray"
        shape = Shape.from_(val.shape)
    else:
        if shape is None:
            shape = Shape.scalar()
        else:
            shape = Shape.from_(shape)

    if domain is not None:
        _logger.warning("Domain parameters are ignored for const.")

    st = SymbolicTensor.full(val, shape, dtype)
    return RecurrentTensor(st, requires_grad=requires_grad)


def const_like(
    x: float | bool | int,
    y: RecurrentTensor,
    requires_grad: bool | None = None,
) -> RecurrentTensor:
    return const(x, shape=y.shape, requires_grad=requires_grad)


def ones(
    shape: ShapeLike = (),
    dtype: DataTypeLike = None,
    domain: DomainLike = None,
    requires_grad: bool | None = None,
) -> RecurrentTensor:
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    return const(1.0, shape, dtype, domain, requires_grad)


def ones_like(
    y: RecurrentTensor,
    dtype: DataTypeLike = None,
    requires_grad: bool | None = None,
) -> RecurrentTensor:
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    return ones(y.shape, dtype, y.domain, requires_grad)


def arange(
    stop: int,
    start: int = 0,
    step: int = 1,
    dtype: DataTypeLike = None,
    requires_grad: bool | None = None,
) -> RecurrentTensor:
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_int)

    st = SymbolicTensor.arange(stop, start, step, dtype)
    return RecurrentTensor(st, requires_grad=requires_grad)


# TODO a version of arange that accepts RecurrentTensors for start, stop, step
# def arange(
#    stop: MaybeRecurrentTensor,
#    start: MaybeRecurrentTensor = 0,
#    step: MaybeRecurrentTensor = 1,
#    dtype: DataType = dtypes.int64,
#    requires_grad: Optional[bool] = None,
# ) -> RecurrentTensor:
#
#    if isinstance(start, int) and isinstance(stop, int) and isinstance(step, int):
#        return _int_arange(stop, start, step, dtype, requires_grad)
#
#    start, stop, step = lift_all_to_rt(start, stop, step)
#    size = ceil((stop - start) / step).cast(dtypes.int64)
#
#    a = step.expand()
#    st = SymbolicTensor.arange(stop, start, step, dtype)
#    return RecurrentTensor(st, requires_grad=requires_grad)


def zeros(
    shape: ShapeLike = (),
    dtype: DataTypeLike = None,
    domain: DomainLike = None,
    requires_grad: bool | None = None,
) -> RecurrentTensor:
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    return const(0.0, shape, dtype, domain, requires_grad)


def zeros_like(
    y: RecurrentTensor,
    dtype: DataTypeLike = None,
    requires_grad: bool | None = None,
) -> RecurrentTensor:
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    return zeros(y.shape, dtype, y.domain, requires_grad)


@overload
def min(  # noqa: A001, A003
    x: MaybeRecurrentTensor, dim: int = 1, keepdim: bool = False
) -> tuple[RecurrentTensor, RecurrentTensor]: ...


@overload
def min(  # noqa: A001, A003
    *xs: MaybeRecurrentTensor,
) -> tuple[RecurrentTensor, RecurrentTensor]: ...


def min(  # noqa: A001, A003
    *args: MaybeRecurrentTensor, **kwargs: Any
) -> tuple[RecurrentTensor, RecurrentTensor]:
    if len(args) == 1:
        x_tt = lift(args[0])
        vals, idxs = max(-x_tt, **kwargs)
        return -vals, idxs
    elif len(args) > 1:
        xs_tt = lift_all_to_rt(*args)
        xs_tt = broadcast_tensors(*xs_tt)
        dim = 0
        xs_unsq = tuple(x.unsqueeze(dim) for x in xs_tt)
        x = RecurrentTensor.cat(*xs_unsq, dim=dim)
        vals, idxs = max(-x, dim=dim, keepdim=False)
        return -vals, idxs
    else:
        raise Exception("Must pass at least one tensor to min")


@overload
def max(  # noqa: A001, A003
    x: MaybeRecurrentTensor, dim: int = 1, keepdim: bool = False
) -> tuple[RecurrentTensor, RecurrentTensor]: ...


@overload
def max(  # noqa: A001, A003
    *xs: MaybeRecurrentTensor,
) -> tuple[RecurrentTensor, RecurrentTensor]: ...


def max(  # noqa: A001, A003
    *args: MaybeRecurrentTensor, **kwargs: Any
) -> tuple[RecurrentTensor, RecurrentTensor]:
    if len(args) == 1:
        x_tt = lift(args[0])
        dim: int = int(kwargs.get("dim", 1))
        dim = normalize_negative_dim(dim, x_tt.shape)
        keepdim: bool = bool(kwargs.get("keepdim", False))
        vals, idxs = ad.Max.apply(x_tt, dim=dim, keepdim=keepdim)
        return vals, idxs
    elif len(args) > 1:
        xs_tt = lift_all_to_rt(*args)
        xs_tt = broadcast_tensors(*xs_tt)
        dim = 0
        xs_unsq = tuple(x.unsqueeze(dim) for x in xs_tt)
        x = RecurrentTensor.cat(*xs_unsq, dim=dim)
        vals, idxs = ad.Max.apply(x, dim=dim, keepdim=False)
        return vals, idxs
    else:
        raise Exception("Must pass at least one tensor to max")


def argmax(x: MaybeRecurrentTensor, dim: int = 1, keepdim: bool = False) -> RecurrentTensor:
    return max(x, dim=dim, keepdim=keepdim)[1]


def argmin(x: MaybeRecurrentTensor, dim: int = 1, keepdim: bool = False) -> RecurrentTensor:
    return min(x, dim=dim, keepdim=keepdim)[1]


def clip(
    x: MaybeRecurrentTensor,
    lb: MaybeRecurrentTensor | None = None,
    ub: MaybeRecurrentTensor | None = None,
) -> RecurrentTensor:
    if lb is not None:
        x = max(*broadcast_tensors(*lift_all_to_rt(x, lb)))[0]
    if ub is not None:
        x = min(*broadcast_tensors(*lift_all_to_rt(x, ub)))[0]
    return lift(x)


def _mul_generic(
    left: int | ie.IntIndexValue, right: int | ie.IntIndexValue
) -> int | ie.IntIndexValue:
    return left * right


def mean(
    x: MaybeRecurrentTensor,
    dims: DIM_TYPE = None,
    keepdim: bool = False,
) -> RecurrentTensor:
    x = lift(x)
    dims = normalize_dims(dims, x.shape)

    size = x.size(dims)
    sizes = (size,) if not isinstance(size, tuple) else size
    denominator = lift(functools.reduce(_mul_generic, sizes))

    summed = x.sum(dims=dims, keepdim=keepdim)
    x_mean = summed / denominator
    return x_mean  # type: ignore


def std(
    x: MaybeRecurrentTensor,
    dims: DIM_TYPE = None,
    correction: float = 1.0,
) -> RecurrentTensor:
    x = lift(x)
    dims = normalize_dims(dims, x.shape)

    size = x.size(dims)
    sizes = (size,) if not isinstance(size, tuple) else size
    denominator = lift(functools.reduce(_mul_generic, sizes) - 1)

    x_mean = x.mean(dims=dims)

    x_std = (((x - x_mean) ** 2).sum(dims=dims, keepdim=False) / denominator).sqrt()

    return x_std  # type: ignore


def normalize(
    x: MaybeRecurrentTensor,
    mean: MaybeRecurrentTensor,
    std: MaybeRecurrentTensor,
    eps: MaybeRecurrentTensor = 1e-8,
) -> RecurrentTensor:
    x = lift(x)
    return (x - mean) / (lift(std) + eps)


def abs(x: MaybeRecurrentTensor) -> RecurrentTensor:  # noqa: A001, A003
    x = lift(x)
    return x.relu() + (-x).relu()


def erf(x: MaybeRecurrentTensor) -> RecurrentTensor:
    """See: https://pytorch.org/docs/stable/special.html#torch.special.erf"""
    x = lift(x)
    t = 1.0 / (1.0 + 0.3275911 * x.abs())
    term1 = 0.254829592 * t
    term2 = -0.284496736 * t**2
    term3 = 1.421413741 * t**3
    term4 = -1.453152027 * t**4
    term5 = 1.061405429 * t**5
    y = term1 + term2 + term3 + term4 + term5
    z = 1.0 - y * RecurrentTensor.exp(-x * x)
    return (x > 0).where(z, -z)


# ============= Non differentiable ops


def logical_not(x: MaybeRecurrentTensor) -> RecurrentTensor:
    x = lift(x)
    assert x._underlying is not None, "Can't apply logical_not to uninitialized tensor"
    return RecurrentTensor(SymbolicTensor.logical_not(x._underlying))


def logical_or(x: MaybeRecurrentTensor, y: MaybeRecurrentTensor) -> RecurrentTensor:
    x, y = lift_all_to_rt(x, y)
    x, y = broadcast_tensors(x, y)
    assert x._underlying is not None and y._underlying is not None, (
        "Can't apply logical_or to uninitialized tensor"
    )
    return RecurrentTensor(SymbolicTensor.logical_or(x._underlying, y._underlying))


def logical_and(x: MaybeRecurrentTensor, y: MaybeRecurrentTensor) -> RecurrentTensor:
    x, y = lift_all_to_rt(x, y)
    x, y = broadcast_tensors(x, y)
    assert x._underlying is not None and y._underlying is not None, (
        "Can't apply logical_and to uninitialized tensor"
    )
    return RecurrentTensor(SymbolicTensor.logical_and(x._underlying, y._underlying))


def equals(x: MaybeRecurrentTensor, y: MaybeRecurrentTensor) -> RecurrentTensor:
    x, y = lift_all_to_rt(x, y)
    x, y = broadcast_tensors(x, y)
    assert x._underlying is not None and y._underlying is not None, (
        "Can't apply equals to uninitialized tensor"
    )
    return RecurrentTensor(SymbolicTensor.equals(x._underlying, y._underlying))


def less_than(x: MaybeRecurrentTensor, y: MaybeRecurrentTensor) -> RecurrentTensor:
    x, y = lift_all_to_rt(x, y)
    x, y = broadcast_tensors(x, y)
    assert x._underlying is not None and y._underlying is not None, (
        "Can't apply less_than to uninitialized tensor"
    )
    return RecurrentTensor(SymbolicTensor.less_than(x._underlying, y._underlying))


def less_than_or_equal(x: MaybeRecurrentTensor, y: MaybeRecurrentTensor) -> RecurrentTensor:
    x, y = lift_all_to_rt(x, y)
    x, y = broadcast_tensors(x, y)
    assert x._underlying is not None and y._underlying is not None, (
        "Can't apply less_than_or_equal to uninitialized tensor"
    )
    return RecurrentTensor(SymbolicTensor.less_than_or_equal(x._underlying, y._underlying))


def greater_than(x: MaybeRecurrentTensor, y: MaybeRecurrentTensor) -> RecurrentTensor:
    x, y = lift_all_to_rt(x, y)
    x, y = broadcast_tensors(x, y)
    assert x._underlying is not None and y._underlying is not None, (
        "Can't apply greater_than to uninitialized tensor"
    )
    return RecurrentTensor(SymbolicTensor.greater_than(x._underlying, y._underlying))


def greater_than_or_equal(x: MaybeRecurrentTensor, y: MaybeRecurrentTensor) -> RecurrentTensor:
    x, y = lift_all_to_rt(x, y)
    x, y = broadcast_tensors(x, y)
    assert x._underlying is not None and y._underlying is not None, (
        "Can't apply greater_than_or_equal to uninitialized tensor"
    )
    return RecurrentTensor(SymbolicTensor.greater_than_or_equal(x._underlying, y._underlying))


def ident(x: MaybeRecurrentTensor) -> RecurrentTensor:
    x = lift(x)

    # TODO: this will cause problems with setting grads...
    if x._underlying.index_expr.struct_eq(x.unindexed_domain.basis_expr):
        return RecurrentTensor(x._underlying.copy_with_no_index(), x.requires_grad, x._ctx, x.grad)

    res: RecurrentTensor = ad.Ident.apply(x)[0]
    return res


# ============== END Non differentiable ops

# ================ DISTRIBUTIONS ======================


def random(
    shape: ShapeLike = (),
    dtype: DataTypeLike = None,
    domain: DomainLike = None,
    requires_grad: bool | None = None,
) -> RecurrentTensor:
    out_dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    shape = Shape.from_(shape)
    domain = Domain.from_(domain, True)

    # NOTE: We want to use at least float32 precision?
    rand = RecurrentTensor(
        SymbolicTensor.rand(shape, dtypes.float32, domain), requires_grad=requires_grad
    )
    return rand.cast(out_dtype)


def random_bool(
    shape: ShapeLike = (),
    domain: DomainLike = None,
    requires_grad: bool | None = None,
) -> RecurrentTensor:
    return RecurrentTensor.uniform(
        shape,
        low=0,
        high=2,
        dtype=dtypes.int32,  # NOTE: This instance of int32 is okay
        domain=domain,
        requires_grad=requires_grad,
    ).cast(dtypes.bool_)


def random_int(
    low: MaybeRecurrentTensor = 0,
    high: MaybeRecurrentTensor = 10,
    shape: ShapeLike = (),
    dtype: DataTypeLike = None,
    domain: DomainLike = None,
    requires_grad: bool | None = None,
) -> RecurrentTensor:
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_int)
    return RecurrentTensor.uniform(
        shape,
        low=low,
        high=high,
        dtype=dtype,
        domain=domain,
        requires_grad=requires_grad,
    )


def uniform(
    shape: ShapeLike = (),
    low: MaybeRecurrentTensor = 0.0,
    high: MaybeRecurrentTensor = 1.0,
    dtype: DataTypeLike = None,
    domain: DomainLike = None,
    requires_grad: bool | None = None,
) -> RecurrentTensor:
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    high, low = _lift_if_index_value(high), _lift_if_index_value(low)
    return (
        RecurrentTensor.random(shape, dtype=dtype, domain=domain, requires_grad=requires_grad)
        * (lift(high) - low)
        + low
    ).cast(dtype)


def scaled_uniform(
    shape: ShapeLike = (),
    dtype: DataTypeLike = None,
    domain: DomainLike = None,
    requires_grad: bool | None = None,
) -> RecurrentTensor:
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    shape = Shape.from_(shape)
    ret: RecurrentTensor = RecurrentTensor.uniform(
        shape,
        low=-1.0,
        high=1.0,
        dtype=dtype,
        domain=domain,
        requires_grad=requires_grad,
    ) * (shape.prod() ** -0.5)
    return ret


# NOTE: xavier or glorot uniform
def linear_init_uniform(
    shape: ShapeLike = (),
    dtype: DataTypeLike = None,
    domain: DomainLike = None,
    num_input_features: int | None = None,
    requires_grad: bool | None = None,
) -> RecurrentTensor:
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    shape = Shape.from_(shape)
    if num_input_features is None:
        num_input_features = int(shape.at(1))
    c = math.sqrt(1.0 / int(num_input_features))

    return RecurrentTensor.uniform(
        shape, low=-c, high=c, dtype=dtype, domain=domain, requires_grad=requires_grad
    )


def glorot_uniform(
    shape: ShapeLike = (),
    dtype: DataTypeLike = None,
    domain: DomainLike = None,
    requires_grad: bool | None = None,
) -> RecurrentTensor:
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    shape = Shape.from_(shape)

    shape_portion = shape._shape[0] + shape[1:].prod()
    shape_portion = lift(shape_portion)
    ret: RecurrentTensor = RecurrentTensor.uniform(
        shape,
        low=-1.0,
        high=1.0,
        dtype=dtype,
        domain=domain,
        requires_grad=requires_grad,
    ) * ((6 / shape_portion) ** 0.5)
    return ret


def _external_init_as_const(backend_tensor: BackendTensorT) -> RecurrentTensor:
    np_array = np.asarray(backend_tensor)
    return const(np_array)


def init_from_statedict(
    flat_state_dict: dict[str, BackendTensorT],
    key: Callable[[dict[ie.Symbol, int]], str] | str,
    shape: ShapeLike = None,
    dtype: DataTypeLike = None,
    domain: DomainLike = None,
    skip_cast_dev_and_bend: bool = False,
) -> RecurrentTensor:
    dtype_ = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    from tempo.core.dl_backend import DLBackend

    exec_cfg = get_active_exec_cfg()
    user_backend = DLBackend.get_backend(exec_cfg.backend)

    if not isinstance(key, str):
        assert shape is not None, "shape must be provided if key is a callable"
        assert dtype_ is not None, "dtype must be provided if key is a callable"
        # assert domain is not None, "domain must be provided if key is a callable"
    else:
        shape = Shape.from_(tuple(flat_state_dict[key].shape))
        dtype_ = user_backend.to_tpo_dtype(flat_state_dict[key].dtype)

    # NOTE: We will leverage the torch implementation of orthogonal_init
    dev_tpo = device.from_(exec_cfg.dev)
    dev = user_backend.to_backend_device_obj(dev_tpo)

    if isinstance(key, str):
        key_fn = lambda ts: key  # type: ignore
    else:
        key_fn = key  # type: ignore

    if skip_cast_dev_and_bend:

        def udf(ts: Mapping[ie.Symbol, int]) -> BackendTensorT:
            return flat_state_dict[key_fn(ts)]  # type: ignore
    else:

        def udf(ts: Mapping[ie.Symbol, int]) -> BackendTensorT:
            t = flat_state_dict[key_fn(ts)]  # type: ignore

            ret = user_backend.from_dlpack(t)
            if dtype_ is not None:
                bend_dtype = user_backend.to_backend_datatype(dtype_)
                # _logger.info("Casting tensor %s to %s", key_, bend_dtype)
                ret = user_backend.cast_backend_dtype(ret, bend_dtype)

            return user_backend.to_device(  # type: ignore
                ret,
                dev,
            )

    shape = Shape.from_(shape)
    domain = Domain.from_(domain, True)

    return source_with_ts_udf(udf, shape, dtype_, domain)


def init_from_existing_tensor(
    t: RecurrentTensor,
    shape: ShapeLike,
    dtype: DataTypeLike,
    domain: DomainLike = None,
) -> RecurrentTensor:
    domain = Domain.from_(domain, True)
    shape = Shape.from_(shape)
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    assert t.shape == shape, f"Shape mismatch: {t.shape} != {shape}"
    assert t.dtype == dtype, f"Dtype mismatch: {t.dtype} != {dtype}"
    assert t.domain == domain, f"Domain mismatch: {t.domain} != {domain}"
    return t.ident()


def _external_init_as_udf(
    backend_tensor_fn: Callable[[], BackendTensorT],
    shape: ShapeLike,
    dtype: DataType,
    domain: DomainLike = None,
) -> RecurrentTensor:
    # NOTE: We will leverage the torch implementation of orthogonal_init
    from tempo.core.dl_backend import DLBackend

    exec_cfg = get_active_exec_cfg()
    user_backend = DLBackend.get_backend(exec_cfg.backend)
    dev = user_backend.to_backend_device_obj(exec_cfg.dev)

    def udf() -> Any:
        ret = user_backend.to_device(
            user_backend.cast_backend_dtype(user_backend.from_dlpack(backend_tensor_fn()), dtype),
            dev,
        )
        return ret

    domain = Domain.from_(domain, True)
    shape = Shape.from_(shape)
    return source_udf(udf, shape, dtype, domain)


def external_init(
    backend_tensor_fn: Callable[[], BackendTensorT],
    shape: ShapeLike,
    dtype: DataType,
    domain: DomainLike = None,
) -> RecurrentTensor:
    domain = Domain.from_(domain, True)

    if domain.is_empty():
        return _external_init_as_const(backend_tensor_fn())
    else:
        return _external_init_as_udf(backend_tensor_fn, shape, dtype, domain)


def orthogonal_init(
    shape: ShapeLike = (),
    dtype: DataTypeLike = None,
    domain: DomainLike = None,
    gain: float = 1.41421356237,
) -> RecurrentTensor:
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    shape_ = Shape.from_(shape)

    # NOTE: We will leverage the torch implementation of orthogonal_init
    from tempo.runtime.backends.pytorch.pytorch_backend import PyTorchBackend

    dtype_torch = PyTorchBackend.to_backend_datatype(dtype)

    import torch

    def builder() -> Any:
        tensor_buffer = torch.empty(size=shape_.as_static()._shape, dtype=dtype_torch, device="cpu")
        torch.nn.init.orthogonal_(tensor_buffer, gain=gain)  # type: ignore

        return tensor_buffer

    return external_init(builder, shape_, dtype, domain)


# def orthogonal_init(
#    shape: ShapeLike = (),
#    dtype: DataType = dtypes.float32,
#    gain: float = 1.0,
#    domain: DomainLike = None,
#    requires_grad: Optional[bool] = None,
# ) -> RecurrentTensor:
#    shape = Shape.from_(shape)
#    if len(shape) < 2:
#        raise ValueError("Only tensors with 2 or more dimensions are supported")
#
#    assert shape.is_static(), "Shape must be static for orthogonal_init"
#    shape = shape.as_static()
#    rows = shape.at(0)
#    cols = shape[1:].prod() // rows
#    flattened = RecurrentTensor.normal((rows, cols), dtype=dtype, domain=domain,
#        requires_grad=requires_grad)
#
#    if rows < cols:
#        flattened = flattened.T
#    q, r = ...
#    d = ...
#    ph = d.sign()
#    q *= ph
#
#    if rows < cols:
#        q = q.T
#
#    tensor = tenso
#
#    #if tensor.ndimension() < 2:
#    #    raise ValueError("Only tensors with 2 or more dimensions are supported")
#
#    #if tensor.numel() == 0:
#    #    # no-op
#    #    return tensor
#    #rows = tensor.size(0)
#    #cols = tensor.numel() // rows
#    #flattened = tensor.new(rows, cols).normal_(0, 1, generator=generator)
#
#    #if rows < cols:
#    #    flattened.t_()
#
#    ## Compute the qr factorization
#    #q, r = torch.linalg.qr(flattened)
#    ## Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
#    #d = torch.diag(r, 0)
#    #ph = d.sign()
#    #q *= ph
#
#    #if rows < cols:
#    #    q.t_()
#
#    #with torch.no_grad():
#    #    tensor.view_as(q).copy_(q)
#    #    tensor.mul_(gain)
#    #return tensor


def randn(
    shape: ShapeLike = None,
    dtype: DataTypeLike = None,
    domain: DomainLike = None,
    requires_grad: bool | None = None,
) -> RecurrentTensor:
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)
    shape = Shape.from_(shape)

    src1 = random(shape, dtype, domain, requires_grad)
    src2 = random(shape, dtype, domain, requires_grad)
    x = src1.mul(2 * math.pi).cos().mul((1 - src2).log().mul(-2).sqrt()).cast(dtype)
    return x


def normal(
    shape: ShapeLike = None,
    dtype: DataTypeLike = None,
    domain: DomainLike = None,
    mean: MaybeRecurrentTensor = 0.0,
    std: MaybeRecurrentTensor = 1.0,
    requires_grad: bool | None = None,
) -> RecurrentTensor:
    shape = Shape.from_(shape)
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    std, mean = _lift_if_index_value(std), _lift_if_index_value(mean)
    rand = randn(shape, dtype, domain, requires_grad)
    ret = rand * std + mean
    return ret.cast(dtype)


def kaiming_uniform(
    shape: ShapeLike = (),
    a: MaybeRecurrentTensor = 0.01,
    dtype: DataTypeLike = None,
    domain: DomainLike = None,
    requires_grad: bool | None = None,
) -> RecurrentTensor:
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    shape = Shape.from_(shape)

    if shape[1:].is_static():
        denom = math.sqrt(math.prod(shape[1:].as_static()))
    else:
        denom = lift(shape[1:].prod()).sqrt()  # type:ignore

    a = lift(a)

    if isinstance(a, RecurrentTensor):
        bound = math.sqrt(3.0) * RecurrentTensor.sqrt(2.0 / (1 + a**2)) / denom
    else:
        bound = math.sqrt(3.0) * math.sqrt(2.0 / (1 + a**2)) / denom

    return RecurrentTensor.uniform(
        shape,
        low=-bound,
        high=bound,
        dtype=dtype,
        domain=domain,
        requires_grad=requires_grad,
    )


def kaiming_normal(
    shape: tuple[int, ...] | Shape = (),
    dtype: DataTypeLike = None,
    a: MaybeRecurrentTensor = 0.01,
    domain: DomainLike = None,
    requires_grad: bool | None = None,
) -> RecurrentTensor:
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    if not isinstance(shape, Shape):
        shape = Shape(shape)

    if shape[1:].is_static():
        denom = math.sqrt(math.prod(shape[1:].as_static()))
    else:
        denom = lift(shape[1:].prod()).sqrt()  # type:ignore

    a = _lift_if_index_value(a)

    if isinstance(a, RecurrentTensor):
        std = RecurrentTensor.sqrt(2.0 / (1 + a**2)) / denom
    else:
        std = math.sqrt(2.0 / (1 + a**2)) / denom  # type:ignore
    return RecurrentTensor.normal(
        shape,
        mean=0.0,
        std=std,
        dtype=dtype,
        domain=domain,
        requires_grad=requires_grad,
    )


def multinomial(
    weights_: MaybeRecurrentTensor,
    num_samples: int = 1,
    replacement: bool = False,
    domain: DomainLike = None,
) -> RecurrentTensor:
    """Computes the multinomial distribution of a tensor of weights.

    Args:
        weights (MaybeRecurrentTensor): The probabilities of each event.
        num_samples (int, optional): Number of samples. Defaults to 1.
        replacement (bool, optional): Whether to use replacement. Defaults to False.

    Returns:
        RecurrentTensor: An integer tensor of shape (num_samples,)
        containing the indices of the samples.

    """
    weights_ = lift(weights_)
    assert 1 <= weights_.ndim <= 2 and num_samples > 0, (
        f"{weights_.ndim=} of shape {weights_.shape} must be 1D or 2D, "
        f"{num_samples=} must be positive"
    )
    assert replacement or num_samples == 1, "no replacement only supports num_samples = 1"

    dom = Domain.union(Domain.from_(domain, True), weights_.domain)

    weight = weights_.unsqueeze(0) if weights_.ndim == 1 else weights_
    cw = weight.cumsum(dim=1).cast(dtypes.float32)
    cdf = cw / cw[..., -1].unsqueeze(1)
    unif_samples = RecurrentTensor.random((num_samples, cdf.shape.at(0), 1), domain=dom)
    # NOTE: original impl uses >=, but this leads to out-of-bounds errors in llama exmaple
    indices = (unif_samples.expand((-1, -1, cdf.shape.at(1))) >= cdf).sum(2).permute((1, 0))
    return (indices.squeeze(0) if weights_.ndim == 1 else indices).cast(dtypes.default_int)


def l_norm(  # noqa: C901
    x: MaybeRecurrentTensor,
    n: MaybeRecurrentTensor = 2,
    dims: DIM_TYPE = None,
) -> RecurrentTensor:
    x = lift(x)

    dims = normalize_dims(dims, x.shape)

    if isinstance(n, (float, int)) and float(n) == 0.0:
        return lift(x != 0).sum(dims)
    elif isinstance(n, float) and math.isinf(n):
        res = x.abs()
        func = RecurrentTensor.max if n == math.inf else RecurrentTensor.min

        res = x.abs()
        # NOTE: sort-reverse the dims to make dims still make sense as we reduce
        for dim in sorted(dims, reverse=True):
            res = func(dim)[0]  # NOTE: ignore the indices

        res.reshape(Shape(()))  # reshape to scalar

    if isinstance(n, int) and n % 2 == 0:
        # NOTE: can skip the abs if n is even
        return x.pow_(n).sum(dims).pow_(1.0 / n)
    else:
        return x.abs().pow_(n).sum(dims).pow_(1 / lift(n))


def logsumexp(
    x: MaybeRecurrentTensor, sum_dims: DIM_TYPE = None, keepdim: bool = False
) -> RecurrentTensor:
    x = lift(x)
    sum_dims = normalize_dims(sum_dims, x.shape)
    return x.exp().sum(sum_dims, keepdim=keepdim).ln()


def chunk(x: MaybeRecurrentTensor, chunks: int, dim: int = 0) -> list[RecurrentTensor]:
    x = lift(x)
    # Use index
    size_at_dim = x.shape.int_at(dim)

    assert size_at_dim % chunks == 0, "Number of chunks must divide tensor size"
    chunk_size = size_at_dim // chunks

    return [x.slice_dim(dim, chunk_size * i, chunk_size * (i + 1)) for i in range(chunks)]


def dilate(
    x: MaybeRecurrentTensor, dilations: tuple[int, ...] | int, n_dims: int | None = None
) -> RecurrentTensor:
    """
    Dilates the input tensor by the given dilations.

    Must either provide dilations as a tuple of ints, or provide n_dims.
    The operation affects the last n_dims spatial dimensions of the tensor.
    """
    x = lift(x)

    if n_dims is None:
        assert isinstance(dilations, tuple)
        n_dims = len(dilations)
    elif isinstance(dilations, tuple):
        assert n_dims == len(dilations)
    elif isinstance(dilations, int):
        dilations = (dilations,) * n_dims

    num_non_spatial_dims = x.ndim - n_dims
    non_spatial_size = (1,) * num_non_spatial_dims

    kernel_size = (1,) * n_dims

    weight = RecurrentTensor.ones(Shape((*non_spatial_size, *kernel_size)), dtype=x.dtype)

    stride = tuple(d + 1 for d in dilations)
    return x.conv_general(weight, stride=stride, transposed=True, n_dims=n_dims)


def pad_dim(
    x: MaybeRecurrentTensor,
    padding: tuple[int, int] | int,
    dim: int,
    mode: str = "constant",
    value: float | None = None,
) -> RecurrentTensor:
    x = lift(x)
    if isinstance(padding, int):
        padding = (padding, padding)

    assert len(x.shape) > dim
    assert padding[0] >= 0 and padding[1] >= 0
    if padding[0] == 0 and padding[1] == 0:
        return x

    ret: RecurrentTensor = ad.Pad.apply(x, padding=padding, dim=dim, mode=mode, value=value)[0]
    return ret


def pad(
    x: MaybeRecurrentTensor,
    padding: tuple[tuple[int, int] | int, ...],
    mode: str = "constant",
    value: float | None = None,
) -> RecurrentTensor:
    """Pads the input tensor with a constant value.

    Args:
        padding: Tuple of (before, after) or (pad,) padding sizes for each dimension
        value: Value to pad with (default: 0.0)
        mode: Padding mode (default: "constant")
        If value is None, the tensor will be padded with the default value for the dtype
    Returns:
        Padded tensor
    """
    x = lift(x)
    padding = normalize_slice_indexes(padding, x.shape._shape)

    for i, p in enumerate(padding):
        x = pad_dim(x, p, i, mode, value)
    return x


def slice_dim(
    x: MaybeRecurrentTensor,
    dim: int,
    start: ie.IntIndexValueLike,
    stop: ie.IntIndexValueLike,
    step: ie.IntIndexValueLike | None = None,
) -> RecurrentTensor:
    """Slices the input tensor along the given dimension.

    Args:
        dim: Dimension to slice along
        start: Starting index
        stop: Stopping index
        step: Step size
    Returns:
        Sliced tensor
    """
    x = lift(x)
    if step is None:
        step = 1
    assert step == 1
    length = stop - start
    dim = normalize_negative_dim(dim, x.shape)
    ret: RecurrentTensor = ad.Slice.apply(x, dim=dim, start=start, length=length)[0]
    return ret


def index(tensor: MaybeRecurrentTensor, dim: int, index: MaybeRecurrentTensor) -> RecurrentTensor:
    """Index the tensor along the given dimension.

    Args:
        tensor (MaybeRecurrentTensor): The tensor to index.
        dim (int): The dimension to index along.
        index (MaybeRecurrentTensor): The index to index along.

    Returns:
        RecurrentTensor: The indexed tensor.
    """
    tensor = lift(tensor)
    index = lift(index)
    assert index.ndim < 2, "Index must be a scalar or 1D tensor"

    assert dtypes.is_integer(index.dtype), "Index must be integer"
    index = index.cast(dtypes.default_int)

    dim = normalize_negative_dim(dim, tensor.shape)

    # Here, if index is scalar, do we want to apply a squeeze?
    ret: RecurrentTensor = ad.IndexSelect.apply(tensor, index, dim=dim)[0]
    # if index.shape.is_scalar():
    #    ret = ret.squeeze(dim)
    return ret


def index_add(
    tensor: MaybeRecurrentTensor,
    dim: int,
    index: MaybeRecurrentTensor,
    src: MaybeRecurrentTensor,
    alpha: float = 1.0,
) -> RecurrentTensor:
    """Index the tensor along the given dimension and add the source tensor.

    Args:
        tensor (MaybeRecurrentTensor): The tensor to index.
        dim (int): The dimension to index along.
        index (MaybeRecurrentTensor): The index to index along.
        src (MaybeRecurrentTensor): The source tensor to add.
        alpha (float): The alpha (multiplier) for the source tensor.
    Returns:
        RecurrentTensor: The indexed tensor.
    """
    tensor, index, src = lift_all_to_rt(tensor, index, src)
    dim = normalize_negative_dim(dim, tensor.shape)
    return ad.IndexAdd.apply(tensor, index, src, dim=dim, alpha=alpha)[0]


# def index(
#    tensor: MaybeRecurrentTensor, dim: int, index: MaybeRecurrentTensor
# ) -> RecurrentTensor:
#    tensor = lift_to_rt(tensor)
#    index = lift_to_rt(index)
#
#    original = index
#
#    #TODO: this is torch-only right? Move to torch backend
#    #If not, then the user should be doing this themselves
#    if original.shape.is_scalar():
#        index = index.unsqueeze(0)
#
#    assert index.ndim < 2, "Index must be a scalar or 1D tensor"
#
#    # We implement index using gather
#    gather_index = index
#
#    for i in range(tensor.ndim):
#        if i == dim:
#            continue
#        gather_index = gather_index.unsqueeze(i)
#
#    final_shape = list(tensor.shape._shape)
#    final_shape[dim] = index.shape.at(0)
#    gather_index = gather_index.expand(Shape.from_(tuple(final_shape)))
#
#    ret = gather(tensor, dim, gather_index)
#    if original.shape.is_scalar():
#        ret = ret.squeeze(dim)
#    return ret


def slice_many(
    tensor: MaybeRecurrentTensor,
    slices: Sequence[ie.IndexAtom | slice | int] | ie.IndexAtom | slice | int,
) -> RecurrentTensor:
    tensor = lift(tensor)
    slices = as_seq(slices)

    for i, slc in enumerate(slices):
        if isinstance(slc, slice):
            tensor = slice_dim(tensor, i, slc.start, slc.stop, slc.step)
        elif isinstance(slc, ie.Slice):
            tensor = slice_dim(tensor, i, slc.start, slc.stop, 1)
        elif isinstance(slc, (ie.IntIndexValue, int)):
            tensor = slice_dim(tensor, i, slc, slc + 1, 1)
        else:
            raise ValueError(f"Invalid slice {slc}")
    return tensor


def index_many(
    tensor: MaybeRecurrentTensor,
    indexes: Sequence[MaybeRecurrentTensor] | MaybeRecurrentTensor | None,
) -> RecurrentTensor:
    tensor = lift(tensor)

    if isinstance(indexes, ie.IndexSequence):
        indexes = indexes.members
    indexes = as_seq(indexes)
    indexes = [(lift(idx) if idx is not None else None) for idx in indexes]  # type: ignore

    squeezes = 0
    for i, idx in reversed(list(enumerate(indexes))):
        if idx is None:
            continue
        tensor = index(tensor, i - squeezes, idx)
        if idx.shape.is_scalar():
            squeezes += 1
    return tensor


# def index_put(
#    sink: MaybeRecurrentTensor,
#    dim: int,
#    src: MaybeRecurrentTensor,
#    index: MaybeRecurrentTensor,
#    alpha: float = 1.0,
# ) -> RecurrentTensor:
#    sink, src, index = lift_all_to_rt(sink, src, index)
#
#    ret: RecurrentTensor = ad.IndexAdd.apply(sink, src, index, dim=dim, alpha=alpha)[0]
#    return ret


def cast(x: MaybeRecurrentTensor, dtype: DataType) -> RecurrentTensor:
    x = lift(x)
    if x.dtype == dtype:
        return x
    ret: RecurrentTensor = ad.Cast.apply(x, dtype=dtype)[0]
    return ret


def is_like(x: MaybeRecurrentTensor, y: MaybeRecurrentTensor) -> bool:
    x = lift(x)
    y = lift(y)
    return x.shape == y.shape and x.dtype == y.dtype and x.domain == y.domain


def barrier(barrier_name: str | None = None) -> None:
    SymbolicTensor.barrier(barrier_name)


def placeholder_like(
    x: MaybeRecurrentTensor,
    requires_grad: bool | None = None,
) -> RecurrentTensor:
    x = lift(x)
    return placeholder(x.shape, x.dtype, x.domain, requires_grad)


# TODO change to "branching" maybe?
def placeholder(
    shape: tuple[int, ...] | Shape = (),
    dtype: DataTypeLike = None,
    domain: DomainLike = None,
    requires_grad: bool | None = None,
) -> RecurrentTensor:
    dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

    if not isinstance(shape, Shape):
        shape = Shape(shape)

    # if not isinstance(dtype, DataType):
    #    dtype = dtypes.from_np(dtype)
    #    # NOTE: np uses float64 as default, but we want to use float32
    #    if dtype == dtypes.float64:
    #        dtype = dtypes.float32
    return RecurrentTensor(SymbolicTensor.merge(shape, dtype, domain), requires_grad=requires_grad)


def maximum(x: MaybeRecurrentTensor, y: MaybeRecurrentTensor) -> RecurrentTensor:
    """Returns the element-wise maximum of two tensors."""
    x, y = lift_all_to_rt(x, y)
    x, y = broadcast_tensors(x, y)
    x, y = upcast_tensors(x, y)
    return where(x >= y, x, y)


def minimum(x: MaybeRecurrentTensor, y: MaybeRecurrentTensor) -> RecurrentTensor:
    """Returns the element-wise minimum of two tensors."""
    x, y = lift_all_to_rt(x, y)
    x, y = broadcast_tensors(x, y)
    x, y = upcast_tensors(x, y)
    return where(x <= y, x, y)


def unflatten(tensor: MaybeRecurrentTensor, dim: int, sizes: Sequence[int]) -> RecurrentTensor:
    """Unflattens a tensor along the specified dimension into the given sizes."""
    tensor = lift(tensor)
    dim = normalize_negative_dim(dim, tensor.shape)

    # Calculate the new shape
    new_shape = list(tensor.shape._shape)
    new_shape[dim : dim + 1] = sizes
    return tensor.reshape(new_shape)


def _sort_manual(
    tensor: MaybeRecurrentTensor, dim: int = -1, descending: bool = False
) -> tuple[RecurrentTensor, RecurrentTensor]:
    """
    Performs a bitonic sort on the tensor along the specified dimension.

    Order of indices for equivalent elements is always preserved.

    See: https://en.wikipedia.org/wiki/Bitonic_sorter

    Cheers to tinygrad for the bitonic sort implementation!

    Args:
        tensor (MaybeRecurrentTensor): The tensor to sort.
        dim (int): The dimension to sort along. Defaults to -1.
        descending (bool): Whether to sort in descending order. Defaults to False.

    Returns:
        Tuple[RecurrentTensor, RecurrentTensor]: A tuple containing (sorted_values, indices)
    """
    x = lift(tensor)
    dim = normalize_negative_dim(dim, x.shape)

    # pad to power of 2
    orig_len = x.shape.at(dim)
    n_stages = math.ceil(math.log2(orig_len))
    fill_value = dtypes.min(x.dtype) if descending else dtypes.max(x.dtype)
    pads = tuple((0, 2**n_stages - orig_len) if i == dim else (0, 0) for i in range(x.ndim))
    x = x.pad(pads, value=fill_value)
    x = x.unflatten(dim, [2] * n_stages)

    # TODO: use symbolic dimension?
    # https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort1.svg
    for stage in range(1, n_stages + 1):
        if stage != n_stages:
            # flip so arrows of green boxes point the same way as blue boxes
            crossover_dim = dim + n_stages - stage - 1
            blue_box, green_box = x.split(crossover_dim, num_splits=2)
            flip_dims = tuple(-i for i in range(1, stage + 1 + (x.ndim - dim)))
            assert all(isinstance(d, int) for d in flip_dims)
            x = RecurrentTensor.cat(blue_box, green_box.flip(flip_dims), dim=crossover_dim)

        for substage in range(stage - 1, -1, -1):
            partner_dim = dim + n_stages - substage - 1
            x_top, x_bottom = x.split(partner_dim, num_splits=2)
            x_larger, x_smaller = maximum(x_top, x_bottom), minimum(x_top, x_bottom)
            x = (
                RecurrentTensor.cat(x_larger, x_smaller, dim=partner_dim)
                if descending
                else RecurrentTensor.cat(x_smaller, x_larger, dim=partner_dim)
            )

        if stage != n_stages:
            # flip wires back to undo the crossover
            blue_box, flipped_green_box = x.split(crossover_dim, num_splits=2)
            x = RecurrentTensor.cat(blue_box, flipped_green_box.flip(flip_dims), dim=crossover_dim)

    x = x.flatten(dim, dim + n_stages - 1)
    # Remove padding
    x = x.slice_dim(dim, 0, orig_len)

    assert isinstance(orig_len, int)
    # compute indices for sorted values
    idx = arange(orig_len, requires_grad=False).reshape(
        tuple(orig_len if i == dim else 1 for i in range(x.ndim))
    )
    idx = idx.expand(x.shape)

    def compute_counts(t: RecurrentTensor) -> RecurrentTensor:
        return (
            (idx.unsqueeze(dim) <= idx.unsqueeze(dim + 1))
            & (t.unsqueeze(dim) == t.unsqueeze(dim + 1))
        ).sum(dim + 1)

    count_orig, count_sorted = compute_counts(x), compute_counts(x)
    cond = (x.unsqueeze(dim + 1) == x.unsqueeze(dim)) & (
        count_orig.unsqueeze(dim + 1) == count_sorted.unsqueeze(dim)
    )
    idx = (cond * idx.unsqueeze(dim + 1)).sum(dim)

    return x, idx


def sort(
    tensor: MaybeRecurrentTensor, dim: int = -1, descending: bool = False, stable: bool = False
) -> tuple[RecurrentTensor, RecurrentTensor]:
    tensor = lift(tensor)
    dim = normalize_negative_dim(dim, tensor.shape)

    vals, idxs = ad.Sort.apply(tensor, dim=dim, stable=stable, descending=descending)

    # NOTE: This approach can fail when descending == True and stable == True, since the
    # stable order for ascending != stable order for descending.
    # if descending:
    #    vals = vals.flip(dim)
    #    idxs = idxs.flip(dim)

    return vals, idxs


def sample_top_p(probs: MaybeRecurrentTensor, p: float = 0.9) -> RecurrentTensor:
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs = lift(probs)
    probs_sort, probs_idx = sort(probs, dim=-1, descending=True)
    probs_sum = cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort = where(mask, 0.0, probs_sort)
    probs_sort = probs_sort / probs_sort.sum(dims=-1, keepdim=True)
    next_token = multinomial(probs_sort, num_samples=1)
    next_token = gather(probs_idx, -1, next_token)  # TODO: use index_select instead of gather?
    return next_token


# def sample_top_k(probs: MaybeRecurrentTensor, k: int = 40) -> RecurrentTensor:
#    """
#    Perform top-k sampling on a probability distribution.
#
#    Args:
#        probs (torch.Tensor): Probability distribution tensor.
#        k (int): Number of top tokens to sample.
#
#    Returns:
#        torch.Tensor: Sampled token indices.
#
#    Note:
#        Top-k sampling selects the top k tokens with the highest probabilities.
#    """
#    probs = lift(probs)
#    probs_sort, probs_idx = sort(probs, dim=-1, descending=True)
#    probs_sort = probs_sort[:, :k]
#    probs_idx = probs_idx[:, :k]
#    next_token = multinomial(probs_sort, num_samples=1)
#    next_token = gather(probs_idx, -1, next_token) #TODO: use index?
#    return next_token


def sample_likely(probs: MaybeRecurrentTensor, above: float = 0.05) -> RecurrentTensor:
    """
    Perform likely sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        above (float): Probability threshold for likely sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Likely sampling selects tokens with probabilities greater than the threshold.
    """
    probs = lift(probs)
    probs = where(probs > above, probs, 0.0)
    probs_sum = probs.sum(dims=-1, keepdim=True)
    probs = probs / probs_sum  # renormalize to 1
    next_token = multinomial(probs, num_samples=1)
    return next_token


class RecurrentTensor:
    __slots__ = (
        "_underlying",
        "_ctx",
        "requires_grad",
        "grad",
    )

    def __init__(
        self,
        underlying: SymbolicTensor,
        requires_grad: bool | None = None,
        ctx: AutodiffFn | None = None,
        grad: RecurrentTensor | None = None,
    ):
        self._underlying: SymbolicTensor = underlying

        self._ctx: AutodiffFn | None = ctx
        self.requires_grad: bool | None = requires_grad
        self.grad: RecurrentTensor | None = None if grad is None else grad

    def __str__(self) -> str:
        op_name = self._underlying.op.__class__.__name__
        op_id = self.tensor_id.op_id
        op_out = self.tensor_id.output_id

        bwd_fn = type(self._ctx).__name__ if self._ctx is not None else None
        dom = self.domain.variables
        index_expr = self._underlying._index_expr

        return (
            f"RT(op={op_name}[{op_id}-{op_out}], shape={self.shape._shape}, "
            + f"dtype={self.dtype}, dom={dom}, requires_grad={self.requires_grad}, "
            + f"bwd_fn={bwd_fn}, grad={self.grad}, index_expr={index_expr})"
        )

    __repr__ = __str__

    def __hash__(self) -> int:
        return id(self)
        # return (
        #    hash(self._underlying)
        #    + hash(self.requires_grad)
        #    + hash(self._ctx)
        #    + hash(self.grad)
        # )

    # NOTE: No idea why mypy does not accept the returns below...
    @property
    def tensor_id(self) -> TensorId:
        return self._underlying.tensor_id  # type: ignore

    @property
    def shape(self) -> Shape:
        return self._underlying.shape  # type: ignore

    @property
    def domain(self) -> Domain:
        return self._underlying.domain  # type: ignore

    @property
    def unindexed_domain(self) -> Domain:
        return self._underlying.unindexed_domain

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int | ie.IndexValue:
        return self.shape.prod()  # type: ignore

    @property
    def spatial_shape(self) -> Shape:
        return self._underlying.spatial_shape

    def size(self, dim: DIM_TYPE = None) -> ie.IntIndexValueLike | Sequence[ie.IntIndexValueLike]:
        return self._underlying.size(dim)

    @property
    def dtype(self) -> DataType:
        assert self._underlying is not None, "Can't get dtype of uninitialized tensor"
        return self._underlying.dtype

    @property
    def next_(self) -> RecurrentTensor:
        return self[self.domain.lex_next_expr]

    # @lex_next.setter
    # def lex_next(self, value: Any) -> None:
    #    # This allows users to use the following notation, which in the end makes a set-expr
    #    # for lex_next easily: x.lex_next = y

    #    index = self._index_expr
    #    self._underlying = self._underlying.copy_with_index(None)
    #    # assert isinstance(value, (ie.ConstBool, ie.BooleanBinaryExpr)), (
    #    #    f"You are attempting"
    #    # )
    #    # with ctx_man.when(value):

    #    self[index] = value

    @property
    def previous(self) -> RecurrentTensor:
        return self[self.domain.lex_prev_expr]

    # @lex_prev.setter
    # def lex_prev(self, value: Any):
    #    index = self._index_expr
    #    self._underlying = self._underlying.copy_with_index(None)

    #    self[index] = value

    @property
    def init(self) -> RecurrentTensor:
        return self[self.domain.lb_expr]

    @init.setter
    def init(self, value: Any) -> None:
        self[self.domain.lb_expr] = value

    def clear_placeholder_branches(self) -> None:
        op = self._underlying.op
        assert isinstance(op, top.MergeOp), "Not a placeholder tensor"
        from tempo.core.global_objects import get_active_dg

        get_active_dg().ops_by_id[op.op_id].uncommitted_branch_conds.clear()
        op.num_inputs_[0] = 0

    add = __add__ = add
    __radd__ = lambda x, y: RecurrentTensor.lift(y).add(x)  # type: ignore

    subtract = sub = __sub__ = sub
    __rsub__ = lambda x, y: RecurrentTensor.lift(y).sub(x)  # type: ignore

    multiply = mul = __mul__ = mul
    __rmul__ = lambda x, y: RecurrentTensor.lift(y).mul(x)  # type: ignore

    divide = div = __truediv__ = div
    __rtruediv__ = lambda x, y: RecurrentTensor.lift(y).div(x)  # type: ignore

    __mod__ = mod = remainder = remainder
    __floordiv__ = floor_div = floor_divide = floor_div

    round = round  # noqa: A001, A003
    ceil = ceiling = ceil
    floor = floor
    trunc = truncate = trunc

    exp = exp
    exp2 = exp2
    exp10 = exp10
    pow_ = __pow__ = pow_
    __rpow__ = lambda x, y: RecurrentTensor.lift(y).pow_(x)  # type: ignore
    squared = square = square

    log = ln = loge = ln
    log_base = log_base
    log2 = log2
    log10 = log10

    conv_general = convolution = conv_general
    conv1d = conv1d
    conv2d = conv2d
    conv3d = conv3d

    dot = matmul = __matmul__ = dot
    __rmatmul__ = lambda x, y: RecurrentTensor.lift(y).dot(x)  # type: ignore

    logical_or = __or__ = logical_or  # type: ignore[assignment]
    logical_and = __and__ = logical_and  # type: ignore[assignment]
    equals = __eq__ = equals  # type: ignore[assignment]
    not_equals = __ne__ = lambda x, y: not x == y  # type: ignore[assignment]
    less_than = __lt__ = less_than
    less_than_or_equal = __le__ = less_than_or_equal
    greater_than = __gt__ = greater_than
    greater_than_or_equal = __ge__ = greater_than_or_equal
    not_ = __invert__ = logical_not

    negate = neg = __neg__ = neg

    relu = relu
    sigmoid = sigmoid
    swish = silu = swish
    softmax = softmax
    cross_entropy = cross_entropy
    tanh = tanh
    elu = elu
    leakyrelu = leakyrelu
    mish = mish
    softplus = softplus
    sin = sin
    cos = cos
    tan = tan
    abs = abs  # noqa: A001, A003
    erf = erf

    min = min  # noqa: A001, A003
    max = max  # noqa: A001, A003
    argmax = argmax
    argmin = argmin

    sqrt = sqrt
    sum = sum  # noqa: A001, A003
    discounted_sum = discounted_sum
    discounted_cum_sum = discounted_cum_sum
    cumsum = cumsum
    cumprod = cumprod
    permute = permute
    reshape = reshape
    transpose = transpose
    flip = flip
    expand = expand
    squeeze = squeeze
    unsqueeze = unsqueeze
    where = where

    @property
    def T(self) -> RecurrentTensor:  # noqa: N802
        return self.transpose()

    mean = mean
    std = std
    normalize = normalize
    barrier = barrier

    placeholder = staticmethod(placeholder)
    placeholder_like = staticmethod(placeholder_like)
    const = staticmethod(const)
    ones = staticmethod(ones)
    zeros = staticmethod(zeros)
    const_like = staticmethod(const_like)
    ones_like = staticmethod(ones_like)
    zeros_like = staticmethod(zeros_like)
    arange = staticmethod(arange)
    broadcast_tensors = staticmethod(broadcast_tensors)
    upcast_tensors = staticmethod(upcast_tensors)

    cast = cast

    random = rand = staticmethod(random)
    random_int = staticmethod(random_int)
    random_bool = staticmethod(random_bool)

    uniform = staticmethod(uniform)
    scaled_uniform = staticmethod(scaled_uniform)
    linear_init_uniform = staticmethod(linear_init_uniform)
    glorot_uniform = staticmethod(glorot_uniform)
    orthogonal_init = staticmethod(orthogonal_init)
    normal = staticmethod(normal)
    kaiming_uniform = staticmethod(kaiming_uniform)
    kaiming_normal = staticmethod(kaiming_normal)

    multinomial = multinomial

    split = split
    cat = staticmethod(cat)
    stack = staticmethod(stack)
    lift = staticmethod(lift)
    lift_all_to_rt = staticmethod(lift_all_to_rt)

    clip = clamp = clip
    logsumexp = logsumexp

    scatter = scatter
    scatter_add = scatter_add
    gather = gather
    chunk = chunk
    slice_dim = slice_dim
    slice_many = slice_many
    pad_dim = pad_dim
    pad = pad
    dilate = dilate

    index = spatial_index = index_select = index
    index_many = spatial_index_many = index_many
    # index_put = index_put
    index_add = spatial_index_add = index_add

    ident = ident

    sink_udf = sink_udf
    sink_many_udf = staticmethod(sink_many_udf)
    sink_with_ts_udf = sink_with_ts_udf
    sink_many_with_ts_udf = staticmethod(sink_many_with_ts_udf)
    source_udf = source_udf
    source_with_ts_udf = source_with_ts_udf
    map_udf = map_udf
    map_with_ts_udf = map_with_ts_udf

    l_norm = l_norm

    from_symbol = from_symbol
    from_expr = from_expr

    flatten = flatten

    one_hot = one_hot
    is_like = is_like

    init_from_statedict = staticmethod(init_from_statedict)
    init_from_existing_tensor = staticmethod(init_from_existing_tensor)
    repeat = repeat
    repeat_interleave = repeat_interleave

    sort = sort
    sample_top_p = sample_top_p
    sample_likely = sample_likely
    maximum = maximum
    minimum = minimum
    unflatten = unflatten

    def detach(self) -> RecurrentTensor:
        return RecurrentTensor(self._underlying, requires_grad=False)

    def copy_with_no_symbolic_index(self) -> RecurrentTensor:
        return RecurrentTensor(
            self._underlying.copy_with_no_index(),
            self.requires_grad,
            self._ctx,
            self.grad,
        )

    @property
    def creation_traceback(self) -> str:
        return self._underlying.op.creation_traceback

    def _topological_sort_computational_graph(self) -> list[RecurrentTensor]:
        def __topo_sort(
            node: RecurrentTensor,
            visited: set[RecurrentTensor],
            nodes: list[RecurrentTensor],
        ) -> list[RecurrentTensor]:
            visited.add(node)
            if node._ctx is not None:
                for i in node._ctx.parents:
                    if i not in visited:
                        __topo_sort(i, visited, nodes)
                nodes.append(node)
            return nodes

        return __topo_sort(self, set(), [])

    def backward(self, grad: RecurrentTensor | None = None) -> None:
        # NOTE: In case the user calls backward on a symbolically indexed tensor, we call ident.
        from tempo.api.tempo_context_manager import get_active_ctx_manager
        from tempo.core.op_tags import BACKWARD_REGION_TAG

        ctx = get_active_ctx_manager()
        with ctx.tag_region(BACKWARD_REGION_TAG):
            self.ident().backward_(grad)

    def backward_(self, grad: RecurrentTensor | None = None) -> None:
        assert self.requires_grad, "Can't call backward on tensor that does not requires_grad"
        assert self._ctx is not None, "Can't call backward on a tensor without a context"
        assert self._underlying is not None, "Can't call backward on an uninitialized tensor"

        _logger.debug("Starting backward pass for %s", self)
        predecessors = self._topological_sort_computational_graph()
        _logger.debug("Predecessors: %s", predecessors)

        if not grad:
            assert self.shape == Shape(()), (
                "backward can only be called implicitly for scalar tensors,"
                + f" but it has shape {self.shape})"
            )
            # TODO the domain of this needs to match self.domain. But is this needed?
            grad = _expand_domain(RecurrentTensor.ones(), self.domain)
        assert grad.shape == self.shape, (
            f"grad shape ({grad.shape!r}) must match tensor shape {self.shape!r}"
        )
        self.grad = grad

        from tempo.api.backprop import propagate_grad_backwards_simple

        for t in reversed(predecessors):
            assert t.grad is not None, "Gradient should not be None."

            _logger.debug("=== Processing predecessor %s", t)
            # print("Processing predecessor", t)
            # propagate_grad_backwards(t)
            propagate_grad_backwards_simple(t)

    def __getitem__(self, item: Any) -> RecurrentTensor:
        # TODO: we may want to allow condition-based indexing.
        # E.g.: dom =(b,i,k,t) item= (k==0) -> item= (b,i,0,t)
        # E.g.: dom =(b,i,k,t) item= ((k==0) & (t==0)) -> item= (b,i,0,0)

        # TODO: if we ever need tensor indexing
        #     - use Tensor.arange == tensor_index to create a mask
        #     - apply mask to self by mask * self for dims where index is a tensor
        #     - (mask * self).sum(dim) to reduce to correct shape

        dom = self.domain

        symbolic_dim_sizes = dom.ubounds
        spatial_dim_sizes: tuple[ie.IntIndexValue, ...] = tuple(
            s if isinstance(s, ie.IntIndexValue) else ie.ConstInt(s)
            for s in self._underlying.spatial_shape
        )
        all_dim_sizes = (
            *symbolic_dim_sizes,
            *spatial_dim_sizes,
        )

        # By default, use the basis symbols for implicit dims and 0:size for explicit dims
        default_idxs = (
            *dom.variables,
            *[ie.Slice(start=ie.ConstInt(0), stop=s) for s in spatial_dim_sizes],
        )

        # If somehow we are passed an index sequence, pretend we were given its members
        if isinstance(item, ie.IndexSequence):
            item = item.members

        # Make sure we are working with a sequence
        item_seq: tuple[Any] = tuple(as_seq(item))  # type: ignore

        ## Combine with existing index sequence if it exists
        # if self._underlying._index_expr is not None:
        #    item = (*self._underlying._index_expr.members, *item)

        item_seq = normalize_indexes(item_seq, default_idxs, all_dim_sizes)

        if len(item_seq) > len(dom) + len(spatial_dim_sizes):
            raise ValueError(
                f"Can't index with more indexes than there are total dimensions \
                (symbolic + spatial), got {len(item_seq)} indexes for {len(dom)} \
                symbolic dimensions and {len(spatial_dim_sizes)} spatial dimensions."
            )

        ## If user did not index all symbolic dimensions, fill in the rest with defaults
        # if len(item) < len(dom):
        #    item = (*item, *default_idxs[len(item) : len(dom)])

        num_symb_indices = builtins.min(len(dom), len(item_seq))
        symbolic_index = ie.lift_to_ie_seq(item_seq[:num_symb_indices])

        # to_ret = RecurrentTensor(
        #    self._underlying.symbolic_index(symbolic_index),
        #    requires_grad=self.requires_grad,
        #    ctx=self._ctx,
        #    grad=self.grad,
        # )
        to_ret = ad.TemporalIndex.apply(self, e=symbolic_index)[0]

        num_conc_index = len(item_seq) - num_symb_indices

        if num_conc_index > 0:
            for i, idx in enumerate(item_seq[num_symb_indices:]):
                if isinstance(idx, (ie.Slice, slice)):
                    # , idx.step
                    to_ret = to_ret.slice_dim(i, idx.start, idx.stop)  # type: ignore
                else:
                    to_ret = to_ret.index(i, idx)
        return to_ret

    def __setitem__(  # noqa: C901
        self,
        key: Any,
        source_orig: MaybeRecurrentTensor,
    ) -> None:
        if isinstance(source_orig, (int, float, bool)):
            source = RecurrentTensor.const_like(source_orig, self)
        else:
            source = lift(source_orig)
        key_: ie.IndexExpr = ie.lift_to_ie(key)

        from tempo.api.tempo_context_manager import get_active_ctx_manager

        ctx_man = get_active_ctx_manager()

        cond = ctx_man.current_condition
        _logger.debug(
            "Setting %s[%s] =  %s with current condition %s",
            self._underlying.tensor_id,
            key_,
            source._underlying.tensor_id,
            cond,
        )

        # TODO promote self to placeholder if not placeholder. Move self.underlying to placeholder
        # Not sure if this is possible, as we will never get the order right. Whatever the current
        # definition of self, it should come last, and we can't really do that without reformulating
        # things in the RDG a bit. We would need to defer the registration of the definition until
        # compile is called.
        # if not self.is_placeholder():
        #    placeholder = RecurrentTensor.placeholder(
        #          self.shape, self.dtype, requires_grad=self.requires_grad
        #    )
        #    old_underlying = self._underlying
        #    self._underlying = placeholder._underlying
        #    todo_cond = ie.ConstBool(True) #TODO what should this cond be?
        #    ctx_man.get_ctx()._register_conditional_definition(
        #        todo_cond,  self._underlying, old_underlying, ctx_man.get_basis_expr(),
        #    )

        basis = self.unindexed_domain.basis_expr
        if not self._underlying.index_expr.struct_eq(basis):
            raise Exception("Tensor must not be indexed already")

        # Validation
        if not (isinstance(key_, ie.IndexExpr) and key_.is_valid_set_item_expr()):
            raise ValueError(
                f"Invalid key {key_}. Key must be a boolean expression or a valid \
                             initializer index sequence"
            )

        source_index_expr = [*source._underlying.index_expr.members]
        # Easy case, just and expr to cond
        if isinstance(
            key_,
            (ie.ConstBool, ie.BooleanBinaryExpr, ie.Not, ie.BooleanLogicBinaryExpr),
        ):
            cond = cond & key_
        else:  # this is the hard case, where we need to convert the initializer expr to a cond
            if not isinstance(key_, ie.IndexSequence):
                if not isinstance(key_, ie.IndexValue):
                    raise Exception("If key is a single non-boolean index, must be an index value")
                key_ = ie.IndexSequence((key_,))

            if not len(key_.members) <= len(basis):
                raise ValueError(
                    f"Sequence keys must have same or fewer members \
                                  than domain dimensions. len({key_.members=}) > len({basis=})"
                )
            if len(key_.members) < len(basis):
                # Expand with domain variables
                # E.g. domain is (x, y, z) and key is (0, 0), then we expand to (0, 0, z)
                key_ = ie.IndexSequence((*key_.members, *basis.members[len(key_.members) :]))

            # TODO here, we could actually support setting with a slice, with the meaning
            # that if the basis vector is between the start and stop of the slice, then
            # the condition is true.
            for child in key_.children:
                assert isinstance(child, ie.IndexValue), "Can only set with values, not slices"

            # Promote initializer exprs to conditions
            _logger.debug(
                "Key was a valid initializer expr. So equating the following: %s",
                list(zip(basis, key_.members, strict=False)),
            )
            for dom_var, member in zip(basis, key_.members, strict=False):
                if not member.struct_eq(dom_var):
                    # NOTE: special case to support syntactic sugar:
                    # x[t+1] = y[t]
                    # By converting it to x[t] = y[t-1]
                    # TODO add more cases to this special case
                    # TODO abstract this using an "inverter"
                    if (
                        isinstance(member, ie.Add)
                        and isinstance(member.left_operand, ie.Symbol)
                        and isinstance(member.right_operand, ie.ConstInt)
                    ):
                        idx = source._underlying.unindexed_domain.find_variable_index(
                            member.left_operand
                        )
                        source_index_expr[idx] = dom_var - member.right_operand
                        var_ub = (
                            source._underlying.unindexed_domain.parameters[idx]
                            + member.right_operand.const
                        )
                        self._underlying.op.domain.set_ubound_override(
                            member.left_operand.as_bound(), var_ub
                        )
                    else:
                        cond = cond & (dom_var == member)

        assert self.shape == source.shape, (
            f"{self.shape=}({type(self.shape)=}) != {source.shape=}({type(source.shape)=})"
        )
        assert self.dtype == source.dtype, f"{self.dtype=} != {source.dtype=}"
        assert source.domain.is_contained_in(self.domain), (
            f"{source.domain=} is not a subset of {self.domain=}"
        )
        # assert not source.requires_grad

        _logger.debug("Setting %s to source %s", self, source)

        if not isinstance(self._underlying.op, top.MergeOp):
            print(self._underlying.op.creation_traceback)
        assert isinstance(self._underlying.op, top.MergeOp), f"{self._underlying.op=}"

        from tempo.core.global_objects import get_active_dg

        get_active_dg().ops_by_id[self._underlying.op.op_id].uncommitted_branch_conds.append(
            (
                cond,
                source._underlying.tensor_id,
                ie.IndexSequence(tuple(source_index_expr)),
            )
        )
        self._underlying.op.num_inputs_[0] = self._underlying.op.num_inputs_[0] + 1

        # ctx_man.dg.register_conditional_definition(
        #    cond,
        #    self._underlying.op.op_id,
        #    source._underlying.tensor_id,
        #    ie.IndexSequence(tuple(source_index_expr)),
        # )


ManyRecurrentTensors = Sequence[RecurrentTensor]
OneOrManyRecurrentTensors = Union[RecurrentTensor, ManyRecurrentTensors]

# TODO ie.IndexExpr instead/
MaybeRecurrentTensor = Union[RecurrentTensor, ie.IndexValue, int, bool, float]  # , List
ManyMaybeRecurrentTensors = Sequence[MaybeRecurrentTensor]
OneOrManyMaybeRecurrentTensors = Union[MaybeRecurrentTensor, ManyMaybeRecurrentTensors]
