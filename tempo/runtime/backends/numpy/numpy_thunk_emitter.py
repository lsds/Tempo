from collections.abc import Callable
from functools import partial

import numpy as np

import tempo.core.tensor_ops as top
from tempo.core import index_expr as ie
from tempo.core.datatypes import OpInId
from tempo.core.thunk import (
    Thunk,
    ThunkExecutionCtx,
)
from tempo.core.thunk_emitter import OpToThunkTranslationFn, ThunkEmissionCtx
from tempo.runtime.thunk_emitter_base import ThunkEmitterBase
from tempo.utils import logger

log = logger.get_logger(__name__)


def rand_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.RandOp)

    backend = emit_ctx.backend

    dtype = backend.to_backend_datatype(op.dtype)

    shape = op.shape.try_resolve(emit_ctx.compile_time_known_symbol_values)

    if shape.is_dynamic():

        def rand_op_thunk(
            inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> tuple[np.ndarray, ...]:
            shape_ = shape.evaluate(exec_ctx.symbol_values)
            val = np.random.uniform(0, 1, size=shape_).astype(dtype)
            return (val,)

    else:
        shape_ = shape.evaluate(emit_ctx.compile_time_known_symbol_values)

        def rand_op_thunk(
            inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> tuple[np.ndarray, ...]:
            val = np.random.uniform(0, 1, size=shape_).astype(dtype)
            return (val,)

    return rand_op_thunk


def const_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.ConstOp)
    shape = op.shape.try_resolve(emit_ctx.compile_time_known_symbol_values)
    value = op.value

    backend = emit_ctx.backend

    dtype = backend.to_backend_datatype(op.dtype)
    val = np.array(value, dtype=dtype)

    shape_ = shape.evaluate(emit_ctx.compile_time_known_symbol_values)

    def const_op_thunk(
        inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> tuple[np.ndarray, ...]:
        return (np.broadcast_to(val, shape_),)

    return const_op_thunk


def eval_symbol_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.EvalSymbolOp)
    shape = op.shape

    backend = emit_ctx.backend

    assert shape.is_scalar()
    symbol_dtype = op.dtype
    dtype = backend.to_backend_datatype(symbol_dtype)

    def f(x: int) -> np.ndarray:
        return np.array(
            x,
            dtype=dtype,
            copy=False,
        )

    s = op.symbol

    def eval_symbol_op_numpy_thunk(
        inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> tuple[np.ndarray, ...]:
        return (f(exec_ctx.symbol_values[s]),)

    return eval_symbol_op_numpy_thunk


def flip_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.FlipOp)
    dim = op.dim
    assert isinstance(dim, int)

    def flip_op_numpy_thunk(
        inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> tuple[np.ndarray, ...]:
        return (np.flip(inputs[0], axis=dim),)

    return flip_op_numpy_thunk


def expand_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.ExpandOp)
    sizes = op.sizes

    sizes = sizes.try_resolve(emit_ctx.compile_time_known_symbol_values)

    if sizes.is_dynamic():

        def expand_op_numpy_thunk(
            inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> tuple[np.ndarray, ...]:
            sizes_ = sizes.evaluate(exec_ctx.symbol_values)
            return (np.broadcast_to(inputs[0], sizes_),)

    else:
        sizes_ = sizes.evaluate(emit_ctx.compile_time_known_symbol_values)

        def expand_op_numpy_thunk(
            inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> tuple[np.ndarray, ...]:
            return (np.broadcast_to(inputs[0], sizes_),)

    return expand_op_numpy_thunk


def reshape_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.ReshapeOp)
    shape = op.shape

    if shape.is_dynamic():

        def reshape_op_numpy_thunk(
            inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> tuple[np.ndarray, ...]:
            return (np.reshape(inputs[0], shape.evaluate(exec_ctx.symbol_values)),)

    else:
        shape_ = shape.evaluate(emit_ctx.compile_time_known_symbol_values)

        def reshape_op_numpy_thunk(
            inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> tuple[np.ndarray, ...]:
            return (np.reshape(inputs[0], shape_),)

    return reshape_op_numpy_thunk


def permute_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.PermuteOp)
    dims = op.dims

    def permute_op_numpy_thunk(
        inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> tuple[np.ndarray, ...]:
        return (np.transpose(inputs[0], dims),)

    return permute_op_numpy_thunk


def cast_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.CastOp)

    backend = emit_ctx.backend

    dtype = backend.to_backend_datatype(op.output_dtype)

    def cast_op_numpy_thunk(
        inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> tuple[np.ndarray, ...]:
        return (inputs[0].astype(dtype),)

    return cast_op_numpy_thunk


def _elementwise_unary_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
    numpy_op: Callable[[np.ndarray], np.ndarray],
) -> Thunk[np.ndarray]:
    def elementwise_unary_op_numpy_thunk(
        inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> tuple[np.ndarray, ...]:
        return (numpy_op(inputs[0]),)

    return elementwise_unary_op_numpy_thunk


def _elementwise_binary_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
    numpy_op: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> Thunk[np.ndarray]:
    def elementwise_binary_op_numpy_thunk(
        inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> tuple[np.ndarray, ...]:
        return (numpy_op(inputs[0], inputs[1]),)

    return elementwise_binary_op_numpy_thunk


def where_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.WhereOp)

    def where_op_numpy_thunk(
        inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> tuple[np.ndarray, ...]:
        return (np.where(inputs[0], inputs[1], inputs[2]),)

    return where_op_numpy_thunk


def sum_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.SumOp)
    dims = op.dims
    keepdims = op.keepdim

    def sum_op_numpy_thunk(
        inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> tuple[np.ndarray, ...]:
        sum_res = np.sum(inputs[0], axis=dims, keepdims=keepdims)
        return (sum_res,)

    return sum_op_numpy_thunk


def cumsum_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.CumSumOp)
    dim = op.dim
    assert isinstance(dim, int)

    def cumsum_op_numpy_thunk(
        inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> tuple[np.ndarray, ...]:
        return (np.cumsum(inputs[0], axis=dim),)

    return cumsum_op_numpy_thunk


def matmul_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.MatMulOp)

    def matmul_op_numpy_thunk(
        inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> tuple[np.ndarray, ...]:
        return (np.matmul(inputs[0], inputs[1]),)

    return matmul_op_numpy_thunk


# def conv_op_translation(
#    op: top.TensorOp,
#    emit_ctx: ThunkEmissionCtx[np.ndarray],
# ) -> Thunk[np.ndarray]:
#    assert isinstance(op, top.ConvOp)
#
#    out_shape = emit_ctx.dg.get_output_shapes(op)[OpOutId(0)]
#    input_shape = emit_ctx.dg.get_input_shape(op, OpInId(0))
#
#    new_batch_size = input_shape[: -(op.n_dims + 1)].prod()
#    new_in_shape = input_shape[-(op.n_dims + 1) :].prepend_dim(new_batch_size)
#
#    stride = op.stride
#
#    padding = (0, 0) * op.n_dims
#    output_padding = (0, 0) * op.n_dims
#
#    if not all(o == 0 for o in output_padding):
#        raise NotImplementedError("output_padding not supported yet in NumPy backend")
#    groups = 1
#
#    # Simplified convolution implementation for NumPy
#    def conv_func(input_data, kernel):
#        # This is a very basic implementation - in practice you'd want a proper conv function
#        # For now, just return a placeholder
#        return np.zeros(input_data.shape[:2] + kernel.shape[2:])
#
#    if out_shape.is_static():
#        new_in_shape_static = new_in_shape.evaluate(emit_ctx.compile_time_known_symbol_values)
#        new_out_shape_static = out_shape.evaluate(emit_ctx.compile_time_known_symbol_values)
#
#        def conv_op_numpy_thunk(
#            inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
#        ) -> tuple[np.ndarray, ...]:
#            input_reshaped = inputs[0].reshape(new_in_shape_static)
#            kernel = inputs[1]
#            conv_result = conv_func(input_reshaped, kernel)
#            return (conv_result.reshape(new_out_shape_static),)
#
#    else:
#
#        def conv_op_numpy_thunk(
#            inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
#        ) -> tuple[np.ndarray, ...]:
#            input_, kernel_ = inputs
#
#            input_ = input_.reshape(new_in_shape.evaluate(exec_ctx.symbol_values))
#
#            conv_result = conv_func(input_, kernel_)
#
#            conv_result = conv_result.reshape(out_shape.evaluate(exec_ctx.symbol_values))
#            return (conv_result,)
#
#    return conv_op_numpy_thunk


def max_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.MaxOp)
    assert len(op.dims) == 1
    dim = op.dims[0]
    keepdims = op.keepdim

    def max_op_numpy_thunk(
        inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> tuple[np.ndarray, ...]:
        res = np.max(inputs[0], axis=dim, keepdims=keepdims)
        arg_res = np.argmax(inputs[0], axis=dim, keepdims=keepdims)
        return (res, arg_res)

    return max_op_numpy_thunk


def squeeze_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.SqueezeOp)
    dim = op.dim

    def squeeze_op_numpy_thunk(
        inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> tuple[np.ndarray, ...]:
        return (np.squeeze(inputs[0], axis=dim),)

    return squeeze_op_numpy_thunk


def unsqueeze_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.UnsqueezeOp)
    dim = op.dim

    def unsqueeze_op_numpy_thunk(
        inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> tuple[np.ndarray, ...]:
        return (np.expand_dims(inputs[0], axis=dim),)

    return unsqueeze_op_numpy_thunk


def cat_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.CatOp)
    dim = op.dim

    def cat_op_numpy_thunk(
        inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> tuple[np.ndarray, ...]:
        return (np.concatenate(inputs, axis=dim),)

    return cat_op_numpy_thunk


def gather_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.GatherOp)
    dim = op.dim

    def gather_op_numpy_thunk(
        inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> tuple[np.ndarray, ...]:
        # Use advanced indexing to perform the gather
        indices = inputs[1]
        if indices.ndim == 0:
            indices = np.expand_dims(indices, 0)
        gathered = np.take(inputs[0], indices, axis=dim)
        return (gathered,)

    return gather_op_numpy_thunk


def scatter_add_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.ScatterAddOp)
    dim = op.dim

    def scatter_add_op_numpy_thunk(
        inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> tuple[np.ndarray, ...]:
        sink, index, src = inputs[0], inputs[1], inputs[2]

        # Create a copy to avoid modifying the input
        result = sink.copy()

        # Use advanced indexing to perform scatter add
        if index.ndim == 0:
            index = np.expand_dims(index, 0)

        # Create index tuple for advanced indexing
        idx = [slice(None)] * sink.ndim
        idx[dim] = index

        result[tuple(idx)] += src
        return (result,)

    return scatter_add_op_numpy_thunk


def split_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.SplitOp)
    dim = op.dim
    num_splits = op.num_splits

    split_size = emit_ctx.dg.get_input_shape(op, OpInId(0)).at(dim) // num_splits

    if isinstance(split_size, ie.ConstInt):
        split_size = split_size.const

    assert isinstance(split_size, int)

    def split_op_numpy_thunk(
        inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> tuple[np.ndarray, ...]:
        return tuple(np.split(inputs[0], indices_or_sections=num_splits, axis=dim))

    return split_op_numpy_thunk


def index_slice_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.IndexSliceOp)
    dim = op.dim
    length = op.length

    if op.is_static():
        assert isinstance(length, (int, ie.ConstInt)), (
            f"length={length} must be an int, is {type(length)}"
        )
        length = int(length)

        def index_slice_op_numpy_thunk(
            inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> tuple[np.ndarray, ...]:
            start_idx = inputs[1]
            # Create index tuple for advanced indexing along the specified dimension
            idx = (
                [slice(None)] * dim
                + [slice(start_idx, start_idx + length)]
                + [slice(None)] * (len(inputs[0].shape) - dim - 1)
            )
            return (inputs[0][tuple(idx)],)

    else:

        def get_fast_eval(x: int | ie.IntIndexValue) -> Callable[[], int]:
            if isinstance(x, ie.IntIndexValue):
                remapped = x.remap(emit_ctx.domain_map)
                remapped.cache_codegenerated_eval(emit_ctx.loop_counters)
                return remapped.eval_fast  # type: ignore
            return lambda: x

        length_eval = get_fast_eval(length)

        def index_slice_op_numpy_thunk(
            inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> tuple[np.ndarray, ...]:
            start_idx = inputs[1]
            # Create index tuple for advanced indexing along the specified dimension
            idx = (
                [slice(None)] * dim
                + [slice(start_idx, start_idx + length_eval())]
                + [slice(None)] * (len(inputs[0].shape) - dim - 1)
            )
            return (inputs[0][tuple(idx)],)

    return index_slice_op_numpy_thunk


def index_select_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.IndexSelectOp)
    dim = op.dim

    def index_select_op_numpy_thunk(
        inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> tuple[np.ndarray, ...]:
        return (np.take(inputs[0], inputs[1], axis=dim),)

    return index_select_op_numpy_thunk


def index_add_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.IndexAddOp)
    dim = op.dim

    index_is_scalar = emit_ctx.dg.get_input_shape(op, OpInId(1)).is_scalar()

    if index_is_scalar:

        def index_add_op_numpy_thunk(
            inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> tuple[np.ndarray, ...]:
            calculated_dim = dim
            if dim < 0:
                # To make sure negative dim actually represents the desired dim
                shape_length = len(inputs[0].shape)
                calculated_dim = shape_length + dim

            # Create a copy to avoid modifying the input
            result = inputs[0].copy()

            # Use advanced indexing to add the value
            idx = (
                [slice(None)] * calculated_dim
                + [inputs[1]]
                + [slice(None)] * (len(result.shape) - calculated_dim - 1)
            )
            result[tuple(idx)] += inputs[2]
            return (result,)

    else:

        def index_add_op_numpy_thunk(
            inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> tuple[np.ndarray, ...]:
            calculated_dim = dim
            if dim < 0:
                shape_length = len(inputs[0].shape)
                calculated_dim = shape_length + dim

            # Create a copy to avoid modifying the input
            result = inputs[0].copy()

            # Use advanced indexing to add the value
            idx = (
                [slice(None)] * calculated_dim
                + [inputs[1]]
                + [slice(None)] * (len(result.shape) - calculated_dim - 1)
            )
            result[tuple(idx)] += inputs[2]
            return (result,)

    return index_add_op_numpy_thunk


def merge_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.MergeOp)

    def merge_op_numpy_thunk(
        inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> tuple[np.ndarray, ...]:
        return (inputs[0],)

    return merge_op_numpy_thunk


def val_to_val_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.ValToValOp)

    in_val = op.in_val
    out_val = op.out_val

    in_val_array = np.array(in_val)
    out_val_array = np.array(out_val)

    # NOTE: We need special handling for NaNs, because np.equal does not work with NaNs
    # Infinity works fine with np.equal
    if np.isnan(in_val_array):

        def val_to_val_op_numpy_thunk(
            inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> tuple[np.ndarray, ...]:
            return (np.where(np.isnan(inputs[0]), out_val_array, inputs[0]),)
    else:

        def val_to_val_op_numpy_thunk(
            inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> tuple[np.ndarray, ...]:
            return (np.where(np.equal(inputs[0], in_val_array), out_val_array, inputs[0]),)

    return val_to_val_op_numpy_thunk


def sort_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.SortOp)

    dim = op.dim
    stable = op.stable
    descending = op.descending

    def sort_op_numpy_thunk(
        inputs: tuple[np.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> tuple[np.ndarray, ...]:
        sorted_array = np.sort(inputs[0], axis=dim, stable=stable)
        if descending:
            sorted_array = np.flip(sorted_array, axis=dim)

        argsorted = np.argsort(inputs[0], axis=dim, stable=stable)
        if descending:
            argsorted = np.flip(argsorted, axis=dim)

        return (sorted_array, argsorted)

    return sort_op_numpy_thunk


def pad_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[np.ndarray],
) -> Thunk[np.ndarray]:
    assert isinstance(op, top.PadOp)

    pad0 = op.padding[0]
    pad1 = op.padding[1]

    value = op.value if hasattr(op, "value") else 0.0
    mode_enum = op.mode

    mode_table = {
        top.PadMode.CONSTANT: "constant",
        top.PadMode.REFLECT: "reflect",
        top.PadMode.ANY: "constant",
    }
    mode = mode_table[mode_enum]

    in_shape_len = len(emit_ctx.dg.get_input_shape(op, OpInId(0))._shape)

    if op.is_static():
        if not isinstance(pad0, int):
            pad0 = pad0.try_eval({})  # type: ignore
        if not isinstance(pad1, int):
            pad1 = pad1.try_eval({})  # type: ignore

        padding = ((0, 0),) * op.dim + ((pad0, pad1),) + ((0, 0),) * (in_shape_len - op.dim - 1)

        def pad_op_numpy_thunk(
            inputs: tuple[np.ndarray, ...],
            exec_ctx: ThunkExecutionCtx,
        ) -> tuple[np.ndarray, ...]:
            return (
                np.pad(
                    inputs[0],
                    pad_width=padding,
                    mode=mode,
                    constant_values=value,
                ),
            )
    else:
        padding_loc = [
            (0, 0),
        ] * (op.dim + 1) + [
            (0, 0),
        ] * (in_shape_len - op.dim - 1)

        def get_fast_eval(
            pad0: ie.IntIndexValueLike, pad1: ie.IntIndexValueLike
        ) -> Callable[[], tuple[tuple[int, int], ...]]:
            if isinstance(pad0, int):
                pad0_eval = lambda: pad0
            else:
                pad0 = pad0.remap(emit_ctx.domain_map)  # type: ignore
                pad0.cache_codegenerated_eval(emit_ctx.loop_counters)
                pad0_eval = pad0.eval_fast
            if isinstance(pad1, int):
                pad1_eval = lambda: pad1
            else:
                pad1 = pad1.remap(emit_ctx.domain_map)  # type: ignore
                pad1.cache_codegenerated_eval(emit_ctx.loop_counters)
                pad1_eval = pad1.eval_fast

            def eval_fast() -> tuple[tuple[int, int], ...]:
                padding_loc[-1] = (pad0_eval(), pad1_eval())  # type: ignore
                return tuple(reversed(padding_loc))

            return eval_fast

        padding_eval = get_fast_eval(pad0, pad1)

        def pad_op_numpy_thunk(
            inputs: tuple[np.ndarray, ...],
            exec_ctx: ThunkExecutionCtx,
        ) -> tuple[np.ndarray, ...]:
            return (np.pad(inputs[0], pad_width=padding_eval(), mode=mode, constant_values=value),)

    return pad_op_numpy_thunk


TEMPO_TO_NUMPY: dict[type[top.TensorOp], OpToThunkTranslationFn[np.ndarray]] = {
    top.RandOp: rand_op_translation,
    top.ConstOp: const_op_translation,
    top.EvalSymbolOp: eval_symbol_op_translation,
    top.MergeOp: merge_op_translation,
    top.FlipOp: flip_op_translation,
    top.CatOp: cat_op_translation,
    top.SqueezeOp: squeeze_op_translation,
    top.UnsqueezeOp: unsqueeze_op_translation,
    top.ExpandOp: expand_op_translation,
    top.ReshapeOp: reshape_op_translation,
    top.PermuteOp: permute_op_translation,
    top.CastOp: cast_op_translation,
    top.SqrtOp: partial(_elementwise_unary_op_translation, numpy_op=np.sqrt),
    top.NegOp: partial(_elementwise_unary_op_translation, numpy_op=np.negative),
    top.NotOp: partial(_elementwise_unary_op_translation, numpy_op=np.logical_not),
    top.LnOp: partial(_elementwise_unary_op_translation, numpy_op=np.log),
    top.ExpOp: partial(_elementwise_unary_op_translation, numpy_op=np.exp),
    top.SinOp: partial(_elementwise_unary_op_translation, numpy_op=np.sin),
    top.AddOp: partial(_elementwise_binary_op_translation, numpy_op=np.add),
    top.SubOp: partial(_elementwise_binary_op_translation, numpy_op=np.subtract),
    top.MulOp: partial(_elementwise_binary_op_translation, numpy_op=np.multiply),
    top.DivOp: partial(_elementwise_binary_op_translation, numpy_op=np.true_divide),
    top.PowOp: partial(_elementwise_binary_op_translation, numpy_op=np.power),
    top.OrOp: partial(_elementwise_binary_op_translation, numpy_op=np.logical_or),
    top.AndOp: partial(_elementwise_binary_op_translation, numpy_op=np.logical_and),
    top.EqualOp: partial(_elementwise_binary_op_translation, numpy_op=np.equal),
    top.LessThanOp: partial(_elementwise_binary_op_translation, numpy_op=np.less),
    top.WhereOp: where_op_translation,
    top.SumOp: sum_op_translation,
    top.MaxOp: max_op_translation,
    top.CumSumOp: cumsum_op_translation,
    top.GatherOp: gather_op_translation,
    top.ScatterAddOp: scatter_add_op_translation,
    top.SplitOp: split_op_translation,
    top.IndexSliceOp: index_slice_op_translation,
    top.IndexSelectOp: index_select_op_translation,
    top.IndexAddOp: index_add_op_translation,
    top.MatMulOp: matmul_op_translation,
    # top.ConvOp: conv_op_translation,
    top.PadOp: pad_op_translation,
    top.ValToValOp: val_to_val_op_translation,
    top.IdentOp: lambda op, emit_ctx: lambda inputs, exec_ctx: inputs,
    top.SortOp: sort_op_translation,
}


class NumPyThunkEmitter(ThunkEmitterBase[np.ndarray]):
    def _emit_thunk_for_op(
        self, op: top.TensorOp, ctx: ThunkEmissionCtx[np.ndarray]
    ) -> Thunk[np.ndarray]:
        thunk = TEMPO_TO_NUMPY[type(op)](op, ctx)
        return thunk  # type: ignore
