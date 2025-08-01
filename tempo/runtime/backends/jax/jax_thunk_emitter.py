import sys
from functools import partial
from typing import Callable, Dict, Tuple, Type, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tempo.core.tensor_ops as top
from tempo.core import index_expr as ie
from tempo.core.datatypes import OpInId, OpOutId
from tempo.core.thunk import (
    OpToThunkTranslationFn,
    Thunk,
    ThunkEmissionCtx,
    ThunkExecutionCtx,
)
from tempo.runtime.thunk_emitter import ThunkEmitter
from tempo.utils import logger

log = logger.get_logger(__name__)


def rand_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.RandOp)

    from tempo.runtime.backends.jax.jax_backend import JaxBackend

    dtype = JaxBackend.to_backend_datatype(op.dtype)
    device = JaxBackend.to_backend_device_obj(emit_ctx.dev)

    shape = op.shape.try_resolve(emit_ctx.compile_time_known_symbol_values)
    # assert op.dtype == dtypes.float32

    key = [JaxBackend.to_device(jax.random.PRNGKey(np.random.randint(0, sys.maxsize - 1)), device)]

    if shape.is_dynamic():

        def split_and_uniform(
            key: jnp.ndarray, shape_: Tuple[int, ...]
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            key_, subkey = jax.random.split(key)
            val = JaxBackend.to_device(
                jax.random.uniform(subkey, shape=shape_, dtype=dtype), device
            )
            return key_, val

        jitted_split_and_uniform = jax.jit(
            split_and_uniform,
            device=device,
        )

        def rand_op_thunk(
            inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[jnp.ndarray, ...]:
            shape_ = shape.evaluate(exec_ctx.symbol_values)
            key_, val = jitted_split_and_uniform(key[0], shape_)
            key[0] = key_
            return (val,)

    else:
        shape_ = shape.evaluate(emit_ctx.compile_time_known_symbol_values)

        def split_and_uniform2(key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            key_, subkey = jax.random.split(key)
            val = jax.random.uniform(subkey, shape=shape_, dtype=dtype)
            return key_, val

        jitted_split_and_uniform2 = jax.jit(
            split_and_uniform2,
            device=device,
        )
        # warmup
        jitted_split_and_uniform2(key[0])

        def rand_op_thunk(
            inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[jnp.ndarray, ...]:
            key_, val = jitted_split_and_uniform2(key[0])
            key[0] = key_
            return (val,)

    return rand_op_thunk


def const_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.ConstOp)
    shape = op.shape.try_resolve(emit_ctx.compile_time_known_symbol_values)
    value = op.value

    from tempo.runtime.backends.jax.jax_backend import JaxBackend

    device = JaxBackend.to_backend_device_obj(emit_ctx.dev)

    dtype = JaxBackend.to_backend_datatype(op.dtype)
    val = jnp.array(value, dtype=dtype, device=device)

    # if shape.is_dynamic():

    #    def const_op_thunk(
    #        inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
    #    ) -> Tuple[jnp.ndarray, ...]:
    #        shape_ = shape.evaluate(exec_ctx.symbol_values)
    #        return (jnp.broadcast_to(val, shape_),)

    # else:
    shape_ = shape.evaluate(emit_ctx.compile_time_known_symbol_values)
    # if emit_ctx.exec_cfg.debug_mode:

    # else:
    #    if val.shape == ():
    #        if len(shape_) > 0:
    #            val = jnp.expand_dims(val, tuple(range(len(shape_))))
    #    else:
    #        val = jnp.broadcast_to(val, shape_)

    # NOTE: we do it this way so that the jit can recognize the expansion
    def const_op_thunk(
        inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[jnp.ndarray, ...]:
        return (jnp.broadcast_to(val, shape_),)

    return const_op_thunk


def eval_symbol_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.EvalSymbolOp)
    shape = op.shape

    from tempo.runtime.backends.jax.jax_backend import JaxBackend

    device = JaxBackend.to_backend_device_obj(emit_ctx.dev)
    assert shape.is_scalar()
    symbol_dtype = op.dtype
    dtype = JaxBackend.to_backend_datatype(symbol_dtype)

    def f(x: int) -> jnp.ndarray:
        return jnp.array(
            x,
            dtype=dtype,
            copy=False,
            device=device,
        )

    jitted_f = jax.jit(f, device=device)
    # warmup
    jitted_f(0)
    s = op.symbol

    def eval_symbol_op_jax_thunk(
        inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[jnp.ndarray, ...]:
        return (jitted_f(exec_ctx.symbol_values[s]),)

    return eval_symbol_op_jax_thunk


def flip_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.FlipOp)
    dim = op.dim
    assert isinstance(dim, int)

    def flip_op_jax_thunk(
        inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[jnp.ndarray, ...]:
        return (jnp.flip(inputs[0], axis=dim),)

    return flip_op_jax_thunk


def expand_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.ExpandOp)
    sizes = op.sizes

    sizes = sizes.try_resolve(emit_ctx.compile_time_known_symbol_values)

    if sizes.is_dynamic():

        def expand_op_jax_thunk(
            inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[jnp.ndarray, ...]:
            sizes_ = sizes.evaluate(exec_ctx.symbol_values)
            return (jnp.broadcast_to(inputs[0], sizes_),)

    else:
        sizes_ = sizes.evaluate(emit_ctx.compile_time_known_symbol_values)

        def expand_op_jax_thunk(
            inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[jnp.ndarray, ...]:
            return (jnp.broadcast_to(inputs[0], sizes_),)

    return expand_op_jax_thunk


def reshape_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.ReshapeOp)
    shape = op.shape

    if shape.is_dynamic():

        def reshape_op_jax_thunk(
            inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[jnp.ndarray, ...]:
            return (jnp.reshape(inputs[0], shape.evaluate(exec_ctx.symbol_values)),)

    else:
        shape_ = shape.evaluate(emit_ctx.compile_time_known_symbol_values)

        def reshape_op_jax_thunk(
            inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[jnp.ndarray, ...]:
            return (jnp.reshape(inputs[0], shape_),)

    return reshape_op_jax_thunk


def permute_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.PermuteOp)
    dims = op.dims

    def permute_op_jax_thunk(
        inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[jnp.ndarray, ...]:
        return (jnp.permute_dims(inputs[0], dims),)

    return permute_op_jax_thunk


def cast_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.CastOp)

    from tempo.runtime.backends.jax.jax_backend import JaxBackend

    dtype = JaxBackend.to_backend_datatype(op.output_dtype)

    def cast_op_jax_thunk(
        inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[jnp.ndarray, ...]:
        return (inputs[0].astype(dtype),)

    return cast_op_jax_thunk


def _elementwise_unary_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
    jax_op: Callable[[jnp.ndarray], jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    def elementwise_unary_op_jax_thunk(
        inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[jnp.ndarray, ...]:
        return (jax_op(inputs[0]),)

    return elementwise_unary_op_jax_thunk


def _elementwise_binary_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
    jax_op: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    # if emit_ctx.dg.parent_graph
    def elementwise_binary_op_jax_thunk(
        inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[jnp.ndarray, ...]:
        return (jax_op(inputs[0], inputs[1]),)

    return elementwise_binary_op_jax_thunk


def where_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.WhereOp)

    def where_op_jax_thunk(
        inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[jnp.ndarray, ...]:
        return (jnp.where(inputs[0], inputs[1], inputs[2]),)

    return where_op_jax_thunk


def sum_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.SumOp)
    dims = op.dims
    keepdims = op.keepdim

    def sum_op_jax_thunk(
        inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[jnp.ndarray, ...]:
        sum_res = jnp.sum(inputs[0], axis=dims, keepdims=keepdims)
        return (sum_res,)

    return sum_op_jax_thunk


def cumsum_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.CumSumOp)
    dim = op.dim
    assert isinstance(dim, int)

    def cumsum_op_jax_thunk(
        inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[jnp.ndarray, ...]:
        return (jnp.cumsum(inputs[0], axis=dim),)

    return cumsum_op_jax_thunk


def matmul_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.MatMulOp)

    def matmul_op_jax_thunk(
        inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[jnp.ndarray, ...]:
        return (jnp.matmul(inputs[0], inputs[1]),)

    return matmul_op_jax_thunk


def conv_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.ConvOp)

    out_shape = emit_ctx.dg.get_output_shapes(op)[OpOutId(0)]
    input_shape = emit_ctx.dg.get_input_shape(op, OpInId(0))

    new_batch_size = input_shape[: -(op.n_dims + 1)].prod()
    new_in_shape = input_shape[-(op.n_dims + 1) :].prepend_dim(new_batch_size)

    stride = op.stride

    padding = (0, 0) * op.n_dims
    output_padding = (0, 0) * op.n_dims
    # dilation = (1, 1) * op.n_dims

    # padding = tuple((p, p) for p in op.padding)
    # dilation = op.dilation
    # output_padding = op.output_padding

    if not all(o == 0 for o in output_padding):
        raise NotImplementedError("output_padding not supported yet in JAX backend")
    groups = 1  # op.groups

    conv_func = partial(
        lax.conv_general_dilated,
        window_strides=stride,
        padding=padding,
        # rhs_dilation=dilation,
        feature_group_count=groups,
        dimension_numbers=None,
    )

    conv_t_func = partial(
        lax.conv_transpose,
        strides=stride,
        padding=padding,
        # rhs_dilation=dilation,
        dimension_numbers=None,
        # transpose_kernel=True,
    )
    if out_shape.is_static():
        new_in_shape_static = new_in_shape.evaluate(emit_ctx.compile_time_known_symbol_values)
        new_out_shape_static = out_shape.evaluate(emit_ctx.compile_time_known_symbol_values)
        jit_conv_func = jax.jit(
            lambda i, k: (
                conv_func(i.reshape(new_in_shape_static), k).reshape(new_out_shape_static),
            ),
            inline=True,
        )
        jit_conv_t_func = jax.jit(
            lambda i, k: (
                conv_t_func(i.reshape(new_in_shape_static), k).reshape(new_out_shape_static),
            ),
            inline=True,
        )

        fun = jit_conv_func if not op.transposed else jit_conv_t_func

        def conv_op_jax_thunk(
            inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[jnp.ndarray, ...]:
            return fun(inputs[0], inputs[1])  # type: ignore

    else:
        fun = conv_t_func if op.transposed else conv_func

        def conv_op_jax_thunk(
            inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[jnp.ndarray, ...]:
            input_, kernel_ = inputs

            input_ = input_.reshape(new_in_shape.evaluate(exec_ctx.symbol_values))

            conv_result = fun(
                lhs=input_,
                rhs=kernel_,
            )

            conv_result = conv_result.reshape(out_shape.evaluate(exec_ctx.symbol_values))
            return (conv_result,)

    return conv_op_jax_thunk


def max_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.MaxOp)
    assert len(op.dims) == 1
    dim = op.dims[0]
    keepdims = op.keepdim

    def max_op_jax_thunk(
        inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[jnp.ndarray, ...]:
        res = jnp.max(inputs[0], axis=dim, keepdims=keepdims)

        # NOTE: This already returns int32
        arg_res = jnp.argmax(inputs[0], axis=dim, keepdims=keepdims)
        return (res, arg_res)

    return max_op_jax_thunk


def squeeze_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.SqueezeOp)
    dim = op.dim

    def squeeze_op_jax_thunk(
        inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[jnp.ndarray, ...]:
        # return (squeeze_p.impl(inputs[0], dimensions=(dim,)))
        return (jnp.squeeze(inputs[0], axis=dim),)

    return squeeze_op_jax_thunk


def unsqueeze_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.UnsqueezeOp)
    dim = op.dim

    def unsqueeze_op_jax_thunk(
        inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[jnp.ndarray, ...]:
        return (jnp.expand_dims(inputs[0], axis=dim),)

    return unsqueeze_op_jax_thunk


def cat_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.CatOp)
    dim = op.dim

    def cat_op_jax_thunk(
        inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[jnp.ndarray, ...]:
        return (jnp.concatenate(inputs, axis=dim),)

    return cat_op_jax_thunk


def gather_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.GatherOp)
    dim = op.dim

    def gather_op_jax_thunk(
        inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[jnp.ndarray, ...]:
        # Use `take_along_axis` to perform the gather
        gathered = jnp.take_along_axis(inputs[0], inputs[1], axis=dim)

        return (gathered,)

    return gather_op_jax_thunk


def scatter_add_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.ScatterAddOp)
    dim = op.dim

    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    vmap_inner = partial(jax.vmap, in_axes=(0, 0, 0), out_axes=0)

    # Seems to work for both static and dynamic shapes
    def scatter_add_op_jax_thunk(
        inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[jnp.ndarray, ...]:
        sink, index, src = inputs[0], inputs[1], inputs[2]

        scatter_fn = partial(jax.lax.scatter_add, dimension_numbers=dnums)
        for _ in range(len(sink.shape) - 1):
            scatter_fn = vmap_inner(scatter_fn)
        swap = lambda x: jnp.swapaxes(x, dim, -1)
        sink, index, src = list(map(swap, (sink, index, src)))
        return (swap(scatter_fn(sink, jnp.expand_dims(index, axis=-1), src)),)  # type: ignore

    return scatter_add_op_jax_thunk


def split_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.SplitOp)
    dim = op.dim
    num_splits = op.num_splits

    split_size = emit_ctx.dg.get_input_shape(op, OpInId(0)).at(dim) // num_splits

    if isinstance(split_size, ie.ConstInt):
        split_size = split_size.const

    assert isinstance(split_size, int)

    def split_op_jax_thunk(
        inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[jnp.ndarray, ...]:
        return tuple(jnp.split(inputs[0], indices_or_sections=num_splits, axis=dim))

    return split_op_jax_thunk


def index_slice_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.IndexSliceOp)
    dim = op.dim
    length = op.length

    # NOTE: JAX is a bit schizo about inplace views, so we need to do some
    #       gymnastics to get it to work.
    # import torch
    # import torch.utils.dlpack as tdl
    # torch_from_dlpack = tdl.from_dlpack
    # jax_from_dlpack = jax.dlpack.from_dlpack
    # inplace_view = lambda x, idx: jax_from_dlpack(torch_from_dlpack(x)[idx])
    # Unfortunately doesnt work:
    # jaxlib.xla_extension.XlaRuntimeError: UNIMPLEMENTED: Only DLPack tensors with trivial
    # (compact) striding are supported; i.e., tensors whose striding represents a transposition
    # of the underlying buffer but not broadcasting. Dimensions were: [256,5,1,128],
    # strides were [128000,128,1,1].

    dim_slice = jax.lax.dynamic_slice_in_dim
    if op.is_static():
        assert isinstance(length, int)

        def index_slice_op_jax_thunk(
            inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[jnp.ndarray, ...]:
            return (dim_slice(inputs[0], inputs[1], length, dim),)
            # return (inputs[0].at[prebuilt_idx].get(),)
            # return (inplace_view(inputs[0], prebuilt_idx),)

    else:

        def get_fast_eval(x: Union[int, ie.IntIndexValue]) -> Callable[[], int]:
            if isinstance(x, ie.IntIndexValue):
                remapped = x.remap(emit_ctx.domain_map)
                remapped.cache_codegenerated_eval(emit_ctx.loop_counters)
                return remapped.eval_fast  # type: ignore
            return lambda: x

        length_eval = get_fast_eval(length)

        def index_slice_op_jax_thunk(
            inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[jnp.ndarray, ...]:
            # return (inputs[0].at[nones + (slice(start_eval(), stop_eval(), step_eval()),)].get(),)
            return (dim_slice(inputs[0], inputs[1], length_eval(), dim),)
            # return (
            #    inplace_view(
            #        inputs[0], nones + (slice(start_eval(), stop_eval(), step_eval()),)
            #    ),
            # )

    return index_slice_op_jax_thunk


def index_select_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.IndexSelectOp)
    dim = op.dim

    from tempo.runtime.backends.jax.jax_backend import JaxBackend

    device = JaxBackend.to_backend_device_obj(emit_ctx.dev)

    jitted_take = jax.jit(partial(jnp.take, axis=dim), device=device, inline=True)

    def index_select_op_jax_thunk(
        inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[jnp.ndarray, ...]:
        # return (jnp.take(inputs[0], inputs[1], axis=dim),)
        return (jitted_take(inputs[0], inputs[1]),)

    return index_select_op_jax_thunk


def index_add_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.IndexAddOp)
    dim = op.dim

    index_is_scalar = emit_ctx.dg.get_input_shape(op, OpInId(1)).is_scalar()

    if index_is_scalar:

        def index_add_op_jax_thunk(
            inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[jnp.ndarray, ...]:
            calculated_dim = dim
            if dim < 0:
                # To make sure negative dim actually represents the desired dim
                shape_length = len(inputs[0].shape)
                calculated_dim = shape_length + dim
            nones = (slice(None, None),) * calculated_dim
            res = inputs[0].at[(*nones, jnp.expand_dims(inputs[1], 0))].add(inputs[2])
            return (res,)

    else:

        def index_add_op_jax_thunk(
            inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[jnp.ndarray, ...]:
            calculated_dim = dim
            if dim < 0:
                shape_length = len(inputs[0].shape)
                calculated_dim = shape_length + dim
            nones = (slice(None, None),) * calculated_dim
            res = inputs[0].at[(*nones, inputs[1])].add(inputs[2])
            return (res,)

    return index_add_op_jax_thunk


def merge_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.MergeOp)

    def merge_op_jax_thunk(
        inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[jnp.ndarray, ...]:
        return (inputs[0],)

    return merge_op_jax_thunk


def val_to_val_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
    assert isinstance(op, top.ValToValOp)

    in_val = op.in_val
    out_val = op.out_val

    in_val_array = jnp.array(in_val)
    out_val_array = jnp.array(out_val)

    # NOTE: We need special handling for NaNs, because jnp.equal does not work with NaNs
    # Infinity works fine with jnp.equal
    if jnp.isnan(in_val_array):

        def val_to_val_op_jax_thunk(
            inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[jnp.ndarray, ...]:
            return (jnp.where(jnp.isnan(inputs[0]), out_val_array, inputs[0]),)
    else:

        def val_to_val_op_jax_thunk(
            inputs: Tuple[jnp.ndarray, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[jnp.ndarray, ...]:
            return (jnp.where(jnp.equal(inputs[0], in_val_array), out_val_array, inputs[0]),)

    return val_to_val_op_jax_thunk


def pad_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
) -> Thunk[jnp.ndarray]:
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

        def pad_op_jax_thunk(
            inputs: Tuple[jnp.ndarray, ...],
            exec_ctx: ThunkExecutionCtx,
        ) -> Tuple[jnp.ndarray, ...]:
            return (
                jnp.pad(
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
        ) -> Callable[[], Tuple[Tuple[int, int], ...]]:
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

            def eval_fast() -> Tuple[Tuple[int, int], ...]:
                padding_loc[-1] = (pad0_eval(), pad1_eval())  # type: ignore
                return tuple(reversed(padding_loc))

            return eval_fast

        padding_eval = get_fast_eval(pad0, pad1)

        def pad_op_jax_thunk(
            inputs: Tuple[jnp.ndarray, ...],
            exec_ctx: ThunkExecutionCtx,
        ) -> Tuple[jnp.ndarray, ...]:
            return (jnp.pad(inputs[0], pad_width=padding_eval(), mode=mode, constant_values=value),)

    return pad_op_jax_thunk


# def udf_translation(
#    op: top.TensorOp,
#    emit_ctx: ThunkEmissionCtx[jnp.ndarray],
# ) -> Thunk[jnp.ndarray]:
#    assert isinstance(op, top.UDFOp)
#    return op.desc.thunk_translation(op, emit_ctx)  # type: ignore


TEMPO_TO_JAX: Dict[Type[top.TensorOp], OpToThunkTranslationFn[jnp.ndarray]] = {
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
    top.SqrtOp: partial(_elementwise_unary_op_translation, jax_op=jnp.sqrt),
    top.NegOp: partial(_elementwise_unary_op_translation, jax_op=jnp.negative),
    top.NotOp: partial(_elementwise_unary_op_translation, jax_op=jnp.logical_not),
    top.LnOp: partial(_elementwise_unary_op_translation, jax_op=jnp.log),
    top.ExpOp: partial(_elementwise_unary_op_translation, jax_op=jnp.exp),
    top.SinOp: partial(_elementwise_unary_op_translation, jax_op=jnp.sin),
    top.AddOp: partial(_elementwise_binary_op_translation, jax_op=jnp.add),
    top.SubOp: partial(_elementwise_binary_op_translation, jax_op=jnp.subtract),
    top.MulOp: partial(_elementwise_binary_op_translation, jax_op=jnp.multiply),
    top.DivOp: partial(_elementwise_binary_op_translation, jax_op=jnp.true_divide),
    top.PowOp: partial(_elementwise_binary_op_translation, jax_op=jnp.pow),
    top.OrOp: partial(_elementwise_binary_op_translation, jax_op=jnp.logical_or),
    top.AndOp: partial(_elementwise_binary_op_translation, jax_op=jnp.logical_and),
    top.EqualOp: partial(_elementwise_binary_op_translation, jax_op=jnp.equal),
    top.LessThanOp: partial(_elementwise_binary_op_translation, jax_op=jnp.less),
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
    top.ConvOp: conv_op_translation,
    top.PadOp: pad_op_translation,
    top.ValToValOp: val_to_val_op_translation,
    # top.ConvBwdOp: conv_bwd_op_translation,
    # top.ExecuteDataflowSubgraphOp: execute_dataflow_op_translation,
    # top.UDFOp: udf_translation,
    top.IdentOp: lambda op, emit_ctx: lambda inputs, exec_ctx: inputs,
}


class JaxThunkEmitter(ThunkEmitter[jnp.ndarray]):
    def _emit_thunk_for_op(
        self, op: top.TensorOp, ctx: ThunkEmissionCtx[jnp.ndarray]
    ) -> Thunk[jnp.ndarray]:
        thunk = TEMPO_TO_JAX[type(op)](op, ctx)
        return thunk


# RandOp
# ConstOp
# EvalSymbolOp
# MergeOp
# FlipOp
# CatOp
# SqueezeOp
# UnsqueezeOp
# ExpandOp
# ReshapeOp
# PermuteOp
# CastOp
# SqrtOp
# NegOp
# NotOp
# LnOp
# ExpOp
# SinOp
# AddOp
# SubOp
# MulOp
# DivOp
# PowOp
# OrOp
# AndOp
# EqualOp
# LessThanOp
# WhereOp
# SumOp
# MaxOp
# CumSumOp
# GatherOp
# ScatterAddOp
# IndexSliceOp
# IndexSelectOp
# IndexAddOp
# MatMulOp
# ConvOp
# PadOp
# IdentOp
