from functools import partial
from typing import Callable, Dict, Tuple, Type, Union

import torch
import torch.fx

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

# op_src = torch.ops.aten  # type: ignore
# op_src_pad = op_src.pad

op_src = torch
op_src_pad = torch.nn.functional.pad


def rand_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    dev = emit_ctx.dev
    assert isinstance(op, top.RandOp)

    from tempo.runtime.backends.pytorch.pytorch_backend import PyTorchBackend

    dev = PyTorchBackend.to_backend_device_obj(emit_ctx.dev)
    dtype = PyTorchBackend.to_backend_datatype(op.dtype)
    shape = op.shape

    if shape.is_dynamic():

        def rand_op_thunk(
            inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[torch.Tensor, ...]:
            shape_ = shape.evaluate(exec_ctx.symbol_values)
            return (torch.rand(size=shape_, dtype=dtype, device=dev),)
    else:
        shape_ = shape.evaluate(emit_ctx.compile_time_known_symbol_values)

        def rand_op_thunk(
            inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[torch.Tensor, ...]:
            return (torch.rand(size=shape_, dtype=dtype, device=dev),)

    return rand_op_thunk


def const_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.ConstOp)
    from tempo.runtime.backends.pytorch.pytorch_backend import PyTorchBackend

    dtype = PyTorchBackend.to_backend_datatype(op.dtype)
    dev = PyTorchBackend.to_backend_device_obj(emit_ctx.dev)
    shape = op.shape
    t = torch.tensor(data=op.value, dtype=dtype, device=dev)

    if shape.is_dynamic():

        def const_op_thunk(
            inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[torch.Tensor, ...]:
            shape_ = shape.evaluate(exec_ctx.symbol_values)
            t1 = t.expand(shape_)
            return (t1,)
    else:
        static_shape = shape.as_static()._shape
        t = PyTorchBackend.to_device(t, dev)
        t = t.expand(static_shape).clone().contiguous()

        def const_op_thunk(
            inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[torch.Tensor, ...]:
            return (t,)

    return const_op_thunk


def eval_symbol_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    from tempo.runtime.backends.pytorch.pytorch_backend import PyTorchBackend

    dev = emit_ctx.dev
    dev_torch = PyTorchBackend.to_backend_device_obj(dev)
    assert isinstance(op, top.EvalSymbolOp)

    symbol_dtype = op.dtype
    dtype = PyTorchBackend.to_backend_datatype(symbol_dtype)

    symbol = op.symbol

    def eval_symbol_op_torch_thunk(
        inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[torch.Tensor, ...]:
        # return (
        #    torch.tensor(
        #        exec_ctx.symbol_values[op.symbol],
        #        dtype=dtype,
        #        device=dev_torch,
        #    ),
        # )
        # NOTE: This aims to avoid synchronizations
        tensor = torch.empty((), dtype=dtype, device=dev_torch)
        tensor.fill_(exec_ctx.symbol_values[symbol])
        return (tensor,)

    return eval_symbol_op_torch_thunk


def flip_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.FlipOp)
    dim = op.dim
    assert isinstance(dim, int)

    def flip_op_torch_thunk(
        inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[torch.Tensor, ...]:
        return (op_src.flip(inputs[0], (dim,)),)

    return flip_op_torch_thunk


def expand_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.ExpandOp)
    sizes = op.sizes

    sizes = sizes.try_resolve(emit_ctx.compile_time_known_symbol_values)

    if sizes.is_dynamic():

        def expand_op_torch_thunk(
            inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[torch.Tensor, ...]:
            sizes_ = sizes.evaluate(exec_ctx.symbol_values)
            return (inputs[0].expand(sizes_),)
    else:
        sizes_ = sizes.evaluate(emit_ctx.compile_time_known_symbol_values)

        def expand_op_torch_thunk(
            inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[torch.Tensor, ...]:
            return (inputs[0].expand(sizes_),)

    return expand_op_torch_thunk


def reshape_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.ReshapeOp)
    shape = op.shape

    if shape.is_dynamic():

        def reshape_op_torch_thunk(
            inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[torch.Tensor, ...]:
            return (op_src.reshape(inputs[0], shape.evaluate(exec_ctx.symbol_values)),)
    else:
        shape_ = shape.evaluate(emit_ctx.compile_time_known_symbol_values)

        def reshape_op_torch_thunk(
            inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[torch.Tensor, ...]:
            return (op_src.reshape(inputs[0], shape_),)

    return reshape_op_torch_thunk


def permute_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.PermuteOp)
    dims = op.dims

    def permute_op_torch_thunk(
        inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[torch.Tensor, ...]:
        inp = inputs[0]
        return (op_src.permute(inp, dims),)

    return permute_op_torch_thunk


def cast_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.CastOp)
    from tempo.runtime.backends.pytorch.pytorch_backend import PyTorchBackend

    dtype = PyTorchBackend.to_backend_datatype(op.output_dtype)

    def cast_op_torch_thunk(
        inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[torch.Tensor, ...]:
        return (inputs[0].to(dtype),)

    return cast_op_torch_thunk


def _elementwise_unary_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
    torch_op: Callable[[torch.Tensor], torch.Tensor],
) -> Thunk[torch.Tensor]:
    output_is_dynamic = emit_ctx.dg.get_output_shapes(op)[OpOutId(0)].is_dynamic()

    if output_is_dynamic:

        def elementwise_unary_op_torch_thunk(
            inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[torch.Tensor, ...]:
            return (torch_op(inputs[0]),)
    else:

        def elementwise_unary_op_torch_thunk(
            inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[torch.Tensor, ...]:
            return (torch_op(inputs[0]),)

    return elementwise_unary_op_torch_thunk


def _elementwise_binary_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
    torch_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Thunk[torch.Tensor]:
    output_is_dynamic = emit_ctx.dg.get_output_shapes(op)[OpOutId(0)].is_dynamic()

    if output_is_dynamic:

        def elementwise_binary_op_torch_thunk(
            inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[torch.Tensor, ...]:
            return (torch_op(inputs[0], inputs[1]),)
    else:

        def elementwise_binary_op_torch_thunk(
            inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[torch.Tensor, ...]:
            return (torch_op(inputs[0], inputs[1]),)

    return elementwise_binary_op_torch_thunk


def where_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.WhereOp)

    def where_op_torch_thunk(
        inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[torch.Tensor, ...]:
        return (op_src.where(inputs[0], inputs[1], inputs[2]),)

    return where_op_torch_thunk


def sum_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.SumOp)
    dims = op.dims
    keepdim = op.keepdim

    def sum_op_torch_thunk(
        inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[torch.Tensor, ...]:
        return (op_src.sum(inputs[0], dim=dims, keepdim=keepdim),)

    return sum_op_torch_thunk


def cumsum_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.CumSumOp)
    dim = op.dim
    assert isinstance(dim, int)

    def cumsum_op_torch_thunk(
        inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[torch.Tensor, ...]:
        return (op_src.cumsum(inputs[0], dim=dim),)

    return cumsum_op_torch_thunk


def matmul_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.MatMulOp)

    def matmul_torch_thunk(
        inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[torch.Tensor, ...]:
        return (op_src.matmul(inputs[0], inputs[1]),)

    return matmul_torch_thunk


def conv_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.ConvOp)

    out_shape = emit_ctx.dg.get_output_shapes(op)[OpOutId(0)]
    input_shape = emit_ctx.dg.get_input_shape(op, OpInId(0))

    padding = (0,) * (op.n_dims * 2)
    output_padding = (0,) * (op.n_dims * 2)
    dilation = (1,) * op.n_dims

    torch_op = op_src.convolution  # type: ignore
    fn = partial(
        torch_op,  # type: ignore
        bias=None,
        stride=op.stride,
        padding=padding,
        dilation=dilation,
        transposed=op.transposed,
        output_padding=output_padding,
        groups=1,
    )

    reshape_needed = len(input_shape) > 2 + op.n_dims
    new_batch_size = input_shape[: -(op.n_dims + 1)].prod()
    new_in_shape = input_shape[-(op.n_dims + 1) :].prepend_dim(new_batch_size)

    if reshape_needed:
        if new_in_shape.is_dynamic():

            def conv_op_torch_thunk(
                inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
            ) -> Tuple[torch.Tensor, ...]:
                input_ = inputs[0]
                input_ = input_.reshape(new_in_shape.evaluate(exec_ctx.symbol_values))
                weight = inputs[1]
                res = fn(input_, weight)  # type: ignore
                res = res.reshape(out_shape.evaluate(exec_ctx.symbol_values))
                return (res,)
        else:
            new_in_shape_ = new_in_shape.evaluate(emit_ctx.compile_time_known_symbol_values)
            out_shape_ = out_shape.evaluate(emit_ctx.compile_time_known_symbol_values)

            def conv_op_torch_thunk(
                inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
            ) -> Tuple[torch.Tensor, ...]:
                input_ = inputs[0]
                input_ = input_.reshape(new_in_shape_)
                weight = inputs[1]
                res = fn(input_, weight)
                return (res.reshape(out_shape_),)
    else:

        def conv_op_torch_thunk(
            inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[torch.Tensor, ...]:
            return fn(*inputs)  # type: ignore

    return conv_op_torch_thunk


def max_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.MaxOp)
    assert len(op.dims) == 1
    dims = op.dims[0]
    keepdim = op.keepdim

    def max_op_torch_thunk(
        inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[torch.Tensor, ...]:
        res = op_src.max(inputs[0], dim=dims, keepdim=keepdim)
        return (res[0], res[1].to(torch.int32))

    return max_op_torch_thunk


def squeeze_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.SqueezeOp)
    dim = op.dim

    def squeeze_op_torch_thunk(
        inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[torch.Tensor, ...]:
        res = op_src.squeeze(inputs[0], dim=dim)
        return (res,)

    return squeeze_op_torch_thunk


def unsqueeze_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.UnsqueezeOp)
    dim = op.dim

    def unsqueeze_op_torch_thunk(
        inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[torch.Tensor, ...]:
        res = op_src.unsqueeze(inputs[0], dim=dim)
        return (res,)

    return unsqueeze_op_torch_thunk


def cat_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.CatOp)
    dim = op.dim

    def cat_op_torch_thunk(
        inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[torch.Tensor, ...]:
        res = op_src.cat(inputs, dim=dim)
        return (res,)

    return cat_op_torch_thunk


def gather_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.GatherOp)
    dim = op.dim

    def gather_op_torch_thunk(
        inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[torch.Tensor, ...]:
        # print(f"Invoked {op}")
        ##print(inputs[0])
        # print(inputs[1])
        res = op_src.gather(inputs[0], dim=dim, index=inputs[1].to(torch.int64))
        return (res,)

    return gather_op_torch_thunk


def scatter_add_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.ScatterAddOp)
    dim = op.dim

    def scatter_add_op_torch_thunk(
        inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[torch.Tensor, ...]:
        res = op_src.scatter_add(inputs[0], dim=dim, index=inputs[1].to(torch.int64), src=inputs[2])
        return (res,)

    return scatter_add_op_torch_thunk


def split_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.SplitOp)
    dim = op.dim
    num_splits = op.num_splits

    split_size = emit_ctx.dg.get_input_shape(op, OpInId(0)).at(dim) // num_splits

    if isinstance(split_size, ie.ConstInt):
        split_size = split_size.const

    assert isinstance(split_size, int)

    def split_op_torch_thunk(
        inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[torch.Tensor, ...]:
        return torch.split(inputs[0], split_size, dim=dim)  # type: ignore

    return split_op_torch_thunk


def index_slice_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.IndexSliceOp)
    dim = op.dim
    length = op.length

    start_op, start_edge_data = emit_ctx.dg.get_flat_direct_dependencies(op)[1]
    can_optimize_narrow = (
        isinstance(start_op, top.ConstOp)
        and start_op.is_uniform
        and start_edge_data.is_unconditional_basis()
    )


    if op.is_static():
        assert isinstance(length, int)

        if can_optimize_narrow:
            value = start_op.uniform_value

            def index_slice_op_torch_thunk(
                inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
            ) -> Tuple[torch.Tensor, ...]:
                res = inputs[0].narrow(dim, value, length)
                return (res,)
        else:

            def index_slice_op_torch_thunk(
                inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
            ) -> Tuple[torch.Tensor, ...]:
                res = inputs[0].narrow(dim, inputs[1], length)
                return (res,)
    else:

        def get_fast_eval(x: Union[int, ie.IntIndexValue]) -> Callable[[], int]:
            if isinstance(x, ie.IntIndexValue):
                remapped = x.remap(emit_ctx.domain_map)
                remapped.cache_codegenerated_eval(emit_ctx.loop_counters)
                return remapped.eval_fast
            return lambda: x

        len_eval = get_fast_eval(length)

        def index_slice_op_torch_thunk(
            inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[torch.Tensor, ...]:
            x = inputs[0]
            start = int(inputs[1])
            return (x.narrow(dim, start, len_eval()),)

    return index_slice_op_torch_thunk


def index_select_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.IndexSelectOp)
    index_is_scalar = emit_ctx.dg.get_input_shape(op, OpInId(1)).is_scalar()

    dim = op.dim
    sq = (lambda x: x.squeeze(dim)) if index_is_scalar else (lambda x: x)

    def index_op_torch_thunk(
        inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[torch.Tensor, ...]:
        # index_max = inputs[1].max()
        # index_min = inputs[1].min()
        # print(f"index_max={index_max}, index_min={index_min}", flush=True)

        ##Print the index values if small
        # if inputs[1].numel() < 100:
        #    print(f"index values: {inputs[1]}", flush=True)

        res = op_src.index_select(inputs[0], dim, inputs[1].to(torch.int64))  # type: ignore
        return (sq(res),)  # type: ignore

    return index_op_torch_thunk


def index_add_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.IndexAddOp)

    def index_add_op_torch_thunk(
        inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[torch.Tensor, ...]:
        res = op_src.index_add(
            inputs[0], op.dim, inputs[1].to(torch.int64), inputs[2], alpha=op.alpha
        )
        return (res,)

    return index_add_op_torch_thunk


def merge_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.MergeOp)

    def merge_op_torch_thunk(
        inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
    ) -> Tuple[torch.Tensor, ...]:
        return (inputs[0],)

    return merge_op_torch_thunk


def val_to_val_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.ValToValOp)

    in_val = op.in_val
    out_val = op.out_val

    in_val_tensor = torch.tensor(in_val)
    out_val_tensor = torch.tensor(out_val)

    if torch.isnan(in_val_tensor):

        def val_to_val_op_torch_thunk(
            inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[torch.Tensor, ...]:
            return (torch.where(torch.isnan(inputs[0]), out_val_tensor, inputs[0]),)
    else:

        def val_to_val_op_torch_thunk(
            inputs: Tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[torch.Tensor, ...]:
            return (torch.where(inputs[0] == in_val_tensor, out_val_tensor, inputs[0]),)

    return val_to_val_op_torch_thunk


def pad_op_translation(
    op: top.TensorOp,
    emit_ctx: ThunkEmissionCtx[torch.Tensor],
) -> Thunk[torch.Tensor]:
    assert isinstance(op, top.PadOp)

    pad0 = op.padding[0]
    pad1 = op.padding[1]

    value = op.value if hasattr(op, "value") else 0.0
    len_in_shape = len(emit_ctx.dg.get_input_shape(op, OpInId(0)))
    mode_enum = op.mode

    mode_table = {
        top.PadMode.CONSTANT: "constant",
        top.PadMode.REFLECT: "reflect",
        top.PadMode.REPLICATE: "replicate",
        top.PadMode.ANY: "constant",
    }
    mode = mode_table[mode_enum]

    if op.is_static():
        if not isinstance(pad0, int):
            pad0 = pad0.try_eval({})  # type: ignore
        if not isinstance(pad1, int):
            pad1 = pad1.try_eval({})  # type: ignore

        padding_ = [
            0,
        ] * (len_in_shape * 2)
        padding_[len_in_shape * 2 - op.dim * 2 - 2] = pad0  # type: ignore
        padding_[len_in_shape * 2 - op.dim * 2 - 1] = pad1  # type: ignore
        padding_ = tuple(padding_)

        def pad_op_thunk(
            inputs: Tuple[torch.Tensor, ...],
            exec_ctx: ThunkExecutionCtx,
        ) -> Tuple[torch.Tensor, ...]:
            return (
                op_src_pad(
                    inputs[0],
                    pad=padding_,
                    mode=mode,
                    value=value,
                ),
            )
    else:
        padding_loc = [
            0,
        ] * (len_in_shape * 2)

        def get_fast_eval(
            pad0_: Union[int, ie.IntIndexValue], pad1_: Union[int, ie.IntIndexValue]
        ) -> Callable[[], Tuple[int, ...]]:
            if isinstance(pad0_, int):
                pad0_eval = lambda: pad0_
            elif isinstance(pad0_, ie.IntIndexValue):
                pad0__ = pad0_.remap(emit_ctx.domain_map)
                pad0__.cache_codegenerated_eval(emit_ctx.loop_counters)
                pad0_eval = pad0__.eval_fast
            if isinstance(pad1_, int):
                pad1_eval = lambda: pad1_
            elif isinstance(pad1_, ie.IntIndexValue):
                pad1__ = pad1_.remap(emit_ctx.domain_map)
                pad1__.cache_codegenerated_eval(emit_ctx.loop_counters)
                pad1_eval = pad1__.eval_fast

            idx0 = len_in_shape * 2 - op.dim * 2 - 2
            idx1 = len_in_shape * 2 - op.dim * 2 - 1

            def eval_fast() -> Tuple[int, ...]:
                padding_loc[idx0] = pad0_eval()  # type: ignore
                padding_loc[idx1] = pad1_eval()  # type: ignore
                return tuple(padding_loc)

            return eval_fast

        padding_eval = get_fast_eval(pad0, pad1)

        def pad_op_thunk(
            inputs: Tuple[torch.Tensor, ...],
            exec_ctx: ThunkExecutionCtx,
        ) -> Tuple[torch.Tensor, ...]:
            pad_evaled = padding_eval()
            return (
                op_src_pad(
                    inputs[0],
                    pad=pad_evaled,
                    mode=mode,
                    value=value,
                ),
            )

    return pad_op_thunk


TEMPO_TO_PYTORCH: Dict[Type[top.TensorOp], OpToThunkTranslationFn[torch.Tensor]] = {
    # Source
    top.RandOp: rand_op_translation,
    top.ConstOp: const_op_translation,
    top.EvalSymbolOp: eval_symbol_op_translation,
    # Control-flow
    top.MergeOp: merge_op_translation,
    # Movement
    top.FlipOp: flip_op_translation,
    top.CatOp: cat_op_translation,
    top.SqueezeOp: squeeze_op_translation,
    top.UnsqueezeOp: unsqueeze_op_translation,
    top.ExpandOp: expand_op_translation,
    top.ReshapeOp: reshape_op_translation,
    top.PermuteOp: permute_op_translation,
    # Elementwise
    ## Unary
    top.CastOp: cast_op_translation,
    top.SqrtOp: partial(_elementwise_unary_op_translation, torch_op=op_src.sqrt),
    top.NegOp: partial(_elementwise_unary_op_translation, torch_op=op_src.neg),
    top.NotOp: partial(_elementwise_unary_op_translation, torch_op=op_src.logical_not),
    top.LnOp: partial(_elementwise_unary_op_translation, torch_op=op_src.log),
    top.ExpOp: partial(_elementwise_unary_op_translation, torch_op=op_src.exp),
    top.SinOp: partial(_elementwise_unary_op_translation, torch_op=op_src.sin),
    top.IdentOp: lambda op, emit_ctx: lambda inputs, exec_ctx: inputs,
    ## Binary
    top.AddOp: partial(_elementwise_binary_op_translation, torch_op=op_src.add),
    top.SubOp: partial(_elementwise_binary_op_translation, torch_op=op_src.sub),
    top.MulOp: partial(_elementwise_binary_op_translation, torch_op=op_src.mul),
    top.DivOp: partial(_elementwise_binary_op_translation, torch_op=op_src.div),
    top.PowOp: partial(_elementwise_binary_op_translation, torch_op=op_src.pow),
    top.OrOp: partial(_elementwise_binary_op_translation, torch_op=op_src.logical_or),
    top.AndOp: partial(_elementwise_binary_op_translation, torch_op=op_src.logical_and),
    top.EqualOp: partial(_elementwise_binary_op_translation, torch_op=op_src.eq),
    top.LessThanOp: partial(_elementwise_binary_op_translation, torch_op=op_src.less),
    ### Ternary
    top.WhereOp: where_op_translation,
    # Reduction
    top.SumOp: sum_op_translation,
    top.MaxOp: max_op_translation,
    # Dimmed Operations
    top.CumSumOp: cumsum_op_translation,
    top.GatherOp: gather_op_translation,
    top.ScatterAddOp: scatter_add_op_translation,
    # These have indexes
    top.IndexSelectOp: index_select_op_translation,
    top.SplitOp: split_op_translation,
    top.IndexSliceOp: index_slice_op_translation,
    top.IndexAddOp: index_add_op_translation,
    top.MatMulOp: matmul_op_translation,
    top.ConvOp: conv_op_translation,
    # Special
    top.PadOp: pad_op_translation,
    top.ValToValOp: val_to_val_op_translation,
}


class PytorchThunkEmitter(ThunkEmitter[torch.Tensor]):
    def _emit_thunk_for_op(self, op: top.TensorOp, ctx: ThunkEmissionCtx) -> Thunk[torch.Tensor]:
        return TEMPO_TO_PYTORCH[type(op)](op, ctx)
