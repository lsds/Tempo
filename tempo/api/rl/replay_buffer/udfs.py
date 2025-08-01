import dataclasses
import math
from typing import Sequence, Tuple

from tempo.api.rl.replay_buffer.replay_buffer_desc import ReplayBufferDesc
from tempo.api.rl.replay_buffer.runtime_replay_buffer_interface import (
    RuntimeReplayBufferInterface,
)
from tempo.core import tensor_op as top
from tempo.core.datatypes import BackendTensorT
from tempo.core.dtype import dtypes
from tempo.core.shape import Shape
from tempo.core.tensor_ops import UserDefinedThunkDesc
from tempo.core.thunk import (
    Thunk,
    ThunkEmissionCtx,
    ThunkExecutionCtx,
    UDFVectorizationCtx,
)
from tempo.runtime.backends.backend import DLBackend


def _build_and_or_get_replay_memory(
    desc: ReplayBufferDesc, ctx: ThunkEmissionCtx[BackendTensorT]
) -> RuntimeReplayBufferInterface:
    if desc.state_id not in ctx.external_state_store:
        ctx.external_state_store[desc.state_id] = desc.builder(desc.ctx)

    return ctx.external_state_store[desc.state_id]  # type: ignore


def vectorize_insert(
    replay_desc: ReplayBufferDesc, vec_ctx: UDFVectorizationCtx
) -> UserDefinedThunkDesc:
    b_size = vec_ctx.vec_size * math.prod(vec_ctx.prior_vectorizations)
    assert isinstance(b_size, int), "b_size must be an integer"

    output_shape = (Shape((b_size,)),)

    def new_infer_output_shapes(input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        return output_shape

    def insert_replay_op_translation(
        op: top.TensorOp, ctx: ThunkEmissionCtx[BackendTensorT]
    ) -> Thunk[BackendTensorT]:
        replay_memory = _build_and_or_get_replay_memory(replay_desc, ctx)

        # TODO fix this: should be given a backend, not create one
        backend = DLBackend.get_backend(ctx.exec_cfg.backend)
        backend.configure(ctx.exec_cfg)
        dev = backend.to_backend_device_obj(ctx.exec_cfg.dev)

        batched_shapes = tuple(
            (b_size,) + s.as_static()._shape for s in replay_desc.ctx.item_shapes
        )
        bend_bool = backend.to_backend_datatype(dtypes.bool_)

        def insert(
            inputs: Tuple[BackendTensorT, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[BackendTensorT, ...]:
            # TODO avoid using reshape here.
            # JAX does not support reshape without reallocation.
            # https://github.com/jax-ml/jax/issues/11036
            # TODO: instead, just convert stuff to numpy,
            # do the reshape and then convert back to tensor
            data = [
                backend.to_cpu(t)
                # backend.to_cpu(backend.reshape(t, shape))
                for t, shape in zip(inputs, batched_shapes, strict=False)
            ]

            replay_memory.insert_batched(data)
            return (backend.ones_tensor((b_size,), dtype=bend_bool, dev=dev),)

        return insert

    return dataclasses.replace(
        vec_ctx.current_udf_desc,
        infer_output_shapes=new_infer_output_shapes,
        thunk_translation=insert_replay_op_translation,
    )


def get_insert_udf_desc(replay_memory_desc: ReplayBufferDesc) -> UserDefinedThunkDesc:
    def insert_replay_translation(
        op: top.TensorOp, ctx: ThunkEmissionCtx[BackendTensorT]
    ) -> Thunk[BackendTensorT]:
        replay_memory = _build_and_or_get_replay_memory(replay_memory_desc, ctx)

        # TODO fix this: should be given a backend, not create one
        backend = DLBackend.get_backend(ctx.exec_cfg.backend)
        backend.configure(ctx.exec_cfg)
        dev = backend.to_backend_device_obj(ctx.exec_cfg.dev)

        bend_bool = backend.to_backend_datatype(dtypes.bool_)
        # num_inputs_expected = len(replay_memory_desc.item_shapes)
        # on_dev_true = backend.tensor(True, device=dev)

        def insert(
            inputs: Tuple[BackendTensorT, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[BackendTensorT, ...]:
            data = [backend.to_cpu(i) for i in inputs]
            replay_memory.insert(data)
            return (backend.ones_tensor((), dtype=bend_bool, dev=dev),)

        return insert

    return UserDefinedThunkDesc(
        thunk_translation=insert_replay_translation,
        infer_output_shapes=lambda input_shapes: (Shape.from_(()),),
        infer_output_dtypes=lambda input_dtypes: (dtypes.bool_,),
        num_inputs=len(replay_memory_desc.ctx.item_shapes),
        num_outputs=1,
        state_store_access_keys=(replay_memory_desc.state_id,),
        require_simultaneous_vectorization=False,
        vectorize=lambda ctx: vectorize_insert(replay_memory_desc, ctx),
        thunk_name="ReplayInsert",
        clean_up_state={
            replay_memory_desc.state_id: lambda state: state[replay_memory_desc.state_id].clear()
        },
    )


def vectorize_sample(
    replay_desc: ReplayBufferDesc, vec_ctx: UDFVectorizationCtx
) -> UserDefinedThunkDesc:
    # vec_ctx.vec_size is the size of B, I or T, etc.
    # need to use token to show not want to vectorize
    b_size = vec_ctx.vec_size * math.prod(vec_ctx.prior_vectorizations)
    assert isinstance(b_size, int), "b_size must be an integer"

    item_shapes = tuple(replay_desc.ctx.item_shapes)
    # sampled_shapes = tuple(
    #    [Shape((b_size,) + item_shape._shape) for item_shape in item_shapes]
    # )
    output_shapes = tuple(
        [
            Shape((vec_ctx.vec_size,) + vec_ctx.prior_vectorizations + item_shape._shape)
            for item_shape in item_shapes
        ]
    )
    output_shapes_static = tuple([s.as_static()._shape for s in output_shapes])

    def new_infer_output_shapes(input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        return output_shapes

    def sample_replay_op_torch_translation(
        op: top.TensorOp, ctx: ThunkEmissionCtx[BackendTensorT]
    ) -> Thunk[BackendTensorT]:
        replay_memory = _build_and_or_get_replay_memory(replay_desc, ctx)

        # TODO fix this: should be given a backend, not create one
        backend = DLBackend.get_backend(ctx.exec_cfg.backend)
        backend.configure(ctx.exec_cfg)
        device = backend.to_backend_device_obj(ctx.exec_cfg.dev)

        def sample(
            inputs: Tuple[BackendTensorT, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[BackendTensorT, ...]:
            items = replay_memory.sample_batched(b_size)
            on_dev_items = [
                backend.to_device(backend.from_dlpack(item), dev=device) for item in items
            ]

            # TODO if output_shapes == sampled_shapes, we can skip this.
            # TODO: instead, just convert stuff to numpy,
            # do the reshape and then convert back to tensor
            reshaped_items = [
                backend.reshape(item, shape)
                for item, shape in zip(on_dev_items, output_shapes_static, strict=False)
            ]

            return reshaped_items  # type: ignore

        return sample

    return dataclasses.replace(
        vec_ctx.current_udf_desc,
        infer_output_shapes=new_infer_output_shapes,
        thunk_translation=sample_replay_op_torch_translation,
    )


def get_sample_udf_desc(replay_memory_desc: ReplayBufferDesc) -> UserDefinedThunkDesc:
    # NOTE: implementations must support sampling with replacement.
    def sample_replay_op_torch_translation(
        op: top.TensorOp, ctx: ThunkEmissionCtx[BackendTensorT]
    ) -> Thunk[BackendTensorT]:
        replay_memory = _build_and_or_get_replay_memory(replay_memory_desc, ctx)

        # TODO fix this: should be given a backend, not create one
        backend = DLBackend.get_backend(ctx.exec_cfg.backend)
        backend.configure(ctx.exec_cfg)
        device = backend.to_backend_device_obj(ctx.exec_cfg.dev)

        def sample(
            inputs: Tuple[BackendTensorT, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[BackendTensorT, ...]:
            items = replay_memory.sample()
            return tuple(backend.lift_tensor(item, device=device) for item in items)

        return sample

    # outputs batchsize * number of item per sample
    num_item = len(replay_memory_desc.ctx.item_shapes)

    output_shapes = tuple(replay_memory_desc.ctx.item_shapes)
    output_dtypes = tuple(replay_memory_desc.ctx.item_dtypes)

    return UserDefinedThunkDesc(
        thunk_translation=sample_replay_op_torch_translation,
        infer_output_shapes=lambda input_shapes: output_shapes,
        infer_output_dtypes=lambda input_dtypes: output_dtypes,
        num_inputs=1,
        num_outputs=num_item,
        state_store_access_keys=(replay_memory_desc.state_id,),
        require_simultaneous_vectorization=False,
        vectorize=lambda ctx: vectorize_sample(replay_memory_desc, ctx),
        thunk_name="ReplaySample",
        clean_up_state={
            replay_memory_desc.state_id: lambda state: state[replay_memory_desc.state_id].clear()
        },
    )


# TODO: clean up functions. RuntimeReplayBufferInterface should have a cleanup function
# TODO executor reset should clean up all external states.
