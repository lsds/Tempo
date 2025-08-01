import dataclasses
import math
from typing import Callable, Sequence, Tuple

from tempo.api.data.dataloader_desc import DataLoaderDesc
from tempo.api.data.runtime_dataloader import RuntimeDataLoader, get_runtime_dataloader
from tempo.core import tensor_op as top
from tempo.core.datatypes import BackendTensorT
from tempo.core.shape import Shape
from tempo.core.tensor_ops import UserDefinedThunkDesc
from tempo.core.thunk import Thunk, ThunkEmissionCtx, ThunkExecutionCtx, UDFVectorizationCtx


def get_next_batch_translation_for_desc(
    desc: DataLoaderDesc,
) -> Callable[[top.TensorOp, ThunkEmissionCtx[BackendTensorT]], Thunk[BackendTensorT]]:
    def next_batch_translation(
        op: top.TensorOp, ctx: ThunkEmissionCtx[BackendTensorT]
    ) -> Thunk[BackendTensorT]:
        if desc.state_id not in ctx.external_state_store:
            ctx.external_state_store[desc.state_id] = get_runtime_dataloader(desc, ctx.exec_cfg)

        dataloader: RuntimeDataLoader[BackendTensorT] = ctx.external_state_store[desc.state_id]  # type: ignore

        def next_batch(
            inputs: Tuple[BackendTensorT, ...], exec_ctx: ThunkExecutionCtx
        ) -> Tuple[BackendTensorT, ...]:
            return dataloader.next_batch()  # type: ignore

        return next_batch

    return next_batch_translation


def vectorize_next_batch(
    desc: DataLoaderDesc, vec_ctx: UDFVectorizationCtx
) -> UserDefinedThunkDesc:
    """Create a vectorized version of the next_batch UDF."""
    batch_size = vec_ctx.vec_size * math.prod(vec_ctx.prior_vectorizations)
    assert isinstance(batch_size, int), "batch_size must be an integer"

    # Create new desc with batch size
    vectorized_desc = dataclasses.replace(desc, batch_size=batch_size)

    # Update output shapes for batched outputs
    updated_output_shapes = tuple(
        Shape((batch_size,) + shape._shape) for shape in desc.dataset_info.sample_shapes
    )

    def new_infer_output_shapes(input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        return updated_output_shapes

    return dataclasses.replace(
        vec_ctx.current_udf_desc,
        infer_output_shapes=new_infer_output_shapes,
        thunk_translation=get_next_batch_translation_for_desc(vectorized_desc),
    )


def get_next_batch_udf_desc(desc: DataLoaderDesc) -> UserDefinedThunkDesc:
    return UserDefinedThunkDesc(
        thunk_translation=get_next_batch_translation_for_desc(desc),
        infer_output_shapes=lambda input_shapes: tuple(
            shape if desc.batch_size is None else Shape((desc.batch_size,) + shape._shape)
            for shape in desc.dataset_info.sample_shapes
        ),
        infer_output_dtypes=lambda input_dtypes: tuple(desc.dataset_info.sample_dtypes),
        num_inputs=0,
        num_outputs=desc.dataset_info.num_outputs,
        state_store_access_keys=(desc.state_id,),
        require_simultaneous_vectorization=False,
        vectorize=lambda ctx: vectorize_next_batch(desc, ctx),
        thunk_name="DataLoaderNextBatch",
        clean_up_state={desc.state_id: lambda state: state[desc.state_id].close()},
    )
