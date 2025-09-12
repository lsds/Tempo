import typing
from collections.abc import Callable, Sequence

from tempo.core import tensor_ops as top
from tempo.core.analysis_ctx import AnalysisCtx
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import BackendTensorT, OpOutId, TensorId
from tempo.core.dependence_graph import PDG
from tempo.core.device import DeviceGroup
from tempo.core.dl_backends import DLBackendName
from tempo.core.dtype import DataType, dtypes
from tempo.core.shape import Shape
from tempo.core.storage_methods import PreallocBufferStore
from tempo.core.tensor_op import TensorOp


def has_buffer_stored_outputs(
    parent_graph: PDG, graph_op: TensorOp, analysis_ctx: AnalysisCtx
) -> bool:
    res = find_buffer_stored_output_tensors(parent_graph, graph_op, analysis_ctx)
    return len(res) > 0


def find_buffer_stored_output_tensors(
    parent_graph: PDG, graph_op: TensorOp, analysis_ctx: AnalysisCtx
) -> tuple[TensorId, ...]:
    # if analysis_ctx._buffer_stored_output_tensor_positions is None:
    #    return ()

    all_output_shapes = parent_graph.get_output_shapes_list(graph_op)
    block_outs = []

    for i in range(len(all_output_shapes)):
        op_out_id: OpOutId = OpOutId(i)
        tensor_id = TensorId(graph_op.op_id, op_out_id)
        if isinstance(analysis_ctx.tensor_storage_classes[tensor_id], PreallocBufferStore):
            block_outs.append(tensor_id)
    return tuple(block_outs)


# finds out the buffer and index shapes, and the output data types for block stored tensors
def find_buffer_stored_buffer_index_shapes_and_types(
    tensor_ids: tuple[TensorId, ...],
    dataflow_graph: PDG,
    graph_op: TensorOp,
    analysis_ctx: AnalysisCtx,
    output_types: Sequence[DataType],
) -> tuple[tuple[Shape, int, DataType], ...]:
    buffer_index_shapes_and_data_type: list[tuple[Shape, int, DataType]] = []

    all_output_shapes = dataflow_graph.get_output_shapes_list(graph_op)

    for tensor_id in tensor_ids:
        # cast into BlockStore, as we already know tensor_ids are all block stored tensors' ids
        if not isinstance(analysis_ctx.tensor_storage_classes[tensor_id], PreallocBufferStore):
            raise TypeError("Tensor is not buffer stored")

        storage_info = typing.cast(
            PreallocBufferStore, analysis_ctx.tensor_storage_classes[tensor_id]
        )

        # get the spatial shape of the block stored tensor
        spatial_shape = all_output_shapes[int(tensor_id.output_id)]

        buffer_shape = Shape((storage_info.buffer_size, *spatial_shape._shape))
        buffer_index_shapes_and_data_type.append(
            (buffer_shape, storage_info.num_buffer_writes_needed, output_types[tensor_id.output_id])
        )
    return tuple(buffer_index_shapes_and_data_type)


def make_inplace_write_wrapper(
    exec_cfg: ExecutionConfig,
    analysis_ctx: AnalysisCtx,
    parent_graph: PDG,
    self_op: top.TensorOp,
    output_dtypes: Sequence[DataType],
    interp_exec_func: Callable[[tuple[BackendTensorT, ...]], tuple[BackendTensorT, ...]],
    example_inputs: tuple[BackendTensorT, ...],
    donatable_args: tuple[int, ...],
) -> tuple[
    Callable[[tuple[BackendTensorT, ...]], tuple[BackendTensorT, ...]],
    tuple[BackendTensorT, ...],
    tuple[int, ...],
]:
    """
    Handles inplace write logic for buffer-stored output tensors.
    """
    from tempo.core.dl_backend import DLBackend

    backend: type[DLBackend] = DLBackend.get_backend(exec_cfg.backend)

    buffer_stored_tensor_ids = find_buffer_stored_output_tensors(
        parent_graph=parent_graph, graph_op=self_op, analysis_ctx=analysis_ctx
    )
    buffer_stored_output_tensor_positions = tuple(
        block_stored_tensor_id.output_id for block_stored_tensor_id in buffer_stored_tensor_ids
    )
    buffer_index_shapes_and_data_type = find_buffer_stored_buffer_index_shapes_and_types(
        buffer_stored_tensor_ids, parent_graph, self_op, analysis_ctx, output_dtypes
    )
    comp_ctx = CompilationCtx(parent_graph, analysis_ctx, exec_cfg)
    output_devs = [
        comp_ctx.get_tensor_device(buffer_stored_tensor_id)  # .backing_device
        for buffer_stored_tensor_id in buffer_stored_tensor_ids
    ]
    if len(output_devs) != len(buffer_index_shapes_and_data_type):
        raise ValueError("Block stored output devices do not match with the number of buffers")

    dummy_buffers: tuple[BackendTensorT, ...]
    dummy_buffers, dummy_buffer_indices = _build_dummy_inputs(
        exec_cfg, buffer_index_shapes_and_data_type, output_devs
    )
    number_of_original_inputs = len(example_inputs)
    example_inputs = example_inputs + dummy_buffers + dummy_buffer_indices
    first_buf = number_of_original_inputs
    buf_args = range(first_buf, first_buf + len(dummy_buffers))
    donatable_args = donatable_args + tuple(buf_args)
    all_output_shapes = parent_graph.get_output_shapes_list(self_op)
    inplace_set_fns: list[
        Callable[
            [BackendTensorT, Sequence[int | slice], BackendTensorT],
            BackendTensorT,
        ]
    ] = []
    num_idxs = tuple(num for _, num, _ in buffer_index_shapes_and_data_type)

    for slot, pos in enumerate(buffer_stored_output_tensor_positions):
        dummy_output_val = backend.zeros_tensor(
            backend.to_backend_shape(all_output_shapes[pos].as_static()._shape),
            backend.to_backend_datatype(output_dtypes[pos]),
            dev=backend.to_backend_device_obj(output_devs[slot]),
        )
        inplace_set_fns.append(
            backend.get_inplace_set_fn(
                backend.copy(dummy_buffers[slot]),
                dummy_buffer_indices[slot][0],  # Zero because an inplace-set fn sets only one index
                backend.copy(dummy_output_val),
                traceable=True,
            )
        )
    interp_exec_func = _make_inplace_buffer_write_thunk_wrapper(
        interp_exec_func,
        number_of_original_inputs,
        buffer_stored_output_tensor_positions,
        tuple(inplace_set_fns),
        num_idxs,
    )

    if analysis_ctx._buffer_stored_output_tensor_positions is None:
        analysis_ctx._buffer_stored_output_tensor_positions = {}
    analysis_ctx._buffer_stored_output_tensor_positions[self_op.op_id] = (
        buffer_stored_output_tensor_positions
    )
    return interp_exec_func, example_inputs, tuple(donatable_args)  # type: ignore


def _build_dummy_inputs(
    exec_cfg: ExecutionConfig,
    buffer_index_shapes_and_data_type: tuple[tuple[Shape, int, DataType], ...],
    output_devs: list[DeviceGroup],
) -> tuple[tuple[BackendTensorT, ...], tuple[tuple[tuple[int, ...], ...], ...]]:
    from tempo.core.dl_backend import DLBackend

    backend: type[DLBackend] = DLBackend.get_backend(exec_cfg.backend)

    dummy_buffers: list[BackendTensorT] = []
    dummy_buffer_indices: list[tuple[int, ...]] = []
    for (buf_shape, num_indexes, data_type), output_dev in zip(
        buffer_index_shapes_and_data_type, output_devs, strict=True
    ):
        dummy_buffers.append(
            backend.zeros_tensor(
                backend.to_backend_shape(buf_shape.as_static()),
                backend.to_backend_datatype(data_type),
                dev=backend.to_backend_device_obj(output_dev),
            ),
        )

        indexes = []
        for _ in range(num_indexes):
            if backend.get_backend_name() == DLBackendName.TORCH:
                indexes.append(
                    (
                        backend.fast_int_lift(
                            0,
                            dtype=backend.to_backend_datatype(dtypes.int64),
                            device=backend.to_backend_device_obj(exec_cfg.dev),
                        ),
                    )
                )
            else:
                indexes.append((0,))
        dummy_buffer_indices.append(tuple(indexes))  # type: ignore
    return tuple(dummy_buffers), tuple(dummy_buffer_indices)  # type: ignore


def _make_inplace_buffer_write_thunk_wrapper(
    orig_fn: Callable[[tuple[BackendTensorT, ...]], tuple[BackendTensorT, ...]],
    n_input: int,
    output_positions: tuple[int, ...],
    # get_write_idxs_fn: Tuple[Callable[[BackendTensorT], Tuple[Tuple[int, ...], ...]], ...],
    inplace_set_fns: tuple[
        Callable[[BackendTensorT, Sequence[int | slice], BackendTensorT], BackendTensorT],
        ...,
    ],
    num_idxs: tuple[int, ...],
) -> Callable[[tuple[BackendTensorT, ...]], tuple[BackendTensorT, ...]]:
    n_buffers = len(output_positions)

    def _wrapper(example_inputs: tuple[BackendTensorT, ...]) -> tuple[BackendTensorT, ...]:
        inputs = example_inputs[:n_input]
        buffer_tensors = example_inputs[n_input : n_input + n_buffers]
        buffer_index_tensors = example_inputs[n_input + n_buffers : n_input + n_buffers + n_buffers]
        outputs = orig_fn(inputs)
        updated_buffers = []
        for slot in range(n_buffers):
            buf = buffer_tensors[slot]
            idxs = buffer_index_tensors[slot]
            value = outputs[output_positions[slot]]
            newbuf = buf
            for i in range(num_idxs[slot]):  # type: ignore
                idx = idxs[i]
                newbuf = inplace_set_fns[slot](newbuf, idx, value)  # type: ignore[arg-type]
            updated_buffers.append(newbuf)
        final_out = tuple(
            updated_buffers[output_positions.index(i)] if i in output_positions else outputs[i]
            for i in range(len(outputs))
        )
        return final_out

    return _wrapper
