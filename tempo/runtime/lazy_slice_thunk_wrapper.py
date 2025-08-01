import typing
from typing import Callable, Dict, List, Sequence, Tuple, Union

from tempo.core import tensor_ops as top
from tempo.core.analysis_ctx import AnalysisCtx
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import BackendTensorT, OpInId, OpOutId, TensorId
from tempo.core.dependence_graph import PDG
from tempo.core.dtype import DataType
from tempo.core.shape import Shape
from tempo.core.storage_methods import CircularBufferStore
from tempo.core.tensor_op import TensorOp
from tempo.utils import logger
from tempo.utils.dg_utils import is_window_access

log = logger.get_logger(__name__)

"""JAX will copy on any slice operation even if it is a read-only view. This can lead to great
overhead. To avoid this, we wrap thunks in "lazy slicing" wrappers which receive the base backing
buffer, the start index to slice and a static slice size, and perform the slice within JIT.

This is most import for windows, but can also be important for other prealloc types.
For now, this works through collaboration with the tensor stores, which instead of performing the
slice themselves, return a tuple (buffer, start, length) if the backend is JAX
"""


def has_lazy_sliced_outputs(
    parent_graph: PDG, graph_op: top.ExecDataflowOp, analysis_ctx: AnalysisCtx
) -> bool:
    res = find_lazy_sliced_input_tensors(parent_graph, graph_op, analysis_ctx)
    return len(res) > 0


def find_lazy_sliced_input_tensors(
    parent_graph: PDG, graph_op: TensorOp, analysis_ctx: AnalysisCtx
) -> Tuple[
    Tuple[OpInId, ...],
    Tuple[TensorId, ...],
    Tuple[Shape, ...],
    Tuple[DataType, ...],
    Tuple[int, ...],
]:
    lazy_ins = []
    lazy_tensor_ids = []
    lazy_shapes = []
    lazy_dtypes = []
    lazy_static_lengths = []

    for depy_op, depy_data in parent_graph.get_flat_direct_dependencies(graph_op):
        in_tid = TensorId(depy_op.op_id, depy_data.src_out_idx)
        store_info = analysis_ctx.tensor_storage_classes[in_tid]
        is_window_slice = any(is_window_access(e) for e in depy_data.expr.members)
        if isinstance(store_info, CircularBufferStore) and is_window_slice:
            dtype_ = parent_graph.get_output_dtypes(depy_op)[depy_data.src_out_idx]
            shape_ = parent_graph.get_output_shapes(depy_op)[depy_data.src_out_idx]
            shape_ = shape_.prepend_dim(store_info.buffer_size)
            lazy_ins.append(depy_data.sink_in_idx)
            lazy_tensor_ids.append(in_tid)
            lazy_shapes.append(shape_)
            lazy_dtypes.append(dtype_)
            lazy_static_lengths.append(store_info.window_size)

    return (
        tuple(lazy_ins),
        tuple(lazy_tensor_ids),
        tuple(lazy_shapes),
        tuple(lazy_dtypes),
        tuple(lazy_static_lengths),
    )


def find_window_stored_buffer_index_shapes_and_types(
    tensor_ids: Tuple[TensorId, ...],
    dataflow_graph: PDG,
    graph_op: top.ExecDataflowOp,
    analysis_ctx: AnalysisCtx,
    output_types: Dict[OpOutId, DataType],
) -> Tuple[Tuple[Shape, int, DataType], ...]:
    buffer_index_shapes_and_data_type: list[Tuple[Shape, int, DataType]] = []

    all_output_shapes = dataflow_graph.get_output_shapes_list(graph_op)

    for tensor_id in tensor_ids:
        # cast into BlockStore, as we already know tensor_ids are all block stored tensors' ids
        if not isinstance(analysis_ctx.tensor_storage_classes[tensor_id], CircularBufferStore):
            raise TypeError("Tensor is not buffer stored")

        storage_info = typing.cast(
            CircularBufferStore, analysis_ctx.tensor_storage_classes[tensor_id]
        )

        # get the spatial shape of the block stored tensor
        spatial_shape = all_output_shapes[int(tensor_id.output_id)]

        w_size = storage_info.window_size

        buffer_shape = Shape((storage_info.buffer_size, *spatial_shape._shape))
        buffer_index_shapes_and_data_type.append(
            (buffer_shape, w_size, output_types[tensor_id.output_id])
        )
    return tuple(buffer_index_shapes_and_data_type)


def make_lazy_slice_thunk_wrapper(
    exec_cfg: ExecutionConfig,
    analysis_ctx: AnalysisCtx,
    parent_graph: PDG,
    df_op: top.TensorOp,
    output_dtypes: Sequence[DataType],
    interp_exec_func: Callable[[Tuple[BackendTensorT, ...]], Tuple[BackendTensorT, ...]],
    example_inputs: Tuple[BackendTensorT, ...],
    donatable_args: Tuple[int, ...],
) -> Tuple[
    Callable[[Tuple[BackendTensorT, ...]], Tuple[BackendTensorT, ...]],
    Tuple[BackendTensorT, ...],
    Tuple[int, ...],
]:
    from tempo.runtime.backends.backend import DLBackend, DLBackendName

    backend = DLBackend.get_backend(exec_cfg.backend)
    dev = backend.to_backend_device_obj(exec_cfg.dev)

    if backend.get_backend_name() != DLBackendName.JAX:
        return interp_exec_func, example_inputs, donatable_args

    if not isinstance(df_op, top.ExecDataflowOp):
        return interp_exec_func, example_inputs, donatable_args

    lazy_idxs, lazy_tids, shapes_, dtypes_, static_lengths = find_lazy_sliced_input_tensors(
        parent_graph, df_op, analysis_ctx
    )

    if len(lazy_idxs) == 0:
        return interp_exec_func, example_inputs, donatable_args

    # NOTE: Now, alter example_inputs so that the lazy_idxs are replaced with tuple(buf, start)

    new_donatable_args = list(donatable_args)
    for i in lazy_idxs:
        if i in new_donatable_args:
            new_donatable_args.remove(i)
    new_donatable_args = tuple(new_donatable_args)

    if analysis_ctx._donatable_args is not None:
        # NOTE: We need to remove the lazy_idxs from the donatable args, as they are not donatable
        num_orig_args = len(analysis_ctx.donatable_args[df_op.op_id])
        analysis_ctx.donatable_args[df_op.op_id] = new_donatable_args[:num_orig_args]

    example_inputs_new: List[Union[Tuple[BackendTensorT, int], BackendTensorT]] = list(
        example_inputs
    )
    for i in range(len(lazy_idxs)):
        store_info = analysis_ctx.tensor_storage_classes[lazy_tids[i]]
        assert isinstance(store_info, CircularBufferStore)
        backend_dtype = backend.to_backend_datatype(dtypes_[i])
        example_buf = backend.zeros_tensor(shapes_[i].as_static()._shape, backend_dtype, dev)
        example_idx = 0  # backend.zeros_tensor(Shape(()), dtypes_[i], dev)
        example_inputs_new[int(lazy_idxs[i])] = (example_buf, example_idx)

    import jax

    example_inputs_new = tuple(example_inputs_new)

    def _wrapper(inputs: Tuple[BackendTensorT, ...]) -> Tuple[BackendTensorT, ...]:
        inputs_ = list(inputs)
        for i in range(len(lazy_idxs)):
            (buf, start_idx) = inputs_[lazy_idxs[i]]  # type: ignore
            static_length = static_lengths[i]
            inp_ = jax.lax.dynamic_slice_in_dim(buf, start_idx, static_length, axis=0)  # type: ignore
            inputs_[lazy_idxs[i]] = inp_

        outs = interp_exec_func(tuple(inputs_))

        return outs

    log.info(
        "Lazy slice thunk wrapper created for Op %s, with %d lazy-sliced inputs",
        df_op.op_id,
        len(lazy_idxs),
    )

    return _wrapper, example_inputs_new, new_donatable_args  # type: ignore
