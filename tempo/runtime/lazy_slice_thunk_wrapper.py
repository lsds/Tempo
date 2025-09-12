from collections.abc import Callable, Sequence

from tempo.core import tensor_ops as top
from tempo.core.analysis_ctx import AnalysisCtx
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import BackendTensorT, OpInId, TensorId
from tempo.core.dependence_graph import PDG
from tempo.core.dl_backend import DLBackend
from tempo.core.dl_backends import DLBackendName
from tempo.core.dtype import DataType
from tempo.core.shape import Shape
from tempo.core.storage_methods import PreallocBufferStore
from tempo.core.tensor_op import TensorOp
from tempo.utils import isl as isl_utils
from tempo.utils import logger

log = logger.get_logger(__name__)

"""JAX will copy on any slice operation even if it is a read-only view. This can lead to great
overhead. To avoid this, we wrap thunks in "lazy slicing" wrappers which receive the base backing
buffer, the start index to slice and a static slice size, and perform the slice within JIT.

This is most import for windows, but can also be important for other prealloc types.
For now, this works through collaboration with the tensor stores, which instead of performing the
slice themselves, return a tuple (buffer, start, length) if the backend is JAX
"""


# TODO: for all methods, just pass in compilation ctx
def has_lazy_sliced_outputs(
    dg: PDG, graph_op: top.ExecDataflowOp, analysis_ctx: AnalysisCtx
) -> bool:
    for depy_op, depy_data in dg.get_flat_direct_dependencies(graph_op):
        tid = TensorId(depy_op.op_id, depy_data.src_out_idx)
        store_info = analysis_ctx.tensor_storage_classes[tid]

        if isinstance(store_info, PreallocBufferStore):
            dim = store_info.temporal_dim_stored
            prealloc_dim_access_expr = depy_data.expr.members[
                depy_op.domain.find_variable_index(dim)
            ]
            is_point_access = prealloc_dim_access_expr.is_point()
            access_length = 1
            if not is_point_access:
                access_length = int(
                    isl_utils.simplify_int_index_value(
                        prealloc_dim_access_expr.evaluate_shape(dg.static_bounds)[0],
                        known_symbols=dg.static_bounds,
                        ctx=analysis_ctx.isl_ctx,
                    )
                )

            if access_length != store_info.buffer_size:
                return True
    return False


def find_lazy_sliced_input_tensors(
    dg: PDG, graph_op: TensorOp, analysis_ctx: AnalysisCtx
) -> tuple[
    tuple[OpInId, ...],
    tuple[TensorId, ...],
    tuple[Shape, ...],
    tuple[DataType, ...],
    tuple[int, ...],
    tuple[bool, ...],
]:
    lazy_ins = []
    lazy_tensor_ids = []
    buffer_shapes = []
    buffer_dtypes = []
    access_lengths = []
    lazy_squeezes = []
    for depy_op, depy_data in dg.get_flat_direct_dependencies(graph_op):
        tid = TensorId(depy_op.op_id, depy_data.src_out_idx)
        store_info = analysis_ctx.tensor_storage_classes[tid]

        if isinstance(store_info, PreallocBufferStore):
            dim = store_info.temporal_dim_stored
            access_expr = depy_data.expr.members[depy_op.domain.find_variable_index(dim)]
            is_point_access = access_expr.is_point()

            access_length = 1
            if not is_point_access:
                access_length = int(
                    isl_utils.simplify_int_index_value(
                        access_expr.evaluate_shape(dg.static_bounds)[0],
                        known_symbols=dg.static_bounds,
                        ctx=analysis_ctx.isl_ctx,
                    )
                )

            if access_length != store_info.buffer_size:
                dtype_ = dg.get_output_dtypes(depy_op)[depy_data.src_out_idx]
                shape_ = dg.get_output_shapes(depy_op)[depy_data.src_out_idx]
                shape_ = shape_.prepend_dim(store_info.buffer_size)
                lazy_ins.append(depy_data.sink_in_idx)
                lazy_tensor_ids.append(tid)
                buffer_shapes.append(shape_)
                buffer_dtypes.append(dtype_)
                access_lengths.append(access_length)

                # NOTE: When we have a point access, we need to squeeze the result of the
                # lazy slice.
                lazy_squeezes.append(is_point_access)

    return (
        tuple(lazy_ins),
        tuple(lazy_tensor_ids),
        tuple(buffer_shapes),
        tuple(buffer_dtypes),
        tuple(access_lengths),
        tuple(lazy_squeezes),
    )


def make_lazy_slice_thunk_wrapper(
    exec_cfg: ExecutionConfig,
    analysis_ctx: AnalysisCtx,
    parent_graph: PDG,
    df_op: top.TensorOp,
    output_dtypes: Sequence[DataType],
    interp_exec_func: Callable[[tuple[BackendTensorT, ...]], tuple[BackendTensorT, ...]],
    example_inputs: tuple[BackendTensorT, ...],
    donatable_args: tuple[int, ...],
) -> tuple[
    Callable[[tuple[BackendTensorT, ...]], tuple[BackendTensorT, ...]],
    tuple[BackendTensorT, ...],
    tuple[int, ...],
]:
    print(f"Making lazy slice thunk wrapper for {df_op.op_id}", flush=True)

    backend = DLBackend.get_backend(exec_cfg.backend)
    dev = backend.to_backend_device_obj(exec_cfg.dev)

    # TODO: To generalize beyond JAX, we need backend to have squeeze and dynamic slice.
    # TODO: I think dynamic slice is already supported...
    if backend.get_backend_name() != DLBackendName.JAX:
        return interp_exec_func, example_inputs, donatable_args

    if not isinstance(df_op, top.ExecDataflowOp):
        return interp_exec_func, example_inputs, donatable_args

    lazy_idxs, lazy_tids, buffer_shapes, buffer_dtypes, access_lengths, squeezes = (
        find_lazy_sliced_input_tensors(parent_graph, df_op, analysis_ctx)
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

    example_inputs_new: list[tuple[BackendTensorT, int] | BackendTensorT] = list(example_inputs)
    for i in range(len(lazy_idxs)):
        store_info = analysis_ctx.tensor_storage_classes[lazy_tids[i]]
        assert isinstance(store_info, PreallocBufferStore)
        backend_dtype = backend.to_backend_datatype(buffer_dtypes[i])
        example_buf = backend.zeros_tensor(buffer_shapes[i].as_static()._shape, backend_dtype, dev)
        example_idx = 0  # backend.zeros_tensor(Shape(()), dtypes_[i], dev)
        example_inputs_new[int(lazy_idxs[i])] = (example_buf, example_idx)

    import jax

    example_inputs_new = tuple(example_inputs_new)

    def _wrapper(inputs: tuple[BackendTensorT, ...]) -> tuple[BackendTensorT, ...]:
        inputs_ = list(inputs)
        for i in range(len(lazy_idxs)):
            (buf, start_idx) = inputs_[lazy_idxs[i]]  # type: ignore
            static_length = access_lengths[i]
            inp_ = jax.lax.dynamic_slice_in_dim(buf, start_idx, static_length, axis=0)  # type: ignore
            if squeezes[i]:
                inp_ = jax.lax.squeeze(inp_, axis=0)
            inputs_[lazy_idxs[i]] = inp_

        outs = interp_exec_func(tuple(inputs_))

        return outs

    print(f"created wrapper for {df_op.op_id}", flush=True)
    log.info(
        "Lazy slice thunk wrapper created for Op %s, with %d lazy-sliced inputs",
        df_op.op_id,
        len(lazy_idxs),
    )

    return _wrapper, example_inputs_new, new_donatable_args  # type: ignore
