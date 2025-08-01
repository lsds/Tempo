from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.datatypes import BackendTensorT, TensorId
from tempo.core.storage_methods import (
    BlockStore,
    CircularBufferStore,
    DontStore,
    EvalSymbolStore,
    PointStore,
    PreallocCircularBufferStore,
)
from tempo.runtime.tensor_store.block_runtime_tensor import BlockRuntimeTensor
from tempo.runtime.tensor_store.circular_runtime_tensor import CircularRuntimeTensor
from tempo.runtime.tensor_store.no_storage_runtime_tensor import (
    NoStorageRuntimeTensor,
)
from tempo.runtime.tensor_store.point_runtime_tensor import PointRuntimeTensor
from tempo.runtime.tensor_store.prealloc_eval_symbol_runtime_tensor import (
    EvalSymbolRuntimeTensor,
)
from tempo.runtime.tensor_store.tensor_store import RuntimeTensor, TensorStore
from tempo.utils import logger

log = logger.get_logger(__name__)


class HybridTensorStore(TensorStore[BackendTensorT]):
    def __init__(
        self,
        ctx: CompilationCtx,
    ):
        super().__init__(ctx)
        dg = ctx.dg
        exec_cfg = ctx.exec_cfg
        analysis_ctx = ctx.analysis_ctx
        storage = analysis_ctx.tensor_storage_classes

        for tensor_id, shape, dtype, domain in dg.get_all_tensor_descriptions():
            storage_method = storage[tensor_id]
            dev = self.ctx.get_tensor_device(tensor_id)
            if isinstance(storage_method, EvalSymbolStore):
                self.tensors[tensor_id] = EvalSymbolRuntimeTensor(
                    exec_cfg,
                    tensor_id,
                    shape,
                    dtype,
                    dev,
                    domain,
                    storage_method.symbol,
                    dg.bound_defs,
                )
            elif isinstance(storage_method, BlockStore):
                self.tensors[tensor_id] = BlockRuntimeTensor(
                    exec_cfg,
                    tensor_id,
                    shape,
                    dtype,
                    dev,
                    domain,
                    storage_method,
                    dg.static_bounds,
                )
            elif isinstance(storage_method, CircularBufferStore):
                self.tensors[tensor_id] = CircularRuntimeTensor(
                    exec_cfg,
                    tensor_id,
                    shape,
                    dtype,
                    dev,
                    domain,
                    storage_method,
                    dg.static_bounds,
                )
            elif isinstance(storage_method, PreallocCircularBufferStore):
                raise NotImplementedError("PreallocCircularBufferStore is currently disabled")
            elif isinstance(storage_method, PointStore):
                self.tensors[tensor_id] = PointRuntimeTensor(
                    exec_cfg, tensor_id, shape, dtype, dev, domain
                )
            elif isinstance(storage_method, DontStore):
                self.tensors[tensor_id] = NoStorageRuntimeTensor(
                    exec_cfg, tensor_id, shape, dtype, domain
                )
            else:
                raise ValueError(f"Unknown storage method {storage_method}")

    def __getitem__(self, item: TensorId) -> RuntimeTensor:
        return self.tensors[item]

    def flush(self) -> None:
        """This method clears the tensor store of any remaining data, preparing
        it for the next execution.
        """
        for tensor in self.tensors.values():
            tensor.flush()
