from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.datatypes import BackendTensorT, TensorId
from tempo.core.storage_methods import EvalSymbolStore, PointStore
from tempo.runtime.tensor_store.point_runtime_tensor import PointRuntimeTensor
from tempo.runtime.tensor_store.prealloc_eval_symbol_runtime_tensor import (
    EvalSymbolRuntimeTensor,
)
from tempo.runtime.tensor_store.tensor_store import TensorStore
from tempo.utils import logger

log = logger.get_logger(__name__)


class PointTensorStore(TensorStore[BackendTensorT]):
    def __init__(
        self,
        ctx: CompilationCtx,
    ):
        super().__init__(ctx)
        dg = ctx.dg
        exec_cfg = ctx.exec_cfg
        analysis_ctx = ctx.analysis_ctx
        storage = analysis_ctx._tensor_storage_classes or {}

        for tensor_id, shape, dtype, domain in dg.get_all_tensor_descriptions():
            dev = self.ctx.get_tensor_device(tensor_id)

            storage_method = storage.get(tensor_id, PointStore())

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
            # elif isinstance(storage_method, PointStore):
            self.tensors[tensor_id] = PointRuntimeTensor(
                exec_cfg, tensor_id, shape, dtype, dev, domain
            )
            # NOTE: This causes tests to fail despite correct
            # elif isinstance(storage_method, DontStore):
            #    self.tensors[tensor_id] = NoStorageRuntimeTensor(
            #        exec_cfg, tensor_id, shape, dtype, domain
            #    )
            # else:
            #    raise ValueError(f"Unknown storage method {storage_method}")

    def __getitem__(self, item: TensorId) -> PointRuntimeTensor:
        return self.tensors[item]  # type: ignore

    def flush(self) -> None:
        """This method clears the tensor store of any remaining data, preparing
        it for the next execution.
        """
        for tensor in self.tensors.values():
            tensor.flush()
