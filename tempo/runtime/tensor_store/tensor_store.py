from abc import ABC, abstractmethod
from typing import Generic

from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.datatypes import BackendTensorT, TensorId


class RuntimeTensor(Generic[BackendTensorT], ABC):
    def __init__(self, tensor_id: TensorId) -> None:
        super().__init__()
        self.tensor_id = tensor_id

    @abstractmethod
    def __getitem__(self, item: tuple[int | slice, ...]) -> BackendTensorT:
        """Gets the value at the given item"""
        ...

    @abstractmethod
    def all_int_fast_path(self, item: tuple[int | slice, ...]) -> BackendTensorT:
        """Gets the value at the given item"""
        ...

    @abstractmethod
    def all_int_fast_path_set(self, item: tuple[int, ...], value: BackendTensorT) -> None:
        """Sets the value at the given item"""
        ...

    @abstractmethod
    def __setitem__(self, item: tuple[int | slice, ...], value: BackendTensorT) -> None:
        """Sets the value at the given item"""
        ...

    @abstractmethod
    def flush(self) -> None:
        """This method clears the tensor of any remaining data."""
        ...

    @abstractmethod
    def deallocate_point(self, item: tuple[int | slice, ...]) -> None:
        """This method deallocates the tensor at the given index."""
        ...

    @abstractmethod
    def offload_point(self, item: tuple[int | slice, ...]) -> None:
        """This method offloads the tensor at the given index."""
        ...

    @abstractmethod
    def fetch_point(self, item: tuple[int | slice, ...]) -> None:
        """This method fetches the tensor at the given index."""
        ...

    @abstractmethod
    def deallocate_block(self, block: tuple[int | slice, ...]) -> None: ...

    @abstractmethod
    def offload_block(self, block: tuple[int | slice, ...]) -> None: ...

    @abstractmethod
    def fetch_block(self, block: tuple[int | slice, ...]) -> None: ...

    @abstractmethod
    def mem_usage_bytes(self) -> int: ...


class PreallocRuntimeTensor(RuntimeTensor[BackendTensorT], ABC):
    """Abstract subclass of RuntimeTensor for tensors that use preallocated buffers."""

    @abstractmethod
    def replace_backing_buffer(self, key: tuple[int, ...], buffer: BackendTensorT) -> None:
        """Replaces the backing buffer for the given key."""
        ...

    @abstractmethod
    def get_backing_buffer(self, key: tuple[int, ...]) -> BackendTensorT:
        """Gets the backing buffer for the given key."""
        ...

    @abstractmethod
    def extract_write_key_and_indexes(
        self, item: tuple[int, ...]
    ) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        """Extracts the write key and indexes from the given item."""
        ...


class TensorStore(Generic[BackendTensorT], ABC):
    @abstractmethod
    def __init__(self, ctx: CompilationCtx):
        self.ctx = ctx
        self.exec_cfg = ctx.exec_cfg
        self.dg = ctx.dg
        self.analysis_ctx = ctx.analysis_ctx

        self.tensors: dict[TensorId, RuntimeTensor[BackendTensorT]] = {}

    @abstractmethod
    def __getitem__(self, item: TensorId) -> RuntimeTensor[BackendTensorT]:
        """Gets the tensor with the given id"""
        ...

    @abstractmethod
    def flush(self) -> None:
        """This method clears the tensor store of any remaining data, preparing
        it for the next execution.
        """
        ...

    def wrap_all_tensors_with_debug_checker(self) -> None:
        from tempo.runtime.tensor_store.debug_runtime_tensor_wrapper import (
            DebugRuntimeTensorWrapper,
        )

        for k, v in self.tensors.items():
            self.tensors[k] = DebugRuntimeTensorWrapper(v)

    def mem_usage_bytes(self) -> dict[TensorId, int]:
        return {tensor_id: tensor.mem_usage_bytes() for tensor_id, tensor in self.tensors.items()}

    def mem_usage_bytes_total(self) -> int:
        return sum(self.mem_usage_bytes().values())

    def mem_usage_bytes_percent(self) -> dict[TensorId, float]:
        total = self.mem_usage_bytes_total()
        return {
            tensor_id: round(tensor_bytes / total, 2)
            for tensor_id, tensor_bytes in self.mem_usage_bytes().items()
        }

    def top10_mem_usage_bytes(self) -> dict[TensorId, int]:
        return dict(
            sorted(self.mem_usage_bytes().items(), key=lambda item: item[1], reverse=True)[:10]
        )

    def top10_mem_usage_percent(self) -> dict[TensorId, float]:
        return dict(
            sorted(
                self.mem_usage_bytes_percent().items(),
                key=lambda item: item[1],
                reverse=True,
            )[:10]
        )
