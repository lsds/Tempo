from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from tempo.core.analysis_ctx import AnalysisCtx
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import BackendTensorT, OpId
from tempo.core.dependence_graph import PDG
from tempo.core.device import DeviceGroup, DeviceLike
from tempo.core.dtype import DataType
from tempo.core.shape import StaticShapeLike
from tempo.core.thunk import Thunk
from tempo.runtime.thunk_emitter import ThunkEmitter

# from dlpack import DLPackObject, asdlpack


class DLBackendName(IntEnum):
    TORCH = 0
    JAX = 1

    @classmethod
    def str_to_enum(cls, name: str) -> DLBackendName:
        lower_name = name.lower()
        if lower_name == "pytorch" or lower_name == "torch":
            return DLBackendName.TORCH
        elif lower_name == "jax" or lower_name == "xla":
            return DLBackendName.JAX
        else:
            raise ValueError(f"Invalid backend name: {name}")


def clear_all_jit_caches() -> None:
    """Clear the jit cache for all backends."""
    for backend in DLBackend.registry.values():
        backend.clear_jit_cache()


class DLBackend(Generic[BackendTensorT], ABC):
    registry: ClassVar[Dict[DLBackendName, Type[DLBackend]]] = {}

    backend_cpu: Any = None

    # =============== Registry Methods ===============
    @staticmethod
    def register_backend(backend_name: DLBackendName, backend_cls: Type[DLBackend]) -> None:
        """Register a backend class in the registry."""
        if backend_name in DLBackend.registry:
            raise ValueError(f"Tried overwriting environment set {backend_name} with new builder")
        DLBackend.registry[backend_name] = backend_cls

    @staticmethod
    def get_backend(backend_name: Union[DLBackendName, str]) -> Type[DLBackend]:
        """Get a backend class from the registry."""
        if isinstance(backend_name, str):
            backend_name = DLBackendName.str_to_enum(backend_name)
        return DLBackend.registry[backend_name]

    @staticmethod
    @abstractmethod
    def get_backend_name() -> DLBackendName:
        """Get the name of the backend."""
        raise NotImplementedError

    # =============== Backend Methods ===============

    @staticmethod
    @abstractmethod
    def clear_jit_cache() -> None:
        """Clear the jit cache."""
        raise NotImplementedError

    # --------- Tempo <-> Backend Conversion Methods ---------
    @staticmethod
    @abstractmethod
    def to_backend_device_obj(dev: DeviceLike) -> Any:
        """Convert a tempo device to a backend-specific device object."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def to_backend_datatype(dtype: DataType) -> Any:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def to_tpo_dtype(backend_dtype: Any) -> DataType:
        """Convert a backend-specific dtype to a Tempo dtype."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def cast_backend_dtype(tensor: BackendTensorT, dtype: Any) -> BackendTensorT:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def to_backend_shape(shape: StaticShapeLike) -> Any:
        # NOTE: has to be static already
        raise NotImplementedError

    # --------- Backend Methods ---------

    @staticmethod
    @abstractmethod
    def get_thunk_emitter_cls() -> Type[ThunkEmitter[BackendTensorT]]:
        """Backends must provide a ThunkEmitter class that will be used to emit executable thunks
        for each TensorOp."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def configure(exec_cfg: ExecutionConfig) -> None:
        """Configure the backend for execution. Called by the Compiler before backend compilation
        begins."""
        pass

    @staticmethod
    def sync() -> None:
        """Sync the backend device with host.
        Invoked, for example, to ensure D2H copies are complete."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def device(tensor: BackendTensorT) -> Any:
        """Get the backend-specific device of a backend tensor."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def copy(tensor: BackendTensorT) -> BackendTensorT:
        """Copy a backend tensor. This is a deep copy."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def to_device(tensor: BackendTensorT, dev: Any) -> BackendTensorT:
        """Move a backend tensor to a different backend device."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_dlpack(ext_tensor: Any) -> BackendTensorT:
        """Convert a DLPack object to a backend tensor."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def fast_int_lift(
        fill_value: int,
        dtype: Optional[Any] = None,
        device: Optional[Any] = None,
    ) -> BackendTensorT:
        """Lift a python int to a backend tensor."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def full_tensor(
        fill_value: Any,
        shape: Optional[StaticShapeLike] = None,
        dtype: Optional[Any] = None,
        device: Optional[Any] = None,
    ) -> BackendTensorT:
        """Create a backend tensor filled with a given value."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def lift_tensor(
        data: Any,
        shape: Optional[StaticShapeLike] = None,
        dtype: Optional[DataType] = None,
        device: Optional[str] = None,
    ) -> BackendTensorT:
        """Lift a python object to a backend tensor.
        For example, numpy arrays become tensors with corresponding shape and dtype,
        and python scalars become tensors with shape () and dtype matching the python scalar.
        Args:
            data: The python object to lift
            shape: The shape of the tensor
            dtype: The dtype of the tensor
            device: The device of the tensor
        Returns:
            A backend tensor
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def trace_codegen_thunk(
        execution_func: Callable[[Tuple[BackendTensorT, ...]], Tuple[BackendTensorT, ...]],
        op_id: OpId,
        dev: DeviceGroup,
        exec_cfg: ExecutionConfig,
        inputs: Sequence[BackendTensorT],
        donatable_args: Sequence[int],
        analysis_ctx: AnalysisCtx,
        parent_graph: PDG,
    ) -> Thunk[BackendTensorT]:
        """Traces and codegenerates a thunk using the backend's tracing utilities.
        Args:
            execution_func: The function to trace
            op_id: The id of the operation
            dev: The device of the operation
            exec_cfg: The execution configuration
            inputs: Example input tensors to the operation
            donatable_args: input indices to donate to the thunk for buffer reuse
        Returns:
            A traced thunk
        """
        pass

    # --------- Tensor Manipulation Methods ---------

    # TODO eventually remove either stack or get_stack_fn. Similar for inplace_set.
    @staticmethod
    @abstractmethod
    def stack(tensors: Sequence[BackendTensorT]) -> BackendTensorT:
        """Stack a sequence of backend tensors along a new axis."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_stack_fn(
        tensors: Sequence[BackendTensorT],
    ) -> Callable[[Sequence[BackendTensorT]], BackendTensorT]:
        """Get a function that stacks a sequence of backend tensors along a new axis."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_inplace_set_fn(
        tensor: BackendTensorT,
        item: Sequence[Union[int, slice]],
        value: BackendTensorT,
        traceable: bool = False,
    ) -> Callable[[BackendTensorT, Sequence[Union[int, slice]], BackendTensorT], BackendTensorT]:
        """Get a function that updates a backend tensor in place."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def unbind(tensor: BackendTensorT, axis: int) -> Sequence[BackendTensorT]:
        """Unbind (split) a backend tensor along a given axis."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def permute(tensor: BackendTensorT, axes: Sequence[int]) -> BackendTensorT:
        """Permute the dimensions of a backend tensor."""
        raise NotImplementedError

    # TODO remove or reinstate
    # @staticmethod
    # @abstractmethod
    # def reshape(tensor: BackendTensorT, shape: Tuple[int, ...]) -> BackendTensorT:
    #    raise NotImplementedError

    # --------- Generic Methods

    @classmethod
    def to_cpu(cls, tensor: BackendTensorT) -> BackendTensorT:
        """Move a backend tensor to CPU. Uses the backend's global CPU device."""
        return cls.to_device(tensor, cls.backend_cpu)

    @classmethod
    def zeros_tensor(cls, shape: Any, dtype: Any, dev: Any) -> BackendTensorT:
        """Create a backend tensor filled with zeros."""
        return cls.full_tensor(0.0, shape, dtype, dev)

    @classmethod
    def ones_tensor(cls, shape: Any, dtype: Any, dev: Any) -> BackendTensorT:
        """Create a backend tensor filled with ones."""
        return cls.full_tensor(1.0, shape, dtype, dev)
