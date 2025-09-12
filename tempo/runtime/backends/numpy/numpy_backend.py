from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from tempo.core.analysis_ctx import AnalysisCtx
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import OpId
from tempo.core.dependence_graph import PDG
from tempo.core.device import DeviceGroup, DeviceLike, device
from tempo.core.dl_backend import DLBackend
from tempo.core.dl_backends import DLBackendName
from tempo.core.dtype import (
    INVERSE_NUMPY_TO_TEMPO_DTYPES_DICT,
    NUMPY_TO_TEMPO_DTYPES_DICT,
    DataType,
    dtypes,
)
from tempo.core.shape import StaticShape, StaticShapeLike
from tempo.core.thunk import Thunk
from tempo.core.thunk_emitter import ThunkEmitter
from tempo.utils import logger

log = logger.get_logger(__name__)

SCALAR_SHAPE = ()

try:

    class NumPyBackend(DLBackend[np.ndarray]):
        backend_cpu = "cpu"  # Set the class-level CPU device
        pinned_memory_enabled = False

        # Use the imported dtype dicts
        NUMPY_TO_TEMPO_DTYPES_DICT = NUMPY_TO_TEMPO_DTYPES_DICT
        INVERSE_NUMPY_TO_TEMPO_DTYPES_DICT = INVERSE_NUMPY_TO_TEMPO_DTYPES_DICT

        @staticmethod
        def configure(exec_cfg: ExecutionConfig) -> None:
            np.random.seed(exec_cfg.seed)

        @staticmethod
        def sync() -> None:
            # NumPy is synchronous, no sync needed
            pass

        @staticmethod
        def get_backend_name() -> DLBackendName:
            """Get the name of the backend."""
            return DLBackendName.NUMPY

        @staticmethod
        def get_thunk_emitter_cls() -> type[ThunkEmitter]:
            from tempo.runtime.backends.numpy.numpy_thunk_emitter import NumPyThunkEmitter

            return NumPyThunkEmitter  # type:ignore

        @staticmethod
        def to_backend_device_obj(dev: DeviceLike) -> Any:
            dev = device.from_(dev)

            if dev == device.cpu:
                return "cpu"
            elif dev == device.fake_gpu:
                return "cpu"  # Treat fake_gpu as CPU for NumPy
            else:
                raise ValueError(f"NumPy backend only supports CPU devices, got {dev}")

        @staticmethod
        def device(tensor: np.ndarray) -> Any:
            return "cpu"

        @staticmethod
        def to_device(tensor: np.ndarray, dev: Any, **kwargs: Any) -> np.ndarray:
            # NumPy backend only supports CPU, so just return the input tensor
            return tensor

        @staticmethod
        def to_backend_datatype(dtype: DataType) -> Any:
            return NumPyBackend.INVERSE_NUMPY_TO_TEMPO_DTYPES_DICT[dtype]

        @staticmethod
        def to_tpo_dtype(backend_dtype: Any) -> DataType:
            """Convert a NumPy dtype to a Tempo dtype."""
            return NumPyBackend.NUMPY_TO_TEMPO_DTYPES_DICT[backend_dtype]

        @staticmethod
        def cast_backend_dtype(tensor: np.ndarray, dtype: Any) -> np.ndarray:
            return tensor.astype(dtype)

        @staticmethod
        def to_backend_shape(shape: StaticShapeLike) -> Any:
            shape = StaticShape.from_(shape)
            return shape._shape

        @staticmethod
        def from_dlpack(ext_tensor: Any) -> np.ndarray:
            # NumPy doesn't have native DLPack support, but we can try to convert
            # This is a simplified implementation
            if hasattr(ext_tensor, "numpy"):
                return ext_tensor.numpy()
            elif hasattr(ext_tensor, "__array__"):
                return np.asarray(ext_tensor)
            else:
                raise ValueError(f"Cannot convert {type(ext_tensor)} to NumPy array")

        @staticmethod
        def zeros_tensor(shape: StaticShapeLike, dtype: Any, dev: Any) -> np.ndarray:
            return np.zeros(shape=shape, dtype=dtype)

        @staticmethod
        def ones_tensor(shape: StaticShapeLike, dtype: Any, dev: Any) -> np.ndarray:
            return np.ones(shape=shape, dtype=dtype)

        @staticmethod
        def fast_int_lift(
            fill_value: int,
            dtype: Any | None = None,
            device: Any | None = None,
        ) -> np.ndarray:
            return np.full(shape=(), fill_value=fill_value, dtype=dtype)

        @staticmethod
        def full_tensor(
            fill_value: Any,
            shape: StaticShapeLike = SCALAR_SHAPE,
            dtype: Any | None = None,
            device: Any | None = None,
        ) -> np.ndarray:
            return np.full(shape=shape, fill_value=fill_value, dtype=dtype)

        @staticmethod
        def lift_tensor(
            data: Any,
            shape: StaticShapeLike = None,
            dtype: Any | None = None,
            device: Any | None = None,
        ) -> np.ndarray:
            shape_int_tuple = StaticShape.from_(shape)._shape
            # If already a numpy array, just cast/move as needed
            if isinstance(data, np.ndarray):
                x = data
                if dtype is not None and x.dtype != dtype:
                    x = x.astype(dtype)
                if shape_int_tuple is not None and tuple(x.shape) != tuple(shape_int_tuple):
                    x = np.broadcast_to(x, shape_int_tuple)
                return x

            # Convert to numpy array for uniform handling
            np_data = np.array(data)
            target_shape = shape_int_tuple if shape_int_tuple is not None else np_data.shape

            numpy_dtype = (
                dtype
                if dtype is not None
                else NumPyBackend.to_backend_datatype(dtypes.implied(np_data))
            )

            if np_data.shape == () or np_data.size == 1:
                return np.full(target_shape, np_data.item(), dtype=numpy_dtype)
            else:
                return np.broadcast_to(np_data, target_shape).astype(numpy_dtype)

        @staticmethod
        def unbind(tensor: np.ndarray, axis: int) -> Sequence[np.ndarray]:
            return np.split(tensor, tensor.shape[axis], axis=axis)  # type: ignore

        @staticmethod
        def stack(tensors: Sequence[np.ndarray]) -> np.ndarray:
            return np.stack(tensors, axis=0)

        @staticmethod
        def get_stack_fn(
            tensors: Sequence[np.ndarray],
        ) -> Callable[[Sequence[np.ndarray]], np.ndarray]:
            return lambda ts: np.stack(ts, axis=0)

        @staticmethod
        def reshape(tensor: np.ndarray, shape: StaticShapeLike) -> np.ndarray:
            return np.reshape(tensor, shape)

        @staticmethod
        def permute(tensor: np.ndarray, axes: Sequence[int]) -> np.ndarray:
            return np.transpose(tensor, axes)

        @staticmethod
        def to_numpy(tensor: np.ndarray) -> np.ndarray:
            return tensor

        @staticmethod
        def get_inplace_set_fn(
            tensor: np.ndarray,
            item: Sequence[int | slice],
            value: np.ndarray,
            traceable: bool = False,
        ) -> Callable[[np.ndarray, Sequence[int | slice], np.ndarray], np.ndarray]:
            def fn_(t: np.ndarray, i: Sequence[int | slice], v: np.ndarray) -> np.ndarray:
                t[i] = v
                return t

            return fn_

        @staticmethod
        def copy(tensor: np.ndarray) -> np.ndarray:
            return tensor.copy()

        @staticmethod
        def trace_codegen_thunk(
            execution_func: Callable[[tuple[np.ndarray, ...]], tuple[np.ndarray, ...]],
            op_id: OpId,
            dev: DeviceGroup,
            exec_cfg: ExecutionConfig,
            inputs: Sequence[np.ndarray],
            donatable_args: Sequence[int],
            analysis_ctx: AnalysisCtx,
            parent_graph: PDG,
        ) -> Thunk[np.ndarray]:
            # TODO: implement numba-based codegen.
            raise NotImplementedError("NumPy backend does not support trace codegen")

    DLBackend.register_backend(DLBackendName.NUMPY, NumPyBackend)

except ImportError as e:
    raise ImportError(
        "NumPy is not installed. Please install NumPy to use the NumPy backend."
    ) from e
except Exception as e:
    raise e
