from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Final, Union

import numpy as np
import optree
from numpy.typing import DTypeLike

from tempo.core.dl_backends import DLBackendName
from tempo.utils.logger import get_logger

log = get_logger(__name__)

# Thanks to https://github.com/tinygrad/tinygrad/blob/master/tinygrad/dtype.py for the clean design.


@dataclass(frozen=True, repr=False)
class DataType:
    name: str
    repr_bytes: int
    priority: int
    eps: float | None = None
    minimum: float | None = None

    def __str__(self) -> str:
        return self.name

    __repr__ = __str__


DataTypeLike = Union[None, DataType, str, np.dtype]


def _implied_int(val: int) -> DataType:
    # Signed bounds inclusive
    if -(2**7) <= val <= 2**7 - 1:
        return dtypes.int8
    elif -(2**15) <= val <= 2**15 - 1:
        return dtypes.int16
    elif -(2**31) <= val <= 2**31 - 1:
        return dtypes.int32
    else:
        return dtypes.int64


def _implied_float(val: float) -> DataType:
    return dtypes.float32

    ## Handle NaN/Inf first — representable at all three IEEE754 formats
    # if math.isnan(val) or math.isinf(val):
    # So do not need any special handling

    ## Round-trip check: smallest float that preserves the value
    ## if float(np.float16(val)) == val:
    ##    return dtypes.float16
    # if float(np.float32(val)) == val:
    #    return dtypes.float32
    # return dtypes.float64


class dtypes:  # noqa: N801
    bool_: Final[DataType] = DataType("bool", 1, 0)

    int8: Final[DataType] = DataType("char", 1, 1)
    uint8: Final[DataType] = DataType("unsigned char", 1, 2)
    int16: Final[DataType] = DataType("short", 2, 3)
    uint16: Final[DataType] = DataType("unsigned short", 2, 4)

    int32: Final[DataType] = DataType("int", 4, 5)
    uint32: Final[DataType] = DataType("unsigned int", 4, 6)

    int64: Final[DataType] = DataType("long", 8, 7)
    uint64: Final[DataType] = DataType("unsigned long", 8, 8)

    float16: Final[DataType] = DataType(
        "half", 2, 11, float(np.finfo(np.float16).eps), float(np.finfo(np.float16).min)
    )
    # bfloat16 could go here if added later with priority 12??

    float32: Final[DataType] = DataType(
        "float", 4, 13, float(np.finfo(np.float32).eps), float(np.finfo(np.float32).min)
    )
    float64: Final[DataType] = DataType(
        "double", 8, 14, float(np.finfo(np.float64).eps), float(np.finfo(np.float64).min)
    )

    @staticmethod
    def _implied_from_tensor(val: Any) -> DataType | None:
        if isinstance(val, np.ndarray):
            return dtypes.from_np(val.dtype)

        try:
            import torch

            if isinstance(val, torch.Tensor):
                return TORCH_TO_TEMPO_DTYPES_DICT[val.dtype]  # type: ignore
        except Exception:
            ...

        try:
            import jax

            if isinstance(val, jax.numpy.ndarray):
                return JAX_TO_TEMPO_DTYPES_DICT[val.dtype]  # type: ignore
        except Exception:
            ...

        return None

    @staticmethod
    def implied(val: Any) -> DataType:
        if isinstance(val, bool):
            return dtypes.bool_
        if isinstance(val, int):
            return _implied_int(val)
        if isinstance(val, float):
            return _implied_float(val)
        if isinstance(val, list):
            return dtypes.upcast(
                *[dtypes.implied(x) for x in optree.tree_flatten(val)[0]]  # type: ignore
            )

        tensor_dtype = dtypes._implied_from_tensor(val)
        if tensor_dtype is not None:
            return tensor_dtype

        # TODO: What case does this cover exactly?
        if hasattr(val, "dtype") and isinstance(val.dtype, DataType):
            return val.dtype  # type: ignore

        raise ValueError(f"Unexpected value: {val}.")

    # https://jax.readthedocs.io/en/latest/jep/9407-type-promotion.html
    promo_lattice: dict[DataType, Sequence[DataType]] = {
        bool_: [int8, uint8],
        int8: [int16],
        int16: [int32],
        int32: [int64],
        int64: [float16],
        uint8: [int16, uint16],
        uint16: [int32, uint32],
        uint32: [int64, uint64],
        uint64: [float16],
        float16: [float32],
        float32: [float64],
    }

    default_float: ClassVar[DataType] = float32
    default_int: ClassVar[DataType] = int32

    @staticmethod
    def _get_recursive_parents(dtype: DataType) -> set[DataType]:
        return (
            set.union(
                *[dtypes._get_recursive_parents(d) for d in dtypes.promo_lattice[dtype]], {dtype}
            )
            if dtype != dtypes.float64
            else {dtypes.float64}
        )

    @staticmethod
    def upcast(*dtypes_: DataType) -> DataType:
        if len(dtypes_) == 1:
            return dtypes_[0]

        result = min(
            set.intersection(*[dtypes._get_recursive_parents(d) for d in dtypes_]),
            key=lambda x: x.priority,
        )

        ## Check for problematic upcasting scenarios
        # for dtype_ in dtypes_:
        #    if dtype_ != result:
        #        dtypes._check_upcast_warning(dtype_, result)

        # if result == dtypes.float32:
        #    print(f"Input dtypes: {dtypes_}")
        #    print(f"Result: {result}")
        return result

    @staticmethod
    def least_upper_float(dt: DataType) -> DataType:
        return dt if dtypes.is_float(dt) else dtypes.upcast(dt, dtypes.default_float)

    @staticmethod
    def least_upper_signed_int(dt: DataType) -> DataType:
        t = (
            dt
            if dtypes.is_signed_int(dt)
            else dtypes.upcast(dt, dtypes.default_int if dtypes.is_signed_int(dt) else dtypes.int32)
        )
        return t  # type: ignore

    @staticmethod
    def is_signed_int(x: DataType) -> bool:
        """Check if dtype is a signed integer type."""
        return x in (
            dtypes.int8,
            dtypes.int16,
            dtypes.int32,
            dtypes.int64,
        )

    @staticmethod
    def is_unsigned_int(x: DataType) -> bool:
        return x in (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)

    @staticmethod
    def is_integer(x: DataType) -> bool:
        """Check if dtype is any integer type (signed or unsigned)."""
        return x in (
            dtypes.int8,
            dtypes.int16,
            dtypes.int32,
            dtypes.int64,
            dtypes.uint8,
            dtypes.uint16,
            dtypes.uint32,
            dtypes.uint64,
        )

    @staticmethod
    def is_float(x: DataType) -> bool:
        return x in (
            dtypes.float16,
            dtypes.float32,
            dtypes.float64,
        )

    @staticmethod
    def is_bool(x: DataType) -> bool:
        return x == dtypes.bool_

    @staticmethod
    def from_np(x: DTypeLike) -> DataType:
        return NUMPY_TO_TEMPO_DTYPES_DICT[np.dtype(x).type]

    @staticmethod
    def to_np(x: DataType) -> DTypeLike:
        return INVERSE_NUMPY_TO_TEMPO_DTYPES_DICT[x]

    @staticmethod
    def fields() -> dict[DTypeLike, DataType]:
        return NUMPY_TO_TEMPO_DTYPES_DICT

    @staticmethod
    def from_(x: DataTypeLike, none_dtype: DataType = default_float) -> DataType:
        if x is None:
            return none_dtype
        if isinstance(x, DataType):
            return x
        if isinstance(x, np.dtype):
            return dtypes.from_np(x)
        if isinstance(x, str):
            np_dtype = np.dtype(x)
            return dtypes.from_np(np_dtype)

        raise ValueError(f"Unexpected value: {x}")

    @staticmethod
    def min(dtype: DataType) -> float | int:
        """Get the minimum value for a given dtype."""
        if dtype == dtypes.float32:
            return float("-inf")
        elif dtype == dtypes.float64:
            return float("-inf")
        elif dtype == dtypes.int32:
            return -2147483648
        elif dtype == dtypes.int64:
            return -9223372036854775808
        elif dtype == dtypes.int16:
            return -32768
        elif dtype == dtypes.int8:
            return -128
        elif dtype in {dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64}:
            return 0
        else:
            return float("-inf")

    @staticmethod
    def max(dtype: DataType) -> float | int:
        """Get the maximum value for a given dtype."""
        if dtype == dtypes.float32:
            return float("inf")
        elif dtype == dtypes.float64:
            return float("inf")
        elif dtype == dtypes.int32:
            return 2147483647
        elif dtype == dtypes.int64:
            return 9223372036854775807
        elif dtype == dtypes.int16:
            return 32767
        elif dtype == dtypes.int8:
            return 127
        elif dtype == dtypes.uint8:
            return 255
        elif dtype == dtypes.uint16:
            return 65535
        elif dtype == dtypes.uint32:
            return 4294967295
        elif dtype == dtypes.uint64:
            return 18446744073709551615
        else:
            return float("inf")


NUMPY_TO_TEMPO_DTYPES_DICT: dict[DTypeLike, DataType] = {
    np.bool_: dtypes.bool_,
    np.float16: dtypes.float16,
    np.float32: dtypes.float32,
    np.float64: dtypes.float64,
    np.uint8: dtypes.uint8,
    np.uint16: dtypes.uint16,
    np.uint32: dtypes.uint32,
    np.uint64: dtypes.uint64,
    np.int8: dtypes.int8,
    np.int16: dtypes.int16,
    np.int32: dtypes.int32,
    np.int64: dtypes.int64,
    #
    np.dtypes.BoolDType: dtypes.bool_,
    np.dtypes.Float16DType: dtypes.float16,
    np.dtypes.Float32DType: dtypes.float32,
    np.dtypes.Float64DType: dtypes.float64,
    np.dtypes.UInt8DType: dtypes.uint8,
    np.dtypes.UInt16DType: dtypes.uint16,
    np.dtypes.UInt32DType: dtypes.uint32,
    np.dtypes.UInt64DType: dtypes.uint64,
    np.dtypes.Int8DType: dtypes.int8,
    np.dtypes.Int16DType: dtypes.int16,
    np.dtypes.Int32DType: dtypes.int32,
    np.dtypes.Int64DType: dtypes.int64,
}

INVERSE_NUMPY_TO_TEMPO_DTYPES_DICT = {
    dtypes.bool_: np.bool_,
    dtypes.float16: np.float16,
    dtypes.float32: np.float32,
    dtypes.float64: np.float64,
    dtypes.uint8: np.uint8,
    dtypes.uint16: np.uint16,
    dtypes.uint32: np.uint32,
    dtypes.uint64: np.uint64,
    dtypes.int8: np.int8,
    dtypes.int16: np.int16,
    dtypes.int32: np.int32,
    dtypes.int64: np.int64,
}

# Backend dtype mappings (for use by both core and runtime, avoiding circular imports)

# PyTorch dtype to Tempo DataType mapping
try:
    import torch

    TORCH_TO_TEMPO_DTYPES_DICT: dict[Any, DataType] = {
        torch.bool: dtypes.bool_,
        torch.float16: dtypes.float16,
        torch.float32: dtypes.float32,
        # torch.bfloat16: dtypes.bfloat16, #TODO
        torch.float64: dtypes.float64,
        torch.uint8: dtypes.uint8,
        # torch.uint16: dtypes.uint16, #NOTE THESE DO NOT EXIST IN TORCH
        # torch.uint32: dtypes.uint32,
        # torch.uint64: dtypes.uint64,
        torch.int8: dtypes.int8,
        torch.int16: dtypes.int16,
        torch.int32: dtypes.int32,
        torch.int64: dtypes.int64,
    }
    INVERSE_TORCH_TO_TEMPO_DTYPES_DICT = {v: k for k, v in TORCH_TO_TEMPO_DTYPES_DICT.items()}
except ImportError:
    TORCH_TO_TEMPO_DTYPES_DICT = {}
    INVERSE_TORCH_TO_TEMPO_DTYPES_DICT = {}

# JAX dtype to Tempo DataType mapping
try:
    import jax.numpy as jnp

    JAX_TO_TEMPO_DTYPES_DICT: dict[Any, DataType] = {
        jnp.bool_: dtypes.bool_,
        jnp.float16: dtypes.float16,
        jnp.float32: dtypes.float32,
        jnp.float64: dtypes.float64,
        jnp.uint8: dtypes.uint8,
        jnp.uint16: dtypes.uint16,
        jnp.uint32: dtypes.uint32,
        jnp.uint64: dtypes.uint64,
        jnp.int8: dtypes.int8,
        jnp.int16: dtypes.int16,
        jnp.int32: dtypes.int32,
        jnp.int64: dtypes.int64,
    }
    INVERSE_JAX_TO_TEMPO_DTYPES_DICT = {v: k for k, v in JAX_TO_TEMPO_DTYPES_DICT.items()}
except ImportError:
    JAX_TO_TEMPO_DTYPES_DICT = {}
    INVERSE_JAX_TO_TEMPO_DTYPES_DICT = {}


def set_default_int_dtype_for_backend(backend_name: DLBackendName) -> None:
    if backend_name == DLBackendName.TORCH:
        dtypes.default_int = dtypes.int64
    else:
        dtypes.default_int = dtypes.int32
