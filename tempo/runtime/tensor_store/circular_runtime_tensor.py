from collections.abc import Mapping
from functools import partial

from tempo.core import index_expr as ie
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import BackendTensorT, TensorId
from tempo.core.device import DeviceGroup
from tempo.core.dl_backend import DLBackend
from tempo.core.domain import Domain
from tempo.core.dtype import DataType
from tempo.core.shape import Shape
from tempo.core.storage_methods import CircularBufferStore
from tempo.core.utils import count_block_points, enum_block_points
from tempo.runtime.tensor_store.tensor_store import PreallocRuntimeTensor
from tempo.utils import logger

log = logger.get_logger(__name__)


class CircularRuntimeTensor(PreallocRuntimeTensor[BackendTensorT]):
    """NOTE: The idea here is to have 2x the window size
      as a single contiguous Nan-pre-allocated buffer.
    Every write is written to two opposite ends of the buffer,
     ensuring that there is always a contiguous
    region of the buffer that is always available for reading.

    #TODO for now this only works if the write order is dense and increasing.

    """

    def __init__(
        self,
        exec_cfg: ExecutionConfig,
        tensor_id: TensorId,
        shape: Shape,
        dtype: DataType,
        dev: DeviceGroup,
        domain: Domain,
        storage_info: CircularBufferStore,
        known_bounds: Mapping[ie.Symbol, int],
    ) -> None:
        super().__init__(tensor_id)
        self.exec_cfg = exec_cfg
        self.shape = shape
        self.dtype = dtype

        self.prefetch_amount = exec_cfg.runtime_tensor_prefetch_amount_point
        self.dims_and_window_sizes = storage_info.dims_and_base_buffer_sizes

        self.full_domain = domain
        self.full_domain_size = len(domain)

        self.backend = DLBackend.get_backend(exec_cfg.backend)
        self.dev = self.backend.to_backend_device_obj(dev)
        self.cpu = self.backend.to_backend_device_obj("cpu")

        # Determine window and pointwise domains
        self.window_domain = domain.select_vars(
            *[d for d, _ in storage_info.dims_and_base_buffer_sizes]
        )
        self.pointwise_domain = Domain.difference(self.full_domain, self.window_domain)

        # Create mapping from domain variables to their indices
        self.domain_var_to_index = {var: idx for idx, var in enumerate(domain.variables)}
        self.window_dim_indices = {
            self.domain_var_to_index[d] for d, _ in storage_info.dims_and_base_buffer_sizes
        }
        self.pointwise_dim_indices = set(range(len(domain.variables))) - self.window_dim_indices

        assert len(storage_info.dims_and_base_buffer_sizes) == 1, (
            "Only one window dimension is supported for now"
        )
        # Removed assertion that window dim is last

        shape: tuple[int, ...] = shape.as_static()._shape

        self.points_per_block = 0
        # Calculate window size and total buffer size
        self.window_size = 1
        self.window_shape: tuple[int, ...] = ()
        for dim, window_size in storage_info.dims_and_base_buffer_sizes:
            size = window_size
            self.window_shape += (size,)
            self.window_size *= size
            self.points_per_block += dim.as_bound().evaluate(known_bounds)

        # Total buffer size is 2x window size
        self.total_buffer_size = self.window_size * 2
        self.buffer_shape = (self.total_buffer_size,) + shape

        # Initialize the circular buffers for each pointwise index
        self.bend_dtype = self.backend.to_backend_datatype(dtype)
        self._buffers: dict[tuple[int, ...], BackendTensorT] = {}
        self._write_positions: dict[tuple[int, ...], int] = {}

        self._dealloc_ticks: dict[tuple[int, ...], int] = {}

        self.alloc_fn = partial(
            self.backend.full_tensor,
            fill_value=storage_info.prealloc_value,
            shape=self.buffer_shape,
            dtype=self.bend_dtype,
            device=self.dev,
        )
        self.inplace_set = self.backend.get_inplace_set_fn(self.alloc_fn(), None, None)  # type: ignore

        self.window_dim_idx = self.full_domain.find_variable_index(
            storage_info.dims_and_base_buffer_sizes[0][0]
        )

        self.should_lazy_slice = self.exec_cfg.can_lazy_slice()

    def extract_key_and_index(
        self, item: tuple[int | slice, ...]
    ) -> tuple[tuple[int, ...], tuple[int | slice, ...]]:
        """Split item into pointwise key and window indices."""
        # buffer_key = []
        # window_indices = []
        # for dim_idx, dim_item in enumerate(item):
        #    if dim_idx == self.window_dim_idx:
        #        window_indices.append(dim_item)
        #    else:
        #        buffer_key.append(dim_item)
        # return tuple(buffer_key), tuple(window_indices)
        buffer_key = item[: self.window_dim_idx] + item[self.window_dim_idx + 1 :]
        window_idx = (item[self.window_dim_idx],)
        return buffer_key, window_idx  # type: ignore

    def _get_or_create_buffer(self, pointwise_key: tuple[int, ...]) -> BackendTensorT:
        """Get or create a circular buffer for the given pointwise key."""
        buf = self._buffers.get(pointwise_key)
        if buf is None:
            buf = self.alloc_fn()
            self._buffers[pointwise_key] = buf
            self._dealloc_ticks[pointwise_key] = 0
        return buf

    def get_read_region(self, item: tuple[int | slice, ...]) -> tuple[tuple[int, ...], int | slice]:
        """Given the full item, return the index or slice to read from the buffer."""
        buffer_key, window_indices = self.extract_key_and_index(item)
        # Only support 1 window dim for now
        window_idx = window_indices[0]
        if isinstance(window_idx, int):
            start = window_idx
            stop = start + 1
        else:
            start = window_idx.start
            stop = window_idx.stop
        user_win_len = stop - start
        start_logical = start % self.total_buffer_size
        stop_logical = (start_logical + user_win_len) % self.total_buffer_size
        if start_logical < stop_logical:
            if isinstance(window_idx, int):
                return buffer_key, start_logical
            else:
                return buffer_key, slice(start_logical, stop_logical)
        else:
            # Wrap-around: return slice as before (may need to be handled by caller)
            if isinstance(window_idx, int):
                return buffer_key, (start_logical + self.window_size) % self.total_buffer_size
            else:
                return buffer_key, slice(
                    (start_logical + self.window_size) % self.total_buffer_size,
                    (stop_logical + self.window_size) % self.total_buffer_size,
                )

    def all_int_fast_path(self, item: tuple[int, ...]) -> BackendTensorT:
        return self.__getitem__(item)

    def __getitem__(self, item: tuple[int | slice, ...]) -> BackendTensorT:
        buffer_key, idx = self.get_read_region(item)
        buffer = self._get_or_create_buffer(buffer_key)

        if self.should_lazy_slice:
            # NOTE: We secretely alter the return type for lazy slicing
            return (buffer, idx.start)  # type: ignore
        else:
            return buffer[idx]  # type: ignore

    def __setitem__(self, item: tuple[int | slice, ...], value: BackendTensorT) -> None:
        # NOTE: type ignore because we know for now all writes are point writes
        # TODO: improve this
        return self.all_int_fast_path_set(item, value)  # type: ignore

    def all_int_fast_path_set(self, item: tuple[int, ...], value: BackendTensorT) -> None:
        # NOTE: writes are always single elements.

        buffer_id, (pos1, pos2) = self.extract_write_key_and_indexes(item)

        # Get or create buffer for this pointwise key
        buffer = self._get_or_create_buffer(buffer_id)

        # Update read region
        self._buffers[buffer_id] = self.inplace_set(buffer, pos1, value)
        self._buffers[buffer_id] = self.inplace_set(self._buffers[buffer_id], pos2, value)

        # Update write position

    def flush(self) -> None:
        """Clear all buffers and reset positions."""
        self._buffers.clear()
        self._dealloc_ticks.clear()

    def mem_usage_bytes(self) -> int:
        return self.total_buffer_size * self.dtype.repr_bytes * len(self._buffers)

    def deallocate_point(self, item: tuple[int | slice, ...]) -> None:
        """This method deallocates the tensor at the given index."""

        buffer_key, _ = self.extract_key_and_index(item)

        val = self._dealloc_ticks[buffer_key] + 1
        self._dealloc_ticks[buffer_key] = val

        if val >= self.points_per_block:
            del self._buffers[buffer_key]
            del self._dealloc_ticks[buffer_key]

    def deallocate_block(self, block: tuple[int | slice, ...]) -> None:
        key, index = self.extract_key_and_index(block)
        # NOTE: class assumes a single block dim, but this may not be the case in mem man reqs.
        keys = enum_block_points(key)

        num_points = count_block_points(index)
        for k in keys:
            self._dealloc_ticks[k] += num_points
            if self._dealloc_ticks[k] >= self.points_per_block:
                del self._buffers[k]
                del self._dealloc_ticks[k]

    def offload_point(self, item: tuple[int | slice, ...]) -> None:
        """This method offloads the tensor at the given index."""
        log.warning(
            "offload_point requested with item %s - does not make sense for circular buffers", item
        )
        raise NotImplementedError("Does not make sense for circular buffers")

    def fetch_point(self, item: tuple[int | slice, ...]) -> None:
        """This method fetches the tensor at the given index."""
        log.warning(
            "fetch_point requested with item %s - does not make sense for circular buffers", item
        )
        raise NotImplementedError("Does not make sense for circular buffers")

    def offload_block(self, block: tuple[int | slice, ...]) -> None:
        log.warning(
            "offload_block requested with item %s - does not make sense for circular buffers", block
        )
        raise NotImplementedError("Does not make sense for circular buffers")

    def fetch_block(self, block: tuple[int | slice, ...]) -> None:
        log.warning(
            "fetch_block requested with item %s - does not make sense for circular buffers", block
        )
        raise NotImplementedError("Does not make sense for circular buffers")

    def __str__(self) -> str:
        return f"CircularTensor( \
                  tensor_id={self.tensor_id}, shape={self.shape}, \
                  dtype={self.dtype}, domain={self.full_domain})"

    def __hash__(self) -> int:
        return hash(self.tensor_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CircularRuntimeTensor):
            return False
        return bool(self.tensor_id == other.tensor_id)

    def replace_backing_buffer(self, key: tuple[int, ...], buffer: BackendTensorT) -> None:
        """Replaces the backing buffer for the given key."""
        self._buffers[key] = buffer

    def get_backing_buffer(self, key: tuple[int, ...]) -> BackendTensorT:
        """Gets the backing buffer for the given key."""
        return self._get_or_create_buffer(key)

    def extract_write_key_and_indexes(
        self, item: tuple[int, ...]
    ) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        """Extracts the write key and indexes from the given item."""
        buffer_id, window_indices = self.extract_key_and_index(item)
        # Only support 1 window dim for now
        window_item = window_indices[0]
        if isinstance(window_item, slice):
            raise NotImplementedError("Write with slice not supported for circular buffer")
        pos1 = window_item % self.window_size
        pos2 = (pos1 + self.window_size) % self.total_buffer_size
        return buffer_id, ((pos1,), (pos2,))
