# from functools import partial
# from typing import Dict, Mapping, Tuple, Union
#
# from tempo.core import index_expr as ie
# from tempo.core.configs import ExecutionConfig
# from tempo.core.datatypes import BackendTensorT, TensorId
# from tempo.core.device import DeviceGroup
# from tempo.core.domain import Domain
# from tempo.core.dtype import DataType
# from tempo.core.shape import Shape
# from tempo.core.storage_methods import PreallocCircularBufferStore
# from tempo.core.utils import enum_block_points
# from tempo.runtime.backends.backend import DLBackend
# from tempo.runtime.tensor_store.tensor_store import RuntimeTensor
# from tempo.utils import logger
#
# log = logger.get_logger(__name__)
#
#
# class FlatCircularRuntimeTensor(RuntimeTensor[BackendTensorT]):
#   """A runtime tensor that preallocates a buffer N times the window size and writes to a
#   single position.
#   When reaching the end of the buffer, it performs an inplace set of the
#   last window_size elements.
#   """
#
#   def __init__(
#       self,
#       exec_cfg: ExecutionConfig,
#       tensor_id: TensorId,
#       shape: Shape,
#       dtype: DataType,
#       dev: DeviceGroup,
#       domain: Domain,
#       storage_info: PreallocCircularBufferStore,
#       known_bounds: Mapping[ie.Symbol, int],
#   ) -> None:
#       super().__init__(tensor_id)
#       self.exec_cfg = exec_cfg
#       self.shape = shape
#       self.dtype = dtype
#       self.buffer_multiplier = storage_info.buffer_multiplier
#
#       self.prefetch_amount = exec_cfg.runtime_tensor_prefetch_amount
#       self.dims_and_window_sizes = storage_info.dims_and_window_sizes
#
#       self.full_domain = domain
#       self.full_domain_size = len(domain)
#
#       self.backend = DLBackend.get_backend(exec_cfg.backend)
#       self.dev = self.backend.to_backend_device_obj(dev)
#       self.cpu = self.backend.to_backend_device_obj("cpu")
#
#       # Determine window and pointwise domains
#       self.window_domain = domain.select_vars(
#          *[d for d, _ in storage_info.dims_and_window_sizes])
#       self.pointwise_domain = Domain.difference(self.full_domain, self.window_domain)
#
#       assert (
#           len(storage_info.dims_and_window_sizes) == 1
#       ), "Only one window dimension is supported for now"
#
#       assert (
#           domain.find_variable_index(storage_info.dims_and_window_sizes[0][0]) == len(domain) - 1
#       ), "Window dimension must be the last dimension"
#
#       shape: Tuple[int, ...] = shape.as_static()._shape
#
#       self.points_per_block = 0
#       # Calculate window size and total buffer size
#       self.window_size = 1
#       self.window_shape: Tuple[int, ...] = ()
#       for dim, window_size in storage_info.dims_and_window_sizes:
#           size = window_size
#           self.window_shape += (size,)
#           self.window_size *= size
#           self.points_per_block += dim.as_bound().evaluate(known_bounds)
#
#       # Total buffer size is N times window size
#       self.total_buffer_size = self.window_size * self.buffer_multiplier
#       self.buffer_shape = (self.total_buffer_size,) + shape
#
#       # Initialize the buffers for each pointwise index
#       self.bend_dtype = self.backend.to_backend_datatype(dtype)
#       self._buffers: Dict[Tuple[int, ...], BackendTensorT] = {}
#       self._write_ticks_total: Dict[Tuple[int, ...], int] = {}
#       self._next_write_pos: Dict[Tuple[int, ...], int] = {}
#       self._dealloc_ticks: Dict[Tuple[int, ...], int] = {}
#
#       self.alloc_fn = partial(
#           self.backend.full_tensor,
#           fill_value=storage_info.prealloc_value,
#           shape=self.buffer_shape,
#           dtype=self.bend_dtype,
#           device=self.dev,
#       )
#       self.inplace_set = self.backend.get_inplace_set_fn(
#                self.alloc_fn(), None, None)  # type: ignore
#       self.inplace_move = self.backend.get_inplace_set_fn(
#                self.alloc_fn(), (0, 5), None)  # type: ignore
#
#       # Extract pointwise dimensions
#       self.pointwise_domain_size = len(self.pointwise_domain)
#
#   def _get_or_create_buffer(self, pointwise_key: Tuple[int, ...]) -> BackendTensorT:
#       """Get or create a buffer for the given pointwise key."""
#       buf = self._buffers.get(pointwise_key)
#       if buf is None:
#           buf = self.alloc_fn()
#           self._buffers[pointwise_key] = buf
#           self._next_write_pos[pointwise_key] = 0
#           self._write_ticks_total[pointwise_key] = 0
#           self._dealloc_ticks[pointwise_key] = 0
#       return buf
#
#   def all_int_fast_path(self, item: Tuple[int, ...]) -> BackendTensorT:
#       return self[item]
#
#   def __getitem__(self, item: Tuple[Union[int, slice], ...]) -> BackendTensorT:
#       # Split into pointwise and window indices
#       pointwise_key = item[: self.pointwise_domain_size]
#       window_item = item[self.pointwise_domain_size]
#
#       # Get or create buffer for this pointwise key
#       buffer = self._get_or_create_buffer(pointwise_key)
#
#       if type(window_item) is int:
#           assert False, "Class not prepared for int indexing yet"
#           start = window_item
#           stop = start + 1
#           # Get the contiguous read region
#           read_start, read_stop = self._get_read_region(pointwise_key, start, stop)
#           return buffer[read_start]
#       else:
#           start = window_item.start
#           stop = window_item.stop
#           # Get the contiguous read region
#           write_ticks_total = self._write_ticks_total[pointwise_key]
#           next_write_pos = self._next_write_pos[pointwise_key]
#           if stop > write_ticks_total:
#               read_start = 0
#               read_stop = stop - start
#           else:
#               read_start = next_write_pos - (stop - start)
#               read_stop = next_write_pos
#
#           return (buffer, read_start, read_stop - read_start)
#           # return buffer[read_start:read_stop]
#
#   def __setitem__(self, item: Tuple[Union[int, slice], ...], value: BackendTensorT) -> None:
#       return self.all_int_fast_path_set(item, value)
#
#   def all_int_fast_path_set(self, item: Tuple[int, ...], value: BackendTensorT) -> None:
#       # Split into pointwise and window indices
#       pointwise_key: Tuple[int, ...] = item[: self.pointwise_domain_size]
#       window_item = item[self.pointwise_domain_size]
#
#       # Get or create buffer for this pointwise key
#       buffer = self._get_or_create_buffer(pointwise_key)
#       write_tick_pos = self._next_write_pos[pointwise_key]
#
#       # Update buffer at write position
#       self._buffers[pointwise_key] = self.inplace_set(buffer, write_tick_pos, value)
#
#       # If we've reached the end of the buffer, perform inplace set of last window_size elements
#       if write_tick_pos + 1 == self.total_buffer_size:
#           last_window_start = self.total_buffer_size - self.window_size
#           last_window_end = self.total_buffer_size
#
#           value = self._buffers[pointwise_key][last_window_start:last_window_end]
#
#           self._buffers[pointwise_key] = self.inplace_move(
#               self._buffers[pointwise_key], (0, self.window_size), value
#           )
#           self._next_write_pos[pointwise_key] = self.window_size
#       else:
#           self._next_write_pos[pointwise_key] = write_tick_pos + 1
#       self._write_ticks_total[pointwise_key] += 1
#
#   def flush(self) -> None:
#       """Clear all buffers and reset positions."""
#       self._buffers.clear()
#       self._next_write_pos.clear()
#       self._write_ticks_total.clear()
#       self._dealloc_ticks.clear()
#
#   def mem_usage_bytes(self) -> int:
#       return self.total_buffer_size * self.dtype.repr_bytes * len(self._buffers)
#
#   def deallocate_point(self, item: Tuple[Union[int, slice], ...]) -> None:
#       """This method deallocates the tensor at the given index."""
#       pointwise_key: Tuple[int, ...] = item[: self.pointwise_domain_size]
#
#       val = self._dealloc_ticks[pointwise_key] + 1
#       self._dealloc_ticks[pointwise_key] = val
#
#       if val >= self.points_per_block:
#           del self._buffers[pointwise_key]
#           del self._next_write_pos[pointwise_key]
#           del self._write_ticks_total[pointwise_key]
#           del self._dealloc_ticks[pointwise_key]
#
#   def deallocate_block(self, block: Tuple[Union[int, slice], ...]) -> None:
#       for point in enum_block_points(block):
#           self.deallocate_point(point)
#
#   def offload_point(self, item: Tuple[Union[int, slice], ...]) -> None:
#       """This method offloads the tensor at the given index."""
#       raise NotImplementedError("Does not make sense for prealloc circular buffers")
#
#   def fetch_point(self, item: Tuple[Union[int, slice], ...]) -> None:
#       """This method fetches the tensor at the given index."""
#       raise NotImplementedError("Does not make sense for prealloc circular buffers")
#
#   def offload_block(self, block: Tuple[Union[int, slice], ...]) -> None:
#       raise NotImplementedError("Does not make sense for prealloc circular buffers")
#
#   def fetch_block(self, block: Tuple[Union[int, slice], ...]) -> None:
#       raise NotImplementedError("Does not make sense for prealloc circular buffers")
#
#   def __str__(self) -> str:
#       return f"PreallocCircularTensor(
#   tensor_id={self.tensor_id}, shape={self.shape}, dtype={self.dtype}, domain={self.full_domain})"
#
#   def __hash__(self) -> int:
#       return hash(self.tensor_id)
#
#   def __eq__(self, other: object) -> bool:
#       if not isinstance(other, FlatCircularRuntimeTensor):
#           return False
#       return self.tensor_id == other.tensor_id
