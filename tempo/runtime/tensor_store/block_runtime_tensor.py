import math
from functools import partial
from typing import Dict, List, Mapping, Set, Tuple, Union

from tempo.core import index_expr as ie
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import BackendTensorT, TensorId
from tempo.core.device import DeviceGroup
from tempo.core.domain import Domain
from tempo.core.dtype import DataType, dtypes
from tempo.core.shape import Shape
from tempo.core.storage_methods import BlockStore
from tempo.core.utils import enum_block_points
from tempo.runtime.backends.backend import DLBackend
from tempo.runtime.tensor_store.tensor_store import PreallocRuntimeTensor
from tempo.utils import isl as isl_utils
from tempo.utils import logger

log = logger.get_logger(__name__)


class BlockRuntimeTensor(PreallocRuntimeTensor[BackendTensorT]):
    def __init__(
        self,
        exec_cfg: ExecutionConfig,
        tensor_id: TensorId,
        shape: Shape,
        dtype: DataType,
        dev: DeviceGroup,
        domain: Domain,
        storage_info: BlockStore,
        known_bounds: Mapping[ie.Symbol, int],
    ) -> None:
        super().__init__(tensor_id)
        self.exec_cfg = exec_cfg
        self.shape = shape
        self.dtype = dtype
        self.domain = domain

        self.prefetch_amount = exec_cfg.runtime_tensor_prefetch_amount_block
        self.dims_and_block_sizes = storage_info.dims_and_base_buffer_sizes

        self.full_domain = domain
        self.full_domain_size = len(domain)

        self.backend = DLBackend.get_backend(exec_cfg.backend)
        self.dev = self.backend.to_backend_device_obj(dev)
        self.cpu = self.backend.to_backend_device_obj("cpu")

        # Determine block and pointwise domains
        self.block_domain = domain.select_vars(
            *[d for d, _ in storage_info.dims_and_base_buffer_sizes]
        )
        self.pointwise_domain = Domain.difference(self.full_domain, self.block_domain)

        # Create mapping from domain variables to their indices
        # and whether they are block dimensions
        self.domain_var_to_index = {var: idx for idx, var in enumerate(domain.variables)}
        self.block_dim_indices = {
            self.domain_var_to_index[d] for d, _ in storage_info.dims_and_base_buffer_sizes
        }
        self.pointwise_dim_indices = set(range(len(domain.variables))) - self.block_dim_indices

        params = {
            param: known_bounds.get(param, self.exec_cfg.default_dim_upper_bound_size)
            for param in domain.parameters
        }
        dim_sizes = {
            p: ub.evaluate(params)
            for p, ub in zip(domain.parameters, domain.parameters, strict=False)
            # TODO fix this issue properly
            # p: ub.evaluate(params) for p, ub in zip(domain.parameters, domain.ubounds)
        }

        self.points_per_block = 0
        self.block_modulos: Tuple[int, ...] = ()
        # Build block shape
        block_shape: Tuple[int, ...] = ()
        self.storage_info = storage_info
        for d, block_size in storage_info.dims_and_base_buffer_sizes:
            size = block_size or dim_sizes[d.as_bound()]
            block_shape += (size,)
            self.block_modulos += (size,)
            self.points_per_block += size

        for d in self.shape:
            if isinstance(d, int):
                block_shape += (d,)
            else:
                max_val = isl_utils.int_index_val_max(
                    d,
                    # self.full_domain,
                    None,
                    known_bounds,
                )
                assert max_val is not None, f"Could not determine max value for {d}"
                assert isinstance(max_val, ie.ConstInt)
                block_shape += (max_val.const,)
        self.block_shape = block_shape

        self.bend_dtype = self.backend.to_backend_datatype(dtype)
        self.bend_int64 = self.backend.to_backend_datatype(dtypes.int64)

        self._storage_map: Dict[Tuple[int, ...], BackendTensorT] = {}
        self._dealloc_counters: Dict[Tuple[int, ...], int] = {}
        self._block_points_on_cpu: Dict[Tuple[int, ...], Set[Tuple[int, ...]]] = {}

        self._mem_used_per_block = math.prod(self.block_shape) * self.dtype.repr_bytes

        self.alloc_fn = partial(
            self.backend.full_tensor,
            fill_value=storage_info.prealloc_value,
            shape=block_shape,
            dtype=self.bend_dtype,
            device=self.dev,
        )

        example_tensor = self.alloc_fn()
        example_val = example_tensor[0]
        self.inplace_set = self.backend.get_inplace_set_fn(
            example_tensor,
            (self.backend.zeros_tensor(shape=(), dtype=self.bend_int64, dev=self.dev),),
            example_val,
        )  # type: ignore

        # Extract pointwise and block dimensions
        self.pointwise_domain_size = len(self.pointwise_domain)

        ## NOTE: ASSUMPTION: Block dims always come after pointwise dims
        # print(f"{self.tensor_id} -> {storage_info}")
        # assert all(
        #    self.full_domain.find_variable_index(d[0]) >= self.pointwise_domain_size
        #    for d in storage_info.dims_and_block_sizes
        # ), f"Block dims must come after pointwise dims. \
        #  {self.full_domain} {self.pointwise_domain}"

    def extract_key_and_index(
        self, item: Tuple[Union[int, slice], ...]
    ) -> Tuple[Tuple[int, ...], Tuple[Union[int, slice], ...]]:
        buffer_key: List[int] = []
        intra_buffer_index: List[Union[int, slice]] = []

        # Process each dimension in the order they appear in the domain
        for dim_idx, dim_item in enumerate(item):
            if dim_idx in self.block_dim_indices:
                # This is a block dimension
                block_size = self.block_modulos[list(self.block_dim_indices).index(dim_idx)]

                if type(dim_item) is slice:
                    start = dim_item.start
                    stop = dim_item.stop
                    step = dim_item.step

                    block = start // block_size
                    buffer_key.append(block)
                    block_end = (block + 1) * block_size
                    adjusted_stop = stop if stop < block_end else block_end
                    new_stop = adjusted_stop % block_size
                    if new_stop == 0:
                        new_stop = block_size
                    intra_buffer_index.append(slice(start % block_size, new_stop, step))
                else:
                    block = dim_item // block_size
                    buffer_key.append(block)
                    adjusted_index = dim_item % block_size
                    intra_buffer_index.append(adjusted_index)
            else:
                # This is a pointwise dimension
                buffer_key.append(dim_item)  # type: ignore
                # index.append(dim_item)

        return tuple(buffer_key), tuple(intra_buffer_index)  # type: ignore

    def _get_or_create_buffer(self, key: Tuple[int, ...]) -> BackendTensorT:
        """Get or create a buffer for the given key."""
        if key not in self._storage_map:
            # If no block, borrow one from the pool
            block = self.alloc_fn()
            self._storage_map[key] = block
            # Initialize deallocation counter
            self._dealloc_counters[key] = self.points_per_block
            self._block_points_on_cpu[key] = set()
        return self._storage_map[key]

    def __str__(self) -> str:
        return f"Tensor(tensor_id={self.tensor_id}, shape={self.shape},\
              dtype={self.dtype}, domain={self.full_domain})"

    def __hash__(self) -> int:
        return hash(self.tensor_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BlockRuntimeTensor):
            return False
        res: bool = self.tensor_id == other.tensor_id
        return res

    def __getitem__(self, item: Tuple[Union[int, slice], ...]) -> BackendTensorT:
        # log.info("%s: GET @ %s", self.tensor_id, item)
        key, index = self.extract_key_and_index(item)

        # log.info("TID %s, GET with item %s led to block key %s and block index %s.",
        #  self.tensor_id, item, key, index)
        block = self._storage_map[key]
        # return block

        if type(index[0]) is slice:
            if index[0].stop - index[0].start == self.block_shape[0]:
                indexed = block
            else:
                # NOTE: add support for block tensor lazy slicing in the wrapper
                # NOTE: this happens when there is an access of static size smaller than
                # the max block access size
                indexed = (block, index[0].start)  # type: ignore
        else:
            indexed = block[index]  # type: ignore
        return indexed  # type: ignore

    def all_int_fast_path(self, item: Tuple[int, ...]) -> BackendTensorT:
        # log.info("%s: GET @ %s", self.tensor_id, item)
        key, index = self.extract_write_key_and_indexes(item)
        index = index[0]  # NOTE: index is a tuple of tuples because of circular buffer compat

        # log.info("TID %s, GET with item %s led to block key %s and block index %s.",
        #  self.tensor_id, item, key, index)
        block = self._storage_map[key]
        return block[index]  # type: ignore

    def __setitem__(self, item: Tuple[Union[int, slice], ...], value: BackendTensorT) -> None:
        # log.info("%s: SET @ %s", self.tensor_id, item)
        key, index = self.extract_key_and_index(item)
        # log.info("TID %s, SET with item %s led to block key %s and block index %s.",
        #  self.tensor_id, item, key, index)

        block = self._get_or_create_buffer(key)
        # Set the value in the block
        self._storage_map[key] = self.inplace_set(
            block,
            (self.backend.full_tensor(index[0], shape=(), dtype=self.bend_int64, device=self.dev),),
            value,
        )
        # assert block.unsafe_buffer_pointer() == self._storage_map[key].unsafe_buffer_pointer()

    def all_int_fast_path_set(self, item: Tuple[int, ...], value: BackendTensorT) -> None:
        # log.info("%s: SET @ %s", self.tensor_id, item)
        key, index = self.extract_write_key_and_indexes(item)
        index = index[0]  # NOTE: index is a tuple of tuples because of circular buffer compat
        # log.info("TID %s, SET with item %s led to block key %s and block index %s.",
        #  self.tensor_id, item, key, index)

        block = self._get_or_create_buffer(key)

        # Set the value in the block
        # print(f"Setting {key} at {index}. {block.shape=} \
        #  {value.shape=}, {self.block_shape=}", flush=True)
        self._storage_map[key] = self.inplace_set(
            block,
            (self.backend.full_tensor(index[0], shape=(), dtype=self.bend_int64, device=self.dev),),
            value,
        )

    def flush(self) -> None:
        """This method clears the tensor of any remaining data."""
        self._storage_map.clear()
        self._dealloc_counters.clear()

    def deallocate_point(self, item: Tuple[int, ...]) -> None:
        # log.info("%s: DEA @ %s", self.tensor_id, item)
        # Decrement the counter and deallocate the block if needed
        key, _ = self.extract_write_key_and_indexes(item)

        # NOTE: If steps were allowed, (s.stop - s.start + (s.step - 1)) // s.step
        self._dealloc_counters[key] -= 1
        # log.info("%s: DEA @ %s - points for block %s before=%s after=%s",
        #  self.tensor_id, item, key, self._dealloc_counters[key] + num_points,
        #  self._dealloc_counters[key])
        if self._dealloc_counters[key] == 0:
            # Deallocate the block
            # log.info("%s: DEA @ %s - deallocating block %s. StoreMap %s",
            #  self.tensor_id, item, key, {k: v.shape for k,v in self._storage_map.items()})
            del self._storage_map[key]  # type: ignore
            del self._dealloc_counters[key]  # type: ignore
            # log.info("%s: DEA @ %s - deallocated block %s. StoreMap %s",
            #  self.tensor_id, item, key, {k: v.shape for k,v in self._storage_map.items()})
            # Delete if present in offload and fetch counters
            self._block_points_on_cpu.pop(key, None)

    def offload_point(self, item: Tuple[int, ...]) -> None:
        # log.info("%s: OFF @ %s", self.tensor_id, item)
        self.offload_block(item)

    def fetch_point(self, item: Tuple[int, ...]) -> None:
        # log.info("%s: FET @ %s", self.tensor_id, item)
        self.fetch_block(item)

    def deallocate_block(self, item: Tuple[Union[int, slice], ...]) -> None:
        # log.info("%s: DEA @ %s", self.tensor_id, item)
        # Decrement the counter and deallocate the block if needed
        key, index = self.extract_key_and_index(item)

        # NOTE: class assumes a single block dim, but this may not be the case in mem man reqs.
        keys = enum_block_points(key)

        for k in keys:
            # If steps were allowed, add this to (s.stop - s.start + (s.step - 1)) // s.step
            num_points = math.prod((s.stop - s.start) if type(s) is slice else 1 for s in index)
            self._dealloc_counters[k] -= num_points
            # log.info("%s: DEA @ %s - points for block %s before=%s after=%s",
            # self.tensor_id, item, key, self._dealloc_counters[key] + num_points,
            #  self._dealloc_counters[key])
            if self._dealloc_counters[k] == 0:
                # Deallocate the block
                # log.info("%s: DEA @ %s - deallocating block %s. StoreMap %s",
                #  self.tensor_id, item, key, {k: v.shape for k,v in self._storage_map.items()})
                del self._storage_map[k]  # type: ignore
                del self._dealloc_counters[k]  # type: ignore
                # log.info("%s: DEA @ %s - deallocated block %s. StoreMap %s", self.tensor_id, item,
                #  key, {k: v.shape for k,v in self._storage_map.items()})
                # Delete if present in offload and fetch counters
                self._block_points_on_cpu.pop(k, None)

    def offload_block(self, item: Tuple[Union[int, slice], ...]) -> None:
        # log.info("%s: OFF @ %s", self.tensor_id, item)
        # Offload the block when the counter reaches the block size
        key, index = self.extract_key_and_index(item)

        # NOTE: class assumes a single block dim, but this may not be the case in mem man reqs.
        keys = enum_block_points(key)

        for k in keys:
            counters = self._block_points_on_cpu[k]
            counters.update(set(enum_block_points(index)))
            size_after = len(counters)
            if size_after == self.points_per_block:
                old_block = self._storage_map[k]  # type: ignore
                item = self.backend.to_cpu(old_block)
                self._storage_map[k] = item  # type: ignore

    def fetch_block(self, item: Tuple[Union[int, slice], ...]) -> None:
        # log.info("%s: FET @ %s", self.tensor_id, item)
        key, index = self.extract_key_and_index(item)

        # NOTE: class assumes a single block dim, but this may not be the case in mem man reqs.
        keys = enum_block_points(key)

        for k in keys:
            counters = self._block_points_on_cpu[k]
            counters.difference_update(set(enum_block_points(index)))

            # We fetch on the first new access to the block
            item = self._storage_map[k]  # type: ignore
            t = self.backend.to_device(item, self.dev)
            self._storage_map[k] = t  # type: ignore

            # Pre-fetch the next block if it exists
            if len(k) >= 2:
                for i in range(1, self.prefetch_amount + 1):
                    next_key = k[:-1] + (k[-1] + i,)
                    if next_key in self._storage_map:
                        self._storage_map[next_key] = self.backend.to_device(
                            self._storage_map[next_key], self.dev
                        )
                    else:
                        break

    def mem_usage_bytes(self) -> int:
        return self._mem_used_per_block * len(self._storage_map)

    def replace_backing_buffer(self, key: Tuple[int, ...], buffer: BackendTensorT) -> None:
        if key not in self._storage_map:
            raise KeyError(f"Key {key} not found when replacing the block buffer.")
        self._storage_map[key] = buffer

    def get_backing_buffer(self, key: Tuple[int, ...]) -> BackendTensorT:
        """Gets the backing buffer for the given key."""
        return self._get_or_create_buffer(key)

    def extract_write_key_and_indexes(
        self, item: Tuple[int, ...]
    ) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]:
        """Extracts the write key and indexes from the given item."""
        buffer_key: List[int] = []
        intra_buffer_index: List[int] = []

        # Process each dimension in the order they appear in the domain
        for dim_idx, dim_item in enumerate(item):
            if dim_idx in self.block_dim_indices:
                # This is a block dimension
                block_size = self.block_modulos[list(self.block_dim_indices).index(dim_idx)]
                block = dim_item // block_size
                adjusted_index = dim_item % block_size
                buffer_key.append(block)
                intra_buffer_index.append(adjusted_index)
            else:
                # This is a pointwise dimension, it's just part of buffer key
                buffer_key.append(dim_item)

        return tuple(buffer_key), (tuple(intra_buffer_index),)
