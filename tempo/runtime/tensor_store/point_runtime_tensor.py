from typing import Dict, Tuple, Union

from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import BackendTensorT, TensorId
from tempo.core.device import DeviceGroup
from tempo.core.domain import Domain
from tempo.core.dtype import DataType
from tempo.core.shape import Shape
from tempo.core.utils import enum_block_points
from tempo.runtime.backends.backend import DLBackend
from tempo.runtime.tensor_store.tensor_store import RuntimeTensor
from tempo.utils import logger

log = logger.get_logger(__name__)


class PointRuntimeTensor(RuntimeTensor[BackendTensorT]):
    def __init__(
        self,
        exec_cfg: ExecutionConfig,
        tensor_id: TensorId,
        shape: Shape,
        dtype: DataType,
        dev: DeviceGroup,
        domain: Domain,
        # indexing_exprs: Sequence[ie.IndexSequence] = ()
    ) -> None:
        super().__init__(tensor_id)
        self.exec_cfg = exec_cfg
        self.shape = shape
        self.dtype = dtype

        self.domain = domain
        self.domain_size = len(domain)

        self.backend = DLBackend.get_backend(exec_cfg.backend)
        self.prefetch_amount = exec_cfg.runtime_tensor_prefetch_amount_point

        # Pre-allocate a large  dictionary to avoid resizing at runtime
        self._data_dict: Dict[Tuple[Union[int, slice], ...], BackendTensorT] = dict.fromkeys(
            range(100_000)  # type: ignore
        )
        self._data_dict.clear()

        self.dev = self.backend.to_backend_device_obj(dev)

        if self.shape.is_static():
            self.shape_prod: int = self.shape.as_static().prod()
        else:
            # self.shape_prod = self.shape.prod().evaluate(  # type: ignore
            #    {k: self.exec_cfg.dim_size_upper_bound for k in self.domain.parameters}
            # )
            self.shape_prod = 0  # TODO

        # static_slice_expr_members = [
        #    m for e in indexing_exprs for m in e if (not m.is_point()) and not m.is_constant()
        # ]
        # is_full_static = self.shape.is_static() and all(
        #    (m.is_point() or (m.is_slice() and m.is_constant())) for e in indexing_exprs for m in e
        # )
        # if is_full_static:
        #    self.stack_impl = self.backend.get_stack_fn(example_tensors)
        # else:
        self.stack_impl = self.backend.stack

    def __str__(self) -> str:
        return f"Tensor(tensor_id={self.tensor_id}, shape={self.shape},\
              dtype={self.dtype}, domain={self.domain})"

    def __hash__(self) -> int:
        return hash(self.tensor_id)

    def __eq__(self, other: object) -> bool:
        if type(other) is not PointRuntimeTensor:
            return False
        is_eq: bool = self.tensor_id == other.tensor_id
        return is_eq

    def mem_usage_bytes(self) -> int:
        num_tensors = len(self._data_dict)
        res: int = num_tensors * self.dtype.repr_bytes * self.shape_prod
        return res

    # TODO implement iterative version?
    def _concat_tensors(  # noqa: C901
        self,
        item: Tuple[Union[int, slice], ...],
        item_len: int,
        depth: int,
        index: Tuple[int, ...] = (),
    ) -> BackendTensorT:
        if depth == item_len:
            return self._data_dict[index]

        current_index = item[depth]

        if current_index.__class__ is slice:
            slice_range = range(
                int(current_index.start),  # type: ignore
                int(current_index.stop),  # type: ignore
                current_index.step if current_index.step is not None else 1,  # type: ignore
            )

            res: BackendTensorT = self.stack_impl(
                [
                    self._concat_tensors(item, item_len, depth + 1, index + (idx,))
                    # self._data_dict[index + (idx,)] #TODO slightly faster but less general
                    for idx in slice_range
                ],
            )
            return res
        else:
            return self._concat_tensors(
                item,
                item_len,
                depth + 1,
                index + (current_index,),  # type: ignore
            )

    def __getitem__(  # noqa: C901
        self, item: Tuple[Union[int, slice], ...]
    ) -> BackendTensorT:
        # log.info("%s: GET @ %s", self.tensor_id, item)
        return self._concat_tensors(item, len(item), 0)

    def all_int_fast_path(self, item: Tuple[Union[int, slice], ...]) -> BackendTensorT:
        # log.info("%s: GET @ %s", self.tensor_id, item)
        return self._data_dict[item]

    def split_and_set(
        self,
        tensor: BackendTensorT,
        item: Tuple[Union[int, slice], ...],
        item_len: int,
        depth: int,
        index: Tuple[int, ...] = (),
    ) -> None:
        if depth == item_len:
            self._data_dict[index] = tensor
            return

        current_index = item[depth]

        if type(current_index) is slice:
            slice_range = range(*current_index.indices(tensor.shape[depth]))
            tensors = self.backend.unbind(tensor, 0)
            for slice_idx, t in zip(slice_range, tensors, strict=False):
                self.split_and_set(
                    t,
                    item,
                    item_len,
                    depth + 1,
                    index + (slice_idx,),
                )
        else:
            self.split_and_set(
                tensor,
                item,
                item_len,
                depth + 1,
                index + (current_index,),
            )

    def __setitem__(  # noqa: C901
        self, item: Tuple[Union[int, slice], ...], value: BackendTensorT
    ) -> None:
        # log.info("%s: SET @ %s", self.tensor_id, item)
        # assert len(item) == self.domain_size
        self.split_and_set(value, item, len(item), 0)

        # TODO we may need to later go back to the split_and_set method,
        # but for now we will use the below method to avoid the overhead, since we never write
        # slices to the tensor store.
        # log.info(f"SET {self.tensor_id}[{item}]")

    def all_int_fast_path_set(self, item: Tuple[int, ...], value: BackendTensorT) -> None:
        # log.info("%s: SET @ %s", self.tensor_id, item)
        self._data_dict[item] = value

    def flush(self) -> None:
        """This method clears the tensor of any remaining data."""
        self._data_dict.clear()

    def deallocate_point(self, item: Tuple[int, ...]) -> None:
        # log.info("%s: DEA @ %s", self.tensor_id, item)
        # log.info("deallocate called for %s at %s", self.tensor_id, item)
        # del self._data_dict[item]

        # item = self._data_dict.pop(item)  # type: ignore
        # if self.pool:
        #    self.pool.recycle(item)  # type: ignore

        self._data_dict[item] = None  # type: ignore

    def offload_point(self, item: Tuple[int, ...]) -> None:
        # log.info("%s: OFF @ %s", self.tensor_id, item)
        self._data_dict[item] = self.backend.to_cpu(self._data_dict[item])

    def fetch_point(self, item: Tuple[int, ...]) -> None:
        # log.info("%s: FET @ %s", self.tensor_id, item)
        self._data_dict[item] = self.backend.to_device(self._data_dict[item], self.dev)

        # Pre-fetch the next point if it exists
        if len(item) >= 2:
            # TODO: This presumes that we are scanning in order, which is not always the case.
            # TODO: This should really be part of scheduling.
            for i in range(1, self.prefetch_amount + 1):
                item_next = item[:-1] + (item[-1] + i,)
                if item_next in self._data_dict:
                    self._data_dict[item_next] = self.backend.to_device(
                        self._data_dict[item_next], self.dev
                    )
                else:
                    break

    def deallocate_block(self, item: Tuple[Union[int, slice], ...]) -> None:
        # log.info("%s: DEA @ %s", self.tensor_id, item)
        for p in enum_block_points(item):
            # self.deallocate(p)
            self._data_dict[p] = None  # type: ignore

    def offload_block(self, item: Tuple[Union[int, slice], ...]) -> None:
        # log.info("%s: OFF @ %s", self.tensor_id, item)
        for p in enum_block_points(item):
            self._data_dict[p] = self.backend.to_cpu(self._data_dict[p])

    def fetch_block(self, item: Tuple[Union[int, slice], ...]) -> None:
        # log.info("%s: FET @ %s", self.tensor_id, item)
        for p in enum_block_points(item):
            self._data_dict[p] = self.backend.to_device(self._data_dict[p], self.dev)
