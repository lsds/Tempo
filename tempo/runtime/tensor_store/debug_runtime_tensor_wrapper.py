from typing import Dict, Tuple, Union

from tempo.core.datatypes import BackendTensorT
from tempo.core.utils import enum_block_points
from tempo.runtime.tensor_store.tensor_store import RuntimeTensor
from tempo.utils import logger

log = logger.get_logger(__name__)


class DebugRuntimeTensorWrapper(RuntimeTensor[BackendTensorT]):
    def __init__(
        self,
        rt: RuntimeTensor[BackendTensorT],
    ) -> None:
        super().__init__(rt.tensor_id)
        self.rt = rt
        self._on_dev_dict: Dict[Tuple[Union[int, slice], ...], bool] = dict.fromkeys(
            range(100_000),  # type: ignore
            False,  # type: ignore
        )
        self._on_dev_dict.clear()

    def __str__(self) -> str:
        return f"Wrapped({str(self.rt)})"

    def __hash__(self) -> int:
        return hash(self.rt.tensor_id)

    def mem_usage_bytes(self) -> int:
        mem_inner: int = self.rt.mem_usage_bytes()
        return mem_inner

    def __getitem__(  # noqa: C901
        self, item: Tuple[Union[int, slice], ...]
    ) -> BackendTensorT:
        for p in enum_block_points(item):
            if not self._on_dev_dict[p]:
                raise ValueError(f"GET: Tensor {self.rt.tensor_id} at {p} is not on device")
        return self.rt[item]  # type: ignore

    def all_int_fast_path(self, item: Tuple[Union[int, slice], ...]) -> BackendTensorT:
        if not self._on_dev_dict[item]:
            raise ValueError(f"GET: Tensor {self.rt.tensor_id} at {item} is not on device")
        return self.rt.all_int_fast_path(item)  # type: ignore

    def __setitem__(  # noqa: C901
        self, item: Tuple[Union[int, slice], ...], value: BackendTensorT
    ) -> None:
        raise AssertionError()

    def all_int_fast_path_set(self, item: Tuple[int, ...], value: BackendTensorT) -> None:
        self._on_dev_dict[item] = True
        self.rt.all_int_fast_path_set(item, value)

    def flush(self) -> None:
        self._on_dev_dict.clear()
        self.rt.flush()

    def deallocate_point(self, item: Tuple[int, ...]) -> None:
        if item not in self._on_dev_dict:
            raise ValueError(f"DEA: Tensor {self.rt.tensor_id} at {item} does not exist")
        self._on_dev_dict[item] = None  # type: ignore
        self.rt.deallocate_point(item)

    def offload_point(self, item: Tuple[int, ...]) -> None:
        if item not in self._on_dev_dict:
            raise ValueError(f"OFF: Tensor {self.rt.tensor_id} at {item} does not exist")
        # if not self._on_dev_dict[item]:
        #    log.warning(f"OFF: Tensor {self.rt.tensor_id} at {item} is not on device to offload")
        self._on_dev_dict[item] = False
        self.rt.offload_point(item)

    def fetch_point(self, item: Tuple[int, ...]) -> None:
        if item not in self._on_dev_dict:
            raise ValueError(f"FET: Tensor {self.rt.tensor_id} at {item} does not exist")
        # if self._on_dev_dict[item]:
        #    log.warning(f"FET: Tensor {self.rt.tensor_id} at {item} is already on device")
        self._on_dev_dict[item] = True
        self.rt.fetch_point(item)

    def deallocate_block(self, item: Tuple[Union[int, slice], ...]) -> None:
        for p in enum_block_points(item):
            self.deallocate_point(p)

    def offload_block(self, item: Tuple[Union[int, slice], ...]) -> None:
        for p in enum_block_points(item):
            self.offload_point(p)

    def fetch_block(self, item: Tuple[Union[int, slice], ...]) -> None:
        for p in enum_block_points(item):
            self.fetch_point(p)
