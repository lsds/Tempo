from typing import Tuple, Union

from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import BackendTensorT, TensorId
from tempo.core.domain import Domain
from tempo.core.dtype import DataType
from tempo.core.shape import Shape
from tempo.runtime.backends.backend import DLBackend
from tempo.runtime.tensor_store.tensor_store import RuntimeTensor
from tempo.utils import logger

log = logger.get_logger(__name__)


class NoStorageRuntimeTensor(RuntimeTensor[BackendTensorT]):
    def __init__(
        self,
        exec_cfg: ExecutionConfig,
        tensor_id: TensorId,
        shape: Shape,
        dtype: DataType,
        domain: Domain,
    ) -> None:
        super().__init__(tensor_id)
        self.exec_cfg = exec_cfg
        self.tensor_id = tensor_id
        self.shape = shape
        self.dtype = dtype

        self.domain = domain
        self.domain_size = len(domain)

        self.backend = DLBackend.get_backend(exec_cfg.backend)

    def __str__(self) -> str:
        return f"Tensor(tensor_id={self.tensor_id}, shape={self.shape},\
              dtype={self.dtype}, domain={self.domain})"

    def __hash__(self) -> int:
        return hash(self.tensor_id)

    def __eq__(self, other: object) -> bool:
        if type(other) is not NoStorageRuntimeTensor:
            return False
        return self.tensor_id == other.tensor_id

    def __getitem__(  # noqa: C901
        self, item: Tuple[Union[int, slice], ...]
    ) -> BackendTensorT:
        raise NotImplementedError

    def all_int_fast_path(self, item: Tuple[Union[int, slice], ...]) -> BackendTensorT:
        raise NotImplementedError

    def __setitem__(  # noqa: C901
        self, item: Tuple[Union[int, slice], ...], value: BackendTensorT
    ) -> None:
        pass

    def all_int_fast_path_set(self, item: Tuple[int, ...], value: BackendTensorT) -> None:
        pass

    def flush(self) -> None:
        pass

    def deallocate_point(self, item: Tuple[int, ...]) -> None:
        pass

    def offload_point(self, item: Tuple[int, ...]) -> None:
        pass

    def fetch_point(self, item: Tuple[int, ...]) -> None:
        pass

    def deallocate_block(self, block: Tuple[Union[int, slice], ...]) -> None:
        pass

    def offload_block(self, block: Tuple[Union[int, slice], ...]) -> None:
        pass

    def fetch_block(self, block: Tuple[Union[int, slice], ...]) -> None:
        pass

    def mem_usage_bytes(self) -> int:
        return 0
