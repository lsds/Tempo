from typing import Any, Dict, Tuple, Type

import torch

from tempo.core.fast_object_pool import ObjectPool
from tempo.runtime.backends.backend import DLBackend

BackendDType = Any


class PageLockedBufferManager:
    def __init__(self, backend: Type[DLBackend]) -> None:
        self.backend = backend
        self._manager_map: Dict[Tuple[Tuple[int, ...], BackendDType], ObjectPool[torch.Tensor]] = {}
