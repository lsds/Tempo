from typing import Any

import torch

from tempo.core.dl_backend import DLBackend
from tempo.core.fast_object_pool import ObjectPool

BackendDType = Any


class PageLockedBufferManager:
    def __init__(self, backend: type[DLBackend]) -> None:
        self.backend = backend
        self._manager_map: dict[tuple[tuple[int, ...], BackendDType], ObjectPool[torch.Tensor]] = {}
