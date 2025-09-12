from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from tempo.core import index_expr as ie

DEFAULT_PREALLOC_VALUE = float("nan")


@dataclass(frozen=True)
class StorageMethod:
    ...

    def is_full_prealloc(self) -> bool:
        return False


@dataclass(frozen=True)
class DontStore(StorageMethod): ...


@dataclass(frozen=True)
class PointStore(StorageMethod): ...


@dataclass(frozen=True)
class EvalSymbolStore(StorageMethod):
    symbol: ie.Symbol


@dataclass(frozen=True)
class PreallocBufferStore(StorageMethod):
    prealloc_value: Any = field(default=DEFAULT_PREALLOC_VALUE)
    dims_and_base_buffer_sizes: tuple[tuple[ie.Symbol, int], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        assert len(self.dims_and_base_buffer_sizes) == 1, "Only one buffer dimension is supported"

    @property
    def temporal_dim_stored(self) -> ie.Symbol:
        return self.dims_and_base_buffer_sizes[0][0]

    @property
    @abstractmethod
    def buffer_size(self) -> int: ...

    @property
    @abstractmethod
    def num_buffer_writes_needed(self) -> int: ...


@dataclass(frozen=True)
class BlockStore(PreallocBufferStore):
    def is_full_prealloc(self) -> bool:
        return all(b is None for _, b in self.dims_and_base_buffer_sizes)

    @property
    def buffer_size(self) -> int:
        return self.dims_and_base_buffer_sizes[0][1]

    @property
    def num_buffer_writes_needed(self) -> int:
        return 1


@dataclass(frozen=True)
class CircularBufferStore(PreallocBufferStore):
    @property
    def buffer_size(self) -> int:
        return 2 * self.dims_and_base_buffer_sizes[0][1]

    @property
    def num_buffer_writes_needed(self) -> int:
        return 2

    @property
    def window_size(self) -> int:
        return self.dims_and_base_buffer_sizes[0][1]


@dataclass(frozen=True)
class PreallocCircularBufferStore(PreallocBufferStore):
    buffer_multiplier: int = field(default=2)

    @property
    def buffer_size(self) -> int:
        return self.buffer_multiplier * self.dims_and_base_buffer_sizes[0][1]

    @property
    def num_buffer_writes_needed(self) -> int:
        return 1
