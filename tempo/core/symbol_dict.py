from __future__ import annotations

from collections.abc import MutableMapping  # Import MutableMapping
from typing import Any, Generator, List, Tuple

from tempo.core import index_expr as ie


class SymbolDict(MutableMapping):  # Inherit from MutableMapping
    __slots__ = ["_keys", "_values", "__pool__"]

    def __init__(self, capacity: int = 32):
        self._keys: List[ie.Symbol] = [None] * capacity  # type: ignore
        self._values: List[int] = [None] * capacity  # type: ignore

    def load_keys(self, keys: List[ie.Symbol]) -> None:
        for key in keys:
            self._keys[key.idx] = key

    def has_key(self, key: ie.Symbol) -> bool:
        return self._keys[key.idx] is not None

    def __getitem__(self, key: ie.Symbol) -> int:
        return self._values[key.idx]

    def __setitem__(self, key: ie.Symbol, value: int) -> None:
        self._values[key.idx] = value

    def __delitem__(self, key: ie.Symbol) -> None:
        # self._keys[key.idx] = None  # type: ignore
        self._values[key.idx] = None  # type: ignore

    def __iter__(self) -> Generator[ie.Symbol, None, None]:
        return (key for key in self._keys if key is not None)

    def __len__(self) -> int:
        return sum(1 for key in self._keys if key is not None)

    # TODO change to ItemsView???
    def items(self) -> Generator[Tuple[ie.Symbol, Any], None, None]:  # type:ignore
        for key, val in zip(self._keys, self._values, strict=False):
            if key is not None:
                yield key, val

    def __str__(self) -> str:
        return str(dict(self.items()))

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> SymbolDict:
        new_dict = SymbolDict(len(self._keys))
        new_dict._keys = self._keys.copy()
        new_dict._values = self._values.copy()
        return new_dict

    def keys(self) -> List[ie.Symbol]:  # type: ignore
        return [key for key in self._keys if key is not None]

    # TODO change to ValuesView???
    def values(self) -> List[int]:  # type: ignore
        return [val for key, val in self.items() if key is not None]

    def __getstate__(self) -> dict:
        return {0: self._keys, 1: self._values}

    def __setstate__(self, state: dict) -> None:
        self._keys = state[0]
        self._values = state[1]
