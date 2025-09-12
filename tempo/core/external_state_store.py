from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

StateStoreKey = str


class ExternalStateStore:
    def __init__(
        self, clean_up_dict: dict[StateStoreKey, Callable[[ExternalStateStore], None]]
    ) -> None:
        self._store: dict[StateStoreKey, Any] = {}
        self._clean_up_dict = clean_up_dict

    def __getitem__(self, key: StateStoreKey) -> Any:
        return self._store[key]

    def __setitem__(self, key: StateStoreKey, value: Any) -> None:
        self._store[key] = value

    def __delitem__(self, key: StateStoreKey) -> None:
        if key in self._clean_up_dict:
            self._clean_up_dict[key](self)
        else:
            raise ValueError(f"Key {key} not found in clean up dict.")
        del self._store[key]

    def get(self, key: StateStoreKey, default: Any = None) -> Any:
        return self._store.get(key, default)

    def __contains__(self, key: StateStoreKey) -> bool:
        return key in self._store

    def keys(self) -> Sequence[StateStoreKey]:
        return list(self._store.keys())

    def values(self) -> Sequence[Any]:
        return list(self._store.values())

    def items(self) -> Sequence[tuple[StateStoreKey, Any]]:
        return list(self._store.items())

    def __iter__(self) -> Iterable[StateStoreKey]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __str__(self) -> str:
        return str(self.keys())

    def global_clean_up(self) -> None:
        for key in self.keys():
            if key in self._clean_up_dict:
                self._clean_up_dict[key](self)
            else:
                raise ValueError(f"Key {key} not found in clean up dict.")
