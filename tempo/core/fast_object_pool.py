from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


def custom_del(obj: Any) -> None:
    if getattr(obj, "__pool__", None) is not None:
        obj.__pool__.recycle(obj)  # type: ignore
    else:
        obj.__original_del__()  # type: ignore


def apply_custom_del(cls: type[T]) -> None:
    if hasattr(cls, "__original_del__"):
        return
    # def repr_custom(self):
    #    raise NotImplementedError("ObjectPool does not support __repr__")
    # cls.__repr__ =  repr_custom

    if hasattr(cls, "__del__"):
        cls.__original_del__ = cls.__del__  # type: ignore
    else:
        cls.__original_del__ = lambda obj: None  # type: ignore

    cls.__del__ = custom_del  # type: ignore


class ObjectPool(Generic[T]):
    # __slots__ = ("builder", "recycler", "pool", "max_unused", "reuses", "news")
    __slots__ = ("builder", "pool", "max_unused")

    def __init__(self, builder: Callable[[], T], max_unused: int = sys.maxsize) -> None:
        self.builder: Callable[[], T] = builder
        self.max_unused: int = max_unused
        self.pool: list[T] = []

    def borrow(self) -> T:
        if len(self.pool) == 0:
            obj = self.builder()
            # self.news += 1
        else:
            obj = self.pool.pop()
            # self.reuses += 1

        return obj

    def recycle(self, obj: T) -> None:
        if len(self.pool) < self.max_unused:
            self.pool.append(obj)

    def clear(self) -> None:
        self.pool.clear()
