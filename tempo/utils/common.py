import os
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    TypeVar,
)

T = TypeVar("T")
K = TypeVar("K")


def nanos() -> int:
    return time.perf_counter_ns()


def nanos_to_s(x: int) -> float:
    return round(x / 1e9, 2)


def nanos_to_ms(x: int) -> int:
    return int(x / 1e6)


def get_all_subclasses(cls: type[Any]) -> list[type[Any]]:
    all_subclasses = [cls]

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


def partition(seq: Sequence[T], key: Callable[[T], K]) -> dict[K, list[T]]:
    # https://stackoverflow.com/questions/12720151/simple-way-to-group-items-into-buckets
    d: dict[K, list[T]] = defaultdict(list)
    for x in seq:
        d[key(x)].append(x)
    return d


def argsort(x: tuple[T]) -> tuple[T]:
    return tuple(sorted(range(len(x)), key=x.__getitem__))  # type: ignore


def get_dir_path() -> str:
    """Gets the path to the script being executed. Also works for jupyter notebooks."""
    return os.path.dirname(os.path.realpath("__file__" if "__file__" in locals() else os.getcwd()))


def get_date_str() -> str:
    return datetime.now().strftime("%m-%d-%Y-%H-%M")


def get_dated_dir_path() -> Path:
    return Path(get_dir_path()) / get_date_str()


def get_log_path() -> Path:
    return Path(get_dir_path()) / get_date_str()


def as_seq(x: T | Sequence[T] | None) -> Sequence[T]:
    if x is None:
        return ()
    if not isinstance(x, Sequence):
        return (x,)
    return x


@dataclass
class Timer:
    """Context manager for timing code execution."""

    start_ns: int = 0
    total_elapsed_ns: int = -1

    def __enter__(self) -> "Timer":
        self.start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *args: Any) -> None:
        self.total_elapsed_ns += time.perf_counter_ns() - self.start_ns

    @property
    def elapsed_ns(self) -> int:
        return self.total_elapsed_ns + 1  # NOTE: To offset the initial -1 value

    @property
    def elapsed_ms(self) -> float:
        return round(self.elapsed_ns / 1e6, 2)

    @property
    def elapsed_s(self) -> float:
        return round(self.elapsed_ns / 1e9, 2)
