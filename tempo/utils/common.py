import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

T = TypeVar("T")
K = TypeVar("K")


def nanos() -> int:
    return time.perf_counter_ns()


def nanos_to_s(x: int) -> float:
    return round(x / 1e9, 2)


def nanos_to_ms(x: int) -> int:
    return int(x / 1e6)


def get_all_subclasses(cls: Type[Any]) -> List[Type[Any]]:
    all_subclasses = [cls]

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


def partition(seq: Sequence[T], key: Callable[[T], K]) -> Dict[K, List[T]]:
    # https://stackoverflow.com/questions/12720151/simple-way-to-group-items-into-buckets
    d: Dict[K, List[T]] = defaultdict(list)
    for x in seq:
        d[key(x)].append(x)
    return d


def argsort(x: Tuple[T]) -> Tuple[T]:
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


def as_seq(x: Optional[Union[T, Sequence[T]]]) -> Sequence[T]:
    if x is None:
        return ()
    if not isinstance(x, Sequence):
        return (x,)
    return x


@dataclass
class Timer:
    """Context manager for timing code execution."""

    start_ns: int = 0
    end_ns: int = 0

    def __enter__(self) -> "Timer":
        self.start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end_ns = time.perf_counter_ns()

    @property
    def elapsed_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def elapsed_ms(self) -> float:
        return round(self.elapsed_ns / 1e6, 2)

    @property
    def elapsed_s(self) -> float:
        return round(self.elapsed_ns / 1e9, 2)
