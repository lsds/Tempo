import itertools
from math import prod
from typing import Iterator, List, Sequence, Tuple, Union

from tempo.core.index_expr import Symbol


def make_symbols(symbols: Sequence[str], start_idx: int = 0) -> List[Tuple[Symbol, Symbol]]:
    """Generates pairs of symbols (e.g. (b, B), (t, T)) for all symbol names
    provided (e.g. "b t").

    Args:
        symbols (str):

    """
    are_all_distinct = len(symbols) == len(set(symbols))
    assert are_all_distinct
    results = []

    for i_, s in enumerate(symbols):
        s = s.strip()
        i = start_idx + i_
        results.append(
            (
                Symbol(s.lower(), idx=i * 2),
                Symbol(s.upper(), is_bound=True, idx=(i * 2) + 1),
            )
        )
    return results


def enum_block_points(space: Tuple[Union[int, slice], ...]) -> Iterator[Tuple[int, ...]]:
    """Generates every point in the given space defined by a tuple of int and slice elements."""
    # Fast path if all elements are integers
    if all(type(dim) is int for dim in space):
        yield space  # type: ignore
        return

    # Slow path: expand slices
    ranges = (
        range(dim.start or 0, dim.stop, dim.step or 1) if type(dim) is slice else (dim,)
        for dim in space
    )
    yield from itertools.product(*ranges)


def count_block_points(space: Tuple[Union[int, slice], ...]) -> int:
    """Counts the number of points in the given space
    defined by a tuple of int and slice elements."""
    return prod((dim.stop - dim.start) // dim.step for dim in space if type(dim) is slice)


def identify_block_from_points(
    points: Sequence[Tuple[int, ...]],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Given a collection of points,
    returns the start and end of the block that contains all points."""
    # This is sort of just a slow hull?

    zip_points = zip(*points, strict=False)
    start: Tuple[int, ...] = tuple(min(p) for p in zip_points)
    end: Tuple[int, ...] = tuple(max(p) for p in zip_points)

    # Assert that every point is covered
    assert all(start[i] <= p[i] <= end[i] for p in points for i in range(len(p))), (
        f"Not all points are covered by the block. Points: {points}, start: {start}, end: {end}"
    )

    return start, end


def bytes_to_human_readable(byte_count: int) -> str:
    """Converts a byte count into a human-readable string.

    Args:
        byte_count (int): The size in bytes.

    Returns:
        str: A human-readable string representing the size.

    """
    if byte_count < 0:
        raise ValueError("Byte count cannot be negative.")

    suffixes = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    index = 0
    size = float(byte_count)

    while size >= 1024 and index < len(suffixes) - 1:
        size /= 1024
        index += 1

    return f"{size:.2f} {suffixes[index]}"
