import itertools
from collections.abc import Iterator, Sequence
from math import prod

from tempo.core.index_expr import Symbol


def make_symbols(symbols: Sequence[str], start_idx: int = 0) -> list[tuple[Symbol, Symbol]]:
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


def enum_block_points(space: tuple[int | slice, ...]) -> Iterator[tuple[int, ...]]:
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


def count_block_points(space: tuple[int | slice, ...]) -> int:
    """Counts the number of points in the given space
    defined by a tuple of int and slice elements."""
    return prod((dim.stop - dim.start) // (dim.step or 1) for dim in space if type(dim) is slice)


def identify_block_from_points(
    points: Sequence[tuple[int, ...]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Given a collection of points,
    returns the start and end of the block that contains all points."""
    # This is sort of just a slow hull?

    zip_points = zip(*points, strict=False)
    start: tuple[int, ...] = tuple(min(p) for p in zip_points)
    end: tuple[int, ...] = tuple(max(p) for p in zip_points)

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


def get_all_subclasses(cls: type) -> list[type]:
    """Recursively get all subclasses of a given class.

    Args:
        cls (Type): The class to get the subclasses of.

    Returns:
        List[Type]: A list of all subclasses of the given class.

    """
    subclasses = cls.__subclasses__()
    all_subclasses = []
    for subclass in subclasses:
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses
