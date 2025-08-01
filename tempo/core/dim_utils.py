import builtins
from typing import Any, Tuple, Union

from tempo.core import index_expr as ie
from tempo.core.datatypes import DIM_TYPE
from tempo.core.shape import Shape


def normalize_negative_dim(dim: int, shape: Shape, unsq: bool = False) -> int:
    if isinstance(dim, int) and dim < 0:
        dim = builtins.max(len(shape) + dim + (1 if unsq else 0), 0)
    return dim


def normalize_negative_dim_tuple(dim: Tuple[int, ...], shape: Shape) -> Tuple[int, ...]:
    return tuple(d + len(shape) if isinstance(d, int) and d < 0 else d for d in dim)


def normalize_negative_dims(dim: DIM_TYPE, shape: Shape) -> DIM_TYPE:
    if isinstance(dim, int):
        return normalize_negative_dim(dim, shape)
    elif isinstance(dim, tuple):
        return normalize_negative_dim_tuple(dim, shape)
    elif dim is None:
        return None
    else:
        raise TypeError(f"Expected None, int or tuple of ints, got {type(dim)}")


def normalize_dims(dim: DIM_TYPE, shape: Shape) -> Tuple[int, ...]:
    dims = normalize_negative_dims(dim, shape)
    if isinstance(dims, int):
        return (dims,)
    elif isinstance(dims, tuple):
        return dims
    elif dims is None:
        return tuple(range(len(shape)))
    else:
        raise TypeError(f"Expected None, int or tuple of ints, got {type(dim)}")


def normalize_ellipsis_indexes(
    item: Tuple[Any, ...], default_idxs: Tuple[ie.IndexAtom, ...]
) -> Tuple[Any, ...]:
    """If there is an ellipsis in the index, expand any missing dims with default_idxs.
    Dims to the left of the ellipsis are assumed to be correctly positioned, while
    dims to the right of the ellipsis are assumed to be the last.
    """
    encountered_ellipsis = False
    num_unspecified_dimensions = len(default_idxs) - (len(item) - 1)
    for i, subitem in enumerate(item):
        if isinstance(subitem, type(Ellipsis)):
            if encountered_ellipsis:
                raise ValueError("Only one ellipsis allowed")
            encountered_ellipsis = True
            item2 = (
                *item[:i],
                *default_idxs[i : i + num_unspecified_dimensions],
                *item[i + 1 :],
            )
            item = item2
    return item


def normalize_empty_colon_indexes(
    item: Tuple[Any, ...], default_idxs: Tuple[ie.IndexAtom, ...]
) -> Tuple[Any, ...]:
    # NOTE: Handles ,:, indexes
    item2 = []
    for i, idx in enumerate(item):
        # TODO will this is None comparison work given that IndexValues do not have real __eq__?
        if isinstance(idx, slice) and idx.start is None and idx.stop is None:
            item2.append(default_idxs[i])
        else:
            item2.append(idx)
    return tuple(item2)


def normalize_negative_indexes(
    item: Tuple[Any, ...], dim_sizes: Tuple[ie.IntIndexValue, ...]
) -> Tuple[Any, ...]:
    item2 = []
    for i, idx in enumerate(item):
        if isinstance(idx, int) and idx < 0:
            item2.append(dim_sizes[i] + idx)
        else:
            item2.append(idx)
    return tuple(item2)


def normalize_indexes(
    item: Tuple[Any, ...],
    default_idxs: Tuple[ie.IndexAtom, ...],
    dim_sizes: Tuple[ie.IntIndexValue, ...],
) -> Tuple[Any, ...]:
    item = normalize_ellipsis_indexes(item, default_idxs)
    item = normalize_empty_colon_indexes(item, default_idxs)
    item = normalize_negative_indexes(item, dim_sizes)
    return item


def normalize_slice_indexes(
    padding: Tuple[Union[Tuple[int, int], int], ...],
    dim_sizes: Tuple[Union[ie.IntIndexValue, int], ...],
) -> Tuple[Any, ...]:
    padding = list(padding)
    padding += [(0, 0)] * (len(dim_sizes) - len(padding))

    new_pad = []
    for pad in padding:
        if isinstance(pad, int):
            new_pad.append((pad, pad))
        elif (
            isinstance(pad, tuple)
            and len(pad) == 2
            and isinstance(pad[0], int)
            and isinstance(pad[1], int)
        ):
            new_pad.append(pad)
        else:
            raise ValueError(f"Invalid padding: {pad}")
    return tuple(new_pad)
