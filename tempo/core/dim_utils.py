import builtins
from typing import Any

from tempo.core import index_expr as ie
from tempo.core.datatypes import DIM_TYPE
from tempo.core.shape import Shape


def normalize_negative_dim(dim: int, shape: Shape, allow_end: bool = False) -> int:
    """
    Normalize possibly-negative `dim`.

    - If allow_end=False: valid dims are [-rank, rank-1].
    - If allow_end=True:  valid dims are [-(rank+1), rank]. This is used for unsqueeze-like ops.
    """
    rank = len(shape)
    lower = -(rank + (1 if allow_end else 0))
    upper = rank if allow_end else rank - 1

    if dim < lower or dim > upper:
        raise IndexError(f"dim {dim} out of range for rank {rank} (allowed range {lower}..{upper})")

    if dim < 0:
        dim += rank + (1 if allow_end else 0)

    return dim


def normalize_negative_dim_tuple(dim: tuple[int, ...], shape: Shape) -> tuple[int, ...]:
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


def normalize_dims(dim: DIM_TYPE, shape: Shape) -> tuple[int, ...]:
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
    item: tuple[Any, ...], default_idxs: tuple[ie.IndexAtom, ...]
) -> tuple[Any, ...]:
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
    item: tuple[Any, ...], default_idxs: tuple[ie.IndexAtom, ...]
) -> tuple[Any, ...]:
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
    item: tuple[Any, ...], dim_sizes: tuple[ie.IntIndexValue, ...]
) -> tuple[Any, ...]:
    item2 = []
    for i, idx in enumerate(item):
        if isinstance(idx, int) and idx < 0:
            item2.append(dim_sizes[i] + idx)
        else:
            item2.append(idx)
    return tuple(item2)


def normalize_indexes(
    item: tuple[Any, ...],
    default_idxs: tuple[ie.IndexAtom, ...],
    dim_sizes: tuple[ie.IntIndexValue, ...],
) -> tuple[Any, ...]:
    item = normalize_ellipsis_indexes(item, default_idxs)
    item = normalize_empty_colon_indexes(item, default_idxs)
    item = normalize_negative_indexes(item, dim_sizes)
    return item


def normalize_slice_indexes(
    padding: tuple[tuple[int, int] | int, ...],
    dim_sizes: tuple[ie.IntIndexValue | int, ...],
) -> tuple[Any, ...]:
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


def normalize_neg_1s_in_shape_reshape(curr_shape: Shape, requested_shape: Shape) -> Shape:
    for s in requested_shape:
        if isinstance(s, int) and s < 0:
            assert s == -1, f"Only negative dimension allowed is -1, got {s}"

    num_neg_dims = builtins.sum(1 for s in requested_shape if isinstance(s, int) and s < 0)
    if num_neg_dims == 0:
        return requested_shape
    if num_neg_dims > 1:
        raise ValueError(
            f"Cannot have more than one negative dimension in reshape, got {requested_shape}"
        )

    total_elems = curr_shape.prod()
    prod_of_known_dims = Shape(
        tuple(
            s for s in requested_shape if (isinstance(s, int) and s >= 0) or not isinstance(s, int)
        )
    ).prod()

    if isinstance(total_elems, int) and isinstance(prod_of_known_dims, int):
        if total_elems % prod_of_known_dims != 0:
            raise ValueError(
                f"Reshape cannot infer the missing dimension for shape {requested_shape} \
                and tensor shape {curr_shape}"
            )
    neg_one_val = total_elems // prod_of_known_dims
    return Shape(tuple(neg_one_val if ie.struct_eq(s, -1) else s for s in requested_shape))


def normalize_neg_1s_in_shape_expand(curr_shape: Shape, sizes: Shape) -> Shape:
    # TODO: for every -1, should just match the curr_shape.
    assert len(curr_shape) == len(sizes), (
        f"{curr_shape=} and {sizes=} must have the same length for expand"
    )
    res_shape = []
    for s1, s2 in zip(curr_shape, sizes, strict=True):
        if ie.struct_eq(s2, -1):
            res_shape.append(s1)
        else:
            res_shape.append(s2)
    return Shape.from_(tuple(res_shape))
