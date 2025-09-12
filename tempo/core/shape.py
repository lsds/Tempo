from __future__ import annotations

import math
import typing
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Union

from tempo.core import index_expr as ie
from tempo.utils.common import as_seq


def _try_resolve_index_value(
    dim: ie.IntIndexValue, symbols: Mapping[ie.Symbol, int]
) -> int | ie.IntIndexValue:
    try:
        return dim.evaluate(symbols)
    except KeyError:
        # Not enough info was available to resolve the index value, so we just return the
        # symbolic index value
        return dim


def dim_is_equal(*input_shapes: Shape, dim: int = 0) -> bool:
    dim1 = input_shapes[0].at(dim)
    dim1_raised = ie.lift_to_int_ie(dim1)
    for input_shape in input_shapes[1:]:
        dim2 = input_shape.at(dim)
        dim2_raised = ie.lift_to_int_ie(dim2)
        if not dim1_raised.logical_eq(dim2_raised):
            return False
    return True


def dim_is_one(input_shape: Shape, dim: int = 0) -> bool:
    dim1 = input_shape.at(dim)
    dim2 = 1
    dim1_raised = ie.lift_to_int_ie(dim1)
    dim2_raised = ie.lift_to_int_ie(dim2)

    # NOTE: Use struct_eq first since cheaper
    return dim1_raised.struct_eq(dim2_raised) or dim1_raised.logical_eq(dim2_raised)


def equal_in_all_but_dim(*input_shapes: Shape, dim: int = 0) -> bool:
    # The shapes should be equal on all dimensions except the dimension
    for input_shape in input_shapes:
        for i in range(len(input_shape)):
            if i != dim and not Shape.dim_is_equal(*input_shapes, dim=i):
                return False
    return True


def can_broadcast(*shapes: Shape, log: bool = False) -> bool:
    try:
        broadcast(*shapes)
        return True
    except Exception as e:
        if log:
            print("Cannot broadcast shapes: ", shapes)
            print(e)
        return False


def broadcast(*shapes: ShapeLike) -> Shape:
    # 1.Dimension Matching: Starting from the trailing dimensions, the dimensions of the arrays
    #  should either be the same or one of them should be 1.
    # 2.Unsqueezing Dimensions: If two arrays do not have the same number of dimensions, the shape
    # of the smaller array is 'padded' with ones on its leading (left) side.
    # 3.Size-1 Dimensions Expansion: If the dimensions are not the same, the array with dimensions
    # of size 1 is stretched or 'broadcast' to match the other shape.
    # 4.Result Shape: The resulting array shape is the maximum size along each dimension from the
    # input arrays.

    from tempo.core.global_objects import get_active_dg

    shapes = tuple(Shape.from_(s) for s in shapes)
    shapes = tuple(s.try_resolve(get_active_dg().static_bounds) for s in shapes)

    if any(p == -1 for s in shapes for p in s._shape if isinstance(p, int)):
        raise Exception("Cannot broadcast with -1")
    padded_shapes = unsq_align_shapes_1_pad_left(shapes)
    final_shape = _bcast_compute_final_shape(padded_shapes)
    return Shape(tuple(final_shape))


def unsq_align_shapes_1_pad_right(shapes: tuple[ShapeLike, ...]) -> list[Shape]:
    shapes = tuple(Shape.from_(shape) for shape in shapes)
    target_length = max(len(shape) for shape in shapes)

    return [Shape(shape._shape + (1,) * (target_length - len(shape))) for shape in shapes]


def unsq_align_shapes_1_pad_left(shapes: tuple[ShapeLike, ...]) -> list[Shape]:
    shapes = tuple(Shape.from_(shape) for shape in shapes)

    target_length = max(len(shape) for shape in shapes)
    return [Shape((1,) * (target_length - len(shape)) + shape._shape) for shape in shapes]


def _bcast_compute_final_shape(
    padded_shapes: Sequence[Shape],
) -> Sequence[int | ie.IntIndexValue]:
    final_shape: list[int | ie.IntIndexValue] = []
    # TODO why are we walking backwards?
    for dim in range(1, len(padded_shapes[0]) + 1):
        dimension_values = [shape._shape[-dim] for shape in padded_shapes]
        if any(isinstance(value, ie.IntIndexValue) for value in dimension_values):
            final_shape.append(handle_symbolic_dimensions(dimension_values))
        else:
            final_shape.append(handle_integer_dimensions(dimension_values))  # type: ignore
    return tuple(reversed(final_shape))


def handle_symbolic_dimensions(
    dimension_values: Sequence[int | ie.IntIndexValue],
) -> int | ie.IntIndexValue:
    index_values = [value for value in dimension_values if isinstance(value, ie.IntIndexValue)]
    if not all(value.logical_eq(index_values[0]) for value in index_values):
        raise ValueError(f"Symbolic dimensions mismatch during broadcast: {index_values}")
    ints = [value for value in dimension_values if isinstance(value, int)]
    if not all(int_ == 1 for int_ in ints):
        raise ValueError("Dimension mismatch during broadcast")
    return index_values[0]


def handle_integer_dimensions(dimension_values: list[int]) -> int:
    max_dim = max(dimension_values)
    if not all(dim == max_dim or dim == 1 for dim in dimension_values):
        raise ValueError(f"Dimension mismatch during broadcast, {dimension_values}")
    return max_dim


@dataclass(frozen=True, eq=False)
class Shape:
    _shape: tuple[ie.IntIndexValue | int, ...]

    @staticmethod
    def from_(shape: ShapeLike, simplify: bool = True) -> Shape:
        if shape is None:
            return Shape(())
        if isinstance(shape, Shape):
            return shape.simplify() if simplify else shape
        if type(shape) is int:
            return StaticShape((shape,))
        elif type(shape) is tuple or type(shape) is list:
            if all(isinstance(d, (int, ie.ConstInt)) for d in shape):
                sh_tup = typing.cast(tuple[Union[ie.ConstInt, int], ...], shape)
                return StaticShape(tuple(int(d) for d in sh_tup))

            sh = Shape(tuple(shape))
            if simplify:
                sh = sh.simplify()
            return sh
        raise ValueError(f"Should be unreachable: {shape}, {type(shape)}")

    def __add__(self, other: ShapeLike) -> Shape:
        other = Shape.from_(other)
        return Shape.from_(tuple(self._shape + other._shape))

    def __len__(self) -> int:
        return len(self._shape)

    def __getitem__(self, item: Any) -> Shape:
        return Shape(as_seq(self._shape[item]))  # type: ignore

    def __iter__(self) -> Iterator[ie.IntIndexValue | int]:
        return iter(self._shape)

    def __repr__(self) -> str:
        return f"{self._shape}"

    def __hash__(self) -> int:
        return hash(self._shape)

    def __eq__(self, other: object) -> bool:
        # NOTE: This will throw an exception if not
        try:
            other = Shape.from_(other)  # type: ignore
        except Exception:
            return False

        if len(self) != len(other):
            return False

        for i in range(len(self)):
            if not Shape.dim_is_equal(self, other, dim=i):
                return False
        return True

    def resize_dim(self, dim: int, new_size: ie.IntIndexValueLike) -> Shape:
        return Shape(tuple(new_size if i == dim else size for i, size in enumerate(self._shape)))

    def permute(self, dim_to_move: int, move_to: int) -> Shape:
        # We would just be moving the dimension to the same place - no change to the shape
        if dim_to_move == move_to:
            return self
        # After we remove dim_to_move from the original shape, each index after dim_to_move will
        # now be one less than it was before, so adjust indexes accordingly
        elif dim_to_move > move_to:
            new_index = move_to - 1
        # If dim_to_move < move_to, the index does not need adjusting
        else:
            new_index = move_to

        dimensions = list(self._shape)
        # Remove the value to move from the list of dimensions
        moved_dim_size = dimensions.pop(dim_to_move)

        # Insert the dim at the new index
        dimensions.insert(new_index, moved_dim_size)

        return Shape(tuple(dimensions))

    def is_scalar(self) -> bool:
        return len(self) == 0

    def is_shape_without_extra_dim(self, maybe_shape_with_extra_one: Shape) -> tuple[bool, int]:
        if len(maybe_shape_with_extra_one) != len(self) + 1:
            return False, -1

        # Attempt to find a single extra dimension of 1
        for i in range(len(maybe_shape_with_extra_one)):
            # Check if removing the `i`th element gives the `shape`
            skip_shape = maybe_shape_with_extra_one[:i] + maybe_shape_with_extra_one[i + 1 :]
            if maybe_shape_with_extra_one[i] == 1 and skip_shape == self:
                return True, i

        return False, -1

    def prepend_dim(self, dim: int | ie.IntIndexValue) -> Shape:
        return Shape.from_((dim,) + self._shape)

    def drop_first_dim(self) -> Shape:
        return Shape.from_(self._shape[1:])

    def try_resolve(self, symbols: Mapping[ie.Symbol, int]) -> Shape:
        return Shape.from_(
            tuple(
                d if isinstance(d, int) else _try_resolve_index_value(d, symbols)
                for d in self._shape
            )
        )

    def evaluate(self, symbols: Mapping[ie.Symbol, int]) -> tuple[int, ...]:
        inner = []

        for dim in self._shape:
            if isinstance(dim, ie.IntIndexValue):
                resolved = _try_resolve_index_value(dim, symbols)
                if not isinstance(resolved, int):
                    raise ValueError(f"Expected int, got {resolved}")
                inner.append(resolved)
            else:
                inner.append(dim)
        return tuple(inner)

    def has_negative_dim(self) -> bool:
        return any(isinstance(dim, int) and dim < 0 for dim in self._shape)

    def vars_used(self) -> list[ie.Symbol]:
        return list(
            {v for dim in self._shape if isinstance(dim, ie.IntIndexValue) for v in dim.vars_used()}
        )

    def is_dynamic(self) -> bool:
        return not self.is_static()

    def _count_cond_branches(self) -> int:
        return sum(
            (len(dim.enumerate_all_cond_branches()) if isinstance(dim, ie.IndexExpr) else 0)
            for dim in self._shape
        )

    def simplify(self) -> Shape:
        if self.is_static():
            return self

        import tempo.core.global_objects as glob
        from tempo.utils.isl import simplify_shape

        known_symbols = dict(glob.get_active_dg().static_bounds) if glob.has_active_dg() else None
        simplified = simplify_shape(self, known_symbols=known_symbols)

        return simplified

    # TODO: sympy does not seem powerful enough to do real simplification.
    # def simplify(self) -> Shape:
    #    import tempo.core.global_objects as glob
    #    bounds = glob.get_static_bounds_or_empty()

    #    sh = ie.IndexSequence([ie.lift_to_int_ie(dim) for dim in self._shape])
    #    simplified = ie.simplify(sh, bounds)

    #    return Shape(tuple(simplified.members))

    def is_static(self) -> bool:
        import tempo.core.global_objects as glob

        bounds = glob.get_active_dg().static_bounds if glob.has_active_dg() else {}
        return all(type(ie.lift_to_int_ie(dim).try_eval(bounds)) is int for dim in self._shape)

    def as_static(self) -> StaticShape:
        assert self.is_static()

        import tempo.core.global_objects as glob

        bounds = glob.get_active_dg().static_bounds if glob.has_active_dg() else {}
        return StaticShape(self.evaluate(bounds))  # type: ignore

    def at(self, index: int) -> int | ie.IntIndexValue:
        return self._shape[index]

    def int_at(self, index: int) -> int:
        x = self._shape[index]
        assert type(x) is int, f"Expected int, got {x}"
        return x

    def prod(self) -> int | ie.IntIndexValue:
        # This is complicated, but just aims to compute the product of the shape, pre-computing
        # the static part completely.

        ints = [dim for dim in self._shape if isinstance(dim, int)]
        nonints = [dim for dim in self._shape if not isinstance(dim, int)]
        int_part = math.prod(ints)
        if len(nonints) == 0:
            return int_part
        nonint_part = nonints[0]
        for dim in nonints[1:]:
            nonint_part = nonint_part * dim
        return nonint_part * int_part

    broadcast = staticmethod(broadcast)
    can_broadcast = staticmethod(can_broadcast)
    dim_is_equal = staticmethod(dim_is_equal)
    dim_is_one = staticmethod(dim_is_one)
    equal_in_all_but_dim = staticmethod(equal_in_all_but_dim)

    def flatten(self, keep_first_dim: bool = False) -> Shape:
        if keep_first_dim:
            return Shape(
                (
                    self.at(0),
                    self[1:].prod(),
                )
            )
        else:
            return Shape((self.prod(),))

    @staticmethod
    def scalar() -> Shape:
        return Shape(())


@dataclass(frozen=True, eq=False)
class StaticShape(Shape):
    _shape: tuple[int, ...]

    @staticmethod
    def from_(shape: StaticShapeLike) -> StaticShape:  # type: ignore
        if shape is None:
            return StaticShape(())
        if isinstance(shape, int):
            return StaticShape((shape,))
        elif isinstance(shape, Sequence):
            if all(isinstance(d, (int, ie.ConstInt)) for d in shape):
                sh = typing.cast(tuple[Union[ie.ConstInt, int], ...], shape)
                return StaticShape(
                    tuple((d.const if isinstance(d, ie.ConstInt) else d) for d in sh)
                )
            else:
                raise ValueError(f"Expected tuple of ints, got {shape}")
        return shape

    def __str__(self) -> str:
        return f"s{self._shape}"

    __repr__ = __str__

    # def __eq__(self, other: object) -> bool:
    #    try:
    #        other_ss = StaticShape.from_(other)  # type: ignore
    #        return other_ss._shape == self._shape
    #    except Exception:
    #        return False

    def __getitem__(self, item: Any) -> StaticShape:
        return StaticShape(as_seq(self._shape[item]))  # type: ignore

    def __iter__(self) -> Iterator[int]:
        return iter(self._shape)

    def prepend_dim(self, dim: int | ie.IntIndexValue) -> Shape:
        if isinstance(dim, ie.IntIndexValue):
            return Shape((dim,) + self._shape)
        return StaticShape((dim,) + self._shape)

    def drop_first_dim(self) -> StaticShape:
        return StaticShape(self._shape[1:])

    def has_negative_dim(self) -> bool:
        return any(dim < 0 for dim in self._shape)

    def vars_used(self) -> list[ie.Symbol]:
        return []

    def is_dynamic(self) -> bool:
        return False

    def is_static(self) -> bool:
        return True

    def as_static(self) -> StaticShape:
        return self

    def at(self, index: int) -> int:
        return self._shape[index]

    def int_at(self, index: int) -> int:
        return self._shape[index]

    def prod(self) -> int:
        return math.prod(self._shape)

    def flatten(self, keep_first_dim: bool = False) -> Shape:
        if keep_first_dim:
            return StaticShape(
                (
                    self.at(0),
                    self[1:].prod(),
                )
            )
        else:
            return StaticShape((self.prod(),))


ShapeLike = Union[int, Sequence[Union[ie.IntIndexValue, int]], Shape, StaticShape, None]
StaticShapeLike = Union[int, Sequence[int], StaticShape, None]
