from __future__ import annotations

import functools
import math
from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np

from tempo.core import index_expr as ie
from tempo.core.dataflow_graph import DataflowGraph
from tempo.core.dtype import DataType, dtypes
from tempo.core.op_tags import STATIFY_PAD_ID_TAG
from tempo.core.shape import Shape
from tempo.core.tensor_op import TensorOp
from tempo.core.thunk_udf import UserDefinedThunkDesc
from tempo.utils import logger

log = logger.get_logger(__name__)


@dataclass(frozen=True, eq=False)
class SortOp(TensorOp):
    dim: int
    stable: bool
    descending: bool

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        return (input_shapes[0], input_shapes[0])

    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return (input_dtypes[0], dtypes.default_int)

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 2


@dataclass(frozen=True, eq=False)
class UDFOp(TensorOp):
    desc: UserDefinedThunkDesc = field(repr=False)

    def __str__(self) -> str:
        return (
            f"UDFOp({self.op_id}, is_stateful={self.is_stateful},"
            + f" num_inputs={self.num_inputs}, num_outputs={self.num_outputs}"
            + f", name={self.desc.thunk_name})"
            if self.desc.thunk_name
            else ")"
        )

    def is_udf(self) -> bool:
        return True

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        return self.desc.infer_output_shapes(input_shapes)  # type: ignore

    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return self.desc.infer_output_dtypes(input_dtypes)  # type: ignore

    @property
    def is_stateful(self) -> bool:
        return self.desc.is_stateful  # type: ignore

    @property
    def num_inputs(self) -> int:
        return self.desc.num_inputs  # type: ignore

    @property
    def num_outputs(self) -> int:
        return self.desc.num_outputs  # type: ignore

    def equivalent(self, other: Any) -> bool:  # noqa: C901
        # NOTE: A UDF is never equivalent to another UDF, since we can't know what it does
        return False


@dataclass(frozen=True, eq=False)
class GatherOp(TensorOp):
    dim: int

    # def __post_init__(self) -> None:
    #    raise NotImplementedError("Gather is not yet implemented")
    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        return (input_shapes[1],)

    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return (input_dtypes[0],)

    @property
    def num_inputs(self) -> int:
        return 2

    @property
    def num_outputs(self) -> int:
        return 1


@dataclass(frozen=True, eq=False)
class MarkerOp(TensorOp):
    """A marker op is used purely for control purposes. For example, a barrier can be implemented
    by inserting a marker op with control-dependencies to every other op in the graph.
    """

    marker_name: str = field(default="anonymous marker")

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        return ()

    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return ()

    @property
    def num_inputs(self) -> int:
        return 0

    @property
    def num_outputs(self) -> int:
        return 0


@dataclass(frozen=True, eq=False)
class ScatterAddOp(TensorOp):
    dim: int

    # def __post_init__(self) -> None:
    #    raise NotImplementedError("ScatterAddOp is not yet implemented")

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        return (input_shapes[0],)

    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return (input_dtypes[0],)

    @property
    def num_inputs(self) -> int:
        return 3

    @property
    def num_outputs(self) -> int:
        return 1


@dataclass(frozen=True, eq=False)
class ExecDataflowOp(TensorOp):
    dataflow: DataflowGraph = field(repr=False)

    def __str__(self) -> str:
        num_nodes = len(list(self.dataflow.nodes))
        return f"ExecuteDataflowSubgraphOp({self.op_id}, domain={self.domain}, num_ops={num_nodes})"

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        return self.dataflow.infer_output_shapes(input_shapes)  # type: ignore

    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return self.dataflow.infer_output_dtypes(input_dtypes)  # type: ignore

    @property
    def num_inputs(self) -> int:
        return self.dataflow.num_inputs  # type: ignore

    @property
    def num_outputs(self) -> int:
        return self.dataflow.num_outputs  # type: ignore

    def is_static(self) -> bool:
        res = all(op.is_static() for op in self.dataflow.subgraph.nodes)
        # assert res, f"ExecDataflowOp {self.op_id} is not static"
        return res


# ======================================== SOURCE OPS ========================================
@dataclass(frozen=True, eq=False)
class SourceOp(TensorOp, ABC):
    shape: Shape
    dtype: DataType

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        return (self.shape,)

    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return (self.dtype,)

    @property
    def num_inputs(self) -> int:
        return 0

    @property
    def num_outputs(self) -> int:
        return 1

    def is_static(self) -> bool:
        return self.shape.is_static()


@dataclass(frozen=True, eq=False)
class MergeOp(TensorOp):
    """Each cond is checked in order. The first one that is true is the branch that is taken.
    The executor has to handle MergeOps by evaluating the conditions in order and then
    fetching the input from the branch that is taken.
    Outputted shapes and dtypes must match.
    """

    # NOTE: These are needed to infer the output shapes and dtypes
    shape: Shape
    dtype: DataType
    # NOTE: This is used to track the number of inputs to the merge op. We need to box it
    # because it's a mutable value.
    num_inputs_: list[int] = field(default_factory=lambda: [0])

    def increment_num_inputs(self) -> None:
        self.num_inputs_[0] += 1

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        return (self.shape,)

    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return (self.dtype,)

    @property
    def num_inputs(self) -> int:
        return self.num_inputs_[0]

    @property
    def num_outputs(self) -> int:
        return 1


@dataclass(frozen=True, eq=False)
class RandOp(SourceOp):
    def __post_init__(self) -> None:
        if self.dtype != dtypes.float32:
            raise ValueError("RandOp only supports float32 dtype")

    def equivalent(self, other: Any) -> bool:  # noqa: C901
        # NOTE: A rand op is never equivalent to any other op, because it is non-deterministic.
        return False

    def is_static(self) -> bool:
        return not self.shape.is_dynamic()


@dataclass(frozen=True, eq=False)
class ConstOp(SourceOp):
    value: np.ndarray = field(repr=False)
    is_uniform: bool = False

    def __post_init__(self) -> None:
        if self.value.dtype == np.object_:
            raise ValueError("Object dtype not supported")

        if not self.is_uniform and self.value.size == 1:
            raise ValueError("Non-uniform const with size 1 is impossible")

    @property
    def uniform_value(self) -> int | float | bool:
        assert self.is_uniform
        return self.value.item()  # type: ignore

    @functools.cached_property
    def is_int_arange(self) -> bool:
        if (
            dtypes.is_integer(self.dtype)
            and len(self.value.shape) == 1
            and self.value.ravel()[0] == 0
            and np.all(np.diff(self.value) == 1)
        ):
            return True
        return False

    @functools.lru_cache(maxsize=4)
    def is_uniform_on_dim(self, dim: int) -> bool:
        if self.is_uniform:
            return True
        diff = np.diff(self.value, axis=dim)
        is_uniform_across_dim: bool = np.sum(diff) == 0  # type: ignore
        return is_uniform_across_dim

    def __str__(self) -> str:
        str_ = f"ConstOp({self.op_id}, shape={self.shape}, dtype={self.dtype}"
        str_ += f", uniform={self.is_uniform}"
        if self.is_uniform:
            str_ += f", value[*0]={self.value.ravel()[0]}"
        elif self.value.size < 10:
            str_ += f", value={self.value}"
        str_ += ")"
        return str_


@dataclass(frozen=True, eq=False)
class EvalSymbolOp(SourceOp):
    """This op is only ever used to compute static bounds or variables.
    Dynamic bounds are computed in graph.
    """

    symbol: ie.Symbol = None  # type: ignore
    shape: Shape = Shape(())
    dtype: DataType = field(default_factory=lambda: dtypes.default_int)

    def __post_init__(self) -> None:
        assert self.shape.is_scalar()
        assert self.symbol is not None

    def vars_used(self) -> set[ie.Symbol]:
        return {self.symbol}

    def is_static(self) -> bool:
        return False


# ======================================== END SOURCE OPS ========================================

# ======================================== START SINK OPS ========================================


@dataclass(frozen=True, eq=False)
class SinkOp(TensorOp, ABC):
    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        return ()

    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return ()

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 0


# ======================================= END SINK OPS ========================================


@dataclass(frozen=True, eq=False)
class MovementOp(TensorOp, ABC):
    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return (dtypes.upcast(*input_dtypes),)

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 1

    def dims_affected(self, input_shapes: Sequence[Shape]) -> Sequence[int]:
        raise NotImplementedError("dims_affected not implemented")


@dataclass(frozen=True, eq=False)
class SplitOp(MovementOp):
    dim: int
    num_splits: int

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        in_shape = input_shapes[0]

        assert len(in_shape._shape) > self.dim

        assert in_shape._shape[self.dim] % self.num_splits == 0

        split_size = in_shape._shape[self.dim] // self.num_splits

        return tuple(
            Shape.from_(
                in_shape._shape[: self.dim] + (split_size,) + (in_shape._shape[self.dim + 1 :]),
            )
            for _ in range(self.num_splits)
        )

    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return tuple(input_dtypes[0] for _ in range(self.num_splits))

    def dims_affected(self, input_shapes: Sequence[Shape]) -> Sequence[int]:
        return (self.dim,)

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return self.num_splits


@dataclass(frozen=True, eq=False)
class IndexSliceOp(MovementOp):
    dim: int

    length: ie.IntIndexValueLike

    def vars_used(self) -> set[ie.Symbol]:
        return ie.lift_to_int_ie(self.length).vars_used()

    def is_static(self) -> bool:
        length = self.length
        if not isinstance(length, int):
            from tempo.core.global_objects import get_static_bounds_or_empty

            length = length.partial_eval(get_static_bounds_or_empty())
            if isinstance(length, ie.ConstInt):
                length = length.const

        return isinstance(length, int)

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        src_shape = input_shapes[0]
        assert input_shapes[1].is_scalar()
        s = list(src_shape._shape)
        s[self.dim] = self.length

        return (Shape.from_(tuple(s)),)

    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return (input_dtypes[0],)

    def dims_affected(self, input_shapes: Sequence[Shape]) -> Sequence[int]:
        return (self.dim,)

    @property
    def num_inputs(self) -> int:
        return 2

    @property
    def num_outputs(self) -> int:
        return 1


class PadMode(IntEnum):
    CONSTANT = 0  # Fill with constant value
    REFLECT = 1  # Reflect the tensor across the padding #TODO improve doc
    REPLICATE = 2  # Replicate the tensor across the padding #TODO improve doc
    ANY = 3  # It does not matter what the padding is: e.g. will gather across non-padded values


@dataclass(frozen=True, eq=False)
class PadOp(MovementOp):
    """Pads the input tensor with the given padding description

    Args:
        pad_width: Tuple[int, int] - The padding width for dim (start, end).
        dim: int - The dimension to pad
        mode: PadMode = PadMode.CONSTANT - The mode to pad the tensor with
        constant_value: Optional[float] = None - The value to pad the tensor for PadMode.CONSTANT
    Returns:
        _type_: _description_
    """

    padding: tuple[ie.IntIndexValueLike, ie.IntIndexValueLike]
    dim: int
    mode: PadMode = PadMode.CONSTANT
    value: float | None = None

    def __post_init__(self) -> None:
        assert self.mode in [PadMode.CONSTANT, PadMode.REFLECT, PadMode.REPLICATE, PadMode.ANY]

    def vars_used(self) -> set[ie.Symbol]:
        return (
            ie.lift_to_int_ie(self.padding[0]).vars_used()
            | ie.lift_to_int_ie(self.padding[1]).vars_used()
        )

    def is_mask_pad(self) -> bool:
        # TODO: could also check that value is a pad_value: int 0, float 0.0, bool false

        has_tag = self.tags.get(STATIFY_PAD_ID_TAG) is not None
        return has_tag
        # return self.mode == PadMode.CONSTANT and not self.is_static()

    def is_nan_pad(self) -> bool:
        return self.mode == PadMode.CONSTANT and math.isnan(self.value or 0.0)

    def is_any_pad(self) -> bool:
        return self.mode == PadMode.ANY

    def equivalent(self, other: Any) -> bool:  # noqa: C901
        if not isinstance(other, PadOp):
            return False
        value = self.value or 0.0
        other_value = other.value or 0.0
        return (
            self.dim == other.dim
            and self.mode == other.mode
            and (
                value == other_value
                or (math.isnan(value) and math.isnan(other_value))
                or (math.isinf(value) and math.isinf(other_value) and value > 0 and other_value > 0)
                or (math.isinf(value) and math.isinf(other_value) and value < 0 and other_value < 0)
            )
            and ie.lift_to_int_ie(self.padding[0]).struct_eq(ie.lift_to_int_ie(other.padding[0]))
            and ie.lift_to_int_ie(self.padding[1]).struct_eq(ie.lift_to_int_ie(other.padding[1]))
        )

    def is_static(self) -> bool:
        pad0: ie.IntIndexValueLike = self.padding[0]
        pad1: ie.IntIndexValueLike = self.padding[1]
        from tempo.core.global_objects import get_static_bounds_or_empty

        if not isinstance(pad0, int):
            pad0 = pad0.partial_eval(get_static_bounds_or_empty())
            if isinstance(pad0, ie.ConstInt):
                pad0 = pad0.const
        if not isinstance(pad1, int):
            pad1 = pad1.partial_eval(get_static_bounds_or_empty())
            if isinstance(pad1, ie.ConstInt):
                pad1 = pad1.const

        return isinstance(pad0, int) and isinstance(pad1, int)

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        assert len(input_shapes) == 1

        in_shape = input_shapes[0]

        assert len(in_shape._shape) > self.dim

        out_shape = list(in_shape._shape)
        res_shape = self.padding[0] + out_shape[self.dim] + self.padding[1]
        out_shape[self.dim] = res_shape
        out_shape = Shape.from_(tuple(out_shape)).simplify()
        return (out_shape,)

    def dims_affected(self, input_shapes: Sequence[Shape]) -> Sequence[int]:
        return (self.dim,)

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 1


@dataclass(frozen=True, eq=False)
class FlipOp(MovementOp):
    dim: int

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        assert len(input_shapes) == 1
        return (input_shapes[0],)

    def dims_affected(self, input_shapes: Sequence[Shape]) -> Sequence[int]:
        return (self.dim,)


@dataclass(frozen=True, eq=False)
class ReshapeOp(MovementOp):
    shape: Shape

    def vars_used(self) -> set[ie.Symbol]:
        return set(self.shape.vars_used())

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        assert len(input_shapes) == 1

        # assert ie.lift_to_int_ie(self.shape.prod()).equivalent(
        #    ie.lift_to_int_ie(input_shapes[0].prod())
        # ), f"{self.shape.prod()=} {input_shapes[0].prod()=}"

        return (self.shape,)

    def dims_affected(self, input_shapes: Sequence[Shape]) -> Sequence[int]:
        in_shape = input_shapes[0]
        out_shape = self.shape

        # Ensure the total number of elements is the same
        assert ie.lift_to_int_ie(in_shape.prod()) == ie.lift_to_int_ie(out_shape.prod()), (
            "Total number of elements must be the same"
        )

        in_len = len(in_shape)
        out_len = len(out_shape)
        min_len = min(in_len, out_len)

        # Identify unaffected dimensions from the start
        i = 0
        while i < min_len and ie.lift_to_int_ie(in_shape.at(i)).struct_eq(
            ie.lift_to_int_ie(out_shape.at(i))
        ):
            i += 1

        # Identify unaffected dimensions from the end
        j = 1
        while j <= min_len - i and ie.lift_to_int_ie(in_shape.at(-j)).struct_eq(
            ie.lift_to_int_ie(out_shape.at(-j))
        ):
            j += 1

        # Calculate affected dimensions
        affected_dims = tuple(range(i, in_len - (j - 1)))

        return affected_dims

    def is_static(self) -> bool:
        return not self.shape.is_dynamic()


@dataclass(frozen=True, eq=False)
class PermuteOp(MovementOp):
    dims: tuple[int, ...]

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        assert len(input_shapes) == 1
        in_shape = input_shapes[0]
        out_shape = tuple(in_shape._shape[i] for i in self.dims)
        return (Shape(out_shape),)

    def dims_affected(self, input_shapes: Sequence[Shape]) -> Sequence[int]:
        # Return the dims in the permutation that are not in the same position
        return tuple(i for i, d in enumerate(self.dims) if i != d)


# TODO if we guarantee that sizes is -1 for all dims staying the same,
#  we can remove input_shapes from dims_affected for all ops.
@dataclass(frozen=True, eq=False)
class ExpandOp(MovementOp):
    sizes: Shape

    def vars_used(self) -> set[ie.Symbol]:
        return set(self.sizes.vars_used())

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        assert len(input_shapes) == 1
        input_shape = input_shapes[0]
        assert len(self.sizes) >= len(input_shape), f"Cannot expand {input_shape} to {self.sizes}"
        # pad input_shape with 1s on the left to match sizes
        diff = len(self.sizes) - len(input_shape)
        input_shape = Shape((*tuple([1] * diff), *input_shape._shape))

        # Replace -1 with 1 so we can just use broadcast to check correctness
        sizes_shape = Shape(
            tuple(
                (1 if (sz == -1 if isinstance(sz, int) else sz.struct_eq(ie.ConstInt(-1))) else sz)
                for sz in self.sizes
            )
        )
        return (Shape.broadcast(input_shape, sizes_shape),)

    def dims_affected(self, input_shapes: Sequence[Shape]) -> Sequence[int]:
        shape = input_shapes[0]
        return tuple(
            i
            for i, (s1, s2) in enumerate(zip(shape._shape, self.sizes._shape, strict=False))
            if not ie.lift_to_int_ie(s1).struct_eq(ie.lift_to_int_ie(s2))
        )

    def is_static(self) -> bool:
        return not self.sizes.is_dynamic()


@dataclass(frozen=True, eq=False)
class CatOp(MovementOp):
    dim: int
    num_input_tensors: int

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        assert len(input_shapes) == self.num_input_tensors
        assert len(input_shapes) >= 2, "CatOp requires at least 2 inputs"
        assert all(len(input_shape) == len(input_shapes[0]) for input_shape in input_shapes), (
            f"All inputs must have the same number of dimensions, but got {input_shapes}"
        )
        assert Shape.equal_in_all_but_dim(*input_shapes, dim=self.dim)
        # The resulting shape is the same as the input shapes, except the concatenation dimension
        #  is the sum of the input shapes
        output_shape = tuple(
            (
                sum(input_shape._shape[i] for input_shape in input_shapes)
                if i == self.dim
                else input_shapes[0]._shape[i]
            )
            for i in range(len(input_shapes[0]))
        )
        return (Shape(output_shape),)

    @property
    def num_inputs(self) -> int:
        return self.num_input_tensors

    @property
    def num_outputs(self) -> int:
        return 1

    def dims_affected(self, input_shapes: Sequence[Shape]) -> Sequence[int]:
        return (self.dim,)


@dataclass(frozen=True, eq=False)
class SqueezeOp(MovementOp):
    dim: int

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        assert len(input_shapes) == 1
        output_shape = []
        for i, dim in enumerate(input_shapes[0]):
            if i != self.dim:
                output_shape.append(dim)
            elif dim != 1:
                raise ValueError(f"Cannot squeeze dimension {i} because it is not 1, but {dim}")
        return (Shape(tuple(output_shape)),)

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 1

    def dims_affected(self, input_shapes: Sequence[Shape]) -> Sequence[int]:
        ##  the answer that every dim after the removed is affected
        # return tuple(i for i in range(len(shape)) if i >= self.dim)
        return (self.dim,)


@dataclass(frozen=True, eq=False)
class UnsqueezeOp(MovementOp):
    dim: int

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        assert len(input_shapes) == 1
        assert self.dim <= len(input_shapes[0]), (
            f"Cannot unsqueeze dimension {self.dim} on shape {input_shapes[0]}"
        )
        output_shape = list(input_shapes[0])
        output_shape.insert(self.dim, 1)
        return (Shape(tuple(output_shape)),)

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 1

    def dims_affected(self, input_shapes: Sequence[Shape]) -> Sequence[int]:
        # return tuple(i for i in range(len(input_shapes[0])) if i >= self.dim)
        return (self.dim,)


# ======================================== ElementWise OPS ========================================
@dataclass(frozen=True, eq=False)
class ElementWiseOp(TensorOp, ABC):
    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        # broadcasted_shape = Shape.broadcast(*input_shapes)
        assert Shape.can_broadcast(*input_shapes), f"Shapes cannot be broadcasted: {input_shapes}"
        return (Shape.broadcast(*input_shapes),)
        # for s in input_shapes:
        #    assert s == input_shapes[0], f"Shapes are not equal: {input_shapes}"
        # return (input_shapes[0],)

    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        if all(input_dtype == input_dtypes[0] for input_dtype in input_dtypes):
            return (input_dtypes[0],)
        else:
            # Return whichever has higher dtype.priority
            dtype: DataType = max(
                input_dtypes,
                key=lambda dtype: dtype.priority,  # type: ignore
            )
            return (dtype,)

    @property
    def num_outputs(self) -> int:
        return 1


@dataclass(frozen=True, eq=False)
class UnaryElementWiseOp(ElementWiseOp, ABC):
    @property
    def num_inputs(self) -> int:
        return 1


@dataclass(frozen=True, eq=False)
class ValToValOp(UnaryElementWiseOp):
    in_val: Any
    out_val: Any

    def is_nan_mask(self) -> bool:
        return math.isnan(self.in_val)

    def is_pos_inf_mask(self) -> bool:
        return math.isinf(self.in_val) and self.in_val > 0

    def is_neg_inf_mask(self) -> bool:
        return math.isinf(self.in_val) and self.in_val < 0

    def equivalent(self, other: Any) -> bool:
        if not isinstance(other, ValToValOp):
            return False
        # NOTE: need to carefully compare float("nan") and float("inf")
        if self.is_nan_mask():
            in_val_eq = other.is_nan_mask()
        elif self.is_pos_inf_mask():
            in_val_eq = other.is_pos_inf_mask()
        elif self.is_neg_inf_mask():
            in_val_eq = other.is_neg_inf_mask()
        else:
            in_val_eq = self.in_val == other.in_val

        # Repeat for out_val
        if math.isnan(self.out_val):
            out_val_eq = math.isnan(other.out_val)
        elif math.isinf(self.out_val) and self.out_val > 0:
            out_val_eq = math.isinf(other.out_val) and other.out_val > 0
        elif math.isinf(self.out_val) and self.out_val < 0:
            out_val_eq = math.isinf(other.out_val) and other.out_val < 0
        else:
            out_val_eq = self.out_val == other.out_val

        return in_val_eq and out_val_eq


@dataclass(frozen=True, eq=False)
class CastOp(UnaryElementWiseOp):
    output_dtype: DataType

    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return (self.output_dtype,)


@dataclass(frozen=True, eq=False)
class SqrtOp(UnaryElementWiseOp): ...


@dataclass(frozen=True, eq=False)
class NegOp(UnaryElementWiseOp): ...


@dataclass(frozen=True, eq=False)
class SinOp(UnaryElementWiseOp): ...


@dataclass(frozen=True, eq=False)
class NotOp(UnaryElementWiseOp):
    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return (dtypes.bool_,)


@dataclass(frozen=True, eq=False)
class LnOp(UnaryElementWiseOp): ...


@dataclass(frozen=True, eq=False)
class ExpOp(UnaryElementWiseOp): ...


@dataclass(frozen=True, eq=False)
class BinaryElementWiseOp(ElementWiseOp, ABC):
    @property
    def num_inputs(self) -> int:
        return 2


@dataclass(frozen=True, eq=False)
class MulOp(BinaryElementWiseOp): ...


@dataclass(frozen=True, eq=False)
class DivOp(BinaryElementWiseOp):
    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return (dtypes.least_upper_float(dtypes.upcast(*input_dtypes)),)


@dataclass(frozen=True, eq=False)
class PowOp(BinaryElementWiseOp): ...


@dataclass(frozen=True, eq=False)
class AddOp(BinaryElementWiseOp): ...


@dataclass(frozen=True, eq=False)
class SubOp(BinaryElementWiseOp): ...


@dataclass(frozen=True, eq=False)
class BooleanBinaryOp(BinaryElementWiseOp, ABC):
    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return (dtypes.bool_,)


@dataclass(frozen=True, eq=False)
class OrOp(BooleanBinaryOp): ...


@dataclass(frozen=True, eq=False)
class AndOp(BooleanBinaryOp): ...


@dataclass(frozen=True, eq=False)
class LessThanOp(BooleanBinaryOp): ...


# NOTE: strictly, equal is unnecessary, but it also convenient, so leaving it in.
@dataclass(frozen=True, eq=False)
class EqualOp(BooleanBinaryOp): ...


@dataclass(frozen=True, eq=False)
class WhereOp(ElementWiseOp):  # NOTE Though ternary, still keeps shape so element-wise
    @property
    def num_inputs(self) -> int:
        return 3

    # def infer_output_dtypes(
    #    self, input_dtypes: Tuple[DataType, ...]
    # ) -> Tuple[DataType, ...]:
    #    assert (
    #        input_dtypes[1] == input_dtypes[2]
    #    )  # TODO this is not true, but might be good enough for now
    #    return (input_dtypes[1],)


# =================== BEGIN HARDCODED OPS ===================
# NOTE: These are ops that we would like to implement in terms of other ops, but currently
# simply push down to the backend. We should eventually implement these in terms of other ops
# in order to simplify backend implementations.


@dataclass(frozen=True, eq=False)
class ScanOp(TensorOp, ABC):
    dim: int = 0

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        return tuple(input_shapes)

    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return tuple(input_dtypes)

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def num_inputs(self) -> int:
        return 1


@dataclass(frozen=True, eq=False)
class CumSumOp(ScanOp):
    pass


@dataclass(frozen=True, eq=False)
class IdentOp(ElementWiseOp):
    @property
    def num_inputs(self) -> int:
        return 1

    ...


@dataclass(frozen=True, eq=False)
class MatMulOp(TensorOp):
    def infer_output_shapes(  # noqa: C901
        self, input_shapes: Sequence[Shape]
    ) -> Sequence[Shape]:
        assert len(input_shapes) == 2
        input_shape1 = input_shapes[0]
        input_shape2 = input_shapes[1]

        # Ensure both inputs have at least 2 dimensions
        if len(input_shape1) < 2 or len(input_shape2) < 2:
            raise ValueError(f"Invalid shapes for MatMul: {input_shape1} and {input_shape2}")

        # Get contracting dimensions
        dim1 = len(input_shape1) - 1  # Last dim of first input
        dim2 = len(input_shape2) - 2  # Second-to-last dim of second input

        assert input_shape1.at(dim1) == input_shape2.at(dim2)

        # Handle batch dimensions
        batch_shape1 = Shape(input_shape1._shape[:-2]) if len(input_shape1) > 2 else Shape(())
        batch_shape2 = Shape(input_shape2._shape[:-2]) if len(input_shape2) > 2 else Shape(())
        batch_shape = Shape.broadcast(batch_shape1, batch_shape2)

        # Output shape is [*batch_shape, m, n] where m is from first input, n from second
        return (Shape((*batch_shape._shape, input_shape1.at(-2), input_shape2.at(-1))),)

        # msg = f"Invalid shapes for MatMul: {input_shape1} and {input_shape2}"
        # if len(input_shape1) == 1 and len(input_shape2) == 1:
        #    assert Shape.dim_is_equal(input_shape1[0], input_shape2[0], dim=0), msg
        #    return (Shape(()),)
        # elif len(input_shape1) == 2 and len(input_shape2) == 2:
        #    assert Shape.dim_is_equal(input_shape1[1], input_shape2[0], dim=0), msg
        #    return (Shape((input_shape1.at(0), input_shape2.at(1))),)
        # elif len(input_shape1) == 1 and len(input_shape2) == 2:
        #    assert Shape.dim_is_equal(input_shape1[0], input_shape2[0], dim=0), msg
        #    return (Shape((input_shape2.at(0), input_shape2.at(1))),)
        # elif len(input_shape1) == 2 and len(input_shape2) == 1:
        #    assert Shape.dim_is_equal(input_shape1[1], input_shape2[0], dim=0), msg
        #    return (Shape((input_shape1.at(0), input_shape2.at(0))),)
        # elif len(input_shape1) >= 2 and len(input_shape2) >= 2:
        #    assert Shape.dim_is_equal(input_shape1[-1], input_shape2[-2], dim=0), msg
        #    # Batched matrix multiply
        #    non_batch_dims1 = input_shape1._shape[:-2]
        #    non_batch_dims2 = input_shape2._shape[:-2]
        #    batch_shape = Shape.broadcast(Shape(non_batch_dims1), Shape(non_batch_dims2))
        #    return (Shape((*batch_shape._shape, input_shape1.at(-2), input_shape2.at(-1))),)
        # else:
        #    raise ValueError(f"Invalid shapes for MatMul: {input_shape1} and {input_shape2}")

    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return (dtypes.upcast(*input_dtypes),)

    @property
    def num_inputs(self) -> int:
        return 2

    @property
    def num_outputs(self) -> int:
        return 1


@dataclass(frozen=True, eq=False)
class ConvOp(TensorOp):
    """This Op performs a N-D Convolution using 3 input tensors: input, weight and optional bias.
    input.shape: (B?, C_in, *N_dims_in)
    weight.shape: (C_out, C_in/groups, *N_dims_kernel)
    bias.shape: (C_out,)

    output.shape: (B?, C_out, *N_dims_out)

    The value of each N_dims_out[i] is:
    N_dims_out[i] = floor((N_dims_in[i] + 2 x padding[i] - dilation[i]* (N_dims_kernel[i] - 1)) + 1)
    """

    stride: tuple[int, ...]
    # padding: Tuple[int, ...]
    # dilation: Tuple[int, ...]
    transposed: bool
    # output_padding: Tuple[int, ...]
    # groups: int
    n_dims: int

    # def __post_init__(self) -> None:
    #    if sum(self.output_padding) != 0 and not self.transposed:
    #        raise ValueError("output_padding is only supported for transposed convolutions")

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        input_shape = input_shapes[0]
        weight_shape = input_shapes[1]
        N = self.n_dims
        kernel_shape = weight_shape[-N:]

        C_out = weight_shape.at(0)
        batch_size = input_shape.at(0) if len(input_shape) > N else None  # Handle no batch dim

        output_size = [batch_size, C_out] if batch_size is not None else [C_out]

        for i in range(N):
            dim_in = input_shape.at(-N + i)
            kernel = kernel_shape.at(i)
            stride = self.stride[i]
            # pad = self.padding[i]
            # dilation = self.dilation[i]

            if self.transposed:
                dim_out = (dim_in - 1) * stride - 2 * 0 + 1 * (kernel - 1) + 0 + 1
            else:
                dim_out = (dim_in + 2 * 0 - 1 * (kernel - 1) - 1) // stride + 1

            output_size.append(dim_out)

        return (Shape(tuple(output_size)),)

    # def infer_output_shapes(self, input_shapes: Tuple[Shape, ...]) -> Tuple[Shape, ...]:
    #    input_shape = input_shapes[0]
    #    weight_shape = input_shapes[1]
    #    N = self.n_dims
    #    kernel_shape = weight_shape[-N:]

    #    C_out = weight_shape.at(0)
    #    batch_size = input_shape.at(0) if len(input_shape) > N else None  # Handle no batch dim

    #    output_size = [batch_size, C_out] if batch_size is not None else [C_out]

    #    for i in range(N):
    #        inp = input_shape.at(-N + i)
    #        kernel = kernel_shape.at(i)
    #        stride = self.stride[i]
    #        pad = self.padding[i]
    #        dilation = self.dilation[i]

    #        if self.transposed:
    #            out_pad = self.output_padding[i]  # Fix incorrect indexing
    #            dim_out = (inp - 1) * stride - 2 * pad + dilation * (kernel - 1) + out_pad + 1
    #        else:
    #            dim_out = (inp + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1

    #        output_size.append(dim_out)

    #    return (Shape(tuple(output_size)),)

    # def infer_output_shapes(self, input_shapes: Tuple[Shape, ...]) -> Tuple[Shape, ...]:
    #    input_shape = input_shapes[0]
    #    weight_shape = input_shapes[1]
    #    N = self.n_dims
    #    kernel_shape = weight_shape[-N:]

    #    C_out = weight_shape.at(0)
    #    batch_size = input_shape.at(0)

    #    output_size = [batch_size, C_out]

    #    for i in range(N):
    #        inp = input_shape.at(-N + i)
    #        kernel = kernel_shape.at(i)
    #        stride = self.stride[i]
    #        pad = self.padding[i]
    #        dilation = self.dilation[i]

    #        if self.transposed:
    #            out_pad = self.output_padding[i - 1] if self.output_padding else 0
    #            # TODO this one may not be correct
    #            dim_out = (inp - 1) * stride - 2 * pad + dilation * (kernel - 1) + out_pad + 1
    #        else:
    #            dim_out = (inp + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
    #            assert not isinstance(dim_out, float)

    #        output_size.append(dim_out)

    #    return (Shape(tuple(output_size)),)

    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return (dtypes.upcast(*input_dtypes),)

    @property
    def num_inputs(self) -> int:
        return 2  # input weight

    @property
    def num_outputs(self) -> int:
        return 1


# NOTE: For now, we implement index ops with gather and scatter ops, which are more general.
# We can add them back in later if needed.
@dataclass(frozen=True, eq=False)
class IndexSelectOp(TensorOp):
    """Takes a source tensor and a 1D (or 0D) index tensor with the indices of source to return.
    The output will lose dimension dim if the index tensor is 0D.
    """

    dim: int

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        src_shape = input_shapes[0]
        idx_shape = input_shapes[1]

        assert len(src_shape) > self.dim, f"IndexSelect: {src_shape=} {self.dim=} dim out of bounds"
        assert len(idx_shape) == 1 or len(idx_shape) == 0

        s = list(src_shape._shape)

        if len(idx_shape) == 1:
            lift_src_shape = ie.lift_to_int_ie(s[self.dim])
            lift_idx_shape = ie.lift_to_int_ie(idx_shape._shape[0])
            if isinstance(lift_src_shape, ie.ConstInt) and isinstance(lift_idx_shape, ie.ConstInt):
                assert lift_src_shape.const >= lift_idx_shape.const, (
                    f"IndexSelectOp: {lift_src_shape} <= {lift_idx_shape}"
                )
            s[self.dim] = idx_shape._shape[0]
        else:
            s.pop(self.dim)

        return (Shape.from_(tuple(s)),)

    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return (input_dtypes[0],)

    @property
    def num_inputs(self) -> int:
        return 2

    @property
    def num_outputs(self) -> int:
        return 1


@dataclass(frozen=True, eq=False)
class IndexAddOp(TensorOp):
    # input, dim, index, source
    # Adds source to input at index along dim
    dim: int
    alpha: float = 1

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        return (input_shapes[0],)

    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return (dtypes.upcast(*input_dtypes),)

    @property
    def num_inputs(self) -> int:
        return 3

    @property
    def num_outputs(self) -> int:
        return 1


# =================== END HARDCODED OPS ===================


# ===================================== END ElementWise OPS =================================


# ======================================== REDUCE OPS ========================================
@dataclass(frozen=True, eq=False)
class ReduceOp(TensorOp, ABC):
    # dims: DIM_TYPE = None
    dims: tuple[int, ...]
    keepdim: bool = False

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        assert len(input_shapes) == 1
        in_shape = input_shapes[0]

        assert max(self.dims) < len(in_shape), f"ReduceOp: {self.dims=} dim out of bounds"
        if self.keepdim:
            return self._infer_with_keepdim(in_shape)
        else:
            return self._infer_without_keepdim(in_shape)

    def _infer_with_keepdim(self, in_shape: Shape) -> Sequence[Shape]:
        # if self.dims is None:
        #    return (Shape(tuple(1 for _ in range(len(in_shape)))),)
        # elif isinstance(self.dims, int):
        #    return (
        #        Shape(
        #            tuple(1 if i == self.dims else d for i, d in enumerate(in_shape))
        #        ),
        #    )
        # else:
        s = (Shape(tuple(1 if i in self.dims else d for i, d in enumerate(in_shape))),)
        return s

    def _infer_without_keepdim(self, in_shape: Shape) -> Sequence[Shape]:
        # if self.dims is None:
        #    return (Shape(()),)
        # elif isinstance(self.dims, int):
        #    return (Shape(tuple(d for i, d in enumerate(in_shape) if i != self.dims)),)
        # else:
        return (Shape(tuple(d for i, d in enumerate(in_shape) if i not in self.dims)),)

    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return (input_dtypes[0],)


@dataclass(frozen=True, eq=False)
class SumOp(ReduceOp):
    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 1


@dataclass(frozen=True, eq=False)
class MaxOp(ReduceOp):
    """A max op returns both the values and indexes"""

    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        shape = super().infer_output_shapes(input_shapes)[0]
        return (shape, shape)

    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        return (input_dtypes[0], dtypes.default_int)

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 2
