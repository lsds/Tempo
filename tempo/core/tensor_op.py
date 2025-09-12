from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import optree

from tempo.core import index_expr as ie
from tempo.core.datatypes import OpId
from tempo.core.debug_utils import get_creation_traceback
from tempo.core.domain import Domain
from tempo.core.dtype import DataType
from tempo.core.shape import Shape
from tempo.utils import logger

log = logger.get_logger(__name__)


@dataclass(frozen=True, eq=False)
class TensorOp(ABC):
    op_id: OpId
    domain: Domain = field(repr=True)
    tags: dict[str, Any] = field(repr=False)
    _creation_traceback: list[str] = field(
        init=False,
        repr=False,
        hash=False,
        compare=False,
        default_factory=lambda: get_creation_traceback(),
    )

    @property
    def flat_tags(self) -> dict[str, list[Any]]:
        return {k: sorted(set(optree.tree_flatten(v)[0])) for k, v in self.tags.items()}

    @property
    def creation_traceback(self) -> str:
        return "\n".join([x.strip() for x in self._creation_traceback])

    def is_udf(self) -> bool:
        return False

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if type(self) is not type(other):
            return False

        eq: bool = self.op_id == other.op_id
        return eq

    def equivalent(self, other: Any) -> bool:  # noqa: C901
        if not isinstance(other, self.__class__):
            return False
        if type(self) is not type(other):
            return False

        # Every field must be equal, except for the op_id
        import numpy as np

        for field_name in self.__dataclass_fields__:
            if field_name == "op_id" or field_name == "_creation_traceback" or field_name == "tags":
                continue
            f1 = getattr(self, field_name)
            f2 = getattr(other, field_name)
            if isinstance(f1, ie.IndexExpr):
                if not f1.struct_eq(f2):
                    return False
            elif isinstance(f1, np.ndarray):
                return bool(np.array_equal(f1, f2))
            else:
                if f1 != f2:
                    return False
        return True

    # def __str__(self) -> str:
    #    str_ = f"{type(self).__name__}({self.op_id},"

    #    for field_name in self.__dataclass_fields__:
    #        if field_name == "op_id" or field_name == "domain" or field_name == "tags":
    #            continue
    #        str_ += f"{field_name}={getattr(self, field_name)}, "
    #    str_ += ")"

    #    return str_

    def __hash__(self) -> int:
        return hash((type(self), self.op_id))

    @abstractmethod
    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]: ...

    @abstractmethod
    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]: ...

    @property
    def is_source(self) -> bool:
        return self.num_inputs == 0

    @property
    def is_sink(self) -> bool:
        return self.num_outputs == 0

    @property
    def is_stateful(self) -> bool:
        # NOTE: primitives are stateless, so we make this the default
        return False

    def is_static(self) -> bool:
        return True

    def is_dynamic(self) -> bool:
        return not self.is_static()

    @abstractproperty
    def num_inputs(self) -> int:
        raise NotImplementedError

    @abstractproperty
    def num_outputs(self) -> int:
        raise NotImplementedError

    def vars_used(self) -> set[ie.Symbol]:
        return set()
