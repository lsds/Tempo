from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NewType, Protocol, TypeVar, Union

from optree import PyTree

OpId = NewType("OpId", int)
OpInId = NewType("OpInId", int)
OpOutId = NewType("OpOutId", int)
PDGId = NewType("PDGId", int)

NestedList = Any

DIM_TYPE = Union[int, tuple[int, ...], None]


@dataclass(frozen=True)
class TensorId:
    op_id: OpId
    output_id: OpOutId


IndexType = Union[int, slice, tuple[Union[int, slice], ...]]


class TempoTensorProtocol(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> Any: ...

    def __getitem__(self, key: IndexType) -> TempoTensorProtocol: ...

    # @property
    # def device(self) -> Any:
    #    pass


BackendTensorT = TypeVar("BackendTensorT", bound=TempoTensorProtocol)
# BackendTensorT = TypeVar("BackendTensorT")


BackendTensorTPyTree = PyTree[BackendTensorT]


# TorchBackendTensorTree = PyTree[torch.Tensor]
# NumpyBackendTensorTree = PyTree[np.ndarray]
# NumpyBackendTensorTree = PyTree[jax.numpy.ndarray]
