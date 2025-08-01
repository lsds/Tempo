from abc import ABC, abstractmethod
from typing import Callable, Generic, Iterable, Sequence, Tuple

from tempo.core.datatypes import BackendTensorT, OpId, OpInId, OpOutId
from tempo.core.dtype import DataType
from tempo.core.shape import Shape
from tempo.core.tensor_op import TensorOp
from tempo.utils import logger

log = logger.get_logger(__name__)


class DataflowGraphI(Generic[BackendTensorT], ABC):
    """Represents a subgraph of ops that can be treated as a dataflow.
    This has many uses, for example, it allows us to interprete it with a lower overhead
    runtime, treat it as a single op during scheduling, and allows us to generate
    efficient code for the subgraph.

    We need to know how to route the inputs and outputs to the dataflow into and from
    the internal ops. This is done by specifying the routers.
    """

    def __init__(
        self,
        irouter: Tuple[Tuple[Tuple[OpId, OpInId], ...], ...],
        orouter: Tuple[Tuple[OpId, OpOutId], ...],
    ) -> None:
        self.irouter = irouter
        self.orouter = orouter

    @abstractmethod
    def infer_output_shapes(self, input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        raise NotImplementedError

    @abstractmethod
    def infer_output_dtypes(self, input_dtypes: Sequence[DataType]) -> Sequence[DataType]:
        raise NotImplementedError

    @property
    @abstractmethod
    def nodes(self) -> Iterable[TensorOp]:
        raise NotImplementedError

    @property
    def num_inputs(self) -> int:
        return len(self.irouter)

    @property
    def num_outputs(self) -> int:
        return len(self.orouter)

    @abstractmethod
    def execute(
        self,
        inputs: Tuple[BackendTensorT, ...],
        thunk_map: Callable[[TensorOp, Tuple[BackendTensorT, ...]], Tuple[BackendTensorT, ...]],
    ) -> Tuple[BackendTensorT, ...]:
        raise NotImplementedError
