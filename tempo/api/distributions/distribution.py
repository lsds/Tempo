from abc import ABC, abstractmethod, abstractproperty

from tempo.api.recurrent_tensor import MaybeRecurrentTensor, RecurrentTensor
from tempo.core.shape import Shape, ShapeLike


class Distribution(ABC):
    def __init__(self, event_shape: ShapeLike = None) -> None:
        self.event_shape = Shape.from_(event_shape)

    def _extended_shape(self, sample_shape: Shape) -> Shape:
        return Shape(sample_shape._shape + self.event_shape._shape)

    @abstractmethod
    def cdf(self, value: MaybeRecurrentTensor) -> RecurrentTensor:
        raise NotImplementedError

    @abstractmethod
    def sample(self, shape: ShapeLike = None) -> RecurrentTensor:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, sample: MaybeRecurrentTensor) -> RecurrentTensor:
        raise NotImplementedError

    @abstractmethod
    def entropy(self) -> RecurrentTensor:
        raise NotImplementedError

    @abstractproperty
    def mean(self) -> MaybeRecurrentTensor:
        raise NotImplementedError

    @abstractproperty
    def mode(self) -> MaybeRecurrentTensor:
        raise NotImplementedError

    @abstractproperty
    def stddev(self) -> MaybeRecurrentTensor:
        raise NotImplementedError

    @abstractproperty
    def variance(self) -> MaybeRecurrentTensor:
        raise NotImplementedError
