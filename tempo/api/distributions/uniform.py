from tempo.api.distributions.distribution import Distribution
from tempo.api.recurrent_tensor import (
    MaybeRecurrentTensor,
    RecurrentTensor,
    _lift_if_index_value,
    lift,
)
from tempo.core.domain import Domain, DomainLike
from tempo.core.shape import Shape, ShapeLike


class Uniform(Distribution):
    def __init__(
        self,
        low: MaybeRecurrentTensor,
        high: MaybeRecurrentTensor,
        domain: DomainLike = None,
    ) -> None:
        self.low: RecurrentTensor = lift(low)
        self.high: RecurrentTensor = lift(high)

        self.domain = (
            domain if domain is not None else Domain.union(self.low.domain, self.high.domain)
        )
        super().__init__()

    def cdf(self, value: MaybeRecurrentTensor) -> RecurrentTensor:
        value = _lift_if_index_value(value)
        result = (value - self.low) / (self.high - self.low)
        return RecurrentTensor.clamp(result, lb=0, ub=1)

    def sample(self, shape: ShapeLike = None) -> RecurrentTensor:
        shape = Shape.from_(shape)
        return RecurrentTensor.uniform(shape, self.low, self.high, domain=self.domain)

    def log_prob(self, sample: MaybeRecurrentTensor) -> RecurrentTensor:
        # NOTE: This should return -inf for values outside the range, but we don't support that yet.
        return -RecurrentTensor.ln(self.high - self.low)

    def entropy(self) -> RecurrentTensor:
        # NOTE: tbf, the correct way of doing all this is to allow every API operation to support
        # int and float, and return int and float if no inputs are recurrent tensors.
        return RecurrentTensor.ln(self.high - self.low)

    @property
    def mean(self) -> MaybeRecurrentTensor:
        return (self.low + self.high) / 2

    @property
    def mode(self) -> MaybeRecurrentTensor:
        raise ValueError("Uniform distribution does not have a mode")

    @property
    def stddev(self) -> MaybeRecurrentTensor:
        return (self.high - self.low) / 12**0.5

    @property
    def variance(self) -> MaybeRecurrentTensor:
        return ((self.high - self.low) ** 2) / 12
