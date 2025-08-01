import math

from tempo.api.distributions.distribution import Distribution
from tempo.api.recurrent_tensor import (
    MaybeRecurrentTensor,
    RecurrentTensor,
    _lift_if_index_value,
    lift,
)
from tempo.core.domain import Domain, DomainLike
from tempo.core.shape import ShapeLike


class Normal(Distribution):
    def __init__(
        self,
        mu: MaybeRecurrentTensor,
        sigma: MaybeRecurrentTensor,
        domain: DomainLike = None,
    ) -> None:
        self.mu: RecurrentTensor = lift(mu)
        self.sigma: RecurrentTensor = lift(sigma)
        self.domain = (
            domain if domain is not None else Domain.union(self.mu.domain, self.sigma.domain)
        )
        super().__init__()

    def cdf(self, value: MaybeRecurrentTensor) -> RecurrentTensor:
        value = _lift_if_index_value(value)
        return RecurrentTensor.const(0.5) * (
            RecurrentTensor.const(1.0)
            + RecurrentTensor.erf((value - self.mu) / (self.sigma * math.sqrt(2)))
        )

    def sample(self, shape: ShapeLike = None) -> RecurrentTensor:
        return RecurrentTensor.normal(shape, mean=self.mu, std=self.sigma, domain=self.domain)

    def log_prob(self, sample: MaybeRecurrentTensor) -> RecurrentTensor:
        sample = lift(sample)
        combined = (
            -((sample.detach() - self.mu) ** 2) / (RecurrentTensor.const(2) * self.variance)
            - self.sigma.ln()
            - RecurrentTensor.const(math.log(math.sqrt(2 * math.pi)))
        )

        return combined

    def entropy(self) -> RecurrentTensor:
        # NOTE: tbf, the correct way of doing all this is to allow every API operation to support
        # int and float, and return int and float if no inputs are recurrent tensors.
        return RecurrentTensor.const(0.5 + 0.5 * math.log(2 * math.pi)) + RecurrentTensor.ln(
            self.sigma
        )

    @property
    def mean(self) -> MaybeRecurrentTensor:
        return self.mu

    @property
    def mode(self) -> MaybeRecurrentTensor:
        return self.mu

    @property
    def stddev(self) -> MaybeRecurrentTensor:
        return self.sigma

    @property
    def variance(self) -> MaybeRecurrentTensor:
        return RecurrentTensor.pow_(self.sigma, 2)
