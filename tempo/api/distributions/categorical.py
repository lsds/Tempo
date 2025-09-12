from tempo.api.distributions.distribution import Distribution
from tempo.api.recurrent_tensor import MaybeRecurrentTensor, RecurrentTensor, lift
from tempo.core.domain import DomainLike
from tempo.core.shape import Shape, ShapeLike


def logits_to_probs(logits: RecurrentTensor, is_binary: bool = False) -> RecurrentTensor:
    if is_binary:
        return logits.sigmoid()
    return logits.softmax(dim=-1)


def clamp_probs(probs: RecurrentTensor) -> RecurrentTensor:
    eps = probs.dtype.eps
    assert eps is not None
    return probs.clamp(lb=eps, ub=1 - eps)


def probs_to_logits(probs: RecurrentTensor, is_binary: bool = False) -> RecurrentTensor:
    ps_clamped = clamp_probs(probs)
    if is_binary:
        return ps_clamped.ln() - (RecurrentTensor.ones() - ps_clamped).ln()
    return ps_clamped.ln()


class Categorical(Distribution):
    def __init__(
        self,
        probs: MaybeRecurrentTensor | None = None,
        logits: MaybeRecurrentTensor | None = None,
        domain: DomainLike = None,
    ) -> None:
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            probs = lift(probs)
            if probs.ndim < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            # TODO this division is unnecessary if already softmaxed
            self.probs = probs / probs.sum(-1, keepdim=True)
            self.logits = probs_to_logits(self.probs)
        else:
            assert logits is not None
            logits = lift(logits)
            if logits.ndim < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            # Normalize
            self.logits = logits - logits.logsumexp(sum_dims=-1, keepdim=True)
            self.probs = logits_to_probs(self.logits)

        self._num_events: int = self.probs.size(-1)  # type: ignore

        self.domain = domain if domain is not None else self.probs.domain

        super().__init__()

    def cdf(self, value: MaybeRecurrentTensor) -> RecurrentTensor:
        raise NotImplementedError

    def sample(self, shape: ShapeLike = None) -> RecurrentTensor:
        num_samples = Shape.from_(shape).prod()
        assert isinstance(num_samples, int)  # TODO: for now
        return RecurrentTensor.multinomial(
            self.probs, num_samples, True, domain=self.domain
        ).reshape(shape)

    def log_prob(self, sample: MaybeRecurrentTensor) -> RecurrentTensor:
        sample = lift(sample)
        sample = sample.unsqueeze(-1)
        gathered = self.logits.gather(-1, sample)
        return gathered.squeeze(-1)

    def entropy(self) -> RecurrentTensor:
        min_real = self.logits.dtype.minimum
        assert min_real is not None
        logits = self.logits.clamp(lb=min_real)
        p_log_p = logits * self.probs
        return -p_log_p.sum(-1)

    @property
    def mean(self) -> MaybeRecurrentTensor:
        raise ValueError("Categorical distribution does not have a mean")

    @property
    def mode(self) -> MaybeRecurrentTensor:
        return self.probs.argmax(-1)

    @property
    def stddev(self) -> MaybeRecurrentTensor:
        raise ValueError("Categorical distribution does not have a stddev")

    @property
    def variance(self) -> MaybeRecurrentTensor:
        raise ValueError("Categorical distribution does not have a variance")
