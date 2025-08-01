from typing import Optional, Tuple, Union

from tempo.api.distributions.categorical import Categorical
from tempo.api.nn.activation import ActivationFunctionLike
from tempo.api.nn.turnkey import FullyConnected, FullyConnectedActorCritic
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.api.rl.datatypes import (
    ActionsRecurrentTensor,
    ObservationsRecurrentTensor,
    ValueRecurrentTensor,
)
from tempo.api.rl.networks.network import Network
from tempo.core.domain import DomainLike
from tempo.core.shape import StaticShapeLike


class DiscreteFFActor(Network):
    def __init__(
        self,
        input_shape: StaticShapeLike,
        hidden_sizes: Tuple[int, ...],
        output_shape: StaticShapeLike,
        act_fun: ActivationFunctionLike = "leakyrelu",
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
    ) -> None:
        super().__init__(domain=domain, independent_domain=independent_domain)

        self.network = FullyConnected(
            input_shape, hidden_sizes, output_shape, act_fun, domain, independent_domain
        )

        self.dist: Optional[Categorical] = None

    def forward(
        self,
        obs: Union[ObservationsRecurrentTensor, RecurrentTensor],
    ) -> Tuple[ActionsRecurrentTensor, Optional[ValueRecurrentTensor]]:
        logits = self.network(obs)
        logits = logits.softmax()

        self.dist = Categorical(logits)

        return ActionsRecurrentTensor(self.dist.sample()), None

    def log_prob(self, actions: Union[ActionsRecurrentTensor, RecurrentTensor]) -> RecurrentTensor:
        if self.dist is None:
            raise ValueError("No distribution has been computed yet. Please call forward first.")
        ret = self.dist.log_prob(actions)  # type: ignore
        return ret

    def entropy(self) -> RecurrentTensor:
        if self.dist is None:
            raise ValueError("No distribution has been computed yet. Please call forward first.")
        return self.dist.entropy()  # type: ignore


class DiscreteFFActorCritic(Network):
    def __init__(
        self,
        input_shape: StaticShapeLike,
        hidden_sizes: Tuple[int, ...],
        output_shape: StaticShapeLike,
        act_fun: ActivationFunctionLike = "leakyrelu",
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
    ) -> None:
        super().__init__(domain=domain, independent_domain=independent_domain)

        self.network = FullyConnectedActorCritic(
            input_shape, hidden_sizes, output_shape, act_fun, domain, independent_domain
        )

        self.dist: Optional[Categorical] = None

    def forward(
        self,
        obs: Union[ObservationsRecurrentTensor, RecurrentTensor],
    ) -> Tuple[ActionsRecurrentTensor, Optional[ValueRecurrentTensor]]:
        logits, value = self.network(obs)
        logits = logits.softmax()

        self.dist = Categorical(logits)

        return ActionsRecurrentTensor(self.dist.sample()), ValueRecurrentTensor(value)

    def log_prob(self, actions: Union[ActionsRecurrentTensor, RecurrentTensor]) -> RecurrentTensor:
        if self.dist is None:
            raise ValueError("No distribution has been computed yet. Please call forward first.")
        return self.dist.log_prob(actions)  # type: ignore

    def entropy(self) -> RecurrentTensor:
        if self.dist is None:
            raise ValueError("No distribution has been computed yet. Please call forward first.")
        return self.dist.entropy()  # type: ignore
