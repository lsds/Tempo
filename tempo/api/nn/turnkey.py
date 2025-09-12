from tempo.api.nn.activation import ActivationFunctionLike, get_act_fun_class
from tempo.api.nn.linear import Linear
from tempo.api.nn.module import Module, Sequential
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.core.domain import DomainLike
from tempo.core.shape import StaticShape, StaticShapeLike


class FullyConnected(Module):
    def __init__(
        self,
        input_shape: StaticShapeLike,
        hidden_sizes: tuple[int, ...],
        output_shape: StaticShapeLike,
        act_fun: ActivationFunctionLike = "leakyrelu",
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
    ) -> None:
        super().__init__(domain=domain, independent_domain=independent_domain)
        Act = get_act_fun_class(act_fun)

        self.input_shape = StaticShape.from_(input_shape)
        self.output_shape = StaticShape.from_(output_shape)

        self.layers: list[Module] = []
        if len(hidden_sizes) == 0:
            self.layers.append(
                Linear(
                    self.input_shape.prod(),
                    self.output_shape.prod(),
                    bias=False,
                    domain=domain,
                    independent_domain=independent_domain,
                )
            )
            self.layers.append(Act())
        else:
            self.layers.append(
                Linear(
                    self.input_shape.prod(),
                    hidden_sizes[0],
                    domain=domain,
                    independent_domain=independent_domain,
                )
            )
            self.layers.append(Act())
            for i in range(1, len(hidden_sizes)):
                self.layers.append(
                    Linear(
                        hidden_sizes[i - 1],
                        hidden_sizes[i],
                        domain=domain,
                        independent_domain=independent_domain,
                    )
                )
                self.layers.append(Act())
            self.layers.append(
                Linear(
                    hidden_sizes[-1],
                    self.output_shape.prod(),
                    domain=domain,
                    independent_domain=independent_domain,
                )
            )
            self.layers.append(Act())
        self.seq = Sequential(*self.layers)

    def forward(self, x: RecurrentTensor) -> RecurrentTensor:
        x = x.reshape(self.input_shape.flatten())
        result: RecurrentTensor = self.seq.forward(x)  # type: ignore
        return result.reshape(self.output_shape)

    def __call__(self, x: RecurrentTensor) -> RecurrentTensor:
        return self.forward(x)


class FullyConnectedActorCritic(Module):
    def __init__(
        self,
        input_shape: StaticShapeLike,
        hidden_sizes: tuple[int, ...],
        output_shape: StaticShapeLike,
        act_fun: ActivationFunctionLike = "leakyrelu",
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
    ) -> None:
        super().__init__(domain=domain, independent_domain=independent_domain)
        Act = get_act_fun_class(act_fun)

        self.input_shape = StaticShape.from_(input_shape)
        self.output_shape = StaticShape.from_(output_shape)

        self.layers: list[Module] = []
        self.layers.append(
            Linear(
                self.input_shape.prod(),
                hidden_sizes[0],
                domain=domain,
                independent_domain=independent_domain,
            )
        )
        self.layers.append(Act())
        for i in range(1, len(hidden_sizes)):
            self.layers.append(
                Linear(
                    hidden_sizes[i - 1],
                    hidden_sizes[i],
                    domain=domain,
                    independent_domain=independent_domain,
                )
            )
            self.layers.append(Act())
        self.shared_backbone = Sequential(*self.layers)
        self.act_head = Linear(
            hidden_sizes[-1],
            self.output_shape.prod(),
            domain=domain,
            independent_domain=independent_domain,
        )
        self.critic_head = Linear(
            hidden_sizes[-1],
            1,
            domain=domain,
            independent_domain=independent_domain,
        )

    def forward(self, x: RecurrentTensor) -> tuple[RecurrentTensor, RecurrentTensor]:
        x = x.reshape(self.input_shape.flatten())
        latent: RecurrentTensor = self.shared_backbone.forward(x)  # type: ignore
        return self.act_head(latent).reshape(self.output_shape), self.critic_head(latent).reshape(
            (1,)
        )

    def __call__(self, x: RecurrentTensor) -> tuple[RecurrentTensor, RecurrentTensor]:
        return self.forward(x)
