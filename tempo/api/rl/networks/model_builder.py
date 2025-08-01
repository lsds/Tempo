# type: ignore
# TODO: remove this
from __future__ import annotations

from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Type, Union

import gymnasium as gym
import numpy as np

from tempo.api.distributions.categorical import Categorical
from tempo.api.distributions.normal import Normal
from tempo.api.nn import rnn
from tempo.api.nn.activation import ActivationFunction, ActivationFunctionLike
from tempo.api.nn.conv import Conv2d
from tempo.api.nn.flatten import Flatten
from tempo.api.nn.linear import Linear
from tempo.api.nn.module import MaybeInitFn, Module, Sequential
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.api.rl.datatypes import (
    ActionsRecurrentTensor,
    ObservationsRecurrentTensor,
    ValueRecurrentTensor,
)
from tempo.api.rl.env.env import Env
from tempo.api.rl.networks.network import Network
from tempo.core import index_expr as ie
from tempo.core.domain import DomainLike
from tempo.core.dtype import DataType
from tempo.core.shape import Shape, StaticShape, StaticShapeLike

NumInChannels = int
NumOutChannels = int
KernelSize = int
Stride = int
Padding = int
ConvDescription = Tuple[
    NumInChannels, NumOutChannels, KernelSize, Stride, Padding, ActivationFunctionLike
]

EPS = 1e-6


class ModelBuilder:
    def __init__(
        self,
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
        w_init_fun: MaybeInitFn = None,
    ) -> None:
        self.domain = domain
        self.independent_domain = independent_domain

        self._encoder: Optional[Module] = None
        self._encoder_latent_size: Optional[int] = None

        self._hidden: Optional[Module] = None
        self._last_hidden_size: Optional[int] = None

        self._decoder1: Optional[Module] = None
        self._decoder2: Optional[Module] = None
        self._has_normal_dist_decoder: bool = False
        self._is_q_pi: bool = False
        self._is_v_pi: bool = False
        self._is_recurrent: bool = False

        if w_init_fun is None:
            # w_init_fun = RecurrentTensor.glorot_uniform
            w_init_fun = partial(RecurrentTensor.orthogonal_init, gain=float(np.sqrt(2)))
        self.w_init_fun = w_init_fun

    @staticmethod
    def from_env(
        hidden: Sequence[int],
        domain: Sequence[ie.Symbol],
        env: Env,
        actor_critic: bool = False,
    ) -> Network:
        mod = (
            ModelBuilder(
                domain=domain,
                w_init_fun=partial(RecurrentTensor.orthogonal_init, gain=np.sqrt(2)),
            )
            .with_ff_encoder(env.observation_shape, hidden[0])
            .with_ff_hidden(hidden)
        )

        if isinstance(env.action_space, gym.spaces.Discrete):
            mod = mod.with_discrete_pi_decoder(int(env.action_space.n))
        else:
            mod = mod.with_continuous_pi_decoder(env.action_shape)  # , squish=True

        if actor_critic:
            return mod.with_vfun_decoder().build()
        else:
            return mod.with_no_second_decoder().build()

        raise NotImplementedError(f"Unsupported action space: {env.action_space}")

    @property
    def encoder(self) -> Module:
        assert self._encoder is not None
        return self._encoder

    @property
    def hidden(self) -> Module:
        assert self._hidden is not None
        return self._hidden

    @property
    def decoder1(self) -> Module:
        assert self._decoder1 is not None
        return self._decoder1

    @property
    def decoder2(self) -> Module:
        assert self._decoder2 is not None
        return self._decoder2

    def with_ff_encoder(
        self,
        obs_shape: StaticShape,
        encoded_latent_size: int,
        bias: bool = True,
        activation_fun: Optional[ActivationFunctionLike] = None,
    ) -> EncoderChosenModelBuilder:
        in_features = obs_shape.prod()

        if not activation_fun:
            activation_fun = "tanh"
        enc_modules = [
            Flatten(start_dim=1),
            Linear(
                in_features,
                encoded_latent_size,
                bias,
                domain=self.domain,
                independent_domain=self.independent_domain,
                w_init_fun=self.w_init_fun,  # type: ignore
            ),
            ActivationFunction.from_(activation_fun),
        ]
        self._encoder = Sequential(
            *enc_modules,
            domain=self.domain,
            independent_domain=self.independent_domain,
        )
        self._encoder_latent_size = encoded_latent_size
        return EncoderChosenModelBuilder(self)

    # CNN layers should init he by default
    def with_cnn_encoder(self, convs: Sequence[ConvDescription]) -> EncoderChosenModelBuilder:
        cnn_layers: List[Module] = []
        for conv_desc in convs:
            (
                num_in_channels,
                num_out_channels,
                kernel_size,
                stride,
                padding,
                activation,
            ) = conv_desc
            cnn_layers.append(
                Conv2d(
                    num_in_channels,
                    num_out_channels,
                    kernel_size,
                    stride,
                    padding,
                    domain=self.domain,
                    independent_domain=self.independent_domain,
                )
            )
            cnn_layers.append(ActivationFunction.from_(activation))

        self._encoder = Sequential(
            *cnn_layers, domain=self.domain, independent_domain=self.independent_domain
        )
        return EncoderChosenModelBuilder(self)


class EncoderChosenModelBuilder:
    def __init__(self, network_builder: ModelBuilder) -> None:
        self._network_builder = network_builder

    def with_ff_hidden(
        self,
        hidden_sizes: Sequence[int],
        bias: bool = True,
        activation_fun: Optional[ActivationFunctionLike] = None,
    ) -> HiddenChosenModelBuilder:
        if not activation_fun:
            activation_fun = "tanh"

        last_sz = self._network_builder._encoder_latent_size
        assert last_sz is not None
        layers: List[Module] = []
        for sz in hidden_sizes:
            layers.append(
                Linear(
                    last_sz,
                    sz,
                    bias,
                    domain=self._network_builder.domain,
                    independent_domain=self._network_builder.independent_domain,
                    w_init_fun=self._network_builder.w_init_fun,  # type: ignore
                )
            )
            layers.append(ActivationFunction.from_(activation_fun))

            last_sz = sz

        self._network_builder._last_hidden_size = last_sz

        self._network_builder._hidden = Sequential(
            *layers,
            domain=self._network_builder.domain,
            independent_domain=self._network_builder.independent_domain,
        )
        return HiddenChosenModelBuilder(self._network_builder)

    def with_recurrent_ff_hidden(
        self,
        hidden_sizes: Sequence[int],
        recurren_hidden_size: int,
        bias: bool = True,
        activation_fun: Optional[ActivationFunctionLike] = None,
        recurrent_type: str = "lstm",
    ) -> HiddenChosenModelBuilder:
        if not activation_fun:
            activation_fun = "tanh"

        last_sz = self._network_builder._encoder_latent_size
        assert last_sz is not None
        layers: List[Module] = []
        for sz in hidden_sizes:
            layers.append(
                Linear(
                    last_sz,
                    sz,
                    bias,
                    domain=self._network_builder.domain,
                    independent_domain=self._network_builder.independent_domain,
                    w_init_fun=self._network_builder.w_init_fun,  # type: ignore
                )
            )
            layers.append(ActivationFunction.from_(activation_fun))

            last_sz = sz

        if recurrent_type == "lstm":
            cell_type: Type[Module] = rnn.LSTMCell
        elif recurrent_type == "gru":
            cell_type = rnn.GRUCell
        else:
            raise ValueError(f"Unknown recurrent type {recurrent_type}")

        layers.append(
            cell_type(
                input_size=last_sz,  # type: ignore
                hidden_size=recurren_hidden_size,  # type: ignore
                bias=bias,  # type: ignore
                domain=self._network_builder.domain,  # type: ignore
                independent_domain=self._network_builder.independent_domain,  # type: ignore
            )
        )
        last_sz = sz

        self._network_builder._last_hidden_size = last_sz

        self._network_builder._is_recurrent = True

        return HiddenChosenModelBuilder(self._network_builder)


class NormalDistDecoder(Module):
    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        bias: bool = True,
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
        squish: bool = False,
        w_init_fun: Optional[Callable[[Shape, DataType], RecurrentTensor]] = None,
    ) -> None:
        super().__init__(domain, independent_domain)
        # Initialization for gaussian distribution:
        # https://github.com/openai/baselines/blob/
        #   ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/distributions.py#L96

        # https://github.com/DLR-RM/stable-baselines3/blob/
        #   085bdd5a68be7748e62ca710162f16ac7b62ddfc/stable_baselines3/common/distributions.py#L138
        self.mu_layers_module = Sequential(
            Linear(
                in_dims,
                out_dims,
                bias,
                domain=domain,
                independent_domain=independent_domain,
                w_init_fun=partial(RecurrentTensor.orthogonal_init, gain=0.01),  # type: ignore
            ),
            domain=domain,
            independent_domain=independent_domain,
        )
        self.log_sigma = self.bias_from_init(RecurrentTensor.zeros((out_dims,)))
        self._squish = squish

    def forward(self, x: RecurrentTensor) -> Tuple[RecurrentTensor, RecurrentTensor]:
        mu: RecurrentTensor = self.mu_layers_module(x)  # type: ignore
        exped = self.log_sigma.exp()
        sigma = exped.expand(mu.shape)

        return mu, sigma


class HiddenChosenModelBuilder:
    def __init__(self, network_builder: ModelBuilder) -> None:
        self._network_builder = network_builder

    def with_qfun_decoder(self, num_outputs: int, bias: bool = True) -> DecoderChosenModelBuilder:
        assert self._network_builder._last_hidden_size is not None
        self._network_builder._decoder1 = Linear(
            self._network_builder._last_hidden_size,
            num_outputs,
            bias,
            domain=self._network_builder.domain,
            independent_domain=self._network_builder.independent_domain,
            w_init_fun=self._network_builder.w_init_fun,  # type: ignore
        )
        self._network_builder._is_q_pi = True
        return DecoderChosenModelBuilder(self._network_builder)

    def with_vfun_decoder_only(self, bias: bool = True) -> DecoderChosenModelBuilder:
        assert self._network_builder._last_hidden_size is not None
        self._network_builder._decoder1 = Linear(
            self._network_builder._last_hidden_size,
            1,
            bias,
            domain=self._network_builder.domain,
            independent_domain=self._network_builder.independent_domain,
            w_init_fun=self._network_builder.w_init_fun,  # type: ignore
        )
        self._network_builder._is_v_pi = True
        return DecoderChosenModelBuilder(self._network_builder)

    # TODO here we should accept an action shape like below and then use a reshape on the output
    # if needed
    def with_discrete_pi_decoder(
        self, num_actions: int, bias: bool = True
    ) -> FirstDecoderHeadChosenModelBuilder:
        assert self._network_builder._last_hidden_size is not None
        layers = [
            Linear(
                self._network_builder._last_hidden_size,
                num_actions,
                bias,
                domain=self._network_builder.domain,
                independent_domain=self._network_builder.independent_domain,
                w_init_fun=self._network_builder.w_init_fun,  # type: ignore
            ),
            # ActivationFunction.from_("softmax"),
        ]
        self._network_builder._decoder1 = Sequential(
            *layers,
            domain=self._network_builder.domain,
            independent_domain=self._network_builder.independent_domain,
        )
        return FirstDecoderHeadChosenModelBuilder(self._network_builder)

    def with_continuous_pi_decoder(
        self, action_shape: StaticShapeLike, bias: bool = False, squish: bool = False
    ) -> FirstDecoderHeadChosenModelBuilder:
        action_shape = StaticShape.from_(action_shape)
        assert len(action_shape) == 1
        assert self._network_builder._last_hidden_size is not None
        self._network_builder._decoder1 = NormalDistDecoder(
            self._network_builder._last_hidden_size,
            action_shape.prod(),
            bias,
            domain=self._network_builder.domain,
            independent_domain=self._network_builder.independent_domain,
            squish=squish,
            w_init_fun=self._network_builder.w_init_fun,  # type: ignore
        )
        self._network_builder._has_normal_dist_decoder = True
        return FirstDecoderHeadChosenModelBuilder(self._network_builder)


class FirstDecoderHeadChosenModelBuilder:
    def __init__(self, network_builder: ModelBuilder) -> None:
        self._network_builder = network_builder

    def with_vfun_decoder(self, bias: bool = True) -> DecoderChosenModelBuilder:
        assert self._network_builder._last_hidden_size is not None
        self._network_builder._decoder2 = Linear(
            self._network_builder._last_hidden_size,
            1,
            bias,
            domain=self._network_builder.domain,
            independent_domain=self._network_builder.independent_domain,
            w_init_fun=partial(RecurrentTensor.orthogonal_init, gain=1.0),  # type: ignore
        )
        return DecoderChosenModelBuilder(self._network_builder)

    def with_no_second_decoder(self) -> DecoderChosenModelBuilder:
        return DecoderChosenModelBuilder(self._network_builder)


class QNetwork(Network):
    def __init__(self, builder: ModelBuilder) -> None:
        super().__init__(builder.domain, builder.independent_domain)
        self.is_recurrent = builder._is_recurrent

        self._encoder: Module = builder.encoder
        self._hidden: Module = builder.hidden
        self._decoder: Module = builder.decoder1
        self._decoder2: Optional[Module] = builder._decoder2

    def forward(
        self,
        obs: Union[ObservationsRecurrentTensor, RecurrentTensor],
    ) -> Tuple[Optional[ActionsRecurrentTensor], Optional[ValueRecurrentTensor]]:
        x = self._encoder(obs)
        x = self._hidden(x)
        self.q_values: RecurrentTensor = self._decoder(x)  # type: ignore
        action = ActionsRecurrentTensor(self.q_values.argmax(dim=-1))

        value = None
        if self._decoder2:
            value = ValueRecurrentTensor(self._decoder2(x))

        return action, value

    def qvalues(
        self,
    ) -> RecurrentTensor:
        return self.q_values

    def log_prob(self, actions: Union[ActionsRecurrentTensor, RecurrentTensor]) -> RecurrentTensor:
        raise ValueError("QNetwork does not have a log_prob method")

    def entropy(self) -> RecurrentTensor:
        raise ValueError("QNetwork does not have an entropy method")


class VNetwork(Network):
    def __init__(self, builder: ModelBuilder) -> None:
        super().__init__(builder.domain, builder.independent_domain)

        self._encoder: Module = builder.encoder
        self._hidden: Module = builder.hidden
        self._decoder: Module = builder.decoder1
        self._decoder2: Optional[Module] = builder._decoder2

    def forward(
        self, obs: Union[ObservationsRecurrentTensor, RecurrentTensor]
    ) -> Tuple[Optional[ActionsRecurrentTensor], ValueRecurrentTensor]:
        x = self._encoder(obs)
        x = self._hidden(x)
        v: RecurrentTensor = self._decoder(x)  # type: ignore
        return None, v  # type: ignore

    def qvalues(
        self,
    ) -> RecurrentTensor:
        raise ValueError("VNetwork does not have a qvalues method")

    def log_prob(self, actions: Union[ActionsRecurrentTensor, RecurrentTensor]) -> RecurrentTensor:
        raise ValueError("VNetwork does not have a log_prob method")

    def entropy(self) -> RecurrentTensor:
        raise ValueError("VNetwork does not have an entropy method")


class CategoricalDistPiDecoder(Network):
    def __init__(self, builder: ModelBuilder) -> None:
        super().__init__(builder.domain, builder.independent_domain)

        self._encoder: Module = builder.encoder
        self._hidden: Module = builder.hidden
        self._decoder: Module = builder.decoder1
        self._decoder2: Optional[Module] = builder._decoder2

    def forward(
        self,
        obs: Union[ObservationsRecurrentTensor, RecurrentTensor],
    ) -> Tuple[Optional[ActionsRecurrentTensor], Optional[ValueRecurrentTensor]]:
        x = self._encoder(obs)
        x = self._hidden(x)
        logits = self._decoder(x)  # type: ignore
        logits = logits.softmax()

        self.dist = Categorical(logits)
        action = ActionsRecurrentTensor(self.dist.sample())

        value = None
        if self._decoder2:
            value = ValueRecurrentTensor(self._decoder2(x))

        return action, value

    def log_prob(self, actions: Union[ActionsRecurrentTensor, RecurrentTensor]) -> RecurrentTensor:
        if self.dist is None:
            raise ValueError("No distribution has been computed yet. Please call forward first.")
        return self.dist.log_prob(actions)  # type: ignore

    def entropy(self) -> RecurrentTensor:
        if self.dist is None:
            raise ValueError("No distribution has been computed yet. Please call forward first.")
        return self.dist.entropy()  # type: ignore

    def qvalues(
        self,
    ) -> RecurrentTensor:
        raise ValueError("Network does not have Q Values")


class NormalDistPiNetwork(Network):
    def __init__(self, builder: ModelBuilder) -> None:
        super().__init__(builder.domain, builder.independent_domain)

        self._encoder: Module = builder.encoder
        self._hidden: Module = builder.hidden
        self._decoder: Module = builder.decoder1
        self._decoder2: Optional[Module] = builder._decoder2
        assert isinstance(self._decoder, NormalDistDecoder)
        self._squish = self._decoder._squish

        self._actions_base: Optional[RecurrentTensor] = None

    def forward(
        self,
        obs: Union[ObservationsRecurrentTensor, RecurrentTensor],
    ) -> Tuple[ActionsRecurrentTensor, Optional[ValueRecurrentTensor]]:
        # obs.domain.ubound_overrides not empty
        x1 = self._encoder(obs)

        x2 = self._hidden(x1)
        mu, sigma = self._decoder(x2)  # type: ignore

        # we shouldn't need these squeeze calls
        self.mu, self.sigma = mu, sigma

        self.dist = Normal(mu, sigma)
        action = self.dist.sample()
        self._actions_base = action
        if self._squish:
            action = action.tanh()

        action = ActionsRecurrentTensor(action)
        value = None
        if self._decoder2:
            value = ValueRecurrentTensor(self._decoder2(x2))

        return action, value

    def log_prob(self, actions: Union[ActionsRecurrentTensor, RecurrentTensor]) -> RecurrentTensor:
        if self.dist is None:
            raise ValueError("No distribution has been computed yet. Please call forward first.")

        if self._squish:
            log_prob = self.dist.log_prob(self._actions_base).sum(-1)  # type: ignore
            log_prob -= (1 - actions.tanh() ** 2 + EPS).log().sum(dims=-1)
            return log_prob  # type: ignore
        else:
            log_pr = self.dist.log_prob(actions)
            summed = log_pr.sum(-1)  # type: ignore
            return summed

    def entropy(self) -> RecurrentTensor:
        if self.dist is None:
            raise ValueError("No distribution has been computed yet. Please call forward first.")
        return self.dist.entropy().sum(-1)  # type: ignore

    def qvalues(
        self,
    ) -> RecurrentTensor:
        raise ValueError("Network does not have Q Values")


class DecoderChosenModelBuilder:
    def __init__(self, network_builder: ModelBuilder) -> None:
        self._network_builder = network_builder

    def build(self) -> Network:
        if self._network_builder._is_q_pi:
            return QNetwork(self._network_builder)
        elif self._network_builder._is_v_pi:
            return VNetwork(self._network_builder)
        elif self._network_builder._has_normal_dist_decoder:
            return NormalDistPiNetwork(self._network_builder)
        else:
            return CategoricalDistPiDecoder(self._network_builder)
