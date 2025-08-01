from __future__ import annotations

from abc import ABC
from typing import Any, Type, Union

from tempo.api.nn.module import Module
from tempo.api.recurrent_tensor import RecurrentTensor


class ActivationFunction(Module, ABC):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    @staticmethod
    def from_(act: ActivationFunctionLike) -> ActivationFunction:
        if isinstance(act, ActivationFunction):
            return act
        return get_act_fun_class(act)()


ActivationFunctionLike = Union[ActivationFunction, Type[ActivationFunction], str]


def get_act_fun_class(activation: ActivationFunctionLike) -> Type[ActivationFunction]:
    mapping = {
        "relu": Relu,
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "swish": Swish,
        "silu": Swish,
        "softmax": Softmax,
        "elu": Elu,
        "leakyrelu": LeakyRelu,
        "mish": Mish,
        "softplus": Softplus,
    }
    if isinstance(activation, str):
        return mapping[activation.lower().strip()]  # type: ignore
    elif isinstance(activation, ActivationFunction):
        return type(activation)
    else:
        return activation


class Relu(ActivationFunction):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, x: RecurrentTensor) -> Any:
        return x.relu()


class Sigmoid(ActivationFunction):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, x: RecurrentTensor) -> Any:
        return x.sigmoid()


class Tanh(ActivationFunction):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, x: RecurrentTensor) -> Any:
        result = x.tanh()
        return result


class Swish(ActivationFunction):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, x: RecurrentTensor) -> Any:
        return x.swish()


class Softmax(ActivationFunction):
    def __init__(
        self,
        dim: int = -1,
        stable: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.stable = stable

    def forward(self, x: RecurrentTensor) -> Any:
        return x.softmax(self.dim, self.stable)


class Elu(ActivationFunction):
    def __init__(
        self,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: RecurrentTensor) -> Any:
        return x.elu(self.alpha)


class LeakyRelu(ActivationFunction):
    def __init__(
        self,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: RecurrentTensor) -> Any:
        return x.leakyrelu(self.negative_slope)


class Mish(ActivationFunction):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, x: RecurrentTensor) -> Any:
        return x.mish()


class Softplus(ActivationFunction):
    def __init__(
        self,
        beta: int = 1,
    ) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, x: RecurrentTensor) -> Any:
        return x.softplus(self.beta)
