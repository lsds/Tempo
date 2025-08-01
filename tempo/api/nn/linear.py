from functools import partial
from typing import Any

from tempo.api.nn.module import MaybeInitFn, Module
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.core.domain import DomainLike
from tempo.core.dtype import DataTypeLike, dtypes

default_initialization_function = RecurrentTensor.linear_init_uniform


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: DataTypeLike = None,
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
        w_init_fun: MaybeInitFn = None,
        b_init_fun: MaybeInitFn = None,
    ) -> None:
        super().__init__(domain, independent_domain)

        dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

        if w_init_fun is None:
            w_init_fun = partial(
                RecurrentTensor.linear_init_uniform,
                num_input_features=in_features,
                # gain=float(np.sqrt(2)),
            )
        if b_init_fun is None:
            b_init_fun = partial(RecurrentTensor.zeros, requires_grad=False)  # type: ignore

        assert b_init_fun is not None

        self.w = self.param_from_init(
            w_init_fun(
                # shape=(in_features, out_features),
                shape=(out_features, in_features),
                dtype=dtype,
                domain=independent_domain,
            )
        )
        self.b = (
            self.bias_from_init(
                b_init_fun(
                    shape=(out_features,),
                    dtype=dtype,
                    domain=independent_domain,
                )
            )
            if bias
            else None
        )

    def forward(self, x: RecurrentTensor) -> Any:
        result = x @ self.w.transpose()
        # result = x @ self.w

        if self.b is not None:
            result = result + self.b

        return result
