from tempo.api.nn import Linear
from tempo.api.nn.module import MaybeInitFn, Module
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.core.domain import DomainLike
from tempo.core.dtype import DataType, dtypes


class SwiGLU(Module):
    """SwiGLU (Swish-Gated Linear Unit) activation.

    SwiGLU(x) = Swish(xW + b) ⊙ (xV + c)

    Where:
    - Swish(x) = x ⊙ sigmoid(x)
    - ⊙ is element-wise multiplication

    Args:
        in_features: Input feature dimensionality
        out_features: Output feature dimensionality
        dtype: Data type
        domain: Domain for the module
        independent_domain: Independent dimensions (not affected by the accumulation domain)
        w_init_fun: Initialization function for gate weights
        v_init_fun: Initialization function for linear weights
        b_init_fun: Initialization function for gate bias
        c_init_fun: Initialization function for linear bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: DataType = dtypes.float32,
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
        bias: bool = True,
        w_init_fun: MaybeInitFn = None,
        v_init_fun: MaybeInitFn = None,
        b_init_fun: MaybeInitFn = None,
        c_init_fun: MaybeInitFn = None,
    ) -> None:
        super().__init__(domain, independent_domain)
        wrapped_w = (
            None
            if w_init_fun is None
            else lambda shape, dtype, domain: w_init_fun(shape=shape, dtype=dtype, domain=domain)
        )
        wrapped_b = (
            None
            if b_init_fun is None
            else lambda shape, dtype, domain: b_init_fun(shape=shape, dtype=dtype, domain=domain)
        )
        wrapped_v = (
            None
            if v_init_fun is None
            else lambda shape, dtype, domain: v_init_fun(shape=shape, dtype=dtype, domain=domain)
        )
        wrapped_c = (
            None
            if c_init_fun is None
            else lambda shape, dtype, domain: c_init_fun(shape=shape, dtype=dtype, domain=domain)
        )

        self.linear_layer = Linear(
            in_features,
            out_features,
            dtype=dtype,
            domain=domain,
            bias=bias,
            independent_domain=independent_domain,
            w_init_fun=wrapped_v,
            b_init_fun=wrapped_c,
        )
        self.swish_linear_layer = Linear(
            in_features,
            out_features,
            dtype=dtype,
            domain=domain,
            bias=bias,
            independent_domain=independent_domain,
            w_init_fun=wrapped_w,
            b_init_fun=wrapped_b,
        )

    def forward(self, x: RecurrentTensor) -> RecurrentTensor:
        # Gate part: Swish(xW + b)
        switch_linear = self.swish_linear_layer(x)
        gate: RecurrentTensor = (
            switch_linear.cast(dtypes.upcast(switch_linear.dtype, dtypes.float32))
            .swish()
            .cast(switch_linear.dtype)
        )

        # Linear part: xV + c
        linear: RecurrentTensor = self.linear_layer(x)

        # Swish(xW + b) ⊙ (xV + c)
        return gate * linear
