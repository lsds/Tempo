from typing import Tuple

from tempo.api.nn.linear import Linear
from tempo.api.nn.module import Module
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.core.domain import DomainLike
from tempo.core.dtype import DataType, dtypes


class GRUCell(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        dtype: DataType = dtypes.float32,
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
        # TODO: add init_fns
    ) -> None:
        super().__init__(domain, independent_domain)

        self.ih_lin = Linear(
            input_size,
            3 * hidden_size,
            bias=bias,
            dtype=dtype,
            domain=domain,
            independent_domain=independent_domain,
        )

        self.hh_lin = Linear(
            hidden_size,
            3 * hidden_size,
            bias=bias,
            dtype=dtype,
            domain=domain,
            independent_domain=independent_domain,
        )

    def forward(self, x: RecurrentTensor, h: RecurrentTensor) -> RecurrentTensor:
        gates: RecurrentTensor = self.ih_lin(x) + self.hh_lin(h)
        z, r, n = gates.chunk(3, dim=1)
        z, r, n = z.sigmoid(), r.sigmoid(), n.tanh()
        new_h: RecurrentTensor = (1 - z) * n + z * h
        return new_h


class LSTMCell(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        dtype: DataType = dtypes.float32,
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
    ) -> None:
        super().__init__(domain, independent_domain)

        self.ih_lin = Linear(
            input_size,
            4 * hidden_size,
            bias=bias,
            dtype=dtype,
            domain=domain,
            independent_domain=independent_domain,
        )

        self.hh_lin = Linear(
            hidden_size,
            4 * hidden_size,
            bias=bias,
            dtype=dtype,
            domain=domain,
            independent_domain=independent_domain,
        )

    def forward(
        self, x: RecurrentTensor, hc: Tuple[RecurrentTensor, RecurrentTensor]
    ) -> Tuple[RecurrentTensor, RecurrentTensor]:
        h, c = hc
        gates: RecurrentTensor = self.ih_lin(x) + self.hh_lin(h)
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o, g = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
        new_c = f * c + i * g
        new_h = o * new_c.tanh()
        return new_h, new_c
