from functools import partial
from typing import Any

from tempo.api.nn.module import MaybeInitFn, Module
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.core.domain import DomainLike
from tempo.core.dtype import DataTypeLike, dtypes


class Embedding(Module):
    def __init__(
        self,
        vocab_size: int,
        encoded_dim_size: int,
        dtype: DataTypeLike = None,
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
        w_init_fun: MaybeInitFn = None,
    ) -> None:
        super().__init__(domain, independent_domain)

        dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

        if w_init_fun is None:
            w_init_fun = partial(
                RecurrentTensor.normal,  # NOTE: torch uses normal init for embeddings
            )

        self.embed_table = self.param_from_init(
            w_init_fun(shape=(vocab_size, encoded_dim_size), dtype=dtype, domain=domain),
            regularize=False,
        )

    def forward(self, x: RecurrentTensor) -> Any:
        assert dtypes.is_integer(x.dtype), "Embedding input must be integer"
        embedded = self.embed_table.index(dim=0, index=x)

        return embedded
