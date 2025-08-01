from functools import partial

from tempo.api import RecurrentTensor
from tempo.api.nn.module import MaybeInitFn, Module
from tempo.core.domain import DomainLike
from tempo.core.dtype import DataTypeLike, dtypes
from tempo.core.shape import Shape, ShapeLike


class RMSNorm(Module):
    def __init__(
        self,
        normalized_shape: ShapeLike,
        eps: float = 1e-7,
        elementwise_affine: bool = True,
        dtype: DataTypeLike = None,
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
        w_init_fun: MaybeInitFn = None,
    ) -> None:
        """
        The RMS Norm is taken on the last N dimensions of an input tensor,
        where the shape of the last N dimension is specified by `normalized_shape`.
        Here it will only be applied on the last N dimensions
        of the spatial shape of the input tensor.
        """
        super().__init__(domain, independent_domain)
        dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)
        self.eps = eps
        self.normalized_shape = Shape.from_(normalized_shape)
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            if w_init_fun is None:
                w_init_fun = partial(
                    RecurrentTensor.ones, requires_grad=True
                )  # TODO is this an actually good init fun?

            # Shape of scaling parameters is the same as normalized_shape
            self.w = self.param_from_init(
                w_init_fun(
                    shape=self.normalized_shape,
                    dtype=dtype,
                    domain=independent_domain,
                )
            )
        else:
            assert w_init_fun is None, "w_init_fun must be None if elementwise_affine is False"

    def forward(self, x: RecurrentTensor) -> RecurrentTensor:
        """
        x is a RecurrentTensor maybe with a domain (iteration,batch...),
        plus a spatial shape that ends with 'normalized_shape'.
        E.g.: domain=(iteration,batch), shape=(A,B,C).
        If normalized_shape=(B,C), this reduces over the last 2 dims
        """
        normed = self._norm(x.cast(dtypes.float32)).cast(x.dtype)

        if self.elementwise_affine:
            return normed * self.w
        else:
            return normed

    def _norm(self, x: RecurrentTensor) -> RecurrentTensor:
        n_dims = len(self.normalized_shape)

        x_last_n_dims = x.shape[-n_dims:]

        # check if normalized_shape matches the last
        # N dimensions of spatial shape of the input tensor
        if x_last_n_dims != self.normalized_shape:
            raise ValueError(
                f"RMSNorm mismatch: last {n_dims} dims of x are {x_last_n_dims}, "
                f"but normalized_shape/weight is {self.normalized_shape}"
            )

        dims_to_reduce = tuple(range(-n_dims, 0))
        mean_sq = (x * x).mean(dims=dims_to_reduce, keepdim=True)
        rms = (mean_sq + self.eps).sqrt()
        normed = x / rms
        return normed
