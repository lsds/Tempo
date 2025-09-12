from functools import partial
from typing import Any

from tempo.api.nn.module import MaybeInitFn, Module
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.core.domain import DomainLike
from tempo.core.dtype import DataType, DataTypeLike, dtypes

# TODO: output_padding is wrong. See note in:
# https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html


class _ConvNd(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: tuple[int, ...],
        dilation: tuple[int, ...],
        transposed: bool,
        output_padding: tuple[int, ...],
        groups: int,
        bias: bool,
        dtype: DataTypeLike = None,
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
        w_init_fun: MaybeInitFn = None,
        b_init_fun: MaybeInitFn = None,
    ) -> None:
        super().__init__(domain, independent_domain)
        dtype = dtypes.from_(dtype, none_dtype=dtypes.default_float)

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        if w_init_fun is None:
            w_init_fun = partial(RecurrentTensor.kaiming_normal, a=0.01)
        if b_init_fun is None:
            b_init_fun = RecurrentTensor.linear_init_uniform

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups

        channels = in_channels if not transposed else out_channels
        other_channels = out_channels if not transposed else in_channels

        self.w = self.param_from_init(
            w_init_fun(
                shape=(other_channels, channels // groups, *kernel_size),
                dtype=dtype,
                domain=independent_domain,
            )
        )

        self.b = (
            self.bias_from_init(
                b_init_fun(
                    shape=(other_channels,) + (1,) * len(self.stride),
                    dtype=dtype,
                    domain=independent_domain,
                )
            )
            if bias
            else None
        )

    def forward(self, x: RecurrentTensor) -> Any:
        n_dims = len(self.stride)
        res = x.conv_general(
            self.w,
            n_dims=n_dims,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            transposed=self.transposed,
            output_padding=self.output_padding,
            groups=self.groups,
        )
        if self.b is not None:
            res = res + self.b

        return res


class Conv1d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: DataType | None = None,
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
        w_init_fun: MaybeInitFn = None,
        b_init_fun: MaybeInitFn = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=(kernel_size,),
            stride=(stride,),
            padding=(padding,),
            dilation=(dilation,),
            transposed=False,
            output_padding=(0,),
            groups=groups,
            bias=bias,
            dtype=dtype,
            domain=domain,
            independent_domain=independent_domain,
            w_init_fun=w_init_fun,
            b_init_fun=b_init_fun,
        )


class Conv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: DataType | None = None,
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
        w_init_fun: MaybeInitFn = None,
        b_init_fun: MaybeInitFn = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=((kernel_size,) * 2 if isinstance(kernel_size, int) else kernel_size),
            stride=(stride,) * 2 if isinstance(stride, int) else stride,
            padding=(padding,) * 2 if isinstance(padding, int) else padding,
            dilation=(dilation,) * 2 if isinstance(dilation, int) else dilation,
            transposed=False,
            output_padding=(0,) * 2,
            groups=groups,
            bias=bias,
            dtype=dtype,
            domain=domain,
            independent_domain=independent_domain,
            w_init_fun=w_init_fun,
            b_init_fun=b_init_fun,
        )


class Conv3d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: DataType | None = None,
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
        w_init_fun: MaybeInitFn = None,
        b_init_fun: MaybeInitFn = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=((kernel_size,) * 3 if isinstance(kernel_size, int) else kernel_size),
            stride=(stride,) * 3 if isinstance(stride, int) else stride,
            padding=(padding,) * 3 if isinstance(padding, int) else padding,
            dilation=(dilation,) * 3 if isinstance(dilation, int) else dilation,
            transposed=False,
            output_padding=(0,) * 3,
            groups=groups,
            bias=bias,
            dtype=dtype,
            domain=domain,
            independent_domain=independent_domain,
            w_init_fun=w_init_fun,
            b_init_fun=b_init_fun,
        )
