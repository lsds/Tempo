from tempo.api.nn.conv import _ConvNd
from tempo.api.nn.module import MaybeInitFn
from tempo.core.domain import DomainLike
from tempo.core.dtype import DataType


class ConvTransposed1d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        output_padding: int = 0,
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
            transposed=True,
            output_padding=(output_padding,),
            groups=groups,
            bias=bias,
            dtype=dtype,
            domain=domain,
            independent_domain=independent_domain,
            w_init_fun=w_init_fun,
            b_init_fun=b_init_fun,
        )


class ConvTransposed2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        output_padding: int | tuple[int, int] = 0,
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
            transposed=True,
            output_padding=(
                (output_padding,) * 2 if isinstance(output_padding, int) else output_padding
            ),
            groups=groups,
            bias=bias,
            dtype=dtype,
            domain=domain,
            independent_domain=independent_domain,
            w_init_fun=w_init_fun,
            b_init_fun=b_init_fun,
        )


class ConvTransposed3d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        output_padding: int | tuple[int, int, int] = 0,
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
            transposed=True,
            output_padding=(
                (output_padding,) * 3 if isinstance(output_padding, int) else output_padding
            ),
            groups=groups,
            bias=bias,
            dtype=dtype,
            domain=domain,
            independent_domain=independent_domain,
            w_init_fun=w_init_fun,
            b_init_fun=b_init_fun,
        )
