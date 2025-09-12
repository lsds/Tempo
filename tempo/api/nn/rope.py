from __future__ import annotations

import math
from dataclasses import dataclass

from tempo.api.nn.module import Module
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.core.domain import DomainLike
from tempo.core.dtype import dtypes
from tempo.core.index_expr import Symbol
from tempo.utils.logger import get_logger

_logger = get_logger(__name__)


@dataclass
class RopeScalingParams:
    factor: float = 32.0
    high_freq_factor: float = 4.0
    low_freq_factor: float = 1.0
    original_max_position_embeddings: int = 8192
    rope_type: str = "llama3"


def _compute_inv_freq(dim: int, theta: int | float) -> RecurrentTensor:
    arange_ = RecurrentTensor.arange(stop=dim, step=2, dtype=dtypes.float32).slice_dim(
        0, start=0, stop=dim // 2
    )
    inv_freq: RecurrentTensor = 1.0 / (RecurrentTensor.lift(theta) ** (arange_ / dim))
    # inv_freq.sink_with_ts_udf(lambda x, ts: print(f"inv_freq: {x}, ts: {ts}", flush=True))
    return inv_freq


## https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/llama/modeling_llama.py#L147
# def _compute_rope_scaling_llama3_dynamic(
#    angles: RecurrentTensor, rope_scaling_params: RopeScalingParams, t: Symbol, dim: int, base: int
# ) -> RecurrentTensor:
#    """
#    Computes the scaled inverse frequencies for Llama 3 RoPE scaling.
#    """
#    scaling_factor = rope_scaling_params.factor
#    max_position_embeddings = rope_scaling_params.original_max_position_embeddings
#
#    seq_len = RecurrentTensor.lift(t)
#    scaled_base = base * (
#        (scaling_factor * seq_len / max_position_embeddings) - (scaling_factor - 1)
#    ) ** (dim / (dim - 2))
#    inv_freq_dyn = 1.0 / (
#        scaled_base ** (RecurrentTensor.arange(dim, start=0, step=2, dtype=dtypes.float32) / dim)
#    )
#
#    inv_freq = RecurrentTensor.placeholder_like(angles)
#    inv_freq[t <= max_position_embeddings] = angles  # TODO < or <=?
#    inv_freq[True] = inv_freq_dyn
#
#    return inv_freq


# https://github.com/meta-llama/llama-models/blob/v0.1.4/models/llama3/reference_impl/model.py#L69
# https://github.com/huggingface/transformers/blob/5111c8ea2f3eb918fc090f7dd4393d4204940e10/src/transformers/modeling_rope_utils.py#L393
def _compute_rope_scaling_llama3(
    rope_scaling_params: RopeScalingParams,
    dim: int,
    base: int | float,
) -> RecurrentTensor:
    """
    Computes the scaled inverse frequencies for Llama 3 RoPE scaling.

    Returns:
        Scaled inverse frequencies
    """

    print(f"Doing llama3 rope scaling with dim={dim}, base={base} and params={rope_scaling_params}")

    inv_freq = _compute_inv_freq(dim, base)  # type: ignore

    # Extract parameters
    factor = rope_scaling_params.factor
    low_freq_factor = rope_scaling_params.low_freq_factor
    high_freq_factor = rope_scaling_params.high_freq_factor
    old_context_len = rope_scaling_params.original_max_position_embeddings

    # Compute wavelength boundaries
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    # Compute wavelength from inverse frequency
    wavelen = 2 * math.pi / inv_freq

    # Apply scaling based on frequency ranges
    # wavelen < high_freq_wavelen: do nothing (keep original angles)
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = RecurrentTensor.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)

    # For medium frequencies: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (
        1 - smooth_factor
    ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama

    # Identify medium frequency range
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)

    # Apply smoothed frequencies for medium range
    inv_freq_llama = RecurrentTensor.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama


class RotaryPositionalEmbeddings(Module):
    """

    Args:
        dim (int): Embedding dimension per head. Must be even.
        t (Symbol): Symbolic representation of sequence timestep.
        base (int): Base for geometric progression used in angles.
        domain (DomainLike): -
        independent_domain (DomainLike): -
        rope_freqs (Optional[RecurrentTensor]): precomputed frequencies for the RoPE.
    """

    def __init__(
        self,
        dim: int,
        t: Symbol,
        base: int | float = 10000,
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
        rope_scaling_params: RopeScalingParams | None = None,
    ) -> None:
        super().__init__(domain, independent_domain)

        if dim % 2 != 0:
            raise ValueError("RotaryPositionalEmbeddings requires `dim` to be even.")
        self.dim = dim
        self.t = t

        if rope_scaling_params is None:
            inv_freq = _compute_inv_freq(dim, base)
        else:
            if rope_scaling_params.rope_type == "llama3":
                _logger.info("Using Llama3 RoPE scaling!!!")
                inv_freq = _compute_rope_scaling_llama3(rope_scaling_params, dim, base)
            else:
                raise ValueError(f"Unknown rope type: {rope_scaling_params.rope_type}")

        self.inv_freq = inv_freq

    def forward(self, x: RecurrentTensor) -> RecurrentTensor:
        """
        x: domain=(b, s), shape=(nHeads or n_kv_heads, head_dim).
        the returned RT has the same shape as `x`.
        """
        x_dtype = x.dtype

        t = RecurrentTensor.lift(self.t)

        freqs = t * self.inv_freq
        # freqs.sink_with_ts_udf(lambda x, ts: print(f"freqs: {x}, ts: {ts}"))
        x = x.cast(dtypes.float64)  # (nHeads, head_dim)

        cos_half = freqs.cos()  # (half_dim,)
        sin_half = freqs.sin()  # (half_dim,)

        x_pairs = x.reshape(x.shape._shape[:-1] + (-1, 2))  # (nHeads, half_dim, 2)

        e = x_pairs.spatial_index(-1, 0)  # even components, (nHeads, half_dim)
        o = x_pairs.spatial_index(-1, 1)  # odd  components, (nHeads, half_dim)

        y_even = (e * cos_half) - (o * sin_half)  # (..., half_dim)
        y_odd = (o * cos_half) + (e * sin_half)  # (..., half_dim)

        y = RecurrentTensor.stack([y_even, y_odd], dim=-1)  # (..., half_dim, 2)
        return y.reshape(x.shape).cast(x_dtype)
