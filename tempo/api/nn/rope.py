from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from tempo.api.nn.module import Module
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.core.domain import DomainLike
from tempo.core.dtype import DataType, dtypes
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


def _compute_angles(dim: int, base: Union[int, float]) -> RecurrentTensor:
    angles: RecurrentTensor = 1.0 / (
        base ** ((2.0 * RecurrentTensor.arange(int(dim // 2), dtype=dtypes.float32)) / float(dim))
    )
    return angles


# https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/llama/modeling_llama.py#L147
def _compute_rope_scaling_llama3_dynamic(
    angles: RecurrentTensor, rope_scaling_params: RopeScalingParams, t: Symbol, dim: int, base: int
) -> RecurrentTensor:
    """
    Computes the scaled inverse frequencies for Llama 3 RoPE scaling.
    """
    scaling_factor = rope_scaling_params.factor
    max_position_embeddings = rope_scaling_params.original_max_position_embeddings

    seq_len = RecurrentTensor.lift(t)
    scaled_base = base * (
        (scaling_factor * seq_len / max_position_embeddings) - (scaling_factor - 1)
    ) ** (dim / (dim - 2))
    inv_freq_dyn = 1.0 / (
        scaled_base ** (RecurrentTensor.arange(dim, start=0, step=2, dtype=dtypes.float32) / dim)
    )

    inv_freq = RecurrentTensor.placeholder_like(angles)
    inv_freq[t <= max_position_embeddings] = angles  # TODO < or <=?
    inv_freq[True] = inv_freq_dyn

    return inv_freq


# https://github.com/meta-llama/llama-models/blob/v0.1.4/models/llama3/reference_impl/model.py#L69
# https://github.com/huggingface/transformers/blob/5111c8ea2f3eb918fc090f7dd4393d4204940e10/src/transformers/modeling_rope_utils.py#L393
def _compute_rope_scaling_llama3(
    rope_scaling_params: RopeScalingParams, dim: int, base: int, n_heads: int
) -> RecurrentTensor:
    """
    Computes the scaled inverse frequencies for Llama 3 RoPE scaling.

    Returns:
        Scaled inverse frequencies
    """
    import math

    dim = dim / n_heads

    angles = _compute_angles(dim, base)  # type: ignore

    # Extract parameters
    factor = rope_scaling_params.factor
    low_freq_factor = rope_scaling_params.low_freq_factor
    high_freq_factor = rope_scaling_params.high_freq_factor
    old_context_len = rope_scaling_params.original_max_position_embeddings * 2

    # Compute wavelength boundaries
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    # Compute wavelength from inverse frequency
    wavelen = 2 * math.pi / angles

    # Apply scaling based on frequency ranges
    # wavelen < high_freq_wavelen: do nothing (keep original angles)
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = RecurrentTensor.where(wavelen > low_freq_wavelen, angles / factor, angles)

    # For medium frequencies: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (
        1 - smooth_factor
    ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama

    # Identify medium frequency range
    is_medium_freq = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)

    # Apply smoothed frequencies for medium range
    inv_freq_llama = RecurrentTensor.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama


class RotaryPositionalEmbeddings(Module):
    """

    Args:
        dim (int): Embedding dimension per head. Must be even.
        t (Symbol): Symbolic representation of sequence timestep.
        base (int): Base for geometric progression used in angles.
        dtype (DataType): -
        domain (DomainLike): -
        independent_domain (DomainLike): -
        rope_freqs (Optional[RecurrentTensor]): precomputed frequencies for the RoPE.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        t: Symbol,
        base: Union[int, float] = 10000,
        dtype: DataType = dtypes.float32,
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
        rope_scaling_params: Optional[RopeScalingParams] = None,
    ) -> None:
        super().__init__(domain, independent_domain)

        if dim % 2 != 0:
            raise ValueError("RotaryPositionalEmbeddings requires `dim` to be even.")
        self.dim = dim
        self.t = t

        base = 10000
        self.base: Union[int, float] = base
        self.angles = _compute_angles(dim, self.base)

        if rope_scaling_params is not None:
            # if rope_scaling_params.rope_type == "llama3":
            #    _logger.info("Using Llama3 RoPE scaling!!!")
            #    angles = _compute_rope_scaling_llama3(rope_scaling_params, dim, base, n_heads)
            # else:
            #    raise ValueError(f"Unknown rope type: {rope_scaling_params.rope_type}")
            pass

    def forward(self, x: RecurrentTensor) -> RecurrentTensor:
        """
        x: domain=(b, s), shape=(nHeads, dim).
        the returned RT has the same shape as `x`.
        """

        angles_for_timestep = RecurrentTensor.lift(self.t) * self.angles

        # get two recurrent tensors, each representing the cos and sin of the above angle
        # therefore have the same domain and shape
        cos_table = angles_for_timestep.cos()
        sin_table = angles_for_timestep.sin()

        x_dtype = x.dtype
        x = x.cast(dtypes.upcast(x_dtype, dtypes.float32))

        # TODO: consider rotate_half-based implementation
        # https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/llama/modeling_llama.py#L180
        return self._forward_reshape_impl(x, cos_table, sin_table).cast(x_dtype)

    def _forward_reshape_impl(
        self, x: RecurrentTensor, cos_table: RecurrentTensor, sin_table: RecurrentTensor
    ) -> RecurrentTensor:
        """
        Reshape-based implementation of RoPE forward pass.

        Args:
            x: Input tensor with domain=(b, s), shape=(nHeads, dim)
            cos_table: Cosine values for rotation
            sin_table: Sine values for rotation

        Returns:
            Rotated tensor with same shape as x
        """
        # Reshape the last dim to (half_dim, 2) to separate [even, odd] interleaved structure
        x_reshaped = x.reshape(x.shape._shape[:-1] + (self.dim // 2, 2))

        x_even = x_reshaped.spatial_index(-1, 0)  # shape: (..., half_dim)
        x_odd = x_reshaped.spatial_index(-1, 1)  # shape: (..., half_dim)

        out_even = x_even * cos_table - x_odd * sin_table
        out_odd = x_odd * cos_table + x_even * sin_table

        # Stack them back together along last axis = 2
        out_rotated = RecurrentTensor.stack([out_even, out_odd], dim=-1)

        # Then reshape back to original shape
        out = out_rotated.reshape(x.shape)

        return out

    def _forward_index_impl(
        self, x: RecurrentTensor, cos_table: RecurrentTensor, sin_table: RecurrentTensor
    ) -> RecurrentTensor:
        """
        Indexing-based implementation of RoPE forward pass.

        Args:
            x: Input tensor with domain=(b, s), shape=(nHeads, dim)
            cos_table: Cosine values for rotation
            sin_table: Sine values for rotation

        Returns:
            Rotated tensor with same shape as x
        """
        # make indices of shape (half_dim,)
        # indices_even has value: (0, 2, 4...)
        indices_even = RecurrentTensor.arange(self.dim // 2) * 2
        # indices_odd has value: (1, 3, 5...)
        indices_odd = indices_even + 1

        # split input x into x_even, x_odd, based on odd and even indices
        # each has domain (b,s), shape (nHeads, half_dim)
        x_even = x.index(-1, indices_even)
        x_odd = x.index(-1, indices_odd)

        # the standard RoPE formula
        # each has domain (b,s), shape (nHeads, half_dim)
        out_even = x_even * cos_table - x_odd * sin_table
        out_odd = x_odd * cos_table + x_even * sin_table

        # merge out_even and out_odd to get final result
        out = RecurrentTensor.zeros(x.shape, x.dtype, requires_grad=False)

        # TODO: ideally we don't want neither of these... could we use where instead?
        # out = out.index_add(-1, indices_even, out_even)
        # out = out.index_add(-1, indices_odd, out_odd)

        out = out.scatter_add(-1, indices_even, out_even)
        out = out.scatter_add(-1, indices_odd, out_odd)

        return out
