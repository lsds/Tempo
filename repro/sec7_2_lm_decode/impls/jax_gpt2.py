from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax

from repro.sec7_2_lm_decode.bench_runner import BenchRunner

"""Simple JAX implementation of the GPT-2-Small core.
"""


def rms_norm(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Applies RMSNorm: x * (weight / rms(x))"""
    rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    normed = x / rms
    return normed


def init_linear(in_features: int, out_features: int, key: jax.random.PRNGKey) -> jnp.ndarray:
    """Initialize a linear layer with zero bias and random weights."""
    weight = jax.random.normal(key, (out_features, in_features)) * 0.02
    return weight


def get_block_fn(
    num_heads: int, embed_size: int, window_size: int, attn_type: str, seq_len: int, batch_size: int
):
    """Create a jitted function for a single block."""
    embed_size_per_head = embed_size // num_heads

    # Precompute mask indices since they depend on static params
    indices = jnp.arange(seq_len)[None, None, None, :]  # (1, 1, 1, seq_len)

    def get_mask(step):
        if attn_type == "causal":
            return indices <= step
        else:
            start = jnp.maximum(0, step - window_size)
            return (indices >= start) & (indices <= step)

    def block_fn(
        carry: Tuple[jnp.ndarray, int],
        params: Tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ],
    ) -> Tuple[jnp.ndarray, int]:
        x, step = carry
        W_Q, W_K, W_V, W_O, W_FF1, W_FF2, K, V = params

        # MHA
        x_start = x
        # Compute Q, K, V projections
        q = x @ W_Q.T  # (batch_size, num_heads, embed_per_head)
        k = x @ W_K.T  # (batch_size, num_heads, embed_per_head)
        v = x @ W_V.T  # (batch_size, num_heads, embed_per_head)

        # Reshape for attention computation
        q = q.reshape(batch_size, num_heads, 1, embed_size_per_head)
        k = k.reshape(batch_size, num_heads, 1, embed_size_per_head)
        v = v.reshape(batch_size, num_heads, 1, embed_size_per_head)

        # Update KV cache using dynamic update slice
        K = lax.dynamic_update_slice(K, k, (0, 0, step, 0))
        V = lax.dynamic_update_slice(V, v, (0, 0, step, 0))

        # Compute attention scores with mask
        mask = get_mask(step)
        # Expand mask to match batch and head dimensions
        mask = jnp.broadcast_to(mask, (batch_size, num_heads, 1, seq_len))

        # Compute attention scores for all positions
        attn_scores = (q @ K.transpose(0, 1, 3, 2)) / jnp.sqrt(embed_size)
        # Apply mask by setting masked positions to -inf
        attn_scores = jnp.where(mask, attn_scores, -jnp.inf)
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)

        # Compute attention output
        attn_output = jnp.matmul(attn_weights, V)  # (batch_size, num_heads, 1, embed_per_head)

        # Project output
        attn_output = attn_output.reshape(batch_size, -1)  # Flatten heads
        x = attn_output @ W_O.T  # (batch_size, embed_size)
        x = x.reshape(batch_size, 1, embed_size)

        # Feed forward
        x_middle = rms_norm(x + x_start)
        x = x_middle @ W_FF1.T
        x = jax.nn.silu(x)
        x = x @ W_FF2.T
        x = rms_norm(x + x_middle)

        return (x, step + 1)

    return block_fn


def get_model_fn(
    num_blocks: int,
    num_heads: int,
    embed_size: int,
    window_size: int,
    attn_type: str,
    seq_len: int,
    batch_size: int,
):
    """Create a jitted function for the causal model."""
    block_fns = [
        get_block_fn(num_heads, embed_size, window_size, attn_type, seq_len, batch_size)
        for _ in range(num_blocks)
    ]

    def model_fn(
        carry: Tuple[jnp.ndarray, int],
        params: Tuple[
            Tuple[
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
            ],
            ...,
        ],
    ) -> Tuple[jnp.ndarray, int]:
        x, step = carry

        for i, block_fn in enumerate(block_fns):
            x, step = block_fn((x, step), params[i])

        return (x, step)

    return model_fn


def get_autoregressive_decode_fn(
    num_blocks: int,
    num_heads: int,
    embed_size: int,
    window_size: int,
    attn_type: str,
    seq_len: int,
    batch_size: int,
):
    """Create a jitted function for autoregressive decoding."""
    model_fn = get_model_fn(
        num_blocks, num_heads, embed_size, window_size, attn_type, seq_len, batch_size
    )

    def decode_fn(
        carry: Tuple[jnp.ndarray, int],
        params: Tuple[
            Tuple[
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
            ],
            ...,
        ],
    ) -> Tuple[jnp.ndarray, int]:
        output, step = carry

        # Get current token using dynamic slice
        token = lax.dynamic_slice(output, (0, step - 1, 0), (batch_size, 1, embed_size))

        # Run through model
        x, step = model_fn((token, step), params)

        # Update output using dynamic update slice
        output = lax.dynamic_update_slice(output, x, (0, step, 0))

        return (output, step)

    def cond_fn(carry):
        _, step = carry
        return step < seq_len

    def body_fn(carry, params):
        return decode_fn(carry, params), None

    def causal_decode_fn(output, params):
        step = 1
        final_output, _ = lax.scan(
            lambda carry, _: (decode_fn(carry, params), None),
            (output, step),
            None,
            length=seq_len - 1,
        )
        return final_output[0]

    return causal_decode_fn


class JaxBenchRunner(BenchRunner):
    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        embed_size: int,
        seq_len: int,
        window_size: int,
        attn_type: str,
        batch_size: int,
        **kwargs,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.seq_len = seq_len
        self.window_size = window_size
        self.attn_type = attn_type
        self.batch_size = batch_size

        # Create the decode function
        self.decode_fn = get_autoregressive_decode_fn(
            num_blocks, num_heads, embed_size, window_size, attn_type, seq_len, batch_size
        )

    def reallocate(self):
        # Initialize parameters
        key = jax.random.PRNGKey(0)
        self.params = []
        for i in range(self.num_blocks):
            block_key = jax.random.fold_in(key, i)
            key_q, key_k, key_v, key_o, key_ff1, key_ff2 = jax.random.split(block_key, 6)

            W_Q = init_linear(
                self.embed_size, self.num_heads * (self.embed_size // self.num_heads), key_q
            )
            W_K = init_linear(
                self.embed_size, self.num_heads * (self.embed_size // self.num_heads), key_k
            )
            W_V = init_linear(
                self.embed_size, self.num_heads * (self.embed_size // self.num_heads), key_v
            )
            W_O = init_linear(
                self.num_heads * (self.embed_size // self.num_heads), self.embed_size, key_o
            )
            W_FF1 = init_linear(self.embed_size, self.embed_size * 4, key_ff1)
            W_FF2 = init_linear(self.embed_size * 4, self.embed_size, key_ff2)

            # Initialize KV caches for this block
            K = jnp.zeros(
                (self.batch_size, self.num_heads, self.seq_len, self.embed_size // self.num_heads)
            )
            V = jnp.zeros(
                (self.batch_size, self.num_heads, self.seq_len, self.embed_size // self.num_heads)
            )

            self.params.append((W_Q, W_K, W_V, W_O, W_FF1, W_FF2, K, V))
        self.output = jax.numpy.zeros((self.batch_size, self.seq_len, self.embed_size))

    def compile(self):
        self.compiled_fn = jax.jit(self.decode_fn)

    def warmup(self):
        self.run()

    def run(self):
        # Initialize state
        self.reallocate()

        # Run autoregressive decoding
        output = self.compiled_fn(self.output, tuple(self.params))
        output.block_until_ready()

        return output


if __name__ == "__main__":
    runner = JaxBenchRunner(
        num_blocks=4,
        num_heads=12,
        embed_size=768,
        seq_len=64,
        window_size=8,
        attn_type="causal",
        batch_size=8,
    )
    runner.compile()
    runner.run()
