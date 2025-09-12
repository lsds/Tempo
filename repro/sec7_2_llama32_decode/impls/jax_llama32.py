import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import torch
from jax import lax, random

from repro.sec7_2_llama32_decode.impls.tokenizer_shared import Tokenizer
from repro.sec7_2_llama32_decode.shared import get_prompts

"""
Adapted from https://github.com/dhyaneesh/awesome-jax-flax-llms/blob/main/models/llama3/llama3_in_jax.py

Added:
- loading weights and config from checkpoint
- window attention
- KV caching
- End-to-end jit compilation
- float16 precision
"""


@dataclass(frozen=True)
class LLaMAConfig:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = "cuda"
    use_scaled_rope: bool = False

    window_size: int = 0
    attn_type: str = "causal"
    checkpoint_dir: str = ""


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
    t = jnp.arange(end, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    freqs_cis = jnp.exp(1j * freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: jnp.ndarray, xk: jnp.ndarray, freqs_cis: jnp.ndarray, start_pos: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply rotary embeddings to the query and key tensors."""
    # Reshape inputs to isolate the last dimension into pairs for complex multiplication
    xq_r, xk_r = jnp.reshape(xq, (*xq.shape[:-1], -1, 2)), jnp.reshape(xk, (*xk.shape[:-1], -1, 2))

    # Convert to complex numbers
    xq_complex = jnp.complex64(xq_r[..., 0] + 1j * xq_r[..., 1])
    xk_complex = jnp.complex64(xk_r[..., 0] + 1j * xk_r[..., 1])

    freqs_cis = jax.lax.dynamic_slice_in_dim(freqs_cis, start_pos, xq.shape[1], axis=0)

    # Reshape frequency cis for broadcasting
    freqs_cis = jnp.reshape(freqs_cis, (1, freqs_cis.shape[0], 1, freqs_cis.shape[1]))

    # Apply rotation through complex multiplication
    xq_out = xq_complex * freqs_cis
    xk_out = xk_complex * freqs_cis

    # Convert back to real tensor and reshape
    xq = jnp.stack([jnp.real(xq_out), jnp.imag(xq_out)], axis=-1).reshape(xq.shape)
    xk = jnp.stack([jnp.real(xk_out), jnp.imag(xk_out)], axis=-1).reshape(xk.shape)

    return xq.astype(jnp.float16), xk.astype(jnp.float16)


class RMSNorm(nn.Module):
    dim: int
    eps: float
    np_checkpoint: Mapping[str, jnp.ndarray]
    chk_key: str

    def __hash__(self) -> int:
        return hash((self.dim, self.eps, self.chk_key))

    def setup(self) -> None:
        self.scale = self.param(
            "scale", from_dict_init(self.np_checkpoint[self.chk_key]), (self.dim,)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        var = jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
        x_hat = x * lax.rsqrt(var + self.eps)
        return (x_hat * self.scale).astype(jnp.float16)


@jax.jit
def flash_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    mask: jnp.ndarray | None = None,
    scale: float | None = None,
) -> jnp.ndarray:
    batch_size, num_heads, seq_len, head_dim = q.shape
    if scale is None:
        scale = 1.0 / jnp.sqrt(head_dim)
    k_t = jnp.transpose(k, (0, 1, 3, 2))  # (B, H, D, S)
    scores = jnp.matmul(q, k_t) * scale  # (B, H, 1, D) @ (B, H, D, S) -> (B, H, 1, S)
    if mask is not None:
        scores = scores + mask
    scores_max = jnp.max(scores, axis=-1, keepdims=True)
    scores = scores - lax.stop_gradient(scores_max)
    attn_weights = jnp.exp(scores.astype(jnp.float32))
    attn_weights = attn_weights / jnp.sum(attn_weights, axis=-1, keepdims=True)
    output = jnp.matmul(attn_weights, v)  # (B, H, 1, S) @ (B, H, S, D) -> (B, H, 1, D)
    return output.astype(jnp.float16)


def swiglu(x: jnp.ndarray, w1: nn.Dense, w2: nn.Dense, w3: nn.Dense) -> jnp.ndarray:
    return w2(jax.nn.silu(w3(x)) * w1(x))


def from_dict_init(array: jnp.ndarray):
    def init(key, shape, dtype=jnp.float16):
        assert shape == array.shape, f"Expected {shape}, got {array.shape}"
        # return jnp.array(array, dtype=dtype)
        return array

    return init


class LLaMACausalSelfAttention(nn.Module):
    args: LLaMAConfig
    np_checkpoint: Mapping[str, jnp.ndarray]
    layer_idx: int

    def __hash__(self) -> int:
        return hash((self.args, self.layer_idx))

    def setup(self) -> None:
        args = self.args
        dim = args.dim
        n_heads = args.n_heads
        n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else n_heads
        head_dim = dim // n_heads
        self.wq = nn.Dense(
            n_heads * head_dim,
            kernel_init=from_dict_init(
                self.np_checkpoint[f"layers.{self.layer_idx}.attention.wq.weight"]
            ),
            use_bias=False,
        )
        self.wk = nn.Dense(
            n_kv_heads * head_dim,
            kernel_init=from_dict_init(
                self.np_checkpoint[f"layers.{self.layer_idx}.attention.wk.weight"]
            ),
            use_bias=False,
        )
        self.wv = nn.Dense(
            n_kv_heads * head_dim,
            kernel_init=from_dict_init(
                self.np_checkpoint[f"layers.{self.layer_idx}.attention.wv.weight"]
            ),
            use_bias=False,
        )
        self.wo = nn.Dense(
            dim,
            kernel_init=from_dict_init(
                self.np_checkpoint[f"layers.{self.layer_idx}.attention.wo.weight"]
            ),
            use_bias=False,
        )

        dev = jax.devices(args.device)[0]
        self.k_cache = self.variable(
            "cache",
            "k",
            lambda: jax.device_put(
                jnp.zeros(
                    (args.max_batch_size, args.max_seq_len, n_kv_heads, head_dim),
                    dtype=jnp.float16,
                ),
                dev,
            ),
        )
        self.v_cache = self.variable(
            "cache",
            "v",
            lambda: jax.device_put(
                jnp.zeros(
                    (args.max_batch_size, args.max_seq_len, n_kv_heads, head_dim),
                    dtype=jnp.float16,
                ),
                dev,
            ),
        )

    def __call__(
        self,
        x: jnp.ndarray,
        freqs_cis: jnp.ndarray,
        start_pos: int,
        mask: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        assert x.shape == (self.args.max_batch_size, 1, self.args.dim), (
            f"Expected {self.args.max_batch_size, 1, self.args.dim}, got {x.shape=}"
        )
        B, _, C = x.shape
        config = self.args
        n_heads = config.n_heads
        n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else n_heads
        head_dim = C // n_heads

        q = self.wq(x).reshape(B, 1, n_heads, head_dim)
        k = self.wk(x).reshape(B, 1, n_kv_heads, head_dim)
        v = self.wv(x).reshape(B, 1, n_kv_heads, head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis, start_pos)

        q = jnp.swapaxes(q, 1, 2)

        self.k_cache.value = jax.lax.dynamic_update_slice_in_dim(
            self.k_cache.value, k, start_pos, axis=1
        )
        self.v_cache.value = jax.lax.dynamic_update_slice_in_dim(
            self.v_cache.value, v, start_pos, axis=1
        )

        k_attn = jnp.swapaxes(self.k_cache.value, 1, 2)  # (B, H, S, D)
        v_attn = jnp.swapaxes(self.v_cache.value, 1, 2)

        if n_heads > n_kv_heads:
            k_attn = jnp.repeat(k_attn, n_heads // n_kv_heads, axis=1)  # (B, H', S, D)
            v_attn = jnp.repeat(v_attn, n_heads // n_kv_heads, axis=1)

        assert mask is not None, "mask is None"
        output = flash_attention(q, k_attn, v_attn, mask, scale=1.0 / jnp.sqrt(head_dim))

        output = jax.lax.dynamic_slice_in_dim(output, start_pos, 1, axis=2)
        output = jnp.swapaxes(output, 1, 2).reshape(B, 1, -1)
        output = self.wo(output).reshape(B, 1, -1)

        assert output.shape == (self.args.max_batch_size, 1, self.args.dim), (
            f"Expected {self.args.max_batch_size, 1, self.args.dim}, got {output.shape=}"
        )
        return output


class LLaMAMLP(nn.Module):
    config: LLaMAConfig
    np_checkpoint: Mapping[str, jnp.ndarray]
    layer_idx: int

    def __hash__(self) -> int:
        return hash((self.config, self.layer_idx))

    def setup(self) -> None:
        dim = self.config.dim
        hidden_dim = 4 * dim
        self.w1 = nn.Dense(
            hidden_dim,
            kernel_init=from_dict_init(
                self.np_checkpoint[f"layers.{self.layer_idx}.feed_forward.w1.weight"]
            ),
            use_bias=False,
        )
        self.w2 = nn.Dense(
            dim,
            kernel_init=from_dict_init(
                self.np_checkpoint[f"layers.{self.layer_idx}.feed_forward.w2.weight"]
            ),
            use_bias=False,
        )
        self.w3 = nn.Dense(
            hidden_dim,
            kernel_init=from_dict_init(
                self.np_checkpoint[f"layers.{self.layer_idx}.feed_forward.w3.weight"]
            ),
            use_bias=False,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return swiglu(x, self.w1, self.w2, self.w3)


class LLaMABlock(nn.Module):
    config: LLaMAConfig
    np_checkpoint: Mapping[str, jnp.ndarray]
    layer_idx: int

    def __hash__(self) -> int:
        return hash((self.config, self.layer_idx))

    def setup(self) -> None:
        self.attention_norm = RMSNorm(
            self.config.dim,
            eps=self.config.norm_eps,
            np_checkpoint=self.np_checkpoint,
            chk_key=f"layers.{self.layer_idx}.attention_norm.weight",
        )
        self.attention = LLaMACausalSelfAttention(self.config, self.np_checkpoint, self.layer_idx)
        self.ffn_norm = RMSNorm(
            self.config.dim,
            eps=self.config.norm_eps,
            np_checkpoint=self.np_checkpoint,
            chk_key=f"layers.{self.layer_idx}.ffn_norm.weight",
        )
        self.ffn = LLaMAMLP(self.config, self.np_checkpoint, self.layer_idx)

    def __call__(
        self,
        x: jnp.ndarray,
        freqs_cis: jnp.ndarray,
        start_pos: int,
        mask: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        h = x + self.attention(self.attention_norm(x), freqs_cis, start_pos, mask, deterministic)
        out = h + self.ffn(self.ffn_norm(h))
        return out


class Llama32(nn.Module):
    config: LLaMAConfig
    np_checkpoint: Mapping[str, jnp.ndarray]
    apply_weight_tying: bool = True

    def __hash__(self) -> int:
        return hash((self.config, self.apply_weight_tying))

    def setup(self) -> None:
        self.vocab_size = self.config.vocab_size
        self.n_layers = self.config.n_layers
        self.token_embedding = nn.Embed(
            self.vocab_size,
            self.config.dim,
            embedding_init=from_dict_init(self.np_checkpoint["tok_embeddings.weight"]),
        )
        self.blocks = [LLaMABlock(self.config, self.np_checkpoint, i) for i in range(self.n_layers)]
        self.norm_f = RMSNorm(
            self.config.dim,
            eps=self.config.norm_eps,
            np_checkpoint=self.np_checkpoint,
            chk_key="norm.weight",
        )
        self.lm_head = nn.Dense(
            self.vocab_size,
            kernel_init=from_dict_init(self.np_checkpoint["tok_embeddings.weight"].T),
            use_bias=False,
        )

        self.freqs_cis = precompute_freqs_cis(
            self.config.dim // self.config.n_heads,
            self.config.max_seq_len,
            self.config.rope_theta,
        )

        # new_params["lm_head"]["kernel"] = new_params["token_embedding"]["embedding"]

    def __call__(self, token_ids: jnp.ndarray, i: int, deterministic: bool = True) -> jnp.ndarray:
        if token_ids.ndim == 1:
            token_ids = token_ids[:, None]
        L = self.config.max_seq_len
        neg_inf = jnp.finfo(jnp.float16).min

        # causal: allow keys ≤ i
        mask = jnp.where(jnp.arange(L) > i, neg_inf, 0.0)

        if self.config.attn_type == "window":
            window_size = self.config.window_size
            start = jnp.maximum(0, i - window_size + 1)
            # additionally forbid keys < start
            mask = jnp.where(jnp.arange(L) < start, neg_inf, mask)

        mask = mask[None, None, None, :]

        input_id_curr = jax.lax.dynamic_slice_in_dim(token_ids, i, 1, axis=1)
        h = self.token_embedding(input_id_curr)

        for block in self.blocks:
            h = block(h, self.freqs_cis, i, mask, deterministic)

        h = self.norm_f(h)
        logits = self.lm_head(h)
        return logits

    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        window_size: int,
        attn_type: str,
        seed: int = 1,
        device: str = "gpu",
    ) -> tuple["Llama32", Tokenizer]:
        assert os.path.isdir(ckpt_dir), f"Checkpoint directory '{ckpt_dir}' does not exist."
        assert os.path.isfile(tokenizer_path), f"Tokenizer file '{tokenizer_path}' does not exist."

        rng_key = jax.random.PRNGKey(seed)
        rng_key = jax.device_put(rng_key, jax.devices(device)[0])

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        ckpt_path = checkpoints[0]

        checkpoint = torch.load(ckpt_path, map_location="cpu")

        import optree

        paths, leaves, _ = optree.tree_flatten_with_path(checkpoint)
        flat_checkpoint = {
            ".".join(str(k) for k in path): leaf for path, leaf in zip(paths, leaves, strict=True)
        }
        flat_checkpoint = {k: v.to(torch.float16).numpy() for k, v in flat_checkpoint.items()}

        # JAX and Torch use different conventions for the weight matrices.
        flat_checkpoint = {
            k: v.T if ("attention.w" in k or "output.weight" in k or "feed_forward.w" in k) else v
            for k, v in flat_checkpoint.items()
        }

        dev = jax.devices(device)[0]

        checkpoint = {k: jax.device_put(v, device=dev) for k, v in flat_checkpoint.items()}
        # Make frozen dict
        from flax.core import freeze

        checkpoint = freeze(checkpoint)

        with open(Path(ckpt_dir) / "params.json") as f:
            params = json.loads(f.read())

        model_args: LLaMAConfig = LLaMAConfig(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            window_size=window_size,
            attn_type=attn_type,
            checkpoint_dir=ckpt_dir,
            **params,
        )
        print(f"model_args: {model_args}")

        tokenizer = Tokenizer(model_path=tokenizer_path)

        model = Llama32(model_args, checkpoint)

        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        return model, tokenizer


def generate_pyloop(
    llama: Llama32,
    vars_in_: Mapping[str, Any],
    input_ids: jnp.ndarray,
    max_seq_len: int,
    rng_key: jax.Array,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> jnp.ndarray:
    B, prompt_len = input_ids.shape
    batch_idx = jnp.arange(B)

    device = "gpu" if llama.config.device == "cuda" else "cpu"
    dev = jax.devices(device)[0]

    tokens_buf = jax.device_put(jnp.zeros((B, max_seq_len), dtype=input_ids.dtype), dev)
    tokens_buf = lax.dynamic_update_slice(tokens_buf, input_ids, (0, 0))

    def step(
        i: int,
        rng: jax.Array,
        vars_in: Mapping[str, Any],
        buf: jnp.ndarray,
    ) -> tuple[jax.Array, Mapping[str, Any], jnp.ndarray]:
        logits, mut = llama.apply(vars_in, buf, i, True, mutable=["cache"])
        vars_out = flax.core.freeze(
            {
                **vars_in,  # params/others unchanged
                **mut,  # updated cache
            }
        )
        # logits = logits[:, -1, :]
        logits = logits[:, 0, :]  # (B, V)
        rng2, sk = random.split(rng)
        if temperature > 0.0:
            logits = logits / temperature

            def top_p_filter(logits_in: jnp.ndarray) -> jnp.ndarray:
                sorted_idx = jnp.argsort(logits_in, axis=-1)[:, ::-1]
                sorted_logits = jnp.take_along_axis(logits_in, sorted_idx, axis=-1)
                probs = jax.nn.softmax(sorted_logits.astype(jnp.float32), axis=-1).astype(
                    jnp.float16
                )
                cum_probs = jnp.cumsum(probs, axis=-1)
                to_remove_sorted = cum_probs > top_p
                to_remove_sorted = jnp.roll(to_remove_sorted, 1, axis=1)
                to_remove_sorted = to_remove_sorted.at[:, 0].set(False)
                mask = jnp.zeros_like(logits_in, dtype=bool)
                mask = mask.at[batch_idx[:, None], sorted_idx].set(to_remove_sorted)
                return jnp.where(mask, jnp.finfo(jnp.float16).min, logits_in)

            logits = jax.lax.cond(top_p < 1.0, lambda l: top_p_filter(l), lambda l: l, logits)

            next_token = random.categorical(sk, logits, shape=(B,))
        else:
            next_token = jnp.argmax(logits, axis=-1)

        next_token = next_token.astype(buf.dtype)

        row = next_token[:, None]
        buf2 = lax.cond(
            i < prompt_len,
            lambda _: buf,
            lambda _: lax.dynamic_update_slice(buf, row, (0, i)),
            operand=None,
        )

        return (rng2, vars_out, buf2)

    jitted_step = jax.jit(step, donate_argnums=(1, 2, 3), device=dev)

    rng = rng_key
    vars_in = vars_in_
    buf = tokens_buf
    for i in range(max_seq_len):
        rng, vars_in, buf = jitted_step(i, rng, vars_in, buf)

    return buf, vars_in


def generate(
    llama: Llama32,
    vars_in_: Mapping[str, Any],
    input_ids: jnp.ndarray,
    max_seq_len: int,
    rng_key: jax.Array,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> jnp.ndarray:
    def _generate(
        variables: Mapping[str, Any],
        input_ids: jnp.ndarray,
        max_seq_len: int,
        rng_key: jax.Array,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> jnp.ndarray:
        B, prompt_len = input_ids.shape
        batch_idx = jnp.arange(B)

        tokens_buf = jnp.zeros((B, max_seq_len), dtype=input_ids.dtype)
        tokens_buf = lax.dynamic_update_slice(tokens_buf, input_ids, (0, 0))

        def step(
            i: int,
            carry: tuple[jax.Array, jnp.ndarray, jnp.ndarray],
        ) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray]:
            rng, vars_in, buf = carry

            logits, mut = llama.apply(vars_in, buf, i, True, mutable=["cache"])
            vars_out = flax.core.freeze(
                {
                    **vars_in,  # params/others unchanged
                    **mut,  # updated cache
                }
            )
            # logits = logits[:, -1, :]
            logits = logits[:, 0, :]  # (B, V)
            rng2, sk = random.split(rng)
            if temperature > 0.0:
                logits = logits / temperature

                def top_p_filter(logits_in: jnp.ndarray) -> jnp.ndarray:
                    sorted_idx = jnp.argsort(logits_in, axis=-1)[:, ::-1]
                    sorted_logits = jnp.take_along_axis(logits_in, sorted_idx, axis=-1)
                    probs = jax.nn.softmax(sorted_logits.astype(jnp.float32), axis=-1).astype(
                        jnp.float16
                    )
                    cum_probs = jnp.cumsum(probs, axis=-1)
                    to_remove_sorted = cum_probs > top_p
                    to_remove_sorted = jnp.roll(to_remove_sorted, 1, axis=1)
                    to_remove_sorted = to_remove_sorted.at[:, 0].set(False)
                    mask = jnp.zeros_like(logits_in, dtype=bool)
                    mask = mask.at[batch_idx[:, None], sorted_idx].set(to_remove_sorted)
                    return jnp.where(mask, jnp.finfo(jnp.float16).min, logits_in)

                logits = jax.lax.cond(top_p < 1.0, lambda l: top_p_filter(l), lambda l: l, logits)

                next_token = random.categorical(sk, logits, shape=(B,))
            else:
                next_token = jnp.argmax(logits, axis=-1)

            next_token = next_token.astype(buf.dtype)

            row = next_token[:, None]
            buf2 = lax.cond(
                i < prompt_len,
                lambda _: buf,
                lambda _: lax.dynamic_update_slice(buf, row, (0, i)),
                operand=None,
            )

            return (rng2, vars_out, buf2)

        _, _, tokens_buf = lax.fori_loop(
            0, max_seq_len, step, (rng_key, variables, tokens_buf), unroll=False
        )
        return tokens_buf

    device = "gpu" if llama.config.device == "cuda" else "cpu"
    dev = jax.devices(device)[0]
    jitted_generate = jax.jit(_generate, static_argnums=(2, 4, 5), device=dev)

    # variables = llama.variables
    return jitted_generate(vars_in_, input_ids, max_seq_len, rng_key, temperature, top_p)


class JAXLlama32InferenceRunner:
    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        window_size: int,
        attn_type: str,
        checkpoint_dir: str,
        prompts: list[str],
        dev: str,
        **kwargs: Any,
    ):
        dev = "cuda" if dev == "gpu" else dev
        dev_jax = jax.devices(dev)[0]
        jax.default_device(dev_jax).__enter__()

        checkpoint_dir = str(Path(checkpoint_dir + "/original").expanduser())

        self.llama, self.tokenizer = Llama32.build(
            ckpt_dir=checkpoint_dir,
            tokenizer_path=checkpoint_dir + "/tokenizer.model",
            max_seq_len=seq_len,
            max_batch_size=batch_size,
            device=dev,
            window_size=window_size,
            attn_type=attn_type,
        )
        self.prompts = prompts
        self.max_gen_len = seq_len
        self.temperature = kwargs.get("temperature", 0.6)
        self.top_p = kwargs.get("top_p", 0.9)

        self.prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        max_prompt_len = max(len(t) for t in self.prompt_tokens)
        pad_id = self.tokenizer.encode(" ", bos=False, eos=False)[0]
        self.padded_prompt_tokens_array = jnp.full(
            (len(self.prompt_tokens), max_prompt_len), pad_id, dtype=jnp.int32
        )
        for i, t in enumerate(self.prompt_tokens):
            start_idx = max_prompt_len - len(t)
            self.padded_prompt_tokens_array = self.padded_prompt_tokens_array.at[i, start_idx:].set(
                jnp.array(t, dtype=jnp.int32)
            )

        self.padded_prompt_tokens_array = jax.device_put(self.padded_prompt_tokens_array, dev_jax)

        self.variables = flax.core.freeze(
            self.llama.init(
                jax.device_put(jax.random.PRNGKey(0), dev_jax),
                jax.device_put(
                    jnp.zeros((batch_size, seq_len), dtype=jnp.int32), dev_jax
                ),  # dummy ids
                0,  # dummy i
                True,
            )
        )

        # treemapped_shapes_and_dtypes = jax.tree_util.tree_map(
        #    lambda x: (x.shape, x.dtype), self.variables
        # )
        # print(f"treemapped_shapes_and_dtypes: {treemapped_shapes_and_dtypes}", flush=True)

    def reset(self) -> None:
        pass

    def compile(self) -> None:
        pass

    def warmup(self) -> None:
        self.run()

    def run(self) -> None:
        dev_jax = jax.devices(self.llama.config.device)[0]
        self.outputs, self.variables = generate_pyloop(
            llama=self.llama,
            vars_in_=self.variables,
            input_ids=jnp.asarray(self.padded_prompt_tokens_array, copy=True),
            max_seq_len=self.max_gen_len,
            rng_key=jax.device_put(jax.random.PRNGKey(0), dev_jax),
            temperature=self.temperature,
            top_p=self.top_p,
        )
        self.outputs.block_until_ready()

    def get_decoded_outputs(self) -> list[str]:
        return [self.tokenizer.decode(output) for output in self.outputs]


if __name__ == "__main__":
    # model = "Llama-2-7b"
    model = "Llama3.2-1B"
    # model = "Llama3.2-3B"

    base_params = {
        "dev": "cpu",
        "results_path": "./results/jax_llama32_decode/",
        "batch_size": 2,
        "seq_len": 64,
        "window_size": 0,
        "attn_type": "causal",
        "max_prompt_len": 16,
        "temperature": 0.6,  # NOTE: 0.0 for greedy
        "checkpoint_dir": f"~/.llama/checkpoints/{model}/",
        # "validate": True,
    }
    prompts = get_prompts(**base_params)
    # prompts = ["Hello, how are you?", "What is the capital of France?"]
    # prompts = [
    #    "What is the most beautiful place in London?",
    #    "What is the best tube station in London?",
    # ]

    runner = JAXLlama32InferenceRunner(
        **base_params,
        prompts=prompts,
    )
    runner.compile()
    runner.run()
    print(f"Tokens: {runner.outputs}", flush=True)
    print("Decoded outputs:")
    print(runner.get_decoded_outputs())
