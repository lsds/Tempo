import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import torch

from .tokenizer_shared import CompletionPrediction, Tokenizer

"""
Adapted from https://github.com/dhyaneesh/awesome-jax-flax-llms/blob/main/models/llama3/llama3_in_jax.py

Added:
- loading weights and config from checkpoint
- window attention
- KV caching
"""


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = "cuda"
    use_scaled_rope: bool = False

    # Window attention parameters
    window_size: int = 0  # 0 means causal attention (no window)
    attn_type: str = "causal"  # "causal" or "window"


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
    t = jnp.arange(end, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    freqs_cis = jnp.exp(1j * freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    start_pos: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    bsz, seqlen, n_heads, head_dim = xq.shape
    # Extract the relevant frequencies for the current sequence position
    freqs_cis_seq = freqs_cis[start_pos : start_pos + seqlen]

    xq_ = jnp.reshape(xq.astype(jnp.float32), (*xq.shape[:-1], -1, 2))
    xk_ = jnp.reshape(xk.astype(jnp.float32), (*xk.shape[:-1], -1, 2))

    # Reshape freqs_cis_seq to match the expected broadcast shape
    # freqs_cis_seq has shape (seqlen, head_dim//2)
    # We need to reshape it to (1, seqlen, 1, head_dim//2) for broadcasting
    freqs_cis_seq = freqs_cis_seq.reshape(1, seqlen, 1, -1)

    xq_out = jnp.reshape(
        jnp.stack([jnp.real(xq_ * freqs_cis_seq), jnp.imag(xq_ * freqs_cis_seq)], axis=-1), xq.shape
    )
    xk_out = jnp.reshape(
        jnp.stack([jnp.real(xk_ * freqs_cis_seq), jnp.imag(xk_ * freqs_cis_seq)], axis=-1), xk.shape
    )
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)


def repeat_kv(x: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    """jnp.repeat(x, repeats=n_rep, axis=2)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .reshape(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class RMSNorm:
    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = jnp.ones(dim)

    def _norm(self, x):
        return x * jax.lax.rsqrt(
            jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + self.eps
        )

    def __call__(self, x):
        output = self._norm(x.astype(jnp.float32)).astype(x.dtype)
        return output * self.weight


class Attention:
    def __init__(self, args: ModelArgs):
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        # Window attention parameters
        self.window_size = args.window_size
        self.attn_type = args.attn_type

        # Initialize weights (will be loaded from checkpoint)
        self.wq = None
        self.wk = None
        self.wv = None
        self.wo = None

        # Cache for key-value pairs
        self.cache_k = jnp.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            )
        )
        self.cache_v = jnp.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            )
        )

    def set_weights(self, wq, wk, wv, wo, print_=False):
        """Set the weights loaded from checkpoint"""
        if print_:
            print(
                f"Attention weights shapes - wq: {wq.shape}, wk: {wk.shape}, wv: {wv.shape}, wo: {wo.shape}"
            )
            print(
                f"Attention weights dtypes - wq: {wq.dtype}, wk: {wk.dtype}, wv: {wv.dtype}, wo: {wo.dtype}"
            )
        # Transpose weights to match JAX matrix multiplication convention
        # PyTorch stores weights as (out_features, in_features)
        # JAX matrix multiplication expects (in_features, out_features) for x @ w
        self.wq = wq.T
        self.wk = wk.T
        self.wv = wv.T
        self.wo = wo.T
        if print_:
            print(
                f"After transpose - wq: {self.wq.shape}, wk: {self.wk.shape}, wv: {self.wv.shape}, wo: {self.wo.shape}"
            )

    def __call__(
        self,
        x: jnp.ndarray,
        start_pos: int,
        freqs_cis: jnp.ndarray,
        mask: jnp.ndarray | None,
    ):
        bsz, seqlen, _ = x.shape
        print(
            f"Attention input shape: {x.shape}, wq shape: {self.wq.shape}, wk shape: {self.wk.shape}, wv shape: {self.wv.shape}"
        )
        print(
            f"Input x dtype: {x.dtype}, wq dtype: {self.wq.dtype}, wk dtype: {self.wk.dtype}, wv dtype: {self.wv.dtype}"
        )
        print(
            f"Input x last dim: {x.shape[-1]}, wq first dim: {self.wq.shape[0]}, wk first dim: {self.wk.shape[0]}, wv first dim: {self.wv.shape[0]}"
        )
        xq, xk, xv = jnp.dot(x, self.wq), jnp.dot(x, self.wk), jnp.dot(x, self.wv)

        xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis, start_pos)

        # Update cache
        self.cache_k = self.cache_k.at[:bsz, start_pos : start_pos + seqlen].set(xk)
        self.cache_v = self.cache_v.at[:bsz, start_pos : start_pos + seqlen].set(xv)

        # Select keys and values based on attention type
        if self.attn_type == "causal":
            # Causal attention: use all tokens up to current position
            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            # Window attention: use only the last window_size tokens
            start_idx = max(0, start_pos + seqlen - self.window_size)
            keys = self.cache_k[:bsz, start_idx : start_pos + seqlen]
            values = self.cache_v[:bsz, start_idx : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_heads, head_dim)

        xq = xq.transpose(0, 2, 1, 3)  # (bs, n_heads, seqlen, head_dim)
        keys = keys.transpose(0, 2, 1, 3)  # (bs, n_heads, cache_len + seqlen, head_dim)
        values = values.transpose(0, 2, 1, 3)  # (bs, n_heads, cache_len + seqlen, head_dim)

        scores = jnp.matmul(xq, keys.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_heads, seqlen, cache_len + seqlen)
        scores = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(xq.dtype)
        output = jnp.matmul(scores, values)  # (bs, n_heads, seqlen, head_dim)
        output = output.transpose(0, 2, 1, 3).reshape(bsz, seqlen, -1)
        return jnp.dot(output, self.wo)


class FeedForward:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
    ):
        self.dim = dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # Initialize weights (will be loaded from checkpoint)
        self.w1 = None
        self.w2 = None
        self.w3 = None

    def set_weights(self, w1, w2, w3):
        """Set the weights loaded from checkpoint"""
        # Transpose weights to match JAX matrix multiplication convention
        # PyTorch stores weights as (out_features, in_features)
        # JAX matrix multiplication expects (in_features, out_features) for x @ w
        self.w1 = w1.T
        self.w2 = w2.T
        self.w3 = w3.T

    def __call__(self, x):
        return jnp.dot(jax.nn.silu(jnp.dot(x, self.w1)) * jnp.dot(x, self.w3), self.w2)


class TransformerBlock:
    def __init__(self, layer_id: int, args: ModelArgs):
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def set_weights(self, attention_weights, ffn_weights, norm_weights, print_=False):
        """Set the weights loaded from checkpoint"""
        self.attention.set_weights(
            attention_weights["wq"],
            attention_weights["wk"],
            attention_weights["wv"],
            attention_weights["wo"],
            print_=print_,
        )
        self.feed_forward.set_weights(ffn_weights["w1"], ffn_weights["w2"], ffn_weights["w3"])
        self.attention_norm.weight = norm_weights["attention_norm"]
        self.ffn_norm.weight = norm_weights["ffn_norm"]

    def __call__(
        self,
        x: jnp.ndarray,
        start_pos: int,
        freqs_cis: jnp.ndarray,
        mask: jnp.ndarray | None,
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer:
    def __init__(self, params: ModelArgs):
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # Initialize weights (will be loaded from checkpoint)
        self.tok_embeddings = None
        self.output = None

        self.layers = []
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    def set_weights(self, weights):
        """Set the weights loaded from checkpoint"""
        # For token embeddings, we want (vocab_size, dim) for direct indexing
        # For output layer, transpose to match JAX matrix multiplication convention
        # PyTorch stores weights as (out_features, in_features)
        # JAX matrix multiplication expects (in_features, out_features) for x @ w
        print(f"Original tok_embeddings.weight shape: {weights['tok_embeddings.weight'].shape}")
        self.tok_embeddings = weights["tok_embeddings.weight"]
        print(f"Final tok_embeddings shape: {self.tok_embeddings.shape}")
        self.output = weights["output.weight"].T

        for i, layer in enumerate(self.layers):
            layer_prefix = f"layers.{i}."
            if i == 0 or i == 27:
                print(f"Loading weights for layer {i}")
                print(
                    f"Available keys: {[k for k in weights.keys() if k.startswith(layer_prefix)]}"
                )
            attention_weights = {
                "wq": weights[f"{layer_prefix}attention.wq.weight"],
                "wk": weights[f"{layer_prefix}attention.wk.weight"],
                "wv": weights[f"{layer_prefix}attention.wv.weight"],
                "wo": weights[f"{layer_prefix}attention.wo.weight"],
            }
            ffn_weights = {
                "w1": weights[f"{layer_prefix}feed_forward.w1.weight"],
                "w2": weights[f"{layer_prefix}feed_forward.w2.weight"],
                "w3": weights[f"{layer_prefix}feed_forward.w3.weight"],
            }
            norm_weights = {
                "attention_norm": weights[f"{layer_prefix}attention_norm.weight"],
                "ffn_norm": weights[f"{layer_prefix}ffn_norm.weight"],
            }
            layer.set_weights(
                attention_weights, ffn_weights, norm_weights, print_=i == 0 or i == 27
            )

        self.norm.weight = weights["norm.weight"]

    def __call__(self, tokens: jnp.ndarray, start_pos: int):
        _bsz, seqlen = tokens.shape
        print(f"Transformer input tokens shape: {tokens.shape}")
        print(f"Token embeddings shape: {self.tok_embeddings.shape}")
        print(f"Token embeddings dtype: {self.tok_embeddings.dtype}")
        # Token embeddings should have shape (vocab_size, dim)
        # tokens has shape (batch_size, seq_len)
        # We want to index into the vocab dimension to get (batch_size, seq_len, dim)
        h = self.tok_embeddings[tokens]
        print(f"After token embedding, h shape: {h.shape}")
        print(f"After token embedding, h dtype: {h.dtype}")
        # Use the full freqs_cis tensor instead of slicing
        # The apply_rotary_emb function will handle the proper indexing

        mask = None
        if seqlen > 1:
            if self.params.attn_type == "causal":
                # Causal attention mask
                mask = jnp.full((seqlen, seqlen), float("-inf"))
                mask = jnp.triu(mask, k=1)

                # When performing key-value caching, we compute the attention scores
                # only for the new sequence. Thus, the matrix of scores is of size
                # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
                # j > cache_len + i, since row i corresponds to token cache_len + i.
                mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask]).astype(h.dtype)
            else:
                # Window attention mask
                # For window attention, we need to create a mask that allows attention
                # only within the window for each token
                cache_len = start_pos + seqlen
                mask = jnp.full((seqlen, cache_len), float("-inf"))

                # For each token in the current sequence, allow attention to tokens
                # within the window size
                for i in range(seqlen):
                    current_pos = start_pos + i
                    window_start = max(0, current_pos - self.params.window_size + 1)
                    window_end = current_pos + 1
                    mask = mask.at[i, window_start:window_end].set(0.0)

        for layer in self.layers:
            h = layer(h, start_pos, self.freqs_cis, mask)
        h = self.norm(h)
        output = jnp.dot(h, self.output.T).astype(jnp.float32)
        return output


def sample_top_p(probs: jnp.ndarray, p: float) -> jnp.ndarray:
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (jnp.ndarray): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        jnp.ndarray: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = jax.lax.sort_key_val(probs, jnp.arange(probs.shape[-1]), reverse=True)
    probs_sum = jnp.cumsum(probs_sort, axis=-1)
    mask = probs_sum - probs_sort > p
    probs_sort = jnp.where(mask, 0.0, probs_sort)
    probs_sort = probs_sort / jnp.sum(probs_sort, axis=-1, keepdims=True)
    next_token = jax.random.categorical(jax.random.PRNGKey(0), probs_sort, shape=(1,))
    next_token = jnp.take_along_axis(probs_idx, next_token, axis=-1)
    return next_token


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        window_size: int,
        attn_type: str,
        seed: int = 1,
        device: str = "cuda",
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a model checkpoint.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            seed (int): Random seed for reproducibility.
            device (str): Device to use for inference.
        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory.

        Note:
            This method loads the pre-trained model and tokenizer.
        """
        assert os.path.isdir(ckpt_dir), f"Checkpoint directory '{ckpt_dir}' does not exist."
        assert os.path.isfile(tokenizer_path), f"Tokenizer file '{tokenizer_path}' does not exist."

        # seed must be the same in all processes
        jax.random.PRNGKey(seed)

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        ckpt_path = checkpoints[0]  # Load the first checkpoint for non-distributed version

        # Load torch checkpoint and convert to numpy
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # Flatten the nested checkpoint using optree (same as StateDictLoader)
        import optree

        paths, leaves, _ = optree.tree_flatten_with_path(checkpoint)
        flat_checkpoint = {
            ".".join(str(k) for k in path): leaf for path, leaf in zip(paths, leaves, strict=True)
        }

        checkpoint_np = {k: v.to(torch.float16).numpy() for k, v in flat_checkpoint.items()}

        # Debug: print some checkpoint keys to understand the structure
        print(f"Checkpoint keys (first 10): {list(checkpoint_np.keys())[:10]}")
        if "layers.0.attention.wq.weight" in checkpoint_np:
            print(f"First layer wq shape: {checkpoint_np['layers.0.attention.wq.weight'].shape}")

        with open(Path(ckpt_dir) / "params.json") as f:
            params = json.loads(f.read())

        print(f"Loaded params from checkpoint: {params}")
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params,
        )
        model_args.window_size = window_size
        model_args.attn_type = attn_type
        print(f"Model args: {model_args}")
        tokenizer = Tokenizer(model_path=tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words

        model = Transformer(model_args)
        model.set_weights(checkpoint_np)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt_tokens: list[list[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
        enable_prefill: bool = False,
        enable_term_check: bool = False,
    ) -> tuple[list[list[int]], list[list[float]] | None]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.
            enable_prefill (bool, optional): Flag indicating whether to enable batch prefill behavior. If False, processes prompts token by token. Defaults to False.
            enable_term_check (bool, optional): Flag indicating whether to enable termination check. If True, the model will check for termination tokens and stop generating. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.
            If enable_prefill is False, prompts are processed token by token instead of using batch prefill.

        """
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = jnp.full((bsz, total_len), pad_id, dtype=jnp.int32)
        for k, t in enumerate(prompt_tokens):
            tokens = tokens.at[k, : len(t)].set(jnp.array(t, dtype=jnp.int32))

        if logprobs:
            token_logprobs = jnp.zeros_like(tokens, dtype=jnp.float32)

        prev_pos = 0
        eos_reached = jnp.array([False] * bsz)
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model(tokens, prev_pos)
            # Note: logprobs computation would need to be implemented here

        stop_tokens = jnp.array(list(self.tokenizer.stop_tokens))

        start_pos = min_prompt_len if enable_prefill else 1

        for cur_pos in range(start_pos, total_len):
            logits = self.model(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = jax.nn.softmax(logits[:, -1] / temperature, axis=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = jnp.argmax(logits[:, -1], axis=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = jnp.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens = tokens.at[:, cur_pos].set(next_token)

            if enable_term_check:
                eos_reached |= (~input_text_mask[:, cur_pos]) & (jnp.isin(next_token, stop_tokens))

            prev_pos = cur_pos
            if enable_term_check:
                if jnp.all(eos_reached):
                    break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to after eos tok if any
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if logprobs else None
                except ValueError:
                    pass
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: list[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: int | None = None,
        logprobs: bool = False,
        echo: bool = False,
        enable_prefill: bool = False,
        enable_term_check: bool = False,
    ) -> list[CompletionPrediction]:
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        return self._text_completion(
            prompt_tokens,
            temperature,
            top_p,
            max_gen_len,
            logprobs,
            echo,
            enable_prefill,
            enable_term_check,
        )

    def _text_completion(
        self,
        prompt_tokens: list[list[int]],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: int | None = None,
        logprobs: bool = False,
        echo: bool = False,
        enable_prefill: bool = False,
        enable_term_check: bool = False,
    ) -> list[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.
            enable_prefill (bool, optional): Flag indicating whether to enable batch prefill behavior. If False, processes prompts token by token. Defaults to False.
            enable_term_check (bool, optional): Flag indicating whether to enable termination check. If True, the model will check for termination tokens and stop generating. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
            enable_prefill=enable_prefill,
            enable_term_check=enable_term_check,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs, strict=False)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]


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
        **kwargs,
    ):
        dev = "cuda" if dev == "gpu" else dev
        checkpoint_dir = str(Path(checkpoint_dir + "/original").expanduser())
        # Create model args with window attention parameters

        self.llama = Llama.build(
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
        self.enable_prefill = kwargs.get("enable_prefill", False)
        self.enable_term_check = kwargs.get("enable_term_check", False)

        self.prompt_tokens = [self.llama.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

    def reset(self):
        # Reset any internal state if needed
        pass

    def compile(self):
        pass

    def warmup(self):
        self.run()

    def run(self):
        self.outputs = self.llama._text_completion(
            prompt_tokens=self.prompt_tokens,
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
            logprobs=False,
            echo=False,
            enable_prefill=self.enable_prefill,
            enable_term_check=self.enable_term_check,
        )

    def get_decoded_outputs(self) -> list[str]:
        assert self.outputs is not None, "No outputs yet, run the model first"
        return [output["generation"] for output in self.outputs]


if __name__ == "__main__":
    chk_dir = str(Path("~/.llama/checkpoints/Llama3.2-1B/original/").expanduser())

    llama = Llama.build(
        ckpt_dir=chk_dir,
        tokenizer_path=chk_dir + "/tokenizer.model",
        max_seq_len=128,
        max_batch_size=2,
        device="cpu",
    )

    print(
        llama.text_completion(
            prompts=["Hello, how are you?", "What is the capital of France?"],
            echo=True,
            temperature=0.6,
            enable_prefill=False,
            enable_term_check=False,
        )
    )
