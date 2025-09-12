import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Literal,
)

import torch
import torch.nn.functional as F
from torch import nn

from .tokenizer_shared import CompletionPrediction, Tokenizer

"""This file is adapted from https://github.com/meta-llama/llama3/tree/main/llama
Fairscale components were replaced with torch equivalents.
An option to disable prefill was added.
"""


Role = Literal["system", "user", "assistant"]


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


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


# def precompute_freqs_cis(
#    dim: int, end: int, theta: float = 10000.0
# ) -> tuple[torch.Tensor, torch.Tensor]:
#    # angles: (end, dim//2)
#    arange_ = torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2]
#    freqs: torch.Tensor = 1.0 / (theta ** (arange_ / dim))
#    t: torch.Tensor = torch.arange(end, dtype=torch.float32)
#    angles: torch.Tensor = torch.outer(t, freqs)
#    return torch.cos(angles), torch.sin(angles)  # (end, dim//2) each


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# def apply_rotary_emb(
#    xq: torch.Tensor,
#    xk: torch.Tensor,
#    cos: torch.Tensor,
#    sin: torch.Tensor,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#    # reshape last dim into pairs
#    xq_pair: torch.Tensor = xq.float().reshape(*xq.shape[:-1], -1, 2)
#    xk_pair: torch.Tensor = xk.float().reshape(*xk.shape[:-1], -1, 2)
#
#    # broadcast cos/sin over batch/heads; they are shaped (seq_len, n_pairs)
#    cos_half: torch.Tensor = reshape_for_broadcast(cos, xq_pair[..., 0])
#    sin_half: torch.Tensor = reshape_for_broadcast(sin, xq_pair[..., 0])
#    cos_k: torch.Tensor = reshape_for_broadcast(cos, xk_pair[..., 0])
#    sin_k: torch.Tensor = reshape_for_broadcast(sin, xk_pair[..., 0])
#
#    # rotate pairs: (x0, x1) -> (x0*cos - x1*sin, x0*sin + x1*cos)
#    e, o = xq_pair[..., 0], xq_pair[..., 1]
#    xk0, xk1 = xk_pair[..., 0], xk_pair[..., 1]
#
#    xq_out0: torch.Tensor = e * cos_half - o * sin_half
#    xq_out1: torch.Tensor = o * cos_half + e * sin_half
#
#    xk_out0: torch.Tensor = xk0 * cos_k - xk1 * sin_k
#    xk_out1: torch.Tensor = xk0 * sin_k + xk1 * cos_k
#
#    xq_out: torch.Tensor = torch.stack((xq_out0, xq_out1), dim=-1).reshape(xq.shape)
#    xk_out: torch.Tensor = torch.stack((xk_out0, xk_out1), dim=-1).reshape(xk.shape)
#    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        # Window attention parameters
        self.window_size = args.window_size
        self.attn_type = args.attn_type

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            )
        ).to(args.device)
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            )
        ).to(args.device)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

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

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
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

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None,
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            if self.params.attn_type == "causal":
                # Causal attention mask
                mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
                mask = torch.triu(mask, diagonal=1)

                # When performing key-value caching, we compute the attention scores
                # only for the new sequence. Thus, the matrix of scores is of size
                # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
                # j > cache_len + i, since row i corresponds to token cache_len + i.
                mask = torch.hstack(
                    [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
                ).type_as(h)
            else:
                # Window attention mask
                # For window attention, we need to create a mask that allows attention
                # only within the window for each token
                cache_len = start_pos + seqlen
                mask = torch.full((seqlen, cache_len), float("-inf"), device=tokens.device)

                # For each token in the current sequence, allow attention to tokens
                # within the window size
                for i in range(seqlen):
                    current_pos = start_pos + i
                    window_start = max(0, current_pos - self.params.window_size + 1)
                    window_end = current_pos + 1
                    mask[i, window_start:window_end] = 0.0

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
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
            This method sets the device to CUDA and loads the pre-trained model and tokenizer.
            For non-distributed version, only the first checkpoint file is loaded.
        """
        assert os.path.isdir(ckpt_dir), f"Checkpoint directory '{ckpt_dir}' does not exist."
        assert os.path.isfile(tokenizer_path), f"Tokenizer file '{tokenizer_path}' does not exist."

        # Set device to CUDA
        if device == "cuda":
            torch.cuda.set_device(0)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        ckpt_path = checkpoints[0]  # Load the first checkpoint for non-distributed version
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json") as f:
            params = json.loads(f.read())

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
        # if torch.cuda.is_bf16_supported():
        #    torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        # else:
        # torch.set_default_tensor_type(torch.cuda.HalfTensor)
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.HalfTensor)

        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
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
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=params.device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=params.device)
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=params.device)
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens))

        start_pos = min_prompt_len if enable_prefill else 1

        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            if enable_term_check:
                eos_reached |= (~input_text_mask[:, cur_pos]) & (
                    torch.isin(next_token, stop_tokens)
                )

            prev_pos = cur_pos
            if enable_term_check:
                if all(eos_reached):
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


class TorchLlama32InferenceRunner:
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
        # self.run()
        # NOTE: Torch does not require warmup
        pass

    def run(self):
        self.outputs, _ = self.llama.generate(
            prompt_tokens=self.prompt_tokens,
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
            logprobs=False,
            echo=False,
            enable_prefill=self.enable_prefill,
            enable_term_check=self.enable_term_check,
        )

        if self.llama.model.params.device == "cuda":
            torch.cuda.synchronize()

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
