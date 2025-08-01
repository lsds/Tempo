import torch
import torch.nn as nn
import torch.nn.functional as F

from repro.sec7_2_lm_decode.bench_runner import BenchRunner

"""Simple Torch implementation of the GPT-2-Small core.
"""


def rms_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Applies RMSNorm: x * (weight / rms(x))"""
    rms = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + eps)
    normed = x / rms
    return normed


def init_linear(in_features: int, out_features: int, device: str = "cuda") -> nn.Linear:
    """Initialize a linear layer with zero bias and random weights."""
    linear = nn.Linear(in_features, out_features, device=device)
    nn.init.normal_(linear.weight, mean=0.0, std=0.02)
    nn.init.zeros_(linear.bias)
    return linear


class MHA(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_size: int,
        window_size: int,
        attn_type: str,
        seq_len: int,
        batch_size: int,
        device: str,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.embed_size_per_head = embed_size // num_heads
        self.window_size = window_size
        self.attn_type = attn_type
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device

        # Initialize QKV projection weights
        self.W_Q = nn.Parameter(
            torch.randn(num_heads * self.embed_size_per_head, embed_size, device=device)
        )
        self.W_K = nn.Parameter(
            torch.randn(num_heads * self.embed_size_per_head, embed_size, device=device)
        )
        self.W_V = nn.Parameter(
            torch.randn(num_heads * self.embed_size_per_head, embed_size, device=device)
        )
        self.W_O = init_linear(num_heads * self.embed_size_per_head, embed_size, device)

        # Preallocate KV cache
        self.key_cache = torch.zeros(
            (batch_size, num_heads, seq_len, self.embed_size_per_head), device=device
        )
        self.value_cache = torch.zeros(
            (batch_size, num_heads, seq_len, self.embed_size_per_head), device=device
        )

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        # Compute Q, K, V projections
        # projections = torch.matmul(x, self.W_QKV)  # (3, num_heads, embed_per_head)
        assert x.shape == (self.batch_size, 1, self.embed_size), f"Input shape: {x.shape}"

        q = x @ self.W_Q.t()  # (num_heads, embed_per_head)
        k = x @ self.W_K.t()  # (num_heads, embed_per_head)
        v = x @ self.W_V.t()  # (num_heads, embed_per_head)

        q = q.reshape(self.batch_size, self.num_heads, 1, self.embed_size_per_head)
        k = k.reshape(self.batch_size, self.num_heads, 1, self.embed_size_per_head)
        v = v.reshape(self.batch_size, self.num_heads, 1, self.embed_size_per_head)

        # Update KV cache
        self.key_cache[:, :, step : step + 1, :] = k
        self.value_cache[:, :, step : step + 1, :] = v

        # Compute attention scores with the appropriate window
        if self.attn_type == "causal":
            keys_to_use = self.key_cache[:, :, 0 : step + 1, :]
            values_to_use = self.value_cache[:, :, 0 : step + 1, :]
        else:
            start = max(0, step - self.window_size + 1)
            keys_to_use = self.key_cache[:, :, start : step + 1, :]
            values_to_use = self.value_cache[:, :, start : step + 1, :]

        # Compute attention
        attn_scores = (q @ keys_to_use.transpose(-2, -1)) / (self.embed_size**0.5)
        if self.attn_type == "causal":
            assert attn_scores.shape == (self.batch_size, self.num_heads, 1, step + 1), (
                f"Attn scores shape: {attn_scores.shape}, expected: {(self.batch_size, self.num_heads, 1, step)}"
            )
        else:
            expected_shape = (self.batch_size, self.num_heads, 1, min(self.window_size, step + 1))
            assert attn_scores.shape == expected_shape, (
                f"Attn scores shape: {attn_scores.shape}, expected: {expected_shape}"
            )
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, values_to_use)  # (num_heads, embed_per_head)

        # Project output
        attn_output = attn_output.reshape(self.batch_size, -1)  # Flatten heads
        output = self.W_O(attn_output)  # (embed_size)
        return output.reshape(self.batch_size, 1, self.embed_size)


class GPT2Block(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_size: int,
        window_size: int,
        attn_type: str,
        seq_len: int,
        batch_size: int,
        device: str,
        first_block: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.embed_size_per_head = embed_size // num_heads
        self.window_size = window_size
        self.attn_type = attn_type
        self.first_block = first_block

        self.mha = MHA(num_heads, embed_size, window_size, attn_type, seq_len, batch_size, device)
        self.ff1 = init_linear(embed_size, embed_size * 4, device)
        self.ff2 = init_linear(embed_size * 4, embed_size, device)

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        x_start = x
        x = self.mha(x, step)
        x_ = x + x_start

        x_middle = rms_norm(x_)

        x = self.ff1(x_middle)
        x = F.silu(x)
        x = self.ff2(x)
        x = rms_norm(x + x_middle)
        return x


class GPT2(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        embed_size: int,
        window_size: int,
        attn_type: str,
        seq_len: int,
        batch_size: int,
        device: str,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                GPT2Block(
                    num_heads,
                    embed_size,
                    window_size,
                    attn_type,
                    seq_len,
                    batch_size,
                    device,
                    first_block=i == 0,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, step)
        return x


class TorchBenchRunner(BenchRunner):
    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        embed_size: int,
        seq_len: int,
        window_size: int,
        attn_type: str,
        batch_size: int,
        device: str = "cuda",
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
        self.device = device

    def compile(self):
        pass

    def warmup(self):
        self.run()

    def run(self):
        self.model = GPT2(
            self.num_blocks,
            self.num_heads,
            self.embed_size,
            self.window_size,
            self.attn_type,
            self.seq_len,
            self.batch_size,
            self.device,
        ).to(self.device)
        # Preallocate output
        output = torch.randn((self.batch_size, self.seq_len, self.embed_size), device=self.device)

        with torch.no_grad():
            # Run autoregressive decoding
            for step in range(1, self.seq_len):
                token = output[:, step - 1 : step, :]  # Shape: (B, 1, E)
                assert token.shape == (
                    self.batch_size,
                    1,
                    self.embed_size,
                ), f"Token shape: {token.shape}"
                output[:, step : step + 1, :] = self.model(token, step)

        torch.cuda.synchronize()
        return output


if __name__ == "__main__":
    runner = TorchBenchRunner(
        num_blocks=4,
        num_heads=12,
        embed_size=768,
        seq_len=64,
        window_size=8,
        attn_type="causal",
        batch_size=8,
        device="cpu",
    )
    runner.compile()
    runner.run()
