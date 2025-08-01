from functools import partial

import numpy as np
import pytest
import torch
from dataclasses import replace
import torch.nn as nn
import tempo.core.index_expr as ie

from tempo.api.nn.multihead_attention import MultiHeadAttention
from tempo.api.tempo_context_manager import TempoContext
from tempo.api.recurrent_tensor import RecurrentTensor, sink_udf
from tempo.core.configs import ExecutionConfig
from tempo.core import dtype
from tempo.utils.make_sink import make_sink


@pytest.mark.parametrize("backend", ["jax"])
@pytest.mark.skip #TODO: unskip
def test_multihead_causal_attention(backend: str) -> None:
    seq_len = 10
    embed_dim = 12
    num_heads = 4
    batch_size = 5

    sink_data = {}

    exec_cfg = replace(ExecutionConfig.default(), backend=backend)
    exec_cfg.enable_inplace_write = True
    exec_cfg.visualize_pipeline_stages = True
    exec_cfg.inc_statify_block_size = 5

    ctx = TempoContext(exec_cfg, num_dims=2)
    (b,s) = ctx.variables
    (B,S) = ctx.upper_bounds

    with ctx:
        x_tpo = RecurrentTensor.random(
            domain=(b,s),
            shape=(embed_dim,),
            dtype=dtype.dtypes.float32,
            requires_grad=False
        )

        w_init_fun = partial(
            RecurrentTensor.random,
            requires_grad=False
        )
        b_init_fun = partial(
            RecurrentTensor.random,
            requires_grad=False
        )

        tpo_mha = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            seq_len=s,
            bias=True,
            w_init_funs=w_init_fun,
            b_init_funs=b_init_fun,
        )

        # Forward pass
        out_tpo = tpo_mha.forward(x_tpo)

        # Store input
        sink_udf(x_tpo[0:B, 0:S], make_sink("x", sink_data))
        with ctx.tag_region("sink_data"):
            # Store parameters
            sink_udf(tpo_mha.q_proj.w, make_sink("q_weight", sink_data))
            sink_udf(tpo_mha.k_proj.w, make_sink("k_weight", sink_data))
            sink_udf(tpo_mha.v_proj.w, make_sink("v_weight", sink_data))
            sink_udf(tpo_mha.output_proj.w, make_sink("o_weight", sink_data))

            sink_udf(tpo_mha.q_proj.b, make_sink("q_bias", sink_data))
            sink_udf(tpo_mha.k_proj.b, make_sink("k_bias", sink_data))
            sink_udf(tpo_mha.v_proj.b, make_sink("v_bias", sink_data))
            sink_udf(tpo_mha.output_proj.b, make_sink("o_bias", sink_data))

            # Store output
            sink_udf(out_tpo[0:B, 0:S], make_sink("out", sink_data))

        # Compile and execute
        executor = ctx.compile({S: seq_len, B: batch_size})

        executor.execute()

    # Convert Tempo tensors to PyTorch
    x_pt = torch.tensor(np.array(sink_data["x"]), dtype=torch.float32)
    out_tpo_pt = torch.tensor(np.array(sink_data["out"]), dtype=torch.float32)
    assert x_pt.shape == (batch_size, seq_len, embed_dim)
    assert out_tpo_pt.shape == (batch_size, seq_len, embed_dim)

    # Create PyTorch MultiheadAttention
    mha_pt = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    # Copy weights from Tempo to PyTorch
    with torch.no_grad():
        q_weight = torch.tensor(np.array(sink_data.get("q_weight")), dtype=torch.float32)
        k_weight = torch.tensor(np.array(sink_data.get("k_weight")), dtype=torch.float32)
        v_weight = torch.tensor(np.array(sink_data.get("v_weight")), dtype=torch.float32)
        o_weight = torch.tensor(np.array(sink_data.get("o_weight")), dtype=torch.float32)

        q_bias = torch.tensor(np.array(sink_data.get("q_bias")), dtype=torch.float32)
        k_bias = torch.tensor(np.array(sink_data.get("k_bias")), dtype=torch.float32)
        v_bias = torch.tensor(np.array(sink_data.get("v_bias")), dtype=torch.float32)
        o_bias = torch.tensor(np.array(sink_data.get("o_bias")), dtype=torch.float32)

        # PyTorch MHA expects: in_proj_weight [3*embed_dim, embed_dim] with (out_feat, in_feat)
        mha_pt.in_proj_weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))
        mha_pt.in_proj_bias.copy_(torch.cat([q_bias, k_bias, v_bias], dim=0))
        mha_pt.out_proj.weight.copy_(o_weight)
        mha_pt.out_proj.bias.copy_(o_bias)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        attn_mask = causal_mask.masked_fill(causal_mask, float('-inf'))

    with torch.no_grad():
        out_pt, _ = mha_pt(query=x_pt, key=x_pt, value=x_pt, attn_mask=attn_mask)

    msg = f"Tempo MHA != PyTorch MHA;"
    #msg+= f"Tempo: {out_tpo_pt};"
    #msg+= f"PyTorch: {out_pt};"
    msg+= f"Difference max: {(out_tpo_pt - out_pt).abs().max().item()}"
    assert torch.allclose(out_tpo_pt, out_pt, atol=1e-4), msg


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.skip #TODO: unskip
def test_window_causal_attention(exec_cfg: ExecutionConfig, backend: str) -> None:
    """
    Demonstrates a windowed causal attention of window-size=3 in Tempo:
      - For each time t, we allow K/V from [t-2 ... t], clamping at 0.
    Then compares with PyTorch's MHA using a custom attn_mask.
    """
    seq_len = 10
    embed_dim = 8
    num_heads = 2
    batch_size = 3
    sink_data = {}

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg, num_dims=2)
    (b, s) = ctx.variables
    (B, S) = ctx.upper_bounds

    with ctx:
        x_tpo = RecurrentTensor.random(
            domain=(b, s),
            shape=(embed_dim,),
            dtype=dtype.dtypes.float32,
            requires_grad=False
        )

        # Windowed causal mask: [max(0, s-2) : s+1] (this means causal window size of 3)
        window_mask_expr = ie.Slice(ie.max(ie.ConstInt(0), s - 2), s + 1)

        w_init_fun = partial(RecurrentTensor.random, requires_grad=False)
        b_init_fun = partial(RecurrentTensor.random, requires_grad=False)

        tpo_mha = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            seq_len=s,
            bias=True,
            w_init_funs=[w_init_fun, w_init_fun, w_init_fun, w_init_fun],
            b_init_funs=[b_init_fun, b_init_fun, b_init_fun, b_init_fun],
        )

        # Forward pass with the custom slice-based mask
        out_tpo = tpo_mha.forward(x_tpo, pattern=window_mask_expr)

        sink_udf(x_tpo[0:B, 0:S], make_sink("x", sink_data))
        sink_udf(out_tpo[0:B, 0:S], make_sink("out", sink_data))

        # Sink the parameters
        sink_udf(tpo_mha.q_proj.w, make_sink("q_w", sink_data))
        sink_udf(tpo_mha.q_proj.b, make_sink("q_b", sink_data))
        sink_udf(tpo_mha.k_proj.w, make_sink("k_w", sink_data))
        sink_udf(tpo_mha.k_proj.b, make_sink("k_b", sink_data))
        sink_udf(tpo_mha.v_proj.w, make_sink("v_w", sink_data))
        sink_udf(tpo_mha.v_proj.b, make_sink("v_b", sink_data))
        sink_udf(tpo_mha.output_proj.w, make_sink("o_w", sink_data))
        sink_udf(tpo_mha.output_proj.b, make_sink("o_b", sink_data))

        executor = ctx.compile({B:batch_size, S: seq_len})
        executor.execute()

    # ---- Convert Tempo outputs to torch Tensors ----
    x_pt = torch.tensor(np.array(sink_data["x"]), dtype=torch.float32)
    out_tpo_pt = torch.tensor(np.array(sink_data["out"]), dtype=torch.float32)
    assert x_pt.shape == (batch_size, seq_len, embed_dim)
    assert out_tpo_pt.shape == (batch_size, seq_len, embed_dim)

    # Build a matching PyTorch MHA
    mha_pt = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    # Copy weights from Tempo to PyTorch
    with torch.no_grad():
        q_w = torch.tensor(np.array(sink_data["q_w"]), dtype=torch.float32).t()
        k_w = torch.tensor(np.array(sink_data["k_w"]), dtype=torch.float32).t()
        v_w = torch.tensor(np.array(sink_data["v_w"]), dtype=torch.float32).t()
        o_w = torch.tensor(np.array(sink_data["o_w"]), dtype=torch.float32).t()

        q_b = torch.tensor(np.array(sink_data["q_b"]), dtype=torch.float32)
        k_b = torch.tensor(np.array(sink_data["k_b"]), dtype=torch.float32)
        v_b = torch.tensor(np.array(sink_data["v_b"]), dtype=torch.float32)
        o_b = torch.tensor(np.array(sink_data["o_b"]), dtype=torch.float32)

        # PyTorch in_proj_weight = cat([q_w, k_w, v_w], dim=0) shape [3*embed_dim, embed_dim]
        mha_pt.in_proj_weight.copy_(torch.cat([q_w, k_w, v_w], dim=0))
        mha_pt.in_proj_bias.copy_(torch.cat([q_b, k_b, v_b], dim=0))
        mha_pt.out_proj.weight.copy_(o_w)
        mha_pt.out_proj.bias.copy_(o_b)

    # ---- Build an equivalent attn_mask for PyTorch that enforces "windowed" [t-2..t] ----
    # seq_len=5, so create a [5,5] mask where row t can only attend to [t-2..t]
    # store 0.0 for allowed positions, -inf for blocked
    attn_mask = torch.full((seq_len, seq_len), float('-inf'))
    for t in range(seq_len):
        start = max(0, t - 2)
        end = t
        attn_mask[t, start:end + 1] = 0.0

    with torch.no_grad():
        # x_pt = (3,5,8), attn_mask is (5,5)
        out_pt, _ = mha_pt(x_pt, x_pt, x_pt, attn_mask=attn_mask)

    # Compare
    print("Tempo MHA result:\n", out_tpo_pt)
    print("PyTorch MHA result:\n", out_pt)

    assert torch.allclose(out_tpo_pt, out_pt, atol=1e-5), (
        f"Windowed causal MHA mismatch!\n\nTempo:\n{out_tpo_pt}\n\n"
        f"PyTorch:\n{out_pt}\n\n"
        f"Max diff: {(out_tpo_pt - out_pt).abs().max().item()}"
    )

if __name__ == "__main__":
    test_multihead_causal_attention("jax")
