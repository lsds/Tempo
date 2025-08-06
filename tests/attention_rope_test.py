import pytest
import torch
import numpy as np
from torch import nn
from dataclasses import replace
from functools import partial

from torchtune.modules import RotaryPositionalEmbeddings as TorchTuneRoPE
from tempo.core import dtype
from tempo.core.configs import ExecutionConfig
from tempo.api.tempo_context_manager import TempoContext
from tempo.api.recurrent_tensor import RecurrentTensor, sink_udf
from tempo.api.nn.multihead_attention import MultiHeadAttention
from tempo.utils.make_sink import make_sink


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_mha_with_rope(exec_cfg: ExecutionConfig, backend: str) -> None:
    """
    Verify that Tempo MultiHeadAttention(apply_rope=True) matches
    an equivalent PyTorch MHA flow that uses TorchTuneRoPE for rotary
    position embeddings on Q,K.

    We'll:
      1) Build & run MHA in Tempo (with RoPE).
      2) Manually compute Q/K/V in PyTorch, then feed them to PyTorch MHA,
         but we overwrite MHA's in_proj with an identity => no double-projection.
      3) Compare final outputs.
    """
    # -------------------------------------------------------------------------
    # Hyperparameters & Setup
    # -------------------------------------------------------------------------
    max_seq_len = 4
    embed_dim = 8
    num_heads = 2
    batch_size = 2
    base = 10000

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg, num_dims=2)
    (b, s) = ctx.variables
    (B, S) = ctx.upper_bounds
    sink_data = {}

    # -------------------------------------------------------------------------
    # 1) Create random input in Tempo
    # -------------------------------------------------------------------------
    with ctx:
        x_tpo = RecurrentTensor.random(
            domain=(b, s),
            shape=(embed_dim,),
            dtype=dtype.dtypes.float32,
            requires_grad=False
        )

        w_init_fun = partial(RecurrentTensor.random, requires_grad=False)
        b_init_fun = partial(RecurrentTensor.random, requires_grad=False)

        tpo_mha = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            seq_len=s,
            bias=True,
            apply_rope=True, # set apply_rope to True
            w_init_funs=[w_init_fun, w_init_fun, w_init_fun, w_init_fun],
            b_init_funs=[b_init_fun, b_init_fun, b_init_fun, b_init_fun],
        )

        out_tpo = tpo_mha.forward(x_tpo)

        sink_udf(x_tpo[0:B, 0:S], make_sink("x", sink_data))
        sink_udf(out_tpo[0:B, 0:S], make_sink("out", sink_data))

        sink_udf(tpo_mha.q_proj.w, make_sink("q_w", sink_data))
        sink_udf(tpo_mha.q_proj.b, make_sink("q_b", sink_data))
        sink_udf(tpo_mha.k_proj.w, make_sink("k_w", sink_data))
        sink_udf(tpo_mha.k_proj.b, make_sink("k_b", sink_data))
        sink_udf(tpo_mha.v_proj.w, make_sink("v_w", sink_data))
        sink_udf(tpo_mha.v_proj.b, make_sink("v_b", sink_data))
        sink_udf(tpo_mha.output_proj.w, make_sink("o_w", sink_data))
        sink_udf(tpo_mha.output_proj.b, make_sink("o_b", sink_data))

        executor = ctx.compile({B: batch_size, S: max_seq_len})
        executor.execute()

    x_pt = torch.tensor(np.asarray(sink_data["x"]), dtype=torch.float32)
    out_tpo_pt = torch.tensor(np.asarray(sink_data["out"]), dtype=torch.float32)

    assert x_pt.shape == (batch_size, max_seq_len, embed_dim)
    assert out_tpo_pt.shape == (batch_size, max_seq_len, embed_dim)

    # Build a PyTorch MHA module
    mha_pt = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
    mha_pt.eval()
    with torch.no_grad():
        # For out_proj, we'll copy the real weights from sink_data
        o_w = torch.tensor(np.asarray(sink_data["o_w"]), dtype=torch.float32)
        o_b = torch.tensor(np.asarray(sink_data["o_b"]), dtype=torch.float32)
        mha_pt.out_proj.weight.copy_(o_w)
        mha_pt.out_proj.bias.copy_(o_b)

        # Instead of the real Q/K/V weights (which would double-project),
        # we store identity blocks in in_proj so that MHA's in_proj => no-op
        id_block = torch.cat([
            torch.eye(embed_dim),  # Q chunk
            torch.eye(embed_dim),  # K chunk
            torch.eye(embed_dim),  # V chunk
        ], dim=0)  # shape: [3*embed_dim, embed_dim]

        zero_bias = torch.zeros(3 * embed_dim, dtype=torch.float32)

        mha_pt.in_proj_weight.copy_(id_block)
        mha_pt.in_proj_bias.copy_(zero_bias)

    causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
    attn_mask = causal_mask.masked_fill(causal_mask, float('-inf'))

    # Manually compute Q, K, V in PyTorch + apply RoPE
    B_ = x_pt.shape[0]
    S_ = x_pt.shape[1]
    head_dim = embed_dim // num_heads

    # Load Q/K/V from sink_data
    q_w = torch.tensor(np.asarray(sink_data["q_w"]), dtype=torch.float32)
    k_w = torch.tensor(np.asarray(sink_data["k_w"]), dtype=torch.float32)
    v_w = torch.tensor(np.asarray(sink_data["v_w"]), dtype=torch.float32)
    q_b = torch.tensor(np.asarray(sink_data["q_b"]), dtype=torch.float32)
    k_b = torch.tensor(np.asarray(sink_data["k_b"]), dtype=torch.float32)
    v_b = torch.tensor(np.asarray(sink_data["v_b"]), dtype=torch.float32)

    # Precompute Q/K/V in python
    x_flat = x_pt.reshape(-1, embed_dim)
    q_pt = (x_flat @ q_w.t() + q_b).reshape(B_, S_, embed_dim)
    k_pt = (x_flat @ k_w.t() + k_b).reshape(B_, S_, embed_dim)
    v_pt = (x_flat @ v_w.t() + v_b).reshape(B_, S_, embed_dim)

    # [B,S,nHeads,head_dim] => apply TorchTuneRoPE
    rope = TorchTuneRoPE(head_dim, max_seq_len=max_seq_len, base=base)
    q_4d = q_pt.view(B_, S_, num_heads, head_dim)
    k_4d = k_pt.view(B_, S_, num_heads, head_dim)

    with torch.no_grad():
        q_4d = rope(q_4d)
        k_4d = rope(k_4d)
    q_3d = q_4d.view(B_, S_, embed_dim)
    k_3d = k_4d.view(B_, S_, embed_dim)
    # V stays the same

    # Now feed (Q,K,V) into MHA, which has identity in_proj => no double-project
    with torch.no_grad():
        out_pt, _ = mha_pt(q_3d, k_3d, v_pt, attn_mask=attn_mask)

    print("Tempo MHA (RoPE) output:\n", out_tpo_pt)
    print("PyTorch MHA (TorchTuneRoPE) output:\n", out_pt)
    assert out_tpo_pt.shape == out_pt.shape, "Shape mismatch!"
    assert torch.allclose(out_tpo_pt, out_pt, atol=1e-4), (
        f"RoPE MHA mismatch!\n\nTempo:\n{out_tpo_pt}\n\n"
        f"PyTorch:\n{out_pt}\n\n"
        f"Max diff: {(out_tpo_pt - out_pt).abs().max().item()}"
    )
