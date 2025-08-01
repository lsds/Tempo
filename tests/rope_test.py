import numpy as np
import torch
import pytest
from dataclasses import replace

from torchtune.modules import RotaryPositionalEmbeddings as TorchTuneRoPE
from tempo.api.tempo_context_manager import TempoContext
from tempo.api.recurrent_tensor import RecurrentTensor, sink_udf
from tempo.core.configs import ExecutionConfig
from tempo.core import dtype
from tempo.api.nn.rope import RotaryPositionalEmbeddings

@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_rope_torchtune(exec_cfg: ExecutionConfig, backend: str) -> None:
    max_seq_len = 100
    num_heads = 4
    dim_per_head = 32
    base = 10000
    batch_size = 2

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg, num_dims=2)
    (b,s) = ctx.variables
    (B,S) = ctx.upper_bounds

    sink_data = {}

    def store_sink(name):
        def _fn(x_val):
            sink_data[name] = x_val
        return _fn

    with ctx:
        x_tpo = RecurrentTensor.random(
            domain=(b,s),
            shape=(num_heads, dim_per_head),
            dtype=dtype.dtypes.float32,
            requires_grad=True
        )

        rope_tpo = RotaryPositionalEmbeddings(
            dim=dim_per_head,
            n_heads=num_heads,
            t=s,
            base=base,
            dtype=dtype.dtypes.float32
        )
        rope_tpo.fixed()

        out_tpo = rope_tpo(x_tpo)
        sink_udf(x_tpo[0:B, 0:S], store_sink("x"))
        sink_udf(out_tpo[0:B, 0:S], store_sink("out"))

        executor = ctx.compile({S: max_seq_len, B: batch_size})
        executor.execute()

    # Compare with torchtune

    x_pt = torch.tensor(np.array(sink_data["x"]), dtype=torch.float32)  # shape = [batch_size, seq_len, nHeads, dim]
    out_tpo_pt = torch.tensor(np.array(sink_data["out"]), dtype=torch.float32)

    assert x_pt.shape == out_tpo_pt.shape
    assert x_pt.shape == (batch_size, max_seq_len, num_heads, dim_per_head)

    # Create torchtune instance
    rope_ref = TorchTuneRoPE(dim_per_head, max_seq_len=max_seq_len, base=base)
    with torch.no_grad():
        out_ref_pt = rope_ref(x_pt)  # [batch_size, seq_len, nHeads, dim]

    assert torch.allclose(out_tpo_pt, out_ref_pt, atol=1e-5), (
        f"Tempo RoPE != torchtune RoPE\n"
        f"Tempo:\n{out_tpo_pt}\n\n"
        f"torchtune:\n{out_ref_pt}"
    )
