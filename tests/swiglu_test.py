import numpy as np
import torch
import pytest
from dataclasses import replace

from tempo.api.tempo_context_manager import TempoContext
from tempo.api.recurrent_tensor import RecurrentTensor, sink_udf
from tempo.core.configs import ExecutionConfig
from tempo.core import dtype
from tempo.api.nn.swiglu import SwiGLU


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_swiglu(exec_cfg: ExecutionConfig, backend: str) -> None:
    batch_size = 2
    seq_len = 5
    in_features = 16
    out_features = 32

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg, num_dims=2)
    (b, s) = ctx.variables
    (B, S) = ctx.upper_bounds

    sink_data = {}

    def store_sink(name):
        def _fn(value):
            sink_data[name] = value

        return _fn

    with ctx:
        x_tpo = RecurrentTensor.random(
            domain=(b, s),
            shape=(in_features,),
            dtype=dtype.dtypes.float32,
            requires_grad=True
        )

        swiglu_tpo = SwiGLU(
            in_features=in_features,
            out_features=out_features,
            dtype=dtype.dtypes.float32
        )
        swiglu_tpo.fixed()

        # Forward pass
        out_tpo = swiglu_tpo(x_tpo)

        # Store inputs, outputs and weights
        sink_udf(x_tpo[0:B, 0:S], store_sink("x"))
        sink_udf(swiglu_tpo.swish_linear_layer.w, store_sink("w"))
        sink_udf(swiglu_tpo.swish_linear_layer.b, store_sink("b"))
        sink_udf(swiglu_tpo.linear_layer.w, store_sink("v"))
        sink_udf(swiglu_tpo.linear_layer.b, store_sink("c"))
        sink_udf(out_tpo[0:B, 0:S], store_sink("out"))

        executor = ctx.compile({S: seq_len, B: batch_size})
        executor.execute()

    # Convert to PyTorch tensors
    x_pt = torch.tensor(np.array(sink_data["x"]), dtype=torch.float32)
    w_pt = torch.tensor(np.array(sink_data["w"]), dtype=torch.float32)
    b_pt = torch.tensor(np.array(sink_data["b"]), dtype=torch.float32)
    v_pt = torch.tensor(np.array(sink_data["v"]), dtype=torch.float32)
    c_pt = torch.tensor(np.array(sink_data["c"]), dtype=torch.float32)
    out_tpo_pt = torch.tensor(np.array(sink_data["out"]), dtype=torch.float32)

    # Reshape for PyTorch linear operations
    x_pt_flat = x_pt.reshape(-1, in_features)  # [batch_size*seq_len, in_features]


    # Compute PyTorch SwiGLU equivalent (PyTorch doesn't have SwiGLU directly, only has Swish)
    with torch.no_grad():
        # Gate part: Swish(xW + b)
        gate = torch.nn.functional.linear(x_pt_flat, w_pt, b_pt)
        gate = gate * torch.sigmoid(gate)

        # Linear part: xV + c
        linear = torch.nn.functional.linear(x_pt_flat, v_pt, c_pt)

        # Swish(xW + b) ⊙ (xV + c)
        out_ref_flat = gate * linear
        out_ref_pt = out_ref_flat.reshape(batch_size, seq_len, out_features)

    assert torch.allclose(out_tpo_pt, out_ref_pt, atol=1e-5), (
        f"Tempo SwiGLU != PyTorch equivalent\n"
        f"Tempo:\n{out_tpo_pt.flatten()[:10]}\n\n"
        f"PyTorch:\n{out_ref_pt.flatten()[:10]}\n\n"
        f"Abs Diff: {(out_tpo_pt - out_ref_pt).abs().max()}"
    )
