from functools import partial

import numpy as np
import pytest
import torch
from dataclasses import replace

from tempo.api.nn.linear import Linear
from tempo.api.tempo_context_manager import TempoContext
from tempo.api.recurrent_tensor import RecurrentTensor, sink_udf
from tempo.core.configs import ExecutionConfig
from tempo.core import dtype

@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_linear_forward_random_weights(exec_cfg: ExecutionConfig, backend: str) -> None:
    """
    Demonstration:
    1) Create random w, b, and X purely in Tempo (using RecurrentTensor.random).
    2) Run a forward pass in Tempo.
    3) Retrieve w, b, X, out as PyTorch tensors.
    4) Build a torch.nn.Linear using those exact w, b.
    5) Forward in torch, compare with Tempo's result.
    """

    # Dimensions
    in_features = 3
    out_features = 2

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg, num_dims=1)
    (b,) = ctx.variables
    (B,) = ctx.upper_bounds

    sink_data = {}

    def make_sink(name):
        def _store(array_value):
            sink_data[name] = array_value

        return _store

    with ctx:
        # Generate random W, b, and X from Tempo
        #    shape=(in_features, out_features) for w,
        #    shape=(out_features,) for b,
        #    shape=(batch_size, in_features) for X
        w_init_fun = partial(
            RecurrentTensor.random,
            requires_grad=True
        )
        b_init_fun = partial(
            RecurrentTensor.random,
            requires_grad=True
        )
        x_tpo = RecurrentTensor.random(
            domain= (b,),
            shape=in_features,
            dtype=dtype.dtypes.float32,
            requires_grad=True
        )

        # Build a Tempo Linear
        tpo_lin = Linear(in_features, out_features, bias=True, dtype=dtype.dtypes.float32, w_init_fun=w_init_fun, b_init_fun=b_init_fun)

        tpo_lin.fixed()
        # Forward pass
        out_tpo = tpo_lin(x_tpo)

        sink_udf(tpo_lin.w, make_sink("w"))
        sink_udf(tpo_lin.b, make_sink("b"))
        sink_udf(x_tpo, make_sink("x"))
        sink_udf(out_tpo, make_sink("out"))

        executor = ctx.compile({B : 5})
        executor.execute()

        # Retrieve tensors as PyTorch tensors
        w_pt = torch.tensor(np.array(sink_data["w"]), dtype=torch.float32)
        b_pt = torch.tensor(np.array(sink_data["b"]), dtype=torch.float32)
        x_pt = torch.tensor(np.array(sink_data["x"]), dtype=torch.float32)
        out_pt = torch.tensor(np.array(sink_data["out"]), dtype=torch.float32)

    # Build a PyTorch Linear with the same shape, then copy w_spatial & b_spatial
    torch_lin = torch.nn.Linear(in_features, out_features)
    with torch.no_grad():
        torch_lin.weight[:] = w_pt
        torch_lin.bias[:] = b_pt

    # Forward in PyTorch
    torch_out = torch_lin(x_pt)

    # Compare
    assert torch.allclose(torch_out, out_pt, atol=1e-6), (
        f"\nTempo output:\n{out_pt}\n!=\nPyTorch output:\n{torch_out}"
    )
