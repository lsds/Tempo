
import numpy as np
import pytest
import torch
from dataclasses import replace

from tempo.api.nn.rms_norm import RMSNorm
from tempo.api.tempo_context_manager import TempoContext
from tempo.api.recurrent_tensor import RecurrentTensor, sink_udf
from tempo.core.configs import ExecutionConfig
from tempo.core import dtype

@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("elementwise_affine", [True, False])
def test_rms_norm(exec_cfg: ExecutionConfig, backend: str, elementwise_affine: bool) -> None:

    batch_size = 2
    num_of_features = 3
    feature_dim = 4
    eps = 1e-8

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
        x_tpo = RecurrentTensor.random(
            domain=(b,),
            shape=(num_of_features, feature_dim),
            dtype=dtype.dtypes.float32,
            requires_grad=True
        )

        tpo_rms = RMSNorm(
            normalized_shape=(feature_dim,),
            eps=eps,
            elementwise_affine=elementwise_affine,
            dtype=dtype.dtypes.float32,
        )

        tpo_rms.fixed()

        out_tpo = tpo_rms(x_tpo)

        sink_udf(x_tpo[0:B], make_sink("x"))
        sink_udf(out_tpo[0:B], make_sink("out"))

        executor = ctx.compile({B: batch_size})
        executor.execute()

    x_pt = torch.tensor(np.array(sink_data["x"]), dtype=torch.float32)
    out_pt = torch.tensor(np.array(sink_data["out"]), dtype=torch.float32)


    torch_rms = torch.nn.RMSNorm(feature_dim, eps=eps, elementwise_affine=elementwise_affine)
    torch_out = torch_rms(x_pt)
    assert torch.allclose(torch_out, out_pt, atol=1e-5), (
        f"\nTempo RMSNorm output:\n{out_pt}\n"
        f"!=\nPyTorch RMSNorm output:\n{torch_out}"
    )
