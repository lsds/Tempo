from dataclasses import replace

import pytest
import torch
import torch.utils
import torch.utils.dlpack

from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.api.tempo_context_manager import TempoContext
from tempo.core.configs import ExecutionConfig
from tempo.core.dtype import dtypes


def test_scatter_add(
    exec_cfg: ExecutionConfig,
):
    results = []
    for backend in ["torch", "jax"]:
        cfg = replace(exec_cfg, backend=backend)
        ctx = TempoContext(cfg)
        with ctx:
            dim = 1
            sink = RecurrentTensor.zeros((3, 5), dtype=dtypes.float32)
            index = RecurrentTensor.lift(
                [[0, 1, 2, 0, 1], [1, 2, 0, 1, 0], [2, 0, 1, 2, 0]]
            ).cast(dtypes.int64)
            src = RecurrentTensor.lift(
                [[1.0, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15.0]]
            )

            res = sink.scatter_add(dim, index, src)

            exec = ctx.compile({})
            exec.execute()

            res_c = exec.get_spatial_tensor_torch(res.tensor_id, ())

            results.append(torch.utils.dlpack.from_dlpack(res_c))

    assert torch.allclose(results[0], results[1]), str(results)


@pytest.mark.parametrize(
    "sink_shape, indices_shape, dim",
    [
        # Cases where sink_shape == indices_shape
        ((3, 5), (3, 5), 1),
        ((5, 4, 3), (5, 4, 3), 2),
        ((2, 3, 4, 5), (2, 3, 4, 5), 0),
        # Cases where sink_shape > indices_shape
        ((5, 5), (3, 5), 0),
        ((6, 5, 4), (6, 2, 4), 1),
        ((7, 6, 5, 4), (7, 6, 3, 4), 2),
    ],
)
def test_scatter_add2(sink_shape, indices_shape, dim, exec_cfg: ExecutionConfig):
    import numpy as np
    import torch

    # Seed for reproducibility
    np.random.seed(42)

    # Generate indices and source tensors
    indices = np.random.randint(0, sink_shape[dim], size=indices_shape)
    src_values = np.random.randn(*indices_shape).astype(np.float32)

    results = []
    for backend in ["torch", "jax"]:
        cfg = replace(exec_cfg, backend=backend)
        ctx = TempoContext(cfg)
        with ctx:
            # Create sink tensor filled with zeros
            sink = RecurrentTensor.zeros(sink_shape, dtype=dtypes.float32)

            # Generate indices tensor
            index_tensor = RecurrentTensor.lift(indices).cast(dtypes.int64)

            # Generate source tensor
            src_tensor = RecurrentTensor.lift(src_values)

            # Perform scatter_add
            res = sink.scatter_add(dim, index_tensor, src_tensor)

            # Compile and execute
            exec = ctx.compile({})
            exec.execute()

            # Get the result tensor
            res_c = exec.get_spatial_tensor_torch(res.tensor_id, ())

            # Convert to PyTorch tensor for comparison
            result_tensor = torch.utils.dlpack.from_dlpack(res_c)
            results.append(result_tensor)

    # Compare the results from both backends
    if not torch.allclose(results[0], results[1], atol=1e-6):
        print(
            f"Results mismatch for sink_shape {sink_shape}, indices_shape {indices_shape} on dim {dim}"
        )
        print(f"Sink shape: {sink_shape}")
        print(f"Indices shape: {indices_shape}")
        print(f"Dimension: {dim}")
        print(f"Indices:\n{indices}")
        print(f"Source values:\n{src_values}")
        print(f"Backend torch result:\n{results[0]}")
        print(f"Backend jax result:\n{results[1]}")
        assert (
            False
        ), f"Results mismatch for sink_shape {sink_shape}, indices_shape {indices_shape} on dim {dim}"
    else:
        print(
            f"Results match for sink_shape {sink_shape}, indices_shape {indices_shape} on dim {dim}"
        )


def test_index_add(
    exec_cfg: ExecutionConfig,
):
    for index_, src_ in [
        (
            [0, 1, 2],
            [
                [
                    1.0,
                    2,
                    3,
                ],
                [
                    6,
                    7,
                    8,
                ],
                [11, 12, 13],
            ],
        ),
        (
            0,
            [
                [
                    1.0,
                ],
                [
                    2.0,
                ],
                [
                    3.0,
                ],
            ],
        ),
    ]:
        results = []
        for backend in ["torch", "jax"]:
            ctx = TempoContext()
            ctx.execution_config = exec_cfg
            ctx.execution_config.backend = backend
            with ctx:
                sink = RecurrentTensor.zeros((3, 5), dtype=dtypes.float32)
                index = RecurrentTensor.lift(index_).cast(dtypes.int64)
                src = RecurrentTensor.lift(src_)

                res = sink.index_add(1, index, src)

                exec = ctx.compile({})
                exec.execute()

                res_c = exec.get_spatial_tensor_torch(res.tensor_id, ())

                results.append(torch.utils.dlpack.from_dlpack(res_c))

        assert torch.allclose(results[0], results[1])
