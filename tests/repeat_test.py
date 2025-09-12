import itertools
from dataclasses import replace
from math import prod
from typing import Any, Tuple, Union
from collections.abc import Sequence

import pytest
import torch

from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.api.tempo_context_manager import TempoContext
from tempo.core.configs import ExecutionConfig
from tempo.core.dtype import DataType, dtypes
from tempo.core.dl_backend import DLBackend

def idfn_2(val: Any) -> str:
    if isinstance(val, tuple):
        if len(val) == 0:
            return "scalar"
        if isinstance(val[0], int):
            return "x".join(str(i) for i in val)
        else:
            return str(val[1]) #op_name for tempo
    if isinstance(val, str):
        return val

    if isinstance(val, (int, float)):
        return str(val)

    raise ValueError(f"Unexpected value: {val}")


def _pad_repeats(num_repeats: int | Sequence[int], tensor_ndim: int) -> tuple[int, ...]:
    """Pad num_repeats with 1s to match tensor dimensions."""
    if isinstance(num_repeats, int):
        return (num_repeats,) + (1,) * (tensor_ndim - 1)
    else:
        num_repeats = tuple(num_repeats)
        if len(num_repeats) < tensor_ndim:
            return num_repeats + (1,) * (tensor_ndim - len(num_repeats))
        return num_repeats


@pytest.mark.parametrize(
    "shape,num_repeats,dtypes,backend",
    list(
        itertools.product(
            [
                (2,),
                (2, 3),
                (2, 3, 4),
            ],
            [
                2,
                (2,),
                (2, 3),
                (2, 3, 4),
                (1, 2, 1),
            ],
            [(torch.float32, dtypes.float32), (torch.int64, dtypes.int64)],
            ["torch", "jax"],
        )
    ),
    ids=idfn_2,
)
def test_repeat(
    shape: tuple[int, ...],
    num_repeats: int | Sequence[int],
    dtypes: tuple[torch.dtype, DataType],
    backend: str,
    exec_cfg: ExecutionConfig,
):
    torch_dtype, tpo_dtype = dtypes

    # Create input tensor
    x_t = torch.arange(prod(shape), dtype=torch_dtype).reshape(shape)
    # Pad num_repeats to match tensor dimensions
    padded_repeats = _pad_repeats(num_repeats, len(shape))
    z_t = x_t.repeat(padded_repeats)

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg)
    with ctx:
        x = RecurrentTensor.arange(prod(shape), dtype=tpo_dtype).reshape(shape)
        z = x.repeat(num_repeats)

        exec = ctx.compile({})
        exec.execute()
        z_computed = exec.get_spatial_tensor_torch(z.tensor_id, ())

        assert z_t.shape == z_computed.shape, f"Shape mismatch: expected {z_t.shape}, got {z_computed.shape}"
        assert torch.allclose(z_t, z_computed), f"Values mismatch: expected {z_t}, got {z_computed}"


@pytest.mark.parametrize(
    "shape,num_repeats,dim,dtypes,backend",
    list(
        itertools.product(
            [
                (2,),
                (2, 3),
                (2, 3, 4),
            ],
            [2, 3, 4],
            [0, -1],  # Test first and last dimensions
            [(torch.float32, dtypes.float32), (torch.int64, dtypes.int64)],
            ["torch", "jax"],
        )
    ),
    ids=idfn_2,
)
def test_repeat_interleave(
    shape: tuple[int, ...],
    num_repeats: int,
    dim: int,
    dtypes: tuple[torch.dtype, DataType],
    backend: str,
    exec_cfg: ExecutionConfig,
):
    torch_dtype, tpo_dtype = dtypes

    # Create input tensor
    x_t = torch.arange(prod(shape), dtype=torch_dtype).reshape(shape)
    z_t = torch.repeat_interleave(x_t, num_repeats, dim=dim)

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg)
    with ctx:
        x = RecurrentTensor.arange(prod(shape), dtype=tpo_dtype).reshape(shape)
        z = x.repeat_interleave(num_repeats, dim=dim)

        exec = ctx.compile({})
    exec.execute()
    z_computed = exec.get_spatial_tensor_torch(z.tensor_id, ())

    assert z_t.shape == z_computed.shape, f"Shape mismatch: expected {z_t.shape}, got {z_computed.shape}"
    assert torch.allclose(z_t, z_computed), f"Values mismatch: expected {z_t}, got {z_computed}"


@pytest.mark.parametrize(
    "shape,num_repeats,dtypes,backend",
    list(
        itertools.product(
            [
                (1,),
                (1, 1),
                (1, 2, 1),
            ],
            [
                1,
                (1,),
                (1, 1),
                (1, 1, 1),
            ],
            [(torch.float32, dtypes.float32), (torch.int64, dtypes.int64)],
            ["torch", "jax"],
        )
    ),
    ids=idfn_2,
)
def test_repeat_identity(
    shape: tuple[int, ...],
    num_repeats: int | Sequence[int],
    dtypes: tuple[torch.dtype, DataType],
    backend: str,
    exec_cfg: ExecutionConfig,
):
    """Test that repeat with all 1s returns the original tensor."""
    torch_dtype, tpo_dtype = dtypes

    # Create input tensor
    x_t = torch.arange(prod(shape), dtype=torch_dtype).reshape(shape)
    # Pad num_repeats to match tensor dimensions
    padded_repeats = _pad_repeats(num_repeats, len(shape))
    z_t = x_t.repeat(padded_repeats)

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg)
    with ctx:
        x = RecurrentTensor.arange(prod(shape), dtype=tpo_dtype).reshape(shape)
        z = x.repeat(num_repeats)

        exec = ctx.compile({})
        exec.execute()
        z_computed = exec.get_spatial_tensor_torch(z.tensor_id, ())

        # Should be identical to input
        assert x_t.shape == z_computed.shape, f"Shape mismatch: expected {x_t.shape}, got {z_computed.shape}"
        assert torch.allclose(x_t, z_computed), f"Values mismatch: expected {x_t}, got {z_computed}"


@pytest.mark.parametrize(
    "shape,num_repeats,dim,dtypes,backend",
    list(
        itertools.product(
            [
                (2,),
                (2, 3),
                (2, 3, 4),
            ],
            [1, 2, 3],
            [0, -1],
            [(torch.float32, dtypes.float32), (torch.int64, dtypes.int64)],
            ["torch", "jax"],
        )
    ),
    ids=idfn_2,
)
def test_repeat_interleave_edge_cases(
    shape: tuple[int, ...],
    num_repeats: int,
    dim: int,
    dtypes: tuple[torch.dtype, DataType],
    backend: str,
    exec_cfg: ExecutionConfig,
):
    """Test edge cases for repeat_interleave."""
    torch_dtype, tpo_dtype = dtypes

    # Create input tensor with some zeros to test edge cases
    x_t = torch.arange(prod(shape), dtype=torch_dtype).reshape(shape)
    # Set some values to zero to test edge cases
    if len(shape) > 0:
        x_t[0] = 0
    z_t = torch.repeat_interleave(x_t, num_repeats, dim=dim)

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg)
    bend = DLBackend.get_backend(backend)
    with ctx:
        x = RecurrentTensor.source_udf(lambda: bend.from_dlpack(x_t), shape=shape, dtype=tpo_dtype, domain=())
        z = x.repeat_interleave(num_repeats, dim=dim)

        exec = ctx.compile({})
        exec.execute()
        z_computed = exec.get_spatial_tensor_torch(z.tensor_id, ())

        assert z_t.shape == z_computed.shape, f"Shape mismatch: expected {z_t.shape}, got {z_computed.shape}"
        assert torch.allclose(z_t, z_computed), f"Values mismatch: expected {z_t}, got {z_computed}"


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_repeat_complex_shapes(exec_cfg: ExecutionConfig, backend: str):
    """Test repeat with complex shapes and different repeat patterns."""
    shape = (2, 3, 4)
    num_repeats = (3, 1, 2)

    x_t = torch.arange(prod(shape), dtype=torch.float32).reshape(shape)
    z_t = x_t.repeat(num_repeats)

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg)
    with ctx:
        x = RecurrentTensor.arange(prod(shape), dtype=dtypes.float32).reshape(shape)
        z = x.repeat(num_repeats)

        exec = ctx.compile({})
        exec.execute()
        z_computed = exec.get_spatial_tensor_torch(z.tensor_id, ())

        assert z_t.shape == z_computed.shape, f"Shape mismatch: expected {z_t.shape}, got {z_computed.shape}"
        assert torch.allclose(z_t, z_computed), f"Values mismatch: expected {z_t}, got {z_computed}"


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_repeat_interleave_middle_dim(exec_cfg: ExecutionConfig, backend: str):
    """Test repeat_interleave on middle dimensions."""
    shape = (2, 3, 4)
    num_repeats = 2
    dim = 1  # Middle dimension

    x_t = torch.arange(prod(shape), dtype=torch.float32).reshape(shape)
    z_t = torch.repeat_interleave(x_t, num_repeats, dim=dim)

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg)
    with ctx:
        x = RecurrentTensor.arange(prod(shape), dtype=dtypes.float32).reshape(shape)
        z = x.repeat_interleave(num_repeats, dim=dim)

        exec = ctx.compile({})
        exec.execute()
        z_computed = exec.get_spatial_tensor_torch(z.tensor_id, ())

        assert z_t.shape == z_computed.shape, f"Shape mismatch: expected {z_t.shape}, got {z_computed.shape}"
        assert torch.allclose(z_t, z_computed), f"Values mismatch: expected {z_t}, got {z_computed}"


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_repeat_scalar_input(exec_cfg: ExecutionConfig, backend: str):
    """Test repeat with scalar input."""
    shape = ()
    num_repeats = (2, 3, 4)

    x_t = torch.tensor(5.0, dtype=torch.float32)
    z_t = x_t.repeat(num_repeats)

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg)
    with ctx:
        x = RecurrentTensor.const(5.0, dtype=dtypes.float32)
        z = x.repeat(num_repeats)

        exec = ctx.compile({})
        exec.execute()
        z_computed = exec.get_spatial_tensor_torch(z.tensor_id, ())

        assert z_t.shape == z_computed.shape, f"Shape mismatch: expected {z_t.shape}, got {z_computed.shape}"
        assert torch.allclose(z_t, z_computed), f"Values mismatch: expected {z_t}, got {z_computed}"


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_repeat_interleave_negative_dim(exec_cfg: ExecutionConfig, backend: str):
    """Test repeat_interleave with negative dimension indexing."""
    shape = (2, 3, 4)
    num_repeats = 2
    dim = -2  # Should be equivalent to dim=1

    x_t = torch.arange(prod(shape), dtype=torch.float32).reshape(shape)
    z_t = torch.repeat_interleave(x_t, num_repeats, dim=dim)

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg)
    with ctx:
        x = RecurrentTensor.arange(prod(shape), dtype=dtypes.float32).reshape(shape)
        z = x.repeat_interleave(num_repeats, dim=dim)

        exec = ctx.compile({})
        exec.execute()
        z_computed = exec.get_spatial_tensor_torch(z.tensor_id, ())

        assert z_t.shape == z_computed.shape, f"Shape mismatch: expected {z_t.shape}, got {z_computed.shape}"
        assert torch.allclose(z_t, z_computed), f"Values mismatch: expected {z_t}, got {z_computed}"
