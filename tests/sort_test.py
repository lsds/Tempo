import itertools
from dataclasses import replace
from math import prod
from typing import Any

import pytest
import torch

import tempo.api.recurrent_tensor as tpo
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.api.tempo_context_manager import TempoContext
from tempo.core.configs import ExecutionConfig
from tempo.core.dtype import DataType, dtypes

BACKENDS = ["torch", "jax"]
BACKENDS = ["torch"]

SHAPES = [
    (5,),
    #(3, 4),
    #(2, 3, 4),
    (4, 5, 6),
]

DTYPES = [
    (torch.float32, dtypes.float32),
    #(torch.int64, dtypes.int64)
    ]

def idfn(val: Any) -> str:
    if isinstance(val, tuple):
        if len(val) == 0:
            return "scalar"
        if isinstance(val[0], int):
            return "x".join(str(i) for i in val)
        else:
            return str(val[1])  # op_name for tempo
    if isinstance(val, str):
        return val
    if isinstance(val, (int, float)):
        return str(val)
    raise ValueError(f"Unexpected value: {val}")


@pytest.mark.parametrize(
    "shape,dtypes,backend,dim,descending",
    list(
        itertools.product(
            SHAPES,
            DTYPES,
            BACKENDS,
            [-1, 0],  # Test sorting along different dimensions
            [True, False],  # Test both ascending and descending
        )
    ),
    ids=idfn,
)
def test_sort(
    shape: tuple[int, ...],
    dtypes: tuple[torch.dtype, DataType],
    backend: str,
    dim: int,
    descending: bool,
    exec_cfg: ExecutionConfig,
):
    torch_dtype, tpo_dtype = dtypes

    # Create test data
    x_t = torch.arange(prod(shape), dtype=torch_dtype).reshape(shape)
    # Add some randomness to make sorting more interesting
    if torch_dtype in [torch.float32, torch.float64]:
        x_t = x_t + torch.rand_like(x_t) * 10
    else:
        # For integer types, add random integer values
        x_t = x_t + torch.randint(0, 10, x_t.shape, dtype=torch_dtype)

    # Normalize dim to positive index
    normalized_dim = dim if dim >= 0 else len(shape) + dim

    # Reference implementation using torch
    sorted_values_t, indices_t = torch.sort(x_t, dim=normalized_dim, descending=descending, stable=True)

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg)
    with ctx:
        x = RecurrentTensor.const(x_t.numpy())

        # Test sort
        sorted_values, indices = tpo.sort(x, dim=dim, descending=descending, stable=True)

        exec = ctx.compile({})
        exec.execute()

        sorted_computed = exec.get_spatial_tensor_torch(sorted_values.tensor_id, ())
        indices_computed = exec.get_spatial_tensor_torch(indices.tensor_id, ())

        # Verify shapes match
        assert sorted_computed.shape == sorted_values_t.shape
        assert indices_computed.shape == indices_t.shape

        # Verify sorted values match
        assert torch.allclose(sorted_computed, sorted_values_t), f"Sorted values don't match: {sorted_computed} vs {sorted_values_t} "

        # Verify indices match
        assert torch.allclose(indices_computed, indices_t), f"Indices don't match: {indices_computed} vs {indices_t} \n {sorted_computed} \n {sorted_values_t} \n {x_t} "


#@pytest.mark.parametrize(
#    "shape,backend,dim,descending",
#    list(
#        itertools.product(
#            [
#                (5,),
#                (3, 4),
#                (2, 3, 4),
#            ],
#            ["torch", "jax"],
#            [-1, 0],
#            [True, False],
#        )
#    ),
#    ids=idfn,
#)
#def test_sort_with_duplicates(
#    shape: Tuple[int, ...],
#    backend: str,
#    dim: int,
#    descending: bool,
#    exec_cfg: ExecutionConfig,
#):
#    # Create test data with duplicates to test stability
#    x_t = torch.tensor([
#        [1.0, 2.0, 1.0, 3.0, 2.0],
#        [3.0, 1.0, 2.0, 1.0, 3.0],
#        [2.0, 3.0, 1.0, 2.0, 1.0]
#    ], dtype=torch.float32)
#
#    # Normalize dim to positive index
#    normalized_dim = dim if dim >= 0 else len(x_t.shape) + dim
#
#    # Reference implementation using torch
#    sorted_values_t, indices_t = torch.sort(x_t, dim=normalized_dim, descending=descending)
#
#    exec_cfg = replace(exec_cfg, backend=backend)
#    ctx = TempoContext(exec_cfg)
#    with ctx:
#        x = RecurrentTensor.const(x_t.numpy())
#
#        # Test sort
#        sorted_values, indices = tpo.sort(x, dim=dim, descending=descending)
#
#        exec = ctx.compile({})
#        exec.execute()
#
#        sorted_computed = exec.get_spatial_tensor_torch(sorted_values.tensor_id, ())
#        indices_computed = exec.get_spatial_tensor_torch(indices.tensor_id, ())
#
#        # Verify shapes match
#        assert sorted_computed.shape == sorted_values_t.shape
#        assert indices_computed.shape == indices_t.shape
#
#        # Verify sorted values match
#        assert torch.allclose(sorted_computed, sorted_values_t)
#
#        # Verify indices match
#        assert torch.allclose(indices_computed, indices_t)
#
#
#@pytest.mark.parametrize("backend", ["torch", "jax"])
#def test_sort_edge_cases(exec_cfg: ExecutionConfig, backend: str):
#    # Test edge cases: empty tensor, single element, etc.
#    test_cases = [
#        torch.tensor([], dtype=torch.float32),  # Empty tensor
#        torch.tensor([1.0], dtype=torch.float32),  # Single element
#        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),  # 2x2 matrix
#    ]
#
#    exec_cfg = replace(exec_cfg, backend=backend)
#    ctx = TempoContext(exec_cfg)
#
#    for x_t in test_cases:
#        with ctx:
#            # Skip empty tensors as they might not be supported
#            if x_t.numel() == 0:
#                continue
#
#            # Reference implementation
#            sorted_values_t, indices_t = torch.sort(x_t, dim=-1, descending=False)
#
#            x = RecurrentTensor.const(x_t.numpy())
#            sorted_values, indices = tpo.sort(x, dim=-1, descending=False)
#
#            exec = ctx.compile({})
#            exec.execute()
#
#            sorted_computed = exec.get_spatial_tensor_torch(sorted_values.tensor_id, ())
#            indices_computed = exec.get_spatial_tensor_torch(indices.tensor_id, ())
#
#            # Verify results
#            assert torch.allclose(sorted_computed, sorted_values_t)
#            assert torch.allclose(indices_computed, indices_t)


@pytest.mark.parametrize(
    "shape,dim,descending,stable,backend",
    list(
        itertools.product(
            SHAPES,
            [-1, 0],  # Test sorting along different dimensions
            [True, False],  # Test both ascending and descending
            [True, False],  # Test both stable and unstable sorting
            BACKENDS,  # Test both backends
        )
    ),
    ids=idfn,
)
def test_sort_autodiff(shape: tuple[int, ...], dim: int, descending: bool, stable: bool, backend: str, exec_cfg: ExecutionConfig):
    """Test that sort autodiff works correctly by comparing with PyTorch gradients."""
    import torch

    # Create test data with randomness
    torch.manual_seed(42)  # For reproducible randomness
    x_t = torch.randn(shape, dtype=torch.float32, requires_grad=True)

    # Normalize dim to positive index
    normalized_dim = dim if dim >= 0 else len(shape) + dim

    # PyTorch reference
    with torch.autograd.set_grad_enabled(True):
        with torch.inference_mode(False):
            sorted_torch, indices_torch = torch.sort(x_t, dim=normalized_dim, descending=descending, stable=stable)
            loss_torch = sorted_torch.sum()
            loss_torch.backward()
    grad_torch = x_t.grad.clone()

    # Reset gradients for Tempo test
    x_t.grad.zero_()

    # Tempo implementation
    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg)
    with ctx:
        x = RecurrentTensor.const(x_t.detach().numpy(), requires_grad=True)
        sorted_vals, indices = tpo.sort(x, dim=dim, descending=descending, stable=stable)
        loss = sorted_vals.sum()
        loss.backward()

        exec = ctx.compile({})
        exec.execute()

        grad_tempo = exec.get_spatial_tensor_torch(x.grad.tensor_id, ())

    # Verify gradients match
    assert torch.allclose(grad_torch, grad_tempo, atol=1e-5), (
        f"Gradients don't match for shape={shape}, dim={dim}, descending={descending}, stable={stable}, backend={backend}\n"
        f"PyTorch grad: {grad_torch}\nTempo grad: {grad_tempo}"
    )
    print(f"Sort autodiff test passed for shape={shape}, dim={dim}, descending={descending}, stable={stable}, backend={backend}!")


@pytest.mark.parametrize("backend", BACKENDS)
def test_sort_autodiff_with_duplicates(backend: str, exec_cfg: ExecutionConfig):
    """Test sort autodiff with duplicate values to ensure stable sorting works correctly."""
    import torch

    # Create test data with intentional duplicates
    x_t = torch.tensor([
        [3.0, 1.0, 4.0, 1.0, 2.0, 1.0],
        [5.0, 9.0, 2.0, 6.0, 2.0, 9.0],
        [1.0, 1.0, 1.0, 2.0, 2.0, 3.0]
    ], dtype=torch.float32, requires_grad=True)

    # Test both stable and unstable sorting
    for stable in [True, False]:
        for descending in [True, False]:
            for dim in [-1, 0]:
                # Normalize dim to positive index
                normalized_dim = dim if dim >= 0 else len(x_t.shape) + dim

                # PyTorch reference
                with torch.autograd.set_grad_enabled(True):
                    with torch.inference_mode(False):
                        sorted_torch, indices_torch = torch.sort(x_t, dim=normalized_dim, descending=descending, stable=stable)
                        loss_torch = sorted_torch.sum()
                        loss_torch.backward()
                grad_torch = x_t.grad.clone()

                # Reset gradients for Tempo test
                x_t.grad.zero_()

                # Tempo implementation
                exec_cfg = replace(exec_cfg, backend=backend)
                ctx = TempoContext(exec_cfg)
                with ctx:
                    x = RecurrentTensor.const(x_t.detach().numpy(), requires_grad=True)
                    sorted_vals, indices = tpo.sort(x, dim=dim, descending=descending, stable=stable)
                    loss = sorted_vals.sum()
                    loss.backward()

                    exec = ctx.compile({})
                    exec.execute()

                    grad_tempo = exec.get_spatial_tensor_torch(x.grad.tensor_id, ())

                # Verify gradients match
                assert torch.allclose(grad_torch, grad_tempo, atol=1e-5), (
                    f"Gradients don't match for duplicates test: dim={dim}, descending={descending}, stable={stable}, backend={backend}\n"
                    f"PyTorch grad: {grad_torch}\nTempo grad: {grad_tempo}"
                )
                print(f"Sort autodiff with duplicates test passed for dim={dim}, descending={descending}, stable={stable}, backend={backend}!")
