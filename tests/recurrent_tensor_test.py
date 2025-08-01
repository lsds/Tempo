import numpy as np
import torch

from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.api.tempo_context_manager import TempoContext
from tempo.core import dtype
from tempo.core.configs import ExecutionConfig


def test_int_const(exec_cfg: ExecutionConfig) -> None:
    ctx = TempoContext()
    with ctx:
        x = RecurrentTensor.ones(dtype=dtype.dtypes.int64)
        y = RecurrentTensor.const(2)
        z = x + y

        assert not z.requires_grad
        assert z._ctx is None
        assert z.grad is None

        ctx.execution_config = exec_cfg
        exec = ctx.compile()
        exec.execute()
        t = exec.get_spatial_tensor_torch(z.tensor_id)

        assert t.shape == ()
        assert t.item() == 3
        assert t.dtype == torch.int64


def test_int_float_const_broadcasting(exec_cfg: ExecutionConfig) -> None:
    ctx = TempoContext()
    with ctx:
        x = RecurrentTensor.ones()
        y = RecurrentTensor.const(2)
        z = x + y

        assert not z.requires_grad
        assert z._ctx is None
        assert z.grad is None

        ctx.execution_config = exec_cfg
        exec = ctx.compile()
        exec.execute()
        t = exec.get_spatial_tensor_torch(z.tensor_id)

        assert t.shape == ()
        assert t.item() == 3.0
        assert t.dtype == torch.float32


def test_nested_list_const(exec_cfg: ExecutionConfig) -> None:
    ctx = TempoContext()
    with ctx:
        x = RecurrentTensor.ones(dtype=dtype.dtypes.int64)
        y = RecurrentTensor.const([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        z = x + y

        assert not z.requires_grad
        assert z._ctx is None
        assert z.grad is None

        ctx.execution_config = exec_cfg
        exec = ctx.compile()
        exec.execute()
        t = exec.get_spatial_tensor_torch(z.tensor_id)

        assert t.shape == (3, 3)
        assert torch.allclose(t, torch.tensor([[2, 3, 4], [5, 6, 7], [8, 9, 10]]))
        assert t.dtype == torch.int64


def test_nested_list_const_one_float_makes_tensor_float(
    exec_cfg: ExecutionConfig,
) -> None:
    ctx = TempoContext()
    with ctx:
        x = RecurrentTensor.ones(dtype=dtype.dtypes.int64)
        y = RecurrentTensor.const([[1, 2, 3], [4, 5, 6.0], [7, 8, 9]])
        z = x + y

        assert not z.requires_grad
        assert z._ctx is None
        assert z.grad is None

        ctx.execution_config = exec_cfg
        exec = ctx.compile()
        exec.execute()
        t = exec.get_spatial_tensor_torch(z.tensor_id)

        assert t.shape == (3, 3)
        assert torch.allclose(t, torch.tensor([[2.0, 3, 4], [5, 6, 7], [8, 9, 10]]))
        assert t.dtype == torch.float32


def test_nparray_const(exec_cfg: ExecutionConfig) -> None:
    ctx = TempoContext()
    with ctx:
        x = RecurrentTensor.ones(dtype=dtype.dtypes.int64)
        y = RecurrentTensor.const(np.array([[1, 2, 3], [4, 5, 6.0], [7, 8, 9]]))
        z = x + y

        assert not z.requires_grad
        assert z._ctx is None
        assert z.grad is None

        ctx.execution_config = exec_cfg
        exec = ctx.compile()
        exec.execute()
        t = exec.get_spatial_tensor_torch(z.tensor_id)

        assert t.shape == (3, 3)
        assert torch.allclose(
            t, torch.tensor([[2.0, 3, 4], [5, 6, 7], [8, 9, 10]], dtype=torch.float64)
        )
        assert t.dtype == torch.float64
