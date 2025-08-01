from functools import partial
import itertools
from dataclasses import replace
from math import prod
from typing import Any, Tuple

import numpy as np
import pytest
import torch

import tempo.api.recurrent_tensor as tpo
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.api.tempo_context_manager import TempoContext
from tempo.core.configs import ExecutionConfig
from tempo.core.dtype import DataType, dtypes


def idfn(val: Any) -> str:
    shape = val
    if isinstance(shape, int):
        return str(shape)
    if isinstance(val, str):
        return val
    return "x".join(str(i) for i in shape)


@pytest.mark.parametrize(
    "shape,dtypes,backend",
    list(
        itertools.product(
            [
                (4, 4),
                (5, 4, 4, 4),
            ],
            [(torch.float32, dtypes.float32), (torch.int64, dtypes.int64)],
            ["torch", "jax"],
        )
    ),
    ids=idfn,
)
def test_mat_mul(
    shape: Tuple[int, ...],
    dtypes: Tuple[torch.dtype, DataType],
    exec_cfg: ExecutionConfig,
    backend: str,
):
    torch_dtype, tpo_dtype = dtypes
    x_t = torch.arange(prod(shape), dtype=torch_dtype).reshape(shape)
    y_t = torch.arange(prod(shape), dtype=torch_dtype).reshape(shape)
    z_t = x_t @ y_t

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg)
    with ctx:
        x = RecurrentTensor.arange(prod(shape), dtype=tpo_dtype).reshape(shape)
        y = RecurrentTensor.arange(prod(shape), dtype=tpo_dtype).reshape(shape)

        z = x @ y

        exec = ctx.compile({})
        exec.execute()
        z_computed = exec.get_spatial_tensor_torch(z.tensor_id, ())

        # shapes match
        assert z_computed.shape == z_t.shape
        assert torch.allclose(z_computed, z_t)


@pytest.mark.parametrize(
    "shape,backend",
    itertools.product([(4, 4),  (5, 4, 4, 4)], ["torch", "jax"]),
)
def test_mat_mul_bwd(shape: Tuple[int, ...], exec_cfg: ExecutionConfig, backend: str):
    with torch.enable_grad():
        x_t = torch.arange(prod(shape), dtype=torch.float32, requires_grad=True)
        x_t_r = x_t.reshape(shape)
        y_t = torch.arange(prod(shape), dtype=torch.float32).reshape(shape)
        z_t = (x_t_r @ y_t).tanh().sum()
        z_t.backward()

    torch.no_grad().__enter__()
    torch.set_grad_enabled(False).__enter__()

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg)
    with ctx:
        x = RecurrentTensor.arange(
            prod(shape), dtype=dtypes.float32, requires_grad=True
        )
        x_r = x.reshape(shape)
        y = RecurrentTensor.arange(prod(shape), dtype=dtypes.float32).reshape(shape)

        z = (x_r @ y).tanh().sum()

        z.backward()

        exec = ctx.compile({})
        exec.execute()

        assert x.grad is not None
        x_grad_computed = exec.get_spatial_tensor_torch(x.grad.tensor_id, ())

        assert x_t.grad is not None
        assert x_grad_computed.shape == x_t.grad.shape
        assert torch.allclose(x_grad_computed, x_t.grad)


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


@pytest.mark.parametrize(
    "shape,op_names,dtypes,backend",
    list(
        itertools.product(
            [
                (1,),
                (5,),
                (5, 5),
            ],
            [
                ("relu", "relu"),
                ("tanh", "tanh"),
                ("sigmoid", "sigmoid"),
                ("Softmax", "softmax"),
                ("SiLU", "swish"),
                ("Mish", "mish"),
                ("ELU", "elu"),
                ("LeakyReLU", "leakyrelu"),
                ("Softplus", "softplus"),
            ],
            [(torch.float32, dtypes.float32), (torch.int64, dtypes.int64)],
            ["torch", "jax"],
        )
    ),
    ids=idfn_2,
)
def test_activations(
    shape: Tuple[int, ...],
    op_names: Tuple[str, str],
    dtypes: Tuple[torch.dtype, DataType],
    backend: str,
    exec_cfg: ExecutionConfig,
):
    torch_dtype, tpo_dtype = dtypes
    torch_name, tpo_name = op_names

    if (torch_name == "Softmax" or torch_name == "SiLU" or torch_name == "Mish" or torch_name == "ELU" or torch_name == "LeakyReLU" or torch_name == "Softplus") and torch_dtype == torch.int64:
        pytest.skip(f"Skipping {torch_name} for int64 cause torch has no kernel.")

    x_t = torch.arange(prod(shape), dtype=torch_dtype).reshape(shape)
    try:
        z_t = getattr(torch, torch_name)(x_t)
    except AttributeError:
        fn_ = getattr(torch.nn, torch_name)
        if torch_name == "Softmax":
            fn_ = fn_(dim=-1)
        elif torch_name == "LeakyReLU":
            fn_ = fn_(negative_slope=0.01)
        else:
            fn_ = fn_()
        z_t = fn_(x_t)

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg)
    with ctx:
        x = RecurrentTensor.arange(prod(shape), dtype=tpo_dtype).reshape(shape)

        fn_ = getattr(tpo, tpo_name)#(x)
        if tpo_name == "softmax":
            fn_ = partial(fn_, dim=-1, stable=True)
        elif tpo_name == "leakyrelu":
            fn_ = partial(fn_, neg_slope=0.01)
        z = fn_(x)

        exec = ctx.compile({})
        exec.execute()
        z_computed = exec.get_spatial_tensor_torch(z.tensor_id, ())

        assert z_t.dtype == z_computed.dtype, f"Dtypes did not match: torch={z_t.dtype}, tpo={z_computed.dtype}."
        assert torch.allclose(z_t, z_computed)


@pytest.mark.parametrize(
    "shape,op_names,dtypes,backend",
    list(
        itertools.product(
            [
                (),
                (1,),
                (5,),
                (15,),
                (10, 10),
            ],
            [
                ("sin", "sin"),
                ("cos", "cos"),
                ("tan", "tan"),
                ("exp", "exp"),
                ("exp2", "exp2"),
                # ("exp10", "exp10"),
                ("log", "ln"),
                ("log2", "log2"),
                ("log10", "log10"),
                ("abs", "abs"),
                ("neg", "neg"),
                ("erf", "erf"),
            ],
            [(torch.float32, dtypes.float32), (torch.int64, dtypes.int64)],
            ["torch", "jax"],
        )
    ),
    ids=idfn_2,
)
def test_elementwise_unary_ops(
    shape: Tuple[int, ...],
    op_names: Tuple[str, str],
    dtypes: Tuple[torch.dtype, DataType],
    backend: str,
    exec_cfg: ExecutionConfig,
) -> None:
    torch_dtype, tpo_dtype = dtypes
    torch_name, tpo_name = op_names

    x_t = torch.arange(prod(shape), dtype=torch_dtype).reshape(shape) - 5
    z_t = getattr(torch, torch_name)(x_t)

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg)

    with ctx:
        x = RecurrentTensor.arange(prod(shape), dtype=tpo_dtype).reshape(shape) - 5
        z = getattr(tpo, tpo_name)(x)

        exec = ctx.compile({})
        exec.execute()
        z_computed = exec.get_spatial_tensor_torch(z.tensor_id, ())

        assert torch.allclose(z_t, z_computed, equal_nan=True)


@pytest.mark.parametrize(
    "dtypes,backend,exponent",
    list(
        itertools.product(
            [(torch.float32, dtypes.float32), (torch.int64, dtypes.int64)],
            ["torch", "jax"],
            [0.0, 0.5, 1.0, 2.0, 3.0, -1.0, -2.0, 0.25, -0.5],
        )
    ),
    ids=idfn_2,
)
def test_pow(
    dtypes: Tuple[torch.dtype, DataType],
    backend: str,
    exponent: float,
    exec_cfg: ExecutionConfig,
):
    torch_dtype, tpo_dtype = dtypes
    x_t = torch.arange(256, dtype=torch_dtype)
    z_t = x_t**exponent

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg)

    with ctx:
        x = RecurrentTensor.arange(stop=256, dtype=tpo_dtype)
        x.sink_udf(lambda x_: print(f"X: {x_}"))
        z = tpo.pow_(x, exponent)

        exec = ctx.compile({})
        exec.execute()
        z_computed = exec.get_spatial_tensor_torch(z.tensor_id, ())

        assert torch.allclose(z_t, z_computed), f"Expected: {z_t}, Got: {z_computed}"


@pytest.mark.parametrize(
    "shape,op_names,dtypes,backend",
    list(
        itertools.product(
            [(), (1,), (5,), (15, 15)],
            [
                ("add", "add"),
                ("sub", "sub"),
                ("mul", "mul"),
                ("div", "div"),
                ("pow", "pow_"),
                # ("min", "min"),
                # ("max", "max"),
            ],
            [(torch.float32, dtypes.float32), (torch.int64, dtypes.int64)],
            ["torch", "jax"],
        )
    ),
    ids=idfn_2,
)
def test_elementwise_binary_ops(
    shape: Tuple[int, ...],
    op_names: Tuple[str, str],
    dtypes: Tuple[torch.dtype, DataType],
    backend: str,
    exec_cfg: ExecutionConfig,
):
    if op_names[0] == "pow" and dtypes[0] == torch.int64 and len(shape) > 1:
        pytest.skip("Skipping int64 pow due to large int representation issues.")
    torch_dtype, tpo_dtype = dtypes
    torch_name, tpo_name = op_names
    x_t = torch.arange(prod(shape), dtype=torch_dtype).reshape(shape)
    y_t = torch.arange(prod(shape), dtype=torch_dtype).reshape(shape)
    if len(shape) >= 2:
        y_t = y_t.T
    z_t = getattr(torch, torch_name)(x_t, y_t)

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg)

    with ctx:
        x = RecurrentTensor.arange(stop=prod(shape), dtype=tpo_dtype).reshape(shape)
        y = RecurrentTensor.arange(stop=prod(shape), dtype=tpo_dtype).reshape(shape)
        if len(shape) >= 2:
            y = y.T
        z = getattr(tpo, tpo_name)(x, y)

        exec = ctx.compile({})
        exec.execute()
        z_computed = exec.get_spatial_tensor_torch(z.tensor_id, ())

        if torch_name == "div":
            assert torch.allclose(
                z_t.to(torch_dtype), z_computed.to(torch_dtype), equal_nan=True
            )
        else:
            assert torch.allclose(z_t, z_computed)


@pytest.mark.parametrize(
    "shape,op_names,dtypes,backend",
    list(
        itertools.product(
            [(5,),  (15, 15)],
            [
                ("min", "min"),
                ("max", "max"),
            ],
            [(torch.float32, dtypes.float32), (torch.int64, dtypes.int64)],
            ["torch", "jax"],
        )
    ),
    ids=idfn_2,
)
def test_min_max_binary(
    shape: Tuple[int, ...],
    op_names: Tuple[str, str],
    dtypes: Tuple[torch.dtype, DataType],
    backend: str,
    exec_cfg: ExecutionConfig,
):
    torch_dtype, tpo_dtype = dtypes
    torch_name, tpo_name = op_names
    x_t = torch.arange(prod(shape), dtype=torch_dtype).reshape(shape)
    y_t = torch.arange(prod(shape), dtype=torch_dtype).reshape(shape) * 2
    if len(shape) >= 2:
        y_t = y_t.T
    torch_vals = getattr(torch, torch_name)(x_t, y_t)

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg)

    with ctx:
        x = RecurrentTensor.arange(stop=prod(shape), dtype=tpo_dtype).reshape(shape)
        y = RecurrentTensor.arange(stop=prod(shape), dtype=tpo_dtype).reshape(shape) * 2
        if len(shape) >= 2:
            y = y.T
        vals, idxs = getattr(tpo, tpo_name)(x, y)

        exec = ctx.compile({})
        exec.execute()
        vals_computed = exec.get_spatial_tensor_torch(vals.tensor_id, ())
        idxs_computed = exec.get_spatial_tensor_torch(idxs.tensor_id, ())

        assert torch.allclose(torch_vals, vals_computed)
        # assert torch.allclose(torch_idxs, idxs_computed)


@pytest.mark.parametrize(
    "shape,op_names,dtypes,backend",
    list(
        itertools.product(
            [(5,),  (15, 15)],
            [
                ("min", "min"),
                ("max", "max"),
            ],
            [(torch.float32, dtypes.float32), (torch.int64, dtypes.int64)],
            ["torch", "jax"],
        )
    ),
    ids=idfn_2,
)
def test_min_max_unary(
    shape: Tuple[int, ...],
    op_names: Tuple[str, str],
    dtypes: Tuple[torch.dtype, DataType],
    backend: str,
    exec_cfg: ExecutionConfig,
):
    torch_dtype, tpo_dtype = dtypes
    torch_name, tpo_name = op_names
    x_t = torch.arange(prod(shape), dtype=torch_dtype).reshape(shape)
    torch_vals, torch_idxs = getattr(torch, torch_name)(x_t, dim=0)

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg)

    with ctx:
        x = RecurrentTensor.arange(stop=prod(shape), dtype=tpo_dtype).reshape(shape)
        vals, idxs = getattr(tpo, tpo_name)(x, dim=0)

        exec = ctx.compile({})
        exec.execute()
        vals_computed = exec.get_spatial_tensor_torch(vals.tensor_id, ())

        #NOTE: we cast to int64 because Tempo will cast int64 to int32 by default
        idxs_computed = exec.get_spatial_tensor_torch(idxs.tensor_id, ()).to(torch.int64)

        assert torch.allclose(torch_vals, vals_computed)
        assert torch.allclose(torch_idxs, idxs_computed)


# TODO parametrize
@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_gather(exec_cfg: ExecutionConfig, backend: str):
    source_t = torch.arange(2, dtype=torch.float32)
    dim = -1
    num_indexes = 1
    index_t = torch.arange(num_indexes, dtype=torch.int64)
    gathered_t = torch.gather(source_t, dim, index_t)

    exec_cfg = replace(exec_cfg, backend=backend)
    ctx = TempoContext(exec_cfg)

    with ctx:

        source = tpo.arange(2, dtype=tpo.dtypes.float32)
        dim = -1
        index = tpo.arange(num_indexes, dtype=tpo.dtypes.int64)

        gathered = tpo.gather(source, dim, index)

        exec = ctx.compile({})
        exec.execute()
        gathered_computed = exec.get_spatial_tensor_torch(gathered.tensor_id, ())

        assert torch.allclose(gathered_t, gathered_computed)
        # assert torch.allclose(torch_idxs, idxs_computed)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_disc_cum_sum(exec_cfg: ExecutionConfig, backend: str):

    def reference_impl(
        rewards: torch.Tensor, gamma: float, dim: int = 0
    ) -> torch.Tensor:
        # Slow reference implementation using a for-loop
        discounted_cumsum = torch.zeros_like(rewards)
        cumulative_sum = 0.0
        for t in reversed(range(rewards.size(dim))):
            # cumulative_sum = rewards[t] + gamma * cumulative_sum
            # TODO Need to index rewards at dimension dim
            cumulative_sum = (
                rewards.index_select(dim, torch.tensor(t)) + gamma * cumulative_sum
            )
            # discounted_cumsum[t] = cumulative_sum
            discounted_cumsum.index_add_(dim, torch.tensor(t), cumulative_sum)
        return discounted_cumsum

    for _ in range(5):
        random_rewards_array = np.random.rand(10)

        for disc_factor in [0.99, 0.8]:
            torch_solution = reference_impl(
                torch.tensor(random_rewards_array, dtype=torch.float32), disc_factor
            )

            exec_cfg = replace(exec_cfg, backend=backend)
            ctx = TempoContext(exec_cfg, num_dims=0)

            with ctx:
                rewards = RecurrentTensor.lift(random_rewards_array)

                gamma = RecurrentTensor.const(disc_factor)
                dim = 0
                discounted_cumsum = tpo.discounted_cum_sum(
                    rewards, gamma, dim, keep_symbolic_dim=False
                )

                exec = ctx.compile({})
                exec.execute()

                tpo_result = exec.get_spatial_tensor_torch(
                    discounted_cumsum.tensor_id, ()
                )
            assert torch.allclose(
                torch_solution.to(torch.float32), tpo_result.to(torch.float32)
            ), f"Expected: {torch_solution}, Got: {tpo_result}"
