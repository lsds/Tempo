import itertools
from dataclasses import replace
from math import prod
from typing import Any, Callable, List, Tuple, Union

import pytest
import torch

from tempo.core import index_expr as ie
from tempo.core.configs import ExecutionConfig

from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.api.tempo_context_manager import TempoContext
from tempo.core.dtype import dtypes

def get_torch_t(ins: List[torch.Tensor], pattern: Callable[[int], Any], t: int) -> torch.Tensor:
    catted = torch.stack(ins, dim=0)
    return catted[pattern(t)]

all_patterns = [
    "identity",
    "fixed_range",
    "all",
    #"all_past", #NOTE: This is failing due to some off-by-one error. Let's fix it later.
    "all_future",
    "past_sliding_window",
    "future_sliding_window",

    #"past_future_sliding_window",
    #"mod_8",
    #"floor_div_2"
]

def pattern_table(
        pattern: str, T_ub: int, w_past: int = 4, w_future: int = 2
) -> Tuple[Callable[[int], Union[int, torch.Tensor]], Callable[[ie.Symbol], ie.IndexAtom]]:
    if pattern == "identity":
        return lambda t: t, lambda t: t

    # One-step patterns
    #TODO: these will fail... But need to do one of these for RNN support with merge
    #elif pattern == "past_one":
    #    return lambda t: t-1, lambda t: t-1
    #elif pattern == "future_one":
    #    return lambda t: t+1, lambda t: t+1

    # Eliminating patterns
    elif pattern == "fixed_range":
        return lambda t: slice(0, 8), lambda t: ie.slice_(0, 8)
    elif pattern == "all":
        return lambda t: slice(0, T_ub), lambda t: ie.slice_(0, T_ub)

    # All patterns
    elif pattern == "all_past":
        return lambda t: slice(0, t+1), lambda t: ie.slice_(0, t+1)
    elif pattern == "all_future":
        return lambda t: slice(t, T_ub), lambda t: ie.slice_(t, T_ub)

    # Sliding window patterns
    elif pattern == "past_sliding_window":
        return lambda t: slice(max(0, t-w_past), t+1), lambda t: ie.slice_(ie.max(0, t-w_past), t+1)
    elif pattern == "future_sliding_window":
        return lambda t: slice(t, min(T_ub, t+w_future+1)), lambda t: ie.slice_(t, ie.min(T_ub, t+w_future+1))
    elif pattern == "past_future_sliding_window":
        return (lambda t: slice(max(0, t-w_past), min(T_ub, t+w_future+1)),
                lambda t: ie.slice_(ie.max(0, t-w_past), ie.min(T_ub, t+w_future+1)))

    # Weird patterns
    elif pattern == "mod_8":
        return lambda t: t % 8, lambda t: t % 8
    elif pattern == "floor_div_2":
        return lambda t: t // 2, lambda t: t // 2
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

@pytest.mark.parametrize(
    "shape_str,backend,pattern,enable_optims",
    itertools.product(["4x4"], ["torch"], all_patterns, [False, True]),
)
def test_ad(shape_str: str, backend: str, pattern: str, enable_optims: bool, exec_cfg: ExecutionConfig):
    shape = tuple(int(s) for s in shape_str.split("x"))
    T_ub = 100
    pat_torch, pat_tr = pattern_table(pattern, T_ub)

    x_ts: List[torch.Tensor] = []
    y_ts: List[torch.Tensor] = []
    with torch.enable_grad():
        with torch.set_grad_enabled(True):
            for t_int in range(T_ub):
                x_t = torch.randn(prod(shape), dtype=torch.float32, requires_grad=True)
                x_t_r = x_t.reshape(shape)
                x_t_r.retain_grad()
                assert x_t_r.requires_grad
                x_ts.append(x_t_r)
                y_t = torch.arange(prod(shape), dtype=torch.float32)
                y_t_r = y_t.reshape(shape)
                y_ts.append(y_t_r)

            z_ts = []
            for t_int in range(T_ub):
                x_t_indexed = get_torch_t(x_ts, pat_torch, t_int)
                assert x_t_indexed.requires_grad
                y_t_indexed = get_torch_t(y_ts, pat_torch, t_int)
                z_t = (x_t_indexed * y_t_indexed).sum()
                z_t.backward()
                assert z_t.requires_grad
                z_ts.append(z_t)


            torch_x_grads = []
            for t_int in range(T_ub):
                grad_x_t = x_ts[t_int].grad
                assert grad_x_t is not None, f"grad_x_t is None for t={t_int}"
                torch_x_grads.append(grad_x_t)

    torch.no_grad().__enter__()
    torch.set_grad_enabled(False).__enter__()

    tpo_x_grads = []
    exec_cfg = replace(exec_cfg, backend=backend)
    if enable_optims:
        exec_cfg = replace(exec_cfg,
                           enable_fold_pads_into_storage=True,
                           enable_pad_mask_removal=True,
                           enable_statifying_incrementalization=True,
                           enable_algebraic_optimizer=True,
                           enable_dead_code_elim=True,
                           enable_domain_reduction=True,
                           enable_constant_folding=True,
                           enable_duplicate_code_elim=True,
                           enable_hybrid_tensorstore=True,
                           enable_inplace_write=True,
                           enable_lazy_slice=True,
                           inc_statify_block_size=T_ub // 4,
                           )

    ctx = TempoContext(exec_cfg, num_dims=1)
    with ctx:
        ((t, T),) = ctx.variables_and_bounds
        x = RecurrentTensor.source_with_ts_udf(
            lambda ts: x_ts[ts[t]],
              shape=shape, dtype=dtypes.float32, domain=(t, ), requires_grad=True
            )
        y = RecurrentTensor.source_with_ts_udf(
            lambda ts: y_ts[ts[t]],
              shape=shape, dtype=dtypes.float32, domain=(t, ), requires_grad=False
            )

        z = (x[pat_tr(t)] * y[pat_tr(t)]).sum()
        with ctx.tag_region("backward"):
            z.backward()
        x.grad.sink_udf(lambda x: tpo_x_grads.append(x))

        exec = ctx.compile({T: T_ub})
        exec.execute()

    assert x.grad is not None
    for t_int in range(T_ub):
        x_t_tpo_grad = tpo_x_grads[t_int]
        x_t_torch_grad = torch_x_grads[t_int]

        assert x_t_torch_grad is not None
        assert x_t_tpo_grad.shape == x_ts[t_int].shape
        assert x_t_tpo_grad.shape == x_t_torch_grad.shape
        assert not x_t_torch_grad.sum().isnan(), f"Torch grad contains NaNs"
        assert not x_t_tpo_grad.sum().isnan(), f"Tempo grad contains NaNs"
        #print(f"t={t_int}, x_t_tpo_grad={x_t_tpo_grad}, x_t_torch_grad={x_t_torch_grad}")
        assert torch.allclose(x_t_tpo_grad, x_t_torch_grad, atol=1e-5), f"t={t_int}"
    print("Grads match")

if __name__ == "__main__":
    cfg = ExecutionConfig.default()
    cfg = replace(cfg, visualize_pipeline_stages=True)
    test_ad("4x4", "torch", "all_past", True, cfg)
    #test_ad("4x4", "torch", "all_future", True, cfg)
