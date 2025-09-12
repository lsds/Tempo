import itertools
from dataclasses import replace
from math import prod
from typing import Tuple

import numpy as np
import pytest
import torch
from torch import nn

from tempo.api import RecurrentTensor
from tempo.api.optim.optim import SGD, Adam
from tempo.api.rl.networks.model_builder import ModelBuilder
from tempo.api.tempo_context_manager import TempoContext
from tempo.core.configs import ExecutionConfig
from tempo.core.dtype import dtypes
from tempo.core.shape import StaticShape
from tempo.core.dl_backend import DLBackend

# Skip this entire file for now.
#pytest.skip("Skipping this entire file for now.", allow_module_level=True)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.constant_(layer.weight, 1.0)
    # torch.nn.init.constant_(layer.weight, 1.0)
    # torch.nn.init.uniform_(layer.weight)
    # torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class TorchModel(nn.Module):
    def __init__(self, num_in_feat: int, num_layers: int, num_params_per_layer: int):
        super().__init__()

        # Shared backbone with an encoder layer
        layers = [
            nn.Flatten(),
            layer_init(
                nn.Linear(
                    num_in_feat,
                    num_params_per_layer,
                )
            ),
            nn.Tanh(),
        ]

        # Add specified number of hidden layers
        for _ in range(num_layers):
            layers.append(
                layer_init(nn.Linear(num_params_per_layer, num_params_per_layer))
            )
            layers.append(nn.Tanh())

        # Define the backbone
        self.backbone = nn.Sequential(*layers)

        # Define critic using the shared backbone output
        self.critic = nn.Sequential(
            layer_init(nn.Linear(num_params_per_layer, 1), std=1.0)
        )

    def get_value(self, x):
        x = self.backbone(x)
        return self.critic(x)


@pytest.mark.parametrize(
    "in_shape,backend",
    itertools.product([(4, 4)], ["torch"]),
    #itertools.product([(4, 4), (4, 4, 4)], ["torch", "jax"]),
)
@pytest.mark.skip
def test_model_bwd(in_shape: tuple[int, ...], exec_cfg: ExecutionConfig, backend: str):

    model = TorchModel(
        num_in_feat=prod(in_shape),
        num_layers=2,
        num_params_per_layer=16,
    )

    iterations = 4
    num_timesteps = 32
    bs = 8
    # Use numpy to generate random inputs
    inputs = []
    expected_outputs = []
    for _ in range(num_timesteps):
        inputs.append(np.random.rand(bs, *in_shape).astype(np.float32))
        expected_outputs.append(np.random.rand(bs, 1).astype(np.float32))

    grads_at_each_iter = []
    with torch.enable_grad():
        for i in range(iterations):
            total_loss = torch.tensor(0.0)
            for t in range(num_timesteps):
                x_t = torch.tensor(inputs[t])
                y_t = torch.tensor(expected_outputs[t])

                z_t = model.get_value(x_t)
                loss_t = (z_t - y_t).pow(2).sum()
                total_loss += loss_t
            total_loss.backward()
            grads_at_each_iter.append(
                float(
                    torch.sum(
                        torch.stack([p.grad.sum() for p in model.parameters()])
                    ).item()
                )
            )
            # Set grads to 0
            model.zero_grad()

    with torch.no_grad():
        with torch.set_grad_enabled(False):

            grads_at_each_iter_tpo = []
            exec_cfg = replace(exec_cfg, backend=backend)
            bend = DLBackend.get_backend(exec_cfg.backend)
            ctx = TempoContext(exec_cfg, num_dims=2)
            with ctx:
                ((i, I), (t, T)) = ctx.variables_and_bounds

                with ctx.tag_region("model"):
                    model = (
                        ModelBuilder(
                            domain=(i,),
                            w_init_fun=RecurrentTensor.ones,
                        )
                        .with_ff_encoder(StaticShape.from_(in_shape), 16)
                        .with_ff_hidden((16, 16))
                        .with_vfun_decoder_only()
                        .build()
                    )
                    model.fixed()

                x = RecurrentTensor.source_with_ts_udf(
                    lambda ts: bend.from_dlpack(inputs[ts[t]]),
                    shape=(bs, *in_shape),
                    dtype=dtypes.float32,
                    domain=(t,),
                )
                y = RecurrentTensor.source_with_ts_udf(
                    lambda ts: bend.from_dlpack(expected_outputs[ts[t]]),
                    shape=(bs, 1),
                    dtype=dtypes.float32,
                    domain=(t,),
                )

                with ctx.tag_region("model_forward"):
                    _, y_hat = model(x)
                with ctx.tag_region("model_loss"):
                    loss_tpo = (y_hat - y.detach()).pow_(2).sum()
                    total_loss_tpo = loss_tpo[i, 0:T].sum()
                with ctx.tag_region("model_backward"):
                    total_loss_tpo.backward()

                with ctx.tag_region("model_sum_grads"):
                    summed_grads = RecurrentTensor.stack(
                        *[p.grad.sum() for p in model.parameters]
                    ).sum()
                summed_grads.sink_udf(
                    lambda x: grads_at_each_iter_tpo.append(float(x.item()))
                )

                exec = ctx.compile({I: iterations, T: num_timesteps})
                exec.execute()

    for i in range(iterations):
        assert np.isclose(
            grads_at_each_iter[i], grads_at_each_iter_tpo[i], atol=1e-5
        ), f"grads_at_each_iter[{i}]={grads_at_each_iter[i]} != grads_at_each_iter_tpo[{i}]={grads_at_each_iter_tpo[i]}"

@pytest.mark.parametrize(
    "in_shape,backend,optim_name,enable_optimizations",
    itertools.product(
        [
            (2, 2),
        ],
        ["jax", "torch"],  # "torch",
        ["adam"],  # , "sgd", TODO: fix sgd
        [False, True],
    ),
)
def test_model_bwd_optim(
    in_shape: tuple[int, ...],
    exec_cfg: ExecutionConfig,
    backend: str,
    optim_name: str,
    enable_optimizations: bool,
):

    model = TorchModel(
        num_in_feat=prod(in_shape),
        num_layers=1,
        num_params_per_layer=8,
    )
    if optim_name == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=1e-1)
    elif optim_name == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=1e-1)
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")

    iterations = 5
    num_timesteps = 10
    bs = 16
    # Use numpy to generate random inputs
    inputs = []
    expected_outputs = []
    for _ in range(num_timesteps):
        inputs.append(np.random.rand(bs, *in_shape).astype(np.float32))
        expected_outputs.append(np.random.rand(bs, 1).astype(np.float32))

    weight_sum_at_each_iter = []

    weight_sum = torch.mean(torch.stack([p.mean() for p in model.parameters()]))
    weight_sum_at_each_iter.append(float(weight_sum.item()))

    with torch.enable_grad():
        optim.zero_grad()
        for i in range(iterations):
            total_loss = torch.tensor(0.0)
            for t in range(num_timesteps):
                x_t = torch.tensor(inputs[t])
                y_t = torch.tensor(expected_outputs[t])

                z_t = model.get_value(x_t)
                loss_t = (z_t - y_t).pow(2).sum()
                total_loss += loss_t
            total_loss.backward()
            optim.step()
            weight_sum = torch.mean(torch.stack([p.mean() for p in model.parameters()]))
            weight_sum_at_each_iter.append(float(weight_sum.item()))

    with torch.no_grad():
        with torch.set_grad_enabled(False):
            weight_sum_at_each_iter_tpo = []

            # TODO replace vec?
            exec_cfg_ = replace(exec_cfg, backend=backend, enable_x64=False)
            if enable_optimizations:
                exec_cfg_ = replace(
                    exec_cfg_,
                    executor_debug_mode=True,
                    enable_dataflow_grouping=True,
                    enable_group_fusions=True,
                    enable_codegen_dataflows=True,
                    enable_conservative_grouping=True,
                    enable_dead_code_elim=True,
                    enable_duplicate_code_elim=True,
                    enable_domain_reduction=True,
                    enable_algebraic_optimizer=True, #TODO: fix dtype bug here
                    enable_broadcast_elim=True,
                    enable_donation_analysis=False,
                    enable_isolate_loop_conditions=True,
                    enable_vectorization=True,
                )

            bend = DLBackend.get_backend(exec_cfg_.backend)
            ctx = TempoContext(exec_cfg_, num_dims=2)
            with ctx:
                ((i, I), (t, T)) = ctx.variables_and_bounds

                with ctx.tag_region("model_def"):
                    model = (
                        ModelBuilder(
                            domain=(i,),
                            w_init_fun=RecurrentTensor.ones,
                        )
                        .with_ff_encoder(StaticShape.from_(in_shape), 8)
                        .with_ff_hidden((8,))
                        .with_vfun_decoder_only()
                        .build()
                    )
                with ctx.tag_region("optim_def"):
                    if optim_name == "adam":
                        optim = Adam(
                            model.parameters,
                            model.buffers,
                            lr=1e-1,
                        )
                    elif optim_name == "sgd":
                        optim = SGD(
                            model.parameters,
                            model.buffers,
                            lr=1e-1,
                        )
                    else:
                        raise ValueError(f"Unknown optimizer: {optim_name}")

                x = RecurrentTensor.source_with_ts_udf(
                    lambda ts: bend.from_dlpack(inputs[ts[t]]),
                    shape=(bs, *in_shape),
                    dtype=dtypes.float32,
                    domain=(t,),
                )
                y = RecurrentTensor.source_with_ts_udf(
                    lambda ts: bend.from_dlpack(expected_outputs[ts[t]]),
                    shape=(bs, 1),
                    dtype=dtypes.float32,
                    domain=(t,),
                )

                with ctx.tag_region("model_forward"):
                    _, y_hat = model(x)

                print(f"{x.domain=}")
                print(f"{y_hat.domain=}")
                print(f"{model.parameters[0].domain=}")
                with ctx.tag_region("model_loss"):
                    loss_tpo = (y_hat - y.detach()).pow_(2).sum()
                    print(f"{loss_tpo.domain=}")
                    total_loss_tpo = loss_tpo[i, 0:T].sum()

                with ctx.tag_region("model_backward"):
                    total_loss_tpo.backward()
                with ctx.tag_region("optim_step"):
                    optim.step()

                # summed_grads = RecurrentTensor.stack(
                #    *[p.grad.sum() for p in model.parameters]
                # ).sum()
                with ctx.tag_region("stack_sum_params"):
                    summed_params = RecurrentTensor.stack(
                        *[p.mean() for p in model.parameters]
                    ).mean()
                    summed_params.sink_udf(
                        lambda x: weight_sum_at_each_iter_tpo.append(float(x.item()))
                    )

                exec = ctx.compile({I: iterations, T: num_timesteps})
                exec.execute()

    #TODO find ways to reduce tolerance needed
    for i in range(iterations):
        assert np.isclose(
            weight_sum_at_each_iter[i], weight_sum_at_each_iter_tpo[i], atol=1e-1
        ), f"At iteration {i}: {weight_sum_at_each_iter[i]=} != {weight_sum_at_each_iter_tpo[i]=}"
    #assert False, f"{weight_sum_at_each_iter}, {weight_sum_at_each_iter_tpo}"

if __name__ == "__main__":
    cfg = ExecutionConfig.test_cfg()
    cfg = replace(cfg, visualize_pipeline_stages=True)
    test_model_bwd_optim((4, 4), cfg, "torch", "adam", True)
