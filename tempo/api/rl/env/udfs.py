from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import torch

from tempo.api.rl.env.env_desc import EnvDesc

# Define new types based on RecurrentTensor
from tempo.core import tensor_op as top
from tempo.core.shape import Shape
from tempo.core.thunk import (
    Thunk,
    ThunkExecutionCtx,
)
from tempo.core.thunk_emitter import ThunkEmissionCtx
from tempo.core.thunk_udf import UDFVectorizationCtx, UserDefinedThunkDesc


def vectorize_reset(
    env_desc: EnvDesc, seeded: bool, vec_ctx: UDFVectorizationCtx
) -> UserDefinedThunkDesc:
    assert type(vec_ctx.vec_size) is int, "Vectorization size must be known."
    assert all(type(d) is int for d in vec_ctx.prior_vectorizations), (
        "All prior vectorization sizes must be known."
    )

    b_size = (
        vec_ctx.vec_size
        * math.prod(vec_ctx.prior_vectorizations)
        * (env_desc.num_vector_envs if env_desc.num_vector_envs is not None else 1)
    )

    assert isinstance(b_size, int), "b_size should be an integer"

    obs_shape = (
        vec_ctx.prior_vectorizations + (vec_ctx.vec_size,) + env_desc.observation_shape._shape
    )

    def new_infer_output_shapes(input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        return (Shape.from_(obs_shape),)

    def reset_env_op_torch_translation(
        op: top.TensorOp, ctx: ThunkEmissionCtx
    ) -> Thunk[torch.Tensor]:
        if env_desc.state_id not in ctx.external_state_store:
            ctx.external_state_store[env_desc.state_id] = env_desc.builder(b_size)

        env = ctx.external_state_store[env_desc.state_id]

        def seeded_reset(
            inputs: tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
        ) -> tuple[torch.Tensor, ...]:
            # TODO
            # seed_ = inputs[0]
            ## Reshape seed to have a single batch dimension
            # seed_.reshape((b_size,)).tolist()

            # obs, _ = env.reset(seed=seed_ if seeded else None)
            obs, _ = env.reset()
            # Reshape outputs to restore all batch dimensions
            # obs = obs.reshape(obs_shape)
            return (obs,)

        def reset(
            inputs: tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
        ) -> tuple[torch.Tensor, ...]:
            obs, _ = env.reset()
            # Reshape outputs to restore all batch dimensions
            # obs = obs.reshape(obs_shape)
            return (obs,)

        return seeded_reset if seeded else reset

    return dataclasses.replace(
        vec_ctx.current_udf_desc,
        infer_output_shapes=new_infer_output_shapes,
        thunk_translation=reset_env_op_torch_translation,
    )


def get_reset_udf_desc(env_desc: EnvDesc, seeded: bool) -> UserDefinedThunkDesc:
    def reset_env_op_torch_translation(
        op: top.TensorOp, ctx: ThunkEmissionCtx
    ) -> Thunk[torch.Tensor]:
        if env_desc.state_id not in ctx.external_state_store:
            ctx.external_state_store[env_desc.state_id] = env_desc.builder(env_desc.num_vector_envs)

        env = ctx.external_state_store[env_desc.state_id]

        def reset(
            inputs: tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
        ) -> tuple[torch.Tensor, ...]:
            obs, _ = env.reset(
                # seed=int(inputs[0].reshape[-1][0].item()) if seeded else None
            )
            return (obs,)

        return reset

    return UserDefinedThunkDesc(
        thunk_translation=reset_env_op_torch_translation,
        infer_output_shapes=lambda input_shapes: (env_desc.observation_shape,),
        infer_output_dtypes=lambda input_dtypes: (env_desc.observation_dtype,),
        num_inputs=1 if seeded else 0,
        num_outputs=1,
        state_store_access_keys=(env_desc.state_id,),
        vectorize=lambda ctx: vectorize_reset(env_desc, seeded, ctx),
        thunk_name="EnvReset",
        clean_up_state={env_desc.state_id: lambda state: state[env_desc.state_id].close()},
        needs_symbol_setter=False,
    )


def vectorize_step(env_desc: EnvDesc, vec_ctx: UDFVectorizationCtx) -> UserDefinedThunkDesc:
    assert type(vec_ctx.vec_size) is int, "Vectorization size must be known."
    assert all(type(d) is int for d in vec_ctx.prior_vectorizations), (
        "All prior vectorization sizes must be known."
    )

    b_size = (
        vec_ctx.vec_size
        * math.prod(vec_ctx.prior_vectorizations)
        * (env_desc.num_vector_envs if env_desc.num_vector_envs is not None else 1)
    )

    assert isinstance(b_size, int), "b_size should be an integer"

    act_shape = (b_size,) + env_desc.action_shape._shape

    obs_shape = (
        vec_ctx.prior_vectorizations + (vec_ctx.vec_size,) + env_desc.observation_shape._shape
    )

    rew_term_trunc_shape = (
        vec_ctx.prior_vectorizations
        + (vec_ctx.vec_size,)
        + ((env_desc.num_vector_envs,) if env_desc.num_vector_envs is not None else ())
    )
    assert len(vec_ctx.prior_vectorizations) == 0, (
        "To support multiple vectorizations, we need to reshape the outputs again"
    )

    def new_infer_output_shapes(input_shapes: Sequence[Shape]) -> Sequence[Shape]:
        return (
            Shape.from_(obs_shape),
            Shape.from_(rew_term_trunc_shape),
            Shape.from_(rew_term_trunc_shape),
            Shape.from_(rew_term_trunc_shape),
        )

    def step_env_op_torch_translation(
        op: top.TensorOp, ctx: ThunkEmissionCtx
    ) -> Thunk[torch.Tensor]:
        if env_desc.state_id not in ctx.external_state_store:
            ctx.external_state_store[env_desc.state_id] = env_desc.builder(b_size)

        env = ctx.external_state_store[env_desc.state_id]

        if len(vec_ctx.prior_vectorizations) == 0:

            def step(
                inputs: tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
            ) -> tuple[torch.Tensor, ...]:
                action = inputs[0]
                obs, rew, term, trunc, _ = env.step(action)
                return (obs, rew, term, trunc)  # type: ignore

        else:
            raise NotImplementedError("TODO: replace below with backend reshape")

            def step(
                inputs: tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
            ) -> tuple[torch.Tensor, ...]:
                action = inputs[0]
                action.reshape(act_shape)
                obs, rew, term, trunc, _ = env.step(action)
                obs = obs.reshape(obs_shape)
                rew = rew.reshape(rew_term_trunc_shape)
                term = term.reshape(rew_term_trunc_shape)
                trunc = trunc.reshape(rew_term_trunc_shape)

                return (obs, rew, term, trunc)  # type: ignore

        return step

    return dataclasses.replace(
        vec_ctx.current_udf_desc,
        infer_output_shapes=new_infer_output_shapes,
        thunk_translation=step_env_op_torch_translation,
    )


def get_step_udf_desc(env_desc: EnvDesc) -> UserDefinedThunkDesc:
    def step_env_op_torch_translation(
        op: top.TensorOp, ctx: ThunkEmissionCtx
    ) -> Thunk[torch.Tensor]:
        if env_desc.state_id not in ctx.external_state_store:
            ctx.external_state_store[env_desc.state_id] = env_desc.builder(env_desc.num_vector_envs)

        env = ctx.external_state_store[env_desc.state_id]

        def step(
            inputs: tuple[torch.Tensor, ...], exec_ctx: ThunkExecutionCtx
        ) -> tuple[torch.Tensor, ...]:
            action = inputs[0]
            obs, rew, term, trunc, _ = env.step(action)
            return (obs, rew, term, trunc)  # type: ignore

        return step

    return UserDefinedThunkDesc(
        thunk_translation=step_env_op_torch_translation,
        infer_output_shapes=lambda input_shapes: (
            env_desc.observation_shape,
            env_desc.rew_shape,
            env_desc.term_shape,
            env_desc.trunc_shape,
        ),
        infer_output_dtypes=lambda input_dtypes: (
            env_desc.observation_dtype,
            env_desc.rew_dtype,
            env_desc.term_dtype,
            env_desc.trunc_dtype,
        ),
        state_store_access_keys=(env_desc.state_id,),
        num_inputs=1,
        num_outputs=4,
        vectorize=lambda ctx: vectorize_step(env_desc, ctx),
        thunk_name="EnvStep",
        clean_up_state={env_desc.state_id: lambda state: state[env_desc.state_id].close()},
        needs_symbol_setter=False,
    )
