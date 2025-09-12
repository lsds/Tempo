from tempo.api.rl.env.env_registry.env_registry import EnvBuildCtx, EnvRegistry
from tempo.api.rl.env.wrappers import (
    DoneToTermTruncAPIConverterWrapper,
    NoAutoResetWrapper,
    PermuteObservationChannelAxis,
    ToBackendTensorTWrapper,
)
from tempo.core.dl_backends import DLBackendName
from tempo.utils import logger

log = logger.get_logger(__name__)

try:
    import gymnasium as gym

    # import torch
    # import torchcule
    from torchcule.atari import Env as AtariEnv

    # import jax.numpy as jnp
    # from brax import envs
    # from brax.envs.wrappers import gym as gym_wrapper

    def _cule_env_builder(ctx: EnvBuildCtx) -> gym.Env:
        brax_kwargs = {**ctx.kwargs}
        # brax_kwargs.update({"disable_env_checker": True, "autoreset": False})

        backend_dev = ctx.exec_cfg.dev

        for k in ["autoreset", "auto_reset", "disable_env_checker"]:
            if k in brax_kwargs:
                log.warning("%s is not supported for brax environments", k)
                brax_kwargs.pop(k)

        env = AtariEnv(
            ctx.name,
            ctx.num_envs,
            color_mode="rgb",
            repeat_prob=0.0,
            device=backend_dev,
            rescale=False,
            episodic_life=False,
            frameskip=4,
            max_episode_length=ctx.max_episode_steps,
        )

        env = DoneToTermTruncAPIConverterWrapper(env, ctx.exec_cfg)

        if ctx.exec_cfg.get_canonical_backend_name() != DLBackendName.TORCH:
            to_backend = lambda x: ctx.backend.to_device(
                ctx.backend.from_dlpack(x), ctx.exec_cfg.dev
            )
            from tempo.runtime.backends.pytorch.pytorch_backend import PyTorchBackend

            from_backend = lambda x: PyTorchBackend.from_dlpack(x)
            env = ToBackendTensorTWrapper(env, ctx.exec_cfg, to_backend, from_backend)

        env = NoAutoResetWrapper(env, ctx.exec_cfg)

        # For some reason the axis in cule are wrong
        env = PermuteObservationChannelAxis(env, ctx.exec_cfg)
        # return env

        # env = (
        #    gym_wrapper.VectorGymWrapper(env, backend=backend_dev)
        #    if ctx.num_envs
        #    else gym_wrapper.GymWrapper(env, backend=backend_dev)
        # )

        # Call reset/step to trigger compilation
        # env.reset()
        # action = ctx.backend.tensor(env.action_space.sample(), device=ctx.exec_cfg.dev)
        # env.step(action)

        return env

    EnvRegistry.register_env_set("cule", _cule_env_builder)
except ImportError as e:
    log.warning("CULE module not found: %s. Likely not installed.", e)
except Exception as e:
    log.warning("Failed to register CULE environments: %s. Likely not installed.", e)
