import warnings

from tempo.api.rl.env.env_registry.env_registry import EnvBuildCtx, EnvRegistry
from tempo.api.rl.env.wrappers import (
    DoneToTermTruncAPIConverterWrapper,
    ToBackendTensorTWrapper,
)
from tempo.runtime.backends.backend import DLBackendName
from tempo.utils import logger

# Suppress specific deprecation warnings from JAX and related libraries
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"brax.*",
)

log = logger.get_logger(__name__)

try:
    import gymnasium as gym

    # import jax.numpy as jnp
    from brax import envs
    from brax.envs.wrappers import gym as gym_wrapper

    def _brax_env_builder(ctx: EnvBuildCtx) -> gym.Env:
        brax_kwargs = {**ctx.kwargs}
        brax_kwargs.update({"disable_env_checker": True, "autoreset": False})

        backend_dev = "gpu" if ctx.exec_cfg.dev.startswith("cuda") else ctx.exec_cfg.dev
        for k in ["autoreset", "auto_reset", "disable_env_checker"]:
            if k in brax_kwargs:
                log.warning("%s is not supported for brax environments", k)
                brax_kwargs.pop(k)

        env = envs.create(
            ctx.name,
            auto_reset=False,
            batch_size=ctx.num_envs,
            episode_length=ctx.max_episode_steps,
            **brax_kwargs,
        )

        env = (
            gym_wrapper.VectorGymWrapper(env, backend=backend_dev)
            if ctx.num_envs
            else gym_wrapper.GymWrapper(env, backend=backend_dev)
        )

        env = DoneToTermTruncAPIConverterWrapper(env, ctx.exec_cfg)

        if DLBackendName.str_to_enum(ctx.exec_cfg.backend) != DLBackendName.JAX:
            to_backend = lambda x: ctx.backend.to_device(
                ctx.backend.from_dlpack(x), ctx.exec_cfg.dev
            )
            from tempo.runtime.backends.jax.jax_backend import JaxBackend

            from_backend = lambda x: JaxBackend.from_dlpack(x)
            env = ToBackendTensorTWrapper(env, ctx.exec_cfg, to_backend, from_backend)

        # Call reset/step to trigger compilation
        env.reset()
        action = ctx.backend.lift_tensor(
            env.action_space.sample(), device=ctx.backend.to_backend_device_obj(ctx.exec_cfg.dev)
        )
        env.step(action)

        return env

    EnvRegistry.register_env_set("brax", _brax_env_builder)
except ImportError as e:
    log.warning("Brax module not found: %s. Likely not installed.", e)
except Exception as e:
    log.warning("Failed to register Brax environments: %s. Likely not installed.", e)
