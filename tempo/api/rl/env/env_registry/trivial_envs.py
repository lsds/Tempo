from tempo.api.rl.env.env_registry.env_registry import EnvBuildCtx, EnvRegistry
from tempo.api.rl.env.wrappers import ToBackendTensorTWrapper
from tempo.runtime.backends.backend import DLBackendName
from tempo.utils import logger

log = logger.get_logger(__name__)

try:
    import gymnasium as gym

    # import jax
    # import numpy as np
    from jax._src.dlpack import from_dlpack

    from tempo.api.rl.env.trivial_env import TrivialEnv

    def _trivial_env_builder(ctx: EnvBuildCtx) -> gym.Env:
        trivial_kwargs = {**ctx.kwargs}
        # assert ctx.num_envs is not None, "num_envs must be provided"
        # assert ctx.max_episode_steps is not None, "max_episode_steps must be provided"
        # if ctx.num_envs is not None:
        #    assert ctx.num_envs == 256
        # if ctx.max_episode_steps is not None:
        #    assert ctx.max_episode_steps == 512
        num_envs = ctx.num_envs or 1  # or trivial_kwargs["num_envs"]
        ep_len = ctx.max_episode_steps or 1  # or trivial_kwargs["max_ep_len"]
        obs_shape = trivial_kwargs["observation_shape"]

        continuous = trivial_kwargs.pop("continuous", True)

        # print("================= CREATING TRIVIAL ENV =====================")
        # print(f"       num_envs: {num_envs}")
        # print(f"       ep_len: {ep_len}")
        # print(f"       obs_shape: {obs_shape}")
        # print(f"       continuous: {continuous}")

        from tempo.runtime.backends.backend import DLBackend

        dev = DLBackend.get_backend("jax").to_backend_device_obj(ctx.exec_cfg.dev)

        env = TrivialEnv(
            num_envs=num_envs,
            max_ep_len=ep_len,
            continuous=continuous,
            observation_shape=obs_shape,
            dev=dev,
        )

        if DLBackendName.str_to_enum(ctx.exec_cfg.backend) != DLBackendName.JAX:
            # to_backend = lambda x: ctx.backend.to_device(
            #    ctx.backend.from_dlpack(x), dev
            # )
            to_backend = ctx.backend.from_dlpack
            from_backend = from_dlpack
            env = ToBackendTensorTWrapper(env, ctx.exec_cfg, to_backend, from_backend)

        return env

    EnvRegistry.register_env_set("trivial", _trivial_env_builder)
except ImportError as e:
    log.warning("JAX module not found: %s. Likely not installed.", e)
except Exception as e:
    log.warning("Failed to register trivial environments: %s. Likely not installed.", e)
