from functools import partial

from tempo.api.rl.env.env_registry.env_registry import EnvBuildCtx
from tempo.api.rl.env.wrappers import ToBackendTensorTWrapper
from tempo.utils import logger

log = logger.get_logger(__name__)

try:
    import gymnasium as gym
    import jax.numpy as jnp
    import jumanji
    from jumanji.wrappers import JumanjiToGymWrapper, VmapWrapper

    def _jumanji_env_builder(ctx: EnvBuildCtx) -> gym.Env:
        jumanji_kwargs = {**ctx.kwargs}
        backend_dev = "gpu" if ctx.exec_cfg.dev.startswith("cuda") else ctx.exec_cfg.dev

        for k in ["autoreset", "auto_reset", "disable_env_checker"]:
            if k in jumanji_kwargs:
                log.warning("%s is not supported for jumanji environments", k)
                jumanji_kwargs.pop(k)

        env = jumanji.make(ctx.name, **jumanji_kwargs)
        env = VmapWrapper(env) if ctx.num_envs else env
        env = JumanjiToGymWrapper(env, backend=backend_dev)

        # Monkey patching to return info
        env.reset = partial(env.reset, return_info=True)

        # Call reset/step to trigger compilation
        env.reset()
        action = (
            jnp.array(env.action_space.sample())
            .expand_dims(0)
            .broadcast_to((ctx.num_envs, *env.action_space.shape))
        )
        env.step(action)

        to_backend = lambda x: ctx.backend.to_device(ctx.backend.from_dlpack(x), ctx.exec_cfg.dev)
        from_backend = lambda x: jnp.from_dlpack(x)
        env = ToBackendTensorTWrapper(env, ctx.exec_cfg, to_backend, from_backend)

        return env

except ImportError as e:
    log.warning("Jumanji module not found: %s. Likely not installed.", e)
except Exception as e:
    log.warning("Failed to register Jumanji environments: %s. Likely not installed.", e)
