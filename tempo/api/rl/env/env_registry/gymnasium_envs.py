from tempo.api.rl.env.env_registry.env_registry import EnvBuildCtx, EnvRegistry
from tempo.api.rl.env.wrappers import (
    SinglePrecisionRewardWrapper,
    ToBackendTensorTWrapper,
)

try:
    import gymnasium as gym
    import numpy as np

    def _gym_env_builder(ctx: EnvBuildCtx) -> gym.Env:
        gym_kwargs = {**ctx.kwargs}
        gym_kwargs.update({"disable_env_checker": True, "autoreset": False})

        if ctx.num_envs is not None:
            env = gym.vector.make(
                id=ctx.name,
                num_envs=ctx.num_envs,
                asynchronous=False,
                max_episode_steps=ctx.max_episode_steps,
                **gym_kwargs,
            )
        else:
            env = gym.make(id=ctx.name, max_episode_steps=ctx.max_episode_steps, **gym_kwargs)
        bend = ctx.backend

        # if not type(ctx.backend) is NumpyBackend:
        device = ctx.backend.to_backend_device_obj(ctx.exec_cfg.dev)
        to_backend = lambda x: bend.to_device(bend.from_dlpack(np.array(x, copy=False)), device)
        # to_backend = _to_bend
        from_backend = lambda x: np.from_dlpack(bend.to_cpu(x))

        env = ToBackendTensorTWrapper(env, ctx.exec_cfg, to_backend, from_backend)
        env = SinglePrecisionRewardWrapper(env, ctx.exec_cfg)
        return env

    EnvRegistry.register_env_set("gym", _gym_env_builder)

except Exception as e:
    print(f"Failed to register Gym environments: {e}")
