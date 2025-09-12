# type: ignore
from tempo.api.rl.env.env_registry.env_registry import EnvBuildCtx, EnvRegistry
from tempo.api.rl.env.wrappers import ToBackendTensorTWrapper
from tempo.core.dl_backends import DLBackendName
from tempo.utils import logger

log = logger.get_logger(__name__)

try:
    import gymnasium as gym
    import torch
    import torch.utils
    import torch.utils.dlpack

    from tempo.api.rl.env.torch_catch_env import TorchCatchEnv

    def _torchcatch_env_builder(ctx: EnvBuildCtx) -> gym.Env:
        torchcatch_kwargs = {**ctx.kwargs}

        ep_len = torchcatch_kwargs.pop("max_ep_len", 500)
        torchcatch_kwargs.update({"rows": ep_len + 4})  # NOTE: 4 is the default paddle offset
        torchcatch_kwargs.update({"dev": ctx.exec_cfg.dev})

        # for k in ["autoreset", "auto_reset", "disable_env_checker"]:
        #    if k in torchcatch_kwargs:
        #        log.warning(f"{k} is not supported for brax environments")
        #        torchcatch_kwargs.pop(k)

        env = TorchCatchEnv(**torchcatch_kwargs)

        if ctx.exec_cfg.get_canonical_backend_name() != DLBackendName.TORCH:
            to_backend = lambda x: ctx.backend.to_device(
                ctx.backend.from_dlpack(x), ctx.exec_cfg.dev
            )
            from_backend = lambda x: torch.utils.dlpack.from_dlpack(x)
            env = ToBackendTensorTWrapper(env, ctx.exec_cfg, to_backend, from_backend)

        return env

    EnvRegistry.register_env_set("torchcatch", _torchcatch_env_builder)
except ImportError as e:
    log.warning("Torch module not found: %s. Likely not installed.", e)
except Exception as e:
    log.warning("Failed to register Catch environments: %s. Likely not installed.", e)
