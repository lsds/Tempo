import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

from repro.data_loading import ERROR_TXT_FILE, LOG_CSV_FILE, MONITOR_CSV_FILE
from repro.launch_lib import StatsLogger
from tempo.utils.resource_monitor import ResourceMonitorManager

RL_TRAIN_DIR = "rl_train"

""" Shared constants for the RL training experiments."""

SHARED_PPO_HYPERPARAMS = {
    "env_name": "trivial.trivial",
    # NOTE: Default obs shape for trivial env
    "obs_shape": (3, 4, 4),
    "seed": 0,
    "dev": "cuda",
    "iterations": 50,
    # PPO hyperparams
    "gamma": 0.99,
    "start_lr": 1e-4,
    "lambda_": 0.96,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    # NOTE: Fixed param base used in large obs experiments (overwritten in small_to_med_scale)
    "num_envs": 256,
    "ep_len": 1000,
    "params_per_layer": 64,
    "num_layers": 2,
    "use_caching_allocators": True,
}

SYS = [
    "tempo-torch",
    "tempo-jax",
    "cleanrl",
    "cleanrlcache",
    "rlgames",
    "samplefactory",
    "rllib",
]


def is_small_to_med_scale_experiment(obs_shape: Tuple[int, ...]) -> bool:
    return obs_shape[-1] == 4


def is_large_obs(obs_shape: Tuple[int, ...]) -> bool:
    return obs_shape[-1] >= 128


def sweep_param(
    param_name: str,
    values: List[Any],
    params_base: Dict[str, Any],
    base_path: str,
    exclude_sys: Sequence[str] = (),
    name_function: Callable[[Dict[str, Any]], str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    par = []
    seq = []

    if name_function is None:
        name_function = get_experiment_name_and_results_path

    systems = list(SYS)
    for sys in exclude_sys:
        if sys in systems:
            systems.remove(sys)

    for value in values:
        params = params_base.copy()
        params[param_name] = value

        for sys in systems:
            params_sys = params.copy()
            params_sys["sys_cfg"] = sys
            name, results_path = name_function(base_path, params_sys)
            params_sys["name"] = name
            params_sys["results_path"] = results_path
            is_td = "objective" in params_sys and type(params_sys["objective"]) is int
            is_mc = "objective" in params_sys and params_sys["objective"] is None

            # NOTE: Because RLlib is a cpu-based system, we allow it to run in isolation, with all
            # CPUs to itself. Furthermore, when Tempo uses swapping, we also run it in isolation,
            # so all CPU memory is available to it.
            # TD experiments should not use swapping, so we don't run them sequentially.
            if (
                sys == "rllib"
                or ("tempo" in sys and is_large_obs(params["obs_shape"]) and not is_td)
                or is_mc
            ):
                seq.append(params_sys)
            else:
                par.append(params_sys)
    return par, seq


def run_experiment(  # noqa: C901
    **kwargs,
):
    sys_cfg = kwargs["sys_cfg"]
    sys_name = sys_cfg.split("-")[0]
    results_path = Path(kwargs["results_path"])
    results_path.mkdir(parents=True, exist_ok=True)

    stats_logger = StatsLogger(str(results_path / LOG_CSV_FILE))
    stats_logger.set_config(kwargs)

    params = dict(kwargs)
    params["stats_logger"] = stats_logger

    try:
        if sys_name == "tempo":
            from repro.sec7_3_rl_train.impls.tempo_ppo import get_tempo_ppo_executor

            executor = get_tempo_ppo_executor(**params)
            compilation_time_breakdown = executor.get_compilation_time_breakdown()
            with open(results_path / "compilation_time_breakdown.txt", "w") as f:
                f.write(str(compilation_time_breakdown))
            execute_fn = executor.execute
        else:
            params["minibatch_size"] = params["ep_len"] * params["num_envs"]
            if sys_name == "rlgames":
                from repro.sec7_3_rl_train.impls.rl_games_ppo import get_rlgames_ppo_execute_fn

                execute_fn = get_rlgames_ppo_execute_fn(**params)
            elif sys_name == "samplefactory":
                from repro.sec7_3_rl_train.impls.sf_ppo import get_sample_factory_ppo_execute_fn

                execute_fn = get_sample_factory_ppo_execute_fn(**params)
            elif sys_name == "cleanrl":
                from repro.sec7_3_rl_train.impls.cleanrl_ppo import get_cleanrl_ppo_execute_fn

                execute_fn = get_cleanrl_ppo_execute_fn(**params)
            elif sys_name == "cleanrlcache":
                from repro.sec7_3_rl_train.impls.cleanrl_ppo_caching import (
                    get_cleanrl_caching_ppo_execute_fn,
                )

                execute_fn = get_cleanrl_caching_ppo_execute_fn(**params)
            elif sys_name == "rllib":
                from repro.sec7_3_rl_train.impls.rllib_ppo import get_rllib_ppo_execute_fn

                execute_fn = get_rllib_ppo_execute_fn(**params)
            else:
                raise ValueError("Unknown sys")

        gpu_id = kwargs["gpu_id"]
        mon = ResourceMonitorManager(results_path / MONITOR_CSV_FILE, fps=10, gpu_ids=(gpu_id,))
        if kwargs.get("profile", False):
            import torch

            from tempo.utils.torch_profiler import TorchProfiler

            prof = TorchProfiler.get(True, results_path)
            # from tempo.runtime.backends.jax.jax_backend import JaxBackend
            # from tempo.utils.jax_profiler import JaxProfiler
            # prof = JaxProfiler.get(True, results_path)

            with prof:
                execute_fn()
                torch.cuda.synchronize()
                # JaxBackend.sync()
        else:
            with mon:
                execute_fn()

    except Exception as e:
        print(f"Failed to run {sys_cfg}: {e}")
        # traceback.print_exc()

        # Log the exception to a file in results_path
        with open(results_path / ERROR_TXT_FILE, "w") as f:
            f.write(f"Exception: {e}\n")
            traceback.print_exc(file=f)  # Writes the stack trace to the file
            f.write("\n")
    finally:
        stats_logger.finish(quiet=True)


def get_experiment_name_and_results_path(base_path: str, kwargs: Dict[str, Any]) -> Tuple[str, str]:
    s = kwargs["sys_cfg"]
    nl = kwargs["num_layers"]
    pp = kwargs["params_per_layer"]
    ne = kwargs["num_envs"]
    el = kwargs["ep_len"]
    os = kwargs["obs_shape"]
    ca = kwargs["use_caching_allocators"]

    name = name_from_params(s, nl, pp, ne, el, os, ca)

    results_path = Path(base_path) / name
    return name, str(results_path)


def name_from_params(
    s: str, nl: int, pp: int, ne: int, el: int, os: Tuple[int, ...], ca: bool = True
) -> str:
    os_str = "x".join(map(str, os))
    name = f"{s}_num_layers{nl}_params_per_layer{pp}_ep_len{el}_num_envs{ne}_obs_shape{os_str}"
    if not ca:
        name += "_no_caching"
    return name
