from pathlib import Path
from typing import Any, Dict, List, Tuple

import fire

from repro.data_loading import DEFAULT_RESULTS_PATH
from repro.launch_lib import launch_par, launch_seq
from repro.sec7_3_rl_train.shared import (
    RL_TRAIN_DIR,
    SHARED_PPO_HYPERPARAMS,
    SYS,
    is_large_obs,
    run_experiment,
    sweep_param,
)

LARGE_OBS_EXPERIMENT_BASE_NAME = "large_obs"

LARGE_OBS_PPO_SWEEP_PARAM_BASE = {
    "num_envs": 256,
    "ep_len": 1000,
    "params_per_layer": 64,
    "num_layers": 2,
}

OBS_SHAPE_SWEEP = [
    (3, 8, 8),
    (3, 16, 16),
    (3, 32, 32),
    (3, 64, 64),
    (3, 128, 128),
    (3, 256, 256),
]

OBS_SHAPE_SWEEPS = {
    "obs_shape": OBS_SHAPE_SWEEP,
}


# NOTE: Section 7.3: Fig 14
def build_large_obs_configs(
    skip_sys: Tuple[str, ...] = (),
    results_path: str = DEFAULT_RESULTS_PATH,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_path = str(Path(results_path) / RL_TRAIN_DIR / LARGE_OBS_EXPERIMENT_BASE_NAME)

    params_base = {
        "runner": run_experiment,
        **SHARED_PPO_HYPERPARAMS,
    }
    params_base.update(LARGE_OBS_PPO_SWEEP_PARAM_BASE)

    par, seq = [], []
    for use_caching_allocators in [True, False]:
        params = params_base.copy()
        params["use_caching_allocators"] = use_caching_allocators

        par_, seq_ = sweep_param(
            "obs_shape",
            OBS_SHAPE_SWEEP,
            params,
            base_path,
            exclude_sys=["cleanrlcache"] + list(skip_sys),
        )

        # Set iterations for each experiment based on its actual obs_shape
        for experiment in par_ + seq_:
            experiment["iterations"] = _iterations_from_params(experiment)

        par.extend(par_)
        seq.extend(seq_)

    return seq, par


def _iterations_from_params(params: Dict[str, Any]) -> int:
    use_caching_allocators = params["use_caching_allocators"]
    is_large_obs_ = is_large_obs(params["obs_shape"])

    if use_caching_allocators:
        # Large obs cases take very long to execute, so we run them for less iterations
        if is_large_obs_:
            return 4
        else:
            # Around 5 minutes to run
            return 25
    else:
        # Three is sufficient: Ignore the first and last, measure the second for mean and peak
        # memory usage.
        return 3


def launch(
    gpus: str = "0,1,2,3",
    phbgpu: int = None,
    skip_sys: str = "rllib,cleanrlcache",
    results_path: str = DEFAULT_RESULTS_PATH,
) -> None:
    """
    Usage Example:
    python repro/sec7_3_rl_train/run_large_obs.py launch --gpus "0,1,2,3" --phbgpu 1 --skip_sys rllib,cleanrlcache

    Args:
        gpus (str, optional): Comma-separated list of GPU IDs to use. Defaults to "0,1,2,3".
        phbgpu (int, optional): Specific GPU ID to use for sequential execution. Defaults to None.
        skip_sys (str, optional): Comma-separated list of systems to skip. Defaults to "rllib".
        results_path (str, optional): Path to the results directory. Defaults to DEFAULT_RESULTS_PATH.
    """
    if isinstance(gpus, tuple):
        assert all(isinstance(gpu, int) for gpu in gpus), (
            f"gpus must be a string '0,1,2,3', got {type(gpus)}"
        )
        visible_gpus = gpus
    else:
        assert isinstance(gpus, str), f"gpus must be a string '0,1,2,3', got {type(gpus)}"
        visible_gpus = tuple(int(gpu) for gpu in gpus.split(","))

    if isinstance(skip_sys, tuple):
        skip_sys = tuple(skip_sys)
        assert all(sys in SYS for sys in skip_sys), (
            f"Invalid system: {skip_sys}. Supported systems: {SYS}"
        )
    else:
        skip_sys = tuple(skip_sys.split(","))

    seq, par = build_large_obs_configs(skip_sys=skip_sys, results_path=results_path)

    launch_par(par, visible_gpus=visible_gpus)
    launch_seq(seq, timeout_minutes=60, visible_gpus=visible_gpus, phbgpu=phbgpu)


if __name__ == "__main__":
    fire.Fire(launch)
