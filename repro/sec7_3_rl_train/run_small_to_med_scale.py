from pathlib import Path
from typing import Any, Dict, List, Tuple

import fire

from repro.data_loading import DEFAULT_RESULTS_PATH
from repro.launch_lib import launch_par, launch_seq
from repro.sec7_3_rl_train.shared import (
    RL_TRAIN_DIR,
    SHARED_PPO_HYPERPARAMS,
    run_experiment,
    sweep_param,
)

""" Run the small to medium scale experiments from Figure 13 in Section 7.3.
"""

SMALL_TO_MED_EXPERIMENT_BASE_NAME = "small_to_med_scale"

PPO_SWEEP_PARAM_BASE = {
    "obs_shape": (3, 4, 4),
    "num_envs": 512,
    "ep_len": 250,
    "params_per_layer": 64,
    "num_layers": 2,
    "use_caching_allocators": True,
}

PPO_SMALL_TO_MED_SWEEPS = {
    "num_layers": [1, 4, 16],
    "params_per_layer": [16, 64, 256],
    "ep_len": [125, 500, 2000],
    "num_envs": [128, 512, 2048],
}


# NOTE: This is a helper function to deduplicate dictionaries, needed because dicts are not hashable
def dedup_dicts(dicts: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    result: list[dict] = []
    for d in dicts:
        items = tuple(sorted(d.items()))
        if items not in seen:
            seen.add(items)
            result.append(d)
    return result


# NOTE Section 7.3: Fig 13
def build_small_to_med_scale_configs(
    results_path: str = DEFAULT_RESULTS_PATH,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_path = str(Path(results_path) / RL_TRAIN_DIR / SMALL_TO_MED_EXPERIMENT_BASE_NAME)

    params_base: Dict[str, Any] = {
        "runner": run_experiment,
        **SHARED_PPO_HYPERPARAMS,
    }

    # NOTE: Ensure we are using the same base params for all experiments.
    params_base.update(PPO_SWEEP_PARAM_BASE)

    seq = []
    par = []
    for sweep_param_name, sweep_param_values in PPO_SMALL_TO_MED_SWEEPS.items():
        par_, seq_ = sweep_param(sweep_param_name, sweep_param_values, params_base, base_path)
        par.extend(par_)
        seq.extend(seq_)

    # NOTE: remove duplicates
    par = dedup_dicts(par)
    seq = dedup_dicts(seq)

    # Reduce number of RLlib iterations since it takes longer
    for s in seq:
        if s["sys_cfg"] == "rllib":
            s["iterations"] //= 2

    return seq, par


def launch(
    gpus: str = "0,1,2,3", phbgpu: int = None, results_path: str = DEFAULT_RESULTS_PATH
) -> None:
    """
    Usage Example:
    python repro/sec7_3_rl_train/run_small_to_med_scale.py launch --gpus "0,1,2,3"

    Args:
        gpus (str, optional): Comma-separated list of GPU IDs to use. Defaults to "0,1,2,3".
        phbgpu (int, optional): Specific GPU ID to use for sequential execution. Defaults to None.
    """
    if isinstance(gpus, tuple):
        assert all(isinstance(gpu, int) for gpu in gpus), (
            f"gpus must be a string '0,1,2,3', got {type(gpus)}"
        )
        visible_gpus = gpus
    else:
        assert isinstance(gpus, str), f"gpus must be a string '0,1,2,3', got {type(gpus)}"
        visible_gpus = tuple(int(gpu) for gpu in gpus.split(","))

    seq, par = build_small_to_med_scale_configs(results_path=results_path)

    launch_par(par, visible_gpus=visible_gpus)
    # NOTE: Because RLlib is a cpu-based system, we allow it to run in isolation, with all
    # CPUs to itself.
    launch_seq(seq, timeout_minutes=20, retry_attempts=2, visible_gpus=visible_gpus, phbgpu=phbgpu)
    # NOTE: RLlib can be finnicky about exiting cleanly, so we retry a few times.


if __name__ == "__main__":
    fire.Fire(launch)
