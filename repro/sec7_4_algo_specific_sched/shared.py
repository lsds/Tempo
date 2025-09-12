from pathlib import Path
from typing import Any, Dict, Optional, Tuple

BASE_PATH = "./results"
ALGO_SPECIFIC_SCHED_DIR = "algo_specific_sched"
MONITOR_CSV_FILE = "monitor.csv"
LOG_CSV_FILE = "log.csv"
LOG_CONFIG_FILE = "log.config"
ERROR_TXT_FILE = "error.txt"

CACHING_ALLOC_TO_ITERS = {
    True: 10,
    False: 6,
}

SHARED_REINFORCE_HYPERPARAMS = {
    "env_name": "trivial.trivial",
    # NOTE: Default obs shape for trivial env
    "obs_shape": (3, 256, 256),
    "seed": 0,
    "dev": "cuda",
    "iterations": 5,
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

OBJECTIVE_SWEEP = [
    None,  # REINFORCE - Monte Carlo
    1,  # 1-step Temporal Difference
    8,  # 8-step Temporal Difference
    64,  # 64-step Temporal Difference
]

SYS = [
    "tempo-torch",
    "tempo-jax",
    "cleanrl",
    "cleanrlcache",
    "rlgames",
    "samplefactory",
    "rllib",
]


def name_from_params(objective: int | None = None, ca: bool = True) -> str:
    if objective is None:
        name = "objective_monte_carlo"
    else:
        assert isinstance(objective, int), f"objective must be an int, got {type(objective)}"
        name = f"objective_{objective}_temporal_difference"
    if not ca:
        name += "_no_caching"
    return name


def get_experiment_name_and_results_path(base_path: str, kwargs: dict[str, Any]) -> tuple[str, str]:
    ca = kwargs["use_caching_allocators"]
    objective = kwargs["objective"]

    name = name_from_params(objective, ca)

    results_path = Path(base_path) / name
    return name, str(results_path)
