from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import pandas as pd

DEFAULT_RESULTS_PATH = "./results"
DEFAULT_PLOTS_PATH = "./plots"
MONITOR_CSV_FILE = "monitor.csv"
LOG_CSV_FILE = "log.csv"
LOG_CONFIG_FILE = "log.config"
ERROR_TXT_FILE = "error.txt"

"""
This file contains utilities to load and parse the results of the experiments.
"""


def read_csv(path: str) -> Union[pd.DataFrame, None]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def parse_error_file(error_file_path: str) -> str:
    """
    Parse an error file to determine the type of error.
    Returns "OOM" if the error contains memory-related keywords, "MISSING" otherwise.
    """
    try:
        with open(error_file_path, "r") as f:
            error_content = f.read().lower()

        # Keywords that indicate out-of-memory errors
        oom_keywords = [
            "oom",
            "bytes",
            "allocating",
            "out of memory",
            "out-of-memory",
            "cuda out of memory",
            "cuda oom",
            "gpu out of memory",
            "gpu oom",
            "memory allocation failed",
            "insufficient memory",
            "memory error",
        ]

        for keyword in oom_keywords:
            if keyword in error_content:
                return "OOM"

        return "MISSING"
    except Exception as e:
        print(f"Error reading error file {error_file_path}: {e}")
        return "MISSING"


def get_gpu_id_from_run_data(run_data: Dict[str, Any]) -> int:
    if run_data["monitor"] is not None:
        gpu_mem_col = [col for col in run_data["monitor"].columns if "gpu" in col and "mem" in col]
        if gpu_mem_col:
            return int(gpu_mem_col[0].split("_")[0][-1])
    return 0


def get_single_run_data(
    path: str, params: Dict[str, Any], name_function: Callable[[str, Dict[str, Any]], str]
) -> Dict[str, Any]:
    # Generate expected experiment name
    expected_name, experiment_path = name_function(path, params)

    experiment_path = Path(experiment_path)

    if experiment_path.exists():
        # Check for error file
        error_file_path = experiment_path / ERROR_TXT_FILE
        is_error = error_file_path.is_file()
        error_type = parse_error_file(str(error_file_path)) if is_error else ""

        run_data = {
            "monitor": None if is_error else read_csv(experiment_path / MONITOR_CSV_FILE),
            "log": None if is_error else read_csv(experiment_path / LOG_CSV_FILE),
            "config": None if is_error else read_csv(experiment_path / LOG_CONFIG_FILE),
            "error": error_type,
        }

        run_data["gpu_used"] = get_gpu_id_from_run_data(run_data)
    else:
        # Handle missing experiments
        print(f"Missing experiment: {expected_name}")
        run_data = {
            "monitor": None,
            "log": None,
            "config": None,
            "error": "MISSING",
            "gpu_used": 0,
        }
    return run_data


def load_sweep_data(
    base_path: str,
    base_params: Dict[str, Any],
    sweeps: Dict[str, List[Any]],
    systems: List[str],
    name_function: Callable[[Dict[str, Any]], str],
    caching_allocators: bool = True,
) -> Dict[str, Dict[Any, Dict[str, Dict[str, Any]]]]:
    """Load data from the experiment results using the naming scheme from shared.py"""
    # Access the small_to_med_scale subpath
    path = Path(base_path)
    data = {}

    if not path.exists():
        print(f"Path does not exist: {path}")
        return data

    # Iterate through each sweep and system to build expected experiment names
    for sweep_key, sweep_values in sweeps.items():
        for sweep_value in sweep_values:
            for sys in systems:
                # Build parameters for this experiment
                params = base_params.copy()

                # Update the parameter being varied
                params[sweep_key] = sweep_value
                params["sys_cfg"] = sys
                params["use_caching_allocators"] = caching_allocators
                run_data = get_single_run_data(path, params, name_function)

                # Store the data
                val_data = data.setdefault(sweep_key, {}).setdefault(sweep_value, {})
                val_data[sys] = run_data

    return data


def get_sweep_df(
    data: Dict[str, Dict[Any, Dict[str, Dict[str, Any]]]],
    sweeps: Dict[str, List[Any]],
    sweep_key: str,
    systems: List[str],
) -> pd.DataFrame:
    data_list = []
    for sweep_value in sweeps[sweep_key]:
        for sys in systems:
            data_list.append(build_aggregate_metric_df(data, sweep_key, sweep_value, sys))

    df = pd.DataFrame(data_list)
    df = compute_ratios(df)

    return df


def has_error(
    data: Dict[str, Dict[Any, Dict[str, Dict[str, Any]]]],
    framework: str,
    sweep_key: str,
    sweep_value,
) -> bool:
    error_cache = data[sweep_key][sweep_value][framework]["error"]
    has_error_ = error_cache != ""

    log_file = data[sweep_key][sweep_value][framework]["log"]
    monitor_file = data[sweep_key][sweep_value][framework]["monitor"]
    if log_file is None:
        has_error_ = True
    if monitor_file is None:
        has_error_ = True

    return has_error_


def get_error_type(
    data: Dict[str, Dict[Any, Dict[str, Dict[str, Any]]]],
    framework: str,
    sweep_key: str,
    sweep_value,
) -> str:
    """Get the specific error type for a framework and sweep value."""
    return data[sweep_key][sweep_value][framework]["error"]


def get_normalized_dfs(
    data: Dict[str, Dict[Any, Dict[str, Dict[str, Any]]]],
    framework: str,
    sweep_key: str,
    sweep_value,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get normalized dataframes for a specific framework and sweep value"""
    run = data[sweep_key][sweep_value][framework]
    df_monitor = run["monitor"]
    df_log = run["log"]

    if df_monitor is None or df_log is None:
        return None, None

    # NOTE: Remove first and last iterations to avoid warmup/winddown effects
    iteration_0_time_ns = int(df_log["curr_time"].iloc[1])
    iteration_last_time_ns = int(df_log["curr_time"].iloc[-1])

    gpu_used = run["gpu_used"]

    # Filter out data from before start time
    df_log = df_log[df_log["curr_time"] >= iteration_0_time_ns]
    df_monitor = df_monitor[df_monitor["curr_time"] >= iteration_0_time_ns]

    # Filter out data from after end time
    df_log = df_log[df_log["curr_time"] <= iteration_last_time_ns]
    df_monitor = df_monitor[df_monitor["curr_time"] <= iteration_last_time_ns]

    # Create elapse_sec that starts from 0
    df_log["elapsed_sec"] = (df_log["curr_time"] - iteration_0_time_ns) / 1e9
    df_monitor["elapsed_sec"] = (df_monitor["curr_time"] - iteration_0_time_ns) / 1e9

    # Remove gpu number from column names for uniform handling
    df_monitor.columns = [col.replace(f"gpu{gpu_used}_", "gpu_") for col in df_monitor.columns]

    return df_monitor, df_log


def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    # Compute worst values for each sweep_value and metric
    grouped = df.groupby("sweep_value")

    worst_values = grouped.agg(
        {
            "iter_mean": "max",
            "gpu_mem_mean": "max",
            "cpu_mem_mean": "max",
        }
    ).rename(
        columns={
            "iter_mean": "worst_iter_mean",
            "gpu_mem_mean": "worst_gpu_mem_mean",
            "cpu_mem_mean": "worst_cpu_mem_mean",
        }
    )

    # Compute the 95th percentile for gpu_util_mean and cpu_util_mean
    percentile_95 = (
        grouped[["gpu_util_mean", "cpu_util_mean"]]
        .quantile(0.95)
        .rename(
            columns={
                "gpu_util_mean": "worst_gpu_util_mean",
                "cpu_util_mean": "worst_cpu_util_mean",
            }
        )
    )
    worst_values = worst_values.join(percentile_95, how="left")

    # Merge worst values back into the main DataFrame
    df = df.merge(worst_values, on="sweep_value")

    # Compute ratios
    df["iter_mean_ratio"] = 1 / (df["iter_mean"] / df["worst_iter_mean"])
    df["gpu_mem_mean_ratio"] = 1 / (df["gpu_mem_mean"] / df["worst_gpu_mem_mean"])
    df["gpu_util_mean_ratio"] = 1 / (df["gpu_util_mean"] / df["worst_gpu_util_mean"])
    df["cpu_util_mean_ratio"] = 1 / (df["cpu_util_mean"] / df["worst_cpu_util_mean"])
    df["cpu_mem_mean_ratio"] = 1 / (df["cpu_mem_mean"] / df["worst_cpu_mem_mean"])

    # Drop worst values
    df = df.drop(
        columns=[
            "worst_iter_mean",
            "worst_gpu_mem_mean",
            "worst_gpu_util_mean",
            "worst_cpu_util_mean",
            "worst_cpu_mem_mean",
        ]
    )

    return df


def build_aggregate_metric_df(
    data: Dict[str, Dict[Any, Dict[str, Dict[str, Any]]]],
    sweep_key: str,
    sweep_value: Any,
    sys: str,
) -> Dict[str, Any]:
    error = has_error(data, sys, sweep_key, sweep_value)
    error_type = get_error_type(data, sys, sweep_key, sweep_value)

    if not error:
        df_monitor, df_log = get_normalized_dfs(data, sys, sweep_key, sweep_value)

    dict_ = {
        "error": error,
        "error_type": error_type,
        "sweep_value": sweep_value,
        "framework": sys,
        "iter_mean": df_log["elapsed_sec"].diff().mean() if not error else 0,
        "iter_std": df_log["elapsed_sec"].diff().std() if not error else 0,
        "gpu_mem_mean": df_monitor["gpu_mem_util"].mean() if not error else 0,
        "gpu_mem_median": df_monitor["gpu_mem_util"].median() if not error else 0,
        "gpu_mem_peak": df_monitor["gpu_mem_util"].max() if not error else 0,
        "gpu_util_mean": df_monitor["gpu_util"].mean() if not error else 0,
        "gpu_util_median": df_monitor["gpu_util"].median() if not error else 0,
        "gpu_util_peak": df_monitor["gpu_util"].max() if not error else 0,
        "cpu_util_mean": df_monitor["cpu_util"].mean() if not error else 0,
        "cpu_util_median": df_monitor["cpu_util"].median() if not error else 0,
        "cpu_util_peak": df_monitor["cpu_util"].max() if not error else 0,
        "cpu_mem_mean": df_monitor["cpu_mem_util"].mean() if not error else 0,
        "cpu_mem_median": df_monitor["cpu_mem_util"].median() if not error else 0,
        "cpu_mem_peak": df_monitor["cpu_mem_util"].max() if not error else 0,
    }
    return dict_
