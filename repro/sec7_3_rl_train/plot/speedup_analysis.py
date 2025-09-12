from pathlib import Path
from typing import List

import fire
import pandas as pd

from repro.data_loading import DEFAULT_RESULTS_PATH, get_sweep_df, load_sweep_data
from repro.sec7_3_rl_train.plot.plot_shared import (
    FRAMEWORKS_ORDERED,
    PRETTY_NAMES,
)
from repro.sec7_3_rl_train.run_small_to_med_scale import (
    PPO_SMALL_TO_MED_SWEEPS,
    PPO_SWEEP_PARAM_BASE,
    SMALL_TO_MED_EXPERIMENT_BASE_NAME,
)
from repro.sec7_3_rl_train.shared import (
    RL_TRAIN_DIR,
    SYS,
    get_experiment_name_and_results_path,
)


def print_framework_speedup(
    dfs: list[pd.DataFrame], frameworks_ordered: list[str], base_framework: str
):
    """
    Calculate and print the average speedup of a base framework compared to other systems.

    Args:
        dfs: List of dataframes for each sweep
        frameworks_ordered: List of framework names in order
        base_framework: The framework to use as the baseline for speedup calculations
    """
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Filter out rows with errors
    valid_df = combined_df[~combined_df["error"]]

    # Get base framework data
    base_framework_data = valid_df[valid_df["framework"] == base_framework]

    if base_framework_data.empty:
        print(f"No valid {base_framework} data found.")
        return

    # Calculate average speedup for each other framework
    other_frameworks = [fw for fw in frameworks_ordered if fw != base_framework]

    print("\n" + "=" * 60)
    print(f"{PRETTY_NAMES[base_framework].upper()} AVERAGE SPEEDUP ANALYSIS")
    print("=" * 60)

    for framework in other_frameworks:
        framework_data = valid_df[valid_df["framework"] == framework]

        if framework_data.empty:
            print(f"{PRETTY_NAMES[base_framework]} vs {PRETTY_NAMES[framework]}: No valid data")
            continue

        # Calculate speedup for each sweep value where both frameworks have data
        speedups = []

        for sweep_value in framework_data["sweep_value"].unique():
            base_framework_iter_time = (
                base_framework_data[base_framework_data["sweep_value"] == sweep_value][
                    "iter_mean"
                ].iloc[0]
                if not base_framework_data[base_framework_data["sweep_value"] == sweep_value].empty
                else None
            )

            framework_iter_time = (
                framework_data[framework_data["sweep_value"] == sweep_value]["iter_mean"].iloc[0]
                if not framework_data[framework_data["sweep_value"] == sweep_value].empty
                else None
            )

            if base_framework_iter_time is not None and framework_iter_time is not None:
                speedup = framework_iter_time / base_framework_iter_time
                speedups.append(speedup)

        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            # Calculate percentage improvement
            percentage_improvement = (avg_speedup - 1) * 100
            print(
                f"{PRETTY_NAMES[base_framework]} is on average {avg_speedup:.1f}x faster than {PRETTY_NAMES[framework]} ({percentage_improvement:+.1f}%)"
            )
        else:
            print(
                f"{PRETTY_NAMES[base_framework]} vs {PRETTY_NAMES[framework]}: No comparable data points"
            )

    print("=" * 60)


def analyze_speedup(results_path: str = DEFAULT_RESULTS_PATH, base_framework: str = "tempo-jax"):
    """
    Load data and analyze speedup for a given base framework.

    Args:
        base_path (str): Base path to the results directory. Defaults to "./results".
        base_framework (str): The framework to use as baseline. Defaults to "tempo-jax".
    """
    print("Loading data...")
    results_path = str(Path(results_path) / RL_TRAIN_DIR / SMALL_TO_MED_EXPERIMENT_BASE_NAME)

    data = load_sweep_data(
        results_path,
        PPO_SWEEP_PARAM_BASE,
        PPO_SMALL_TO_MED_SWEEPS,
        name_function=get_experiment_name_and_results_path,
        systems=SYS,
    )

    print("Building dataframes...")
    df_ep_len = get_sweep_df(
        data,
        PPO_SMALL_TO_MED_SWEEPS,
        "ep_len",
        SYS,
    )
    df_params = get_sweep_df(
        data,
        PPO_SMALL_TO_MED_SWEEPS,
        "params_per_layer",
        SYS,
    )
    df_layers = get_sweep_df(
        data,
        PPO_SMALL_TO_MED_SWEEPS,
        "num_layers",
        SYS,
    )
    df_envs = get_sweep_df(
        data,
        PPO_SMALL_TO_MED_SWEEPS,
        "num_envs",
        SYS,
    )

    # Print speedup analysis
    print_framework_speedup(
        [df_ep_len, df_params, df_layers, df_envs], FRAMEWORKS_ORDERED, base_framework
    )


if __name__ == "__main__":
    fire.Fire(analyze_speedup)
