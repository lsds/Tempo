from pathlib import Path

import fire
import pandas as pd

from repro.data_loading import (
    DEFAULT_PLOTS_PATH,
    DEFAULT_RESULTS_PATH,
    get_sweep_df,
    load_sweep_data,
)
from repro.sec7_3_rl_train.plot.plot_shared import (
    FRAMEWORKS_ORDERED,
    create_generic_metrics_panel,
)
from repro.sec7_3_rl_train.run_large_obs import (
    LARGE_OBS_EXPERIMENT_BASE_NAME,
    LARGE_OBS_PPO_SWEEP_PARAM_BASE,
    OBS_SHAPE_SWEEPS,
)
from repro.sec7_3_rl_train.shared import RL_TRAIN_DIR, SYS, get_experiment_name_and_results_path


def get_merged_df(base_path: str) -> pd.DataFrame:
    systems = [sys for sys in SYS if sys != "cleanrlcache"]
    data_cache = load_sweep_data(
        base_path,
        LARGE_OBS_PPO_SWEEP_PARAM_BASE,
        OBS_SHAPE_SWEEPS,
        systems=systems,
        name_function=get_experiment_name_and_results_path,
    )
    data_no_cache = load_sweep_data(
        base_path,
        LARGE_OBS_PPO_SWEEP_PARAM_BASE,
        OBS_SHAPE_SWEEPS,
        name_function=get_experiment_name_and_results_path,
        systems=systems,
        caching_allocators=False,
    )

    df_cache = get_sweep_df(data_cache, OBS_SHAPE_SWEEPS, "obs_shape", systems)
    df_no_cache = get_sweep_df(data_no_cache, OBS_SHAPE_SWEEPS, "obs_shape", systems)

    # Copy df_cache and replace GPU memory utilization data with df_no_cache data
    df_merged = df_cache.copy()

    # Replace GPU memory columns with no-cache data
    df_merged["gpu_mem_mean"] = df_no_cache["gpu_mem_mean"]
    df_merged["gpu_mem_median"] = df_no_cache["gpu_mem_median"]
    df_merged["gpu_mem_peak"] = df_no_cache["gpu_mem_peak"]
    df_merged["gpu_mem_mean_ratio"] = df_no_cache["gpu_mem_mean_ratio"]

    return df_merged


def plot_large_obs(results_path: str = DEFAULT_RESULTS_PATH, plots_path: str = DEFAULT_PLOTS_PATH):
    """
    Plot the large observation RL training experiments.

    Args:
        results_path (str): Path to the results directory. Defaults to "./results".
        plots_path (str): Path to the plots directory. Defaults to "./plots".
    """

    print("Loading data...")
    results_path = Path(results_path) / RL_TRAIN_DIR / LARGE_OBS_EXPERIMENT_BASE_NAME
    df_merged = get_merged_df(results_path)

    sweep_values_ordered = [
        OBS_SHAPE_SWEEPS["obs_shape"],
    ]
    sweep_keys = ["obs_shape"]

    framework_order = FRAMEWORKS_ORDERED.copy()
    framework_order.remove("cleanrlcache")

    print("Generating panel plot...")
    fig = create_generic_metrics_panel(
        sweep_dfs=[df_merged],
        frameworks_ordered=framework_order,
        metrics=["iter_mean", "gpu_util", "gpu_mem", "cpu_mem"],
        sweep_keys=sweep_keys,
        sweep_values_ordered=sweep_values_ordered,
        figsize=(13, 9.5),
        bbox_to_anchor=(0.5, 1.05),
        gridspec_kw={"hspace": 0.35, "height_ratios": [1, 1, 1, 1]},
        show_mean_peak_legend_on_percent_plot_number=1,
        y_lim_iter_time=(pow(10, -1), pow(10, 2.5)),
        text_size=18,
    )

    print("Saving plot...")
    # Create plots directory if it doesn't exist
    plot_dir = Path(plots_path) / RL_TRAIN_DIR / LARGE_OBS_EXPERIMENT_BASE_NAME
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = plot_dir / "large_obs.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {out_pdf}")


if __name__ == "__main__":
    fire.Fire(plot_large_obs)
