from pathlib import Path
from typing import Any, Dict, Tuple

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from repro.data_loading import (
    DEFAULT_PLOTS_PATH,
    DEFAULT_RESULTS_PATH,
    get_normalized_dfs,
    get_sweep_df,
    load_sweep_data,
)
from repro.sec7_4_algo_specific_sched.shared import (
    ALGO_SPECIFIC_SCHED_DIR,
    CACHING_ALLOC_TO_ITERS,
    OBJECTIVE_SWEEP,
    SHARED_REINFORCE_HYPERPARAMS,
    get_experiment_name_and_results_path,
)

""" Plot the algorithm-specific scheduling experiments from Figure 15 in Section 7.4.
"""

# --- CONFIG ---
FRAMEWORK = "tempo-jax"
FRAMEWORK_DISPLAY_NAME = "Tempo-JAX"
FRAMEWORK_COLOR = "cadetblue"

OBJECTIVE_DISPLAY_NAMES = {
    None: "Monte Carlo",
    1: "1-step TD",
    8: "8-step TD",
    64: "64-step TD",
}
OBJECTIVE_COLORS = {
    None: "indianred",
    1: "cadetblue",
    8: "cadetblue",
    64: "blue",
}
OBJECTIVE_LINESTYLES = {
    None: "-",
    1: "x",
    8: "--",
    64: "-.",
}

METRICS = ["gpu_util", "gpu_mem_util", "gpu_pcie_tx_util", "gpu_pcie_rx_util"]

METRIC_DISPLAY_NAMES = {
    "gpu_util": "GPU Util. (%)",
    "gpu_mem_util": "GPU Mem. (%)",
    "gpu_pcie_tx_util": "H2D BW (%)",
    "gpu_pcie_rx_util": "D2H BW (%)",
}
METRIC_YLIMS = {
    "gpu_util": (0, 110),
    "gpu_mem_util": (0, 60),
    "gpu_pcie_tx_util": (0, 100),
    "gpu_pcie_rx_util": (0, 100),
}

# Data smoothing factor
SMOOTH_SIGMA = 0.5

# Phase colors for shading
SHADE_1_COLOR = "indianred"
SHADE_2_COLOR = "orange"
SHADE_3_COLOR = "blue"
SHADE_ALPHA = 0.1

# Axis formatting
X_LIM_START, X_LIM_END = 1, 99


## ============== SHADING CONSTANTS (Original) =================
#
## NOTE: This should cover the simulation phase for Monte Carlo (when GPU mem is low)
# RED_SHADE_START, RED_SHADE_END = 15, 33.5
#
## NOTE: This should cover the learning phase for Monte Carlo (when GPU mem is high)
# ORANGE_SHADE_START, ORANGE_SHADE_END = 34, 49
#
## NOTE: This should cover the parallel learning and simulation phase for Temporal Differences
# BLUE_SHADE_START, BLUE_SHADE_END = 66, 98

# ============== SHADING CONSTANTS (NEW) =================

# NOTE: This should cover the simulation phase for Monte Carlo (when GPU mem is low)
RED_SHADE_START, RED_SHADE_END = 16, 32

# NOTE: This should cover the learning phase for Monte Carlo (when GPU mem is high)
ORANGE_SHADE_START, ORANGE_SHADE_END = 33, 50

# NOTE: This should cover the parallel learning and simulation phase for Temporal Differences
BLUE_SHADE_START, BLUE_SHADE_END = 52, 85


## =============== ALIGNMENT OF MEMORY USAGE =================
# In our original runs, no alignment was needed for Temporal differences, as they were generally
# well aligned.
# NOTE: This is used to stretch and compress memory usage, which comes from a different run,
# to align with the GPU utilization/tranfer plots.
ALIGNMENT_X_BEFORE = np.array([13, 28, 43, 58, 72, 87, 100])
ALIGNMENT_X_AFTER = np.array([0, 16, 33, 50, 67, 84, 100])


# --- DATA LOADING ---
def load_algo_specific_sched_data(
    base_path: str,
) -> dict[str, dict[Any, dict[str, dict[str, Any]]]]:
    """Load algorithm-specific scheduling experiment data."""
    data = {}

    for use_caching_allocators in [True, False]:
        cache_str = "caching" if use_caching_allocators else "no_caching"

        # Build sweep parameters
        sweep_params = {
            "objective": OBJECTIVE_SWEEP,
        }

        # Load data using the existing data loading utilities
        sweep_data = load_sweep_data(
            base_path,
            SHARED_REINFORCE_HYPERPARAMS,
            sweep_params,
            systems=["tempo-jax"],
            name_function=get_experiment_name_and_results_path,
            caching_allocators=use_caching_allocators,
        )

        data[cache_str] = sweep_data

    return data


def get_runtime_data(
    data: dict[str, dict[Any, dict[str, dict[str, Any]]]],
    objective: Any,
    use_caching: bool,
    restrict_to_three_iters: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get normalized monitor and log data for a specific configuration."""
    cache_str = "caching" if use_caching else "no_caching"

    total_iters = CACHING_ALLOC_TO_ITERS[use_caching]
    rm_start_iters = 1
    rm_end_iters = 1

    # NOTE: Take 3 iterations from middle of the run to avoid warmup/winddown effects
    if (restrict_to_three_iters) and (total_iters > 6):
        rm_start_iters = 5
        rm_end_iters = total_iters - (rm_start_iters + 3)

    return get_normalized_dfs(
        data[cache_str],
        FRAMEWORK,
        "objective",
        objective,
        iterations_from_start_to_remove=rm_start_iters,
        iterations_from_end_to_remove=rm_end_iters,
    )


def smooth_data(data: np.ndarray, sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    """Apply Gaussian smoothing to data."""
    return gaussian_filter1d(data, sigma)


def get_merged_df(base_path: str) -> pd.DataFrame:
    """Get merged dataframe with caching data for most metrics and no-caching for memory."""

    # Load data with caching allocators
    data_cache = load_sweep_data(
        base_path,
        SHARED_REINFORCE_HYPERPARAMS,
        {"objective": OBJECTIVE_SWEEP},
        systems=["tempo-jax"],
        name_function=get_experiment_name_and_results_path,
        caching_allocators=True,
    )

    # Load data without caching allocators
    data_no_cache = load_sweep_data(
        base_path,
        SHARED_REINFORCE_HYPERPARAMS,
        {"objective": OBJECTIVE_SWEEP},
        systems=["tempo-jax"],
        name_function=get_experiment_name_and_results_path,
        caching_allocators=False,
    )

    # Get sweep dataframes
    df_cache = get_sweep_df(data_cache, {"objective": OBJECTIVE_SWEEP}, "objective", ["tempo-jax"])
    df_no_cache = get_sweep_df(
        data_no_cache, {"objective": OBJECTIVE_SWEEP}, "objective", ["tempo-jax"]
    )

    # Copy df_cache and replace GPU memory utilization data with df_no_cache data
    df_merged = df_cache.copy()

    # Replace GPU memory columns with no-cache data
    df_merged["gpu_mem_mean"] = df_no_cache["gpu_mem_mean"]
    df_merged["gpu_mem_median"] = df_no_cache["gpu_mem_median"]
    df_merged["gpu_mem_peak"] = df_no_cache["gpu_mem_peak"]
    df_merged["gpu_mem_mean_ratio"] = df_no_cache["gpu_mem_mean_ratio"]

    return df_merged


# --- PLOTTING ---
def plot_runtime_metrics(
    data: dict[str, dict[Any, dict[str, dict[str, Any]]]], out_pdf: str
) -> plt.Figure:
    """Create runtime metrics plot showing GPU utilization over time for different objectives."""
    plt.rcParams.update({"font.size": 16})
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, axes = plt.subplots(4, 1, figsize=(10, 6), gridspec_kw={"hspace": 0.25}, sharex=True)

    # Plot configurations to show
    plot_configs = [
        (None, "Monte Carlo", "indianred", "-"),
        (8, "8-step TD", "cadetblue", "--"),
        (64, "64-step TD", "blue", "-."),
    ]

    for i, metric in enumerate(METRICS):
        ax = axes[i]

        for objective, label, color, linestyle in plot_configs:
            try:
                # Get data - use caching data for most metrics, no-caching for memory
                use_caching = metric != "gpu_mem_util"
                df_monitor, _ = get_runtime_data(
                    data, objective, use_caching, restrict_to_three_iters=True
                )

                # Normalize time to 0-100%
                df_monitor["elapsed_percent"] = (
                    df_monitor["elapsed_sec"] / df_monitor["elapsed_sec"].max() * 100
                )

                if metric == "gpu_mem_util":
                    df_monitor["elapsed_percent"] = np.interp(
                        df_monitor["elapsed_percent"], ALIGNMENT_X_BEFORE, ALIGNMENT_X_AFTER
                    )

                # Plot smoothed data
                ax.plot(
                    df_monitor["elapsed_percent"],
                    smooth_data(df_monitor[metric].values),
                    label=label,
                    color=color,
                    linestyle=linestyle,
                )
            except Exception as e:
                print(f"Warning: Could not plot {metric} for {label}: {e}")

        # Configure axis
        ax.text(
            0.01,
            0.2,
            METRIC_DISPLAY_NAMES[metric],
            va="center",
            ha="left",
            rotation=0,
            transform=ax.transAxes,
            bbox={"facecolor": "white", "alpha": 0.5},
            fontsize=14,
        )

        # Formatting
        ax.set_xticks([20, 40, 60, 80])
        if i == 3:
            ax.set_xlabel("Completion of 3 iterations (%)")
        ax.set_xlim(X_LIM_START, X_LIM_END)
        ax.grid(axis="y", which="both", zorder=0, linewidth=0.5)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylim(*METRIC_YLIMS[metric])

        # Shade different phases
        ax.axvspan(RED_SHADE_START, RED_SHADE_END, color=SHADE_1_COLOR, alpha=SHADE_ALPHA, zorder=0)
        ax.axvspan(
            ORANGE_SHADE_START, ORANGE_SHADE_END, color=SHADE_2_COLOR, alpha=SHADE_ALPHA, zorder=0
        )
        ax.axvspan(
            BLUE_SHADE_START, BLUE_SHADE_END, color=SHADE_3_COLOR, alpha=SHADE_ALPHA, zorder=0
        )

    # Add legend
    axes[-1].legend(loc="upper center", bbox_to_anchor=(0.5, 5.3), ncol=3)

    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_pdf}")
    return fig


def plot_iteration_time_comparison(
    data: dict[str, dict[Any, dict[str, dict[str, Any]]]], out_pdf: str
) -> plt.Figure:
    """Create bar chart comparing iteration times across different objectives."""
    plt.rcParams.update({"font.size": 18})
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(1, 1, figsize=(9, 3))

    # Collect iteration times
    iteration_times = []
    objectives = []

    order = [1, 8, 64, None]
    hatch = "x"  # For all

    for objective in order:
        try:
            # Use caching data for iteration time
            df_monitor, df_log = get_runtime_data(data, objective, True)

            # Calculate mean iteration time
            iter_mean = df_log["elapsed_sec"].diff().mean()
            iteration_times.append(iter_mean)
            objectives.append(objective)
        except Exception as e:
            print(f"Warning: Could not get iteration time for {objective}: {e}")

    # Create bar chart
    x_positions = np.arange(len(objectives))
    colors = ["cadetblue" for _ in objectives]

    ax.bar(x_positions, iteration_times, color=colors, zorder=3, hatch=hatch, alpha=0.99)

    # Add speedup labels
    max_time = max(iteration_times)
    for i, (_, time) in enumerate(zip(objectives, iteration_times, strict=False)):
        ratio = max_time / time
        ratio_str = f"{ratio:.1f}x"
        if ratio_str != "1.0x":
            ax.text(i, time * 1.25, ratio_str, ha="center", va="bottom", rotation=90)

    # Formatting
    ax.set_ylabel("Iter. Time (s)")
    ax.grid(axis="y", which="both", zorder=0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([OBJECTIVE_DISPLAY_NAMES.get(obj, str(obj)) for obj in objectives])
    ax.set_yticks([0, 5, 10, 15, 20, 25])

    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_pdf}")
    return fig


# --- MAIN ---
def plot_algo_specific_sched(
    results_path: str = DEFAULT_RESULTS_PATH, plots_path: str = DEFAULT_PLOTS_PATH
):
    """
    Generate plots for algorithm-specific scheduling experiments.

    Args:
        results_path (str, optional): Path to the results directory. Defaults to "./results/".
        plots_path (str, optional): Path to the plots directory. Defaults to "./plots/".
    """
    print("Loading algorithm-specific scheduling data...")
    results_path = Path(results_path) / ALGO_SPECIFIC_SCHED_DIR
    data = load_algo_specific_sched_data(str(results_path))

    # Create plots directory
    plot_dir = Path(plots_path) / ALGO_SPECIFIC_SCHED_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating runtime metrics plot at {str(plot_dir / 'runtime_metrics.pdf')}")
    runtime_pdf = plot_dir / "runtime_metrics.pdf"
    plot_runtime_metrics(data, runtime_pdf)

    print(
        f"Generating iteration time comparison plot at {str(plot_dir / 'iteration_time_comparison.pdf')}"
    )
    iter_time_pdf = plot_dir / "iteration_time_comparison.pdf"
    plot_iteration_time_comparison(data, iter_time_pdf)


if __name__ == "__main__":
    fire.Fire(plot_algo_specific_sched)
