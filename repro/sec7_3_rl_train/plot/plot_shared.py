from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

SWEEP_RENAMINGS = {
    "ep_len": "Episode Length",
    "num_envs": "Number of Environments",
    "params_per_layer": "Parameters per Layer",
    "num_layers": "Number of Hidden Layers",
    "obs_shape": "Observation Shape",
    "objective": "Objective",
}

COLORS = {
    "cleanrl": "olivedrab",
    "cleanrlcache": "green",
    "rlgames": "darkorange",
    "rllib": "royalblue",
    "samplefactory": "darkorchid",
    "tempo-jax": "cadetblue",
    "tempo-torch": "indianred",
}

PRETTY_NAMES = {
    "cleanrl": "CleanRL",
    "cleanrlcache": "CleanRL (C)",
    "rlgames": "RLGames",
    "rllib": "RLlib",
    "samplefactory": "SampleFactory",
    "tempo-jax": "Tempo-JAX",
    "tempo-torch": "Tempo-Torch",
}

SYS = PRETTY_NAMES.keys()

FRAMEWORKS_ORDERED = [
    "rllib",
    "samplefactory",
    "rlgames",
    "cleanrl",
    "cleanrlcache",
    "tempo-torch",
    "tempo-jax",
]

HATCHES = {
    "cleanrl": "",
    "cleanrlcache": "\\",
    "rlgames": ".",
    "rllib": "O",
    "samplefactory": "+",
    "tempo-jax": "x",
    "tempo-torch": "/",
}

METRIC_PRETTY_NAMES = {
    "iter_mean": "Iter. Time (s)",
    "gpu_util": "GPU Util. (%)",
    "gpu_mem": "GPU Mem. (%)",
    "cpu_util": "CPU Util. (%)",
    "cpu_mem": "CPU Mem. (%)",
}


def make_iter_time_subplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    frameworks_ordered: List[str],
    sweep_values_ordered: List[str],
    x_positions: Dict[Any, float],
    show_x_labels: bool = False,
    show_y_label: bool = False,
    y_label: str = "Iter. Time (s)",
    y_lim: tuple = (pow(10, -1), pow(10, 1.6)),
) -> Dict[str, plt.Rectangle]:
    """
    Create an iteration time subplot.

    Args:
        ax: The matplotlib axes to plot on
        df: DataFrame containing the data
        frameworks_ordered: List of framework names in order
        sweep_values_ordered: List of sweep values in order
        x_positions: Dictionary mapping sweep values to x positions
        show_x_labels: Whether to show x-axis labels
        show_y_label: Whether to show y-axis label
        y_label: Y-axis label text
        y_lim: Y-axis limits as (min, max) tuple

    Returns:
        Dictionary mapping framework names to their plot handles for legend
    """
    framework_labels = {}

    def get_x(rv: Any, fw: Any, x_pos: Dict[Any, float]) -> float:
        base = x_pos[rv]
        fw_idx = frameworks_ordered.index(fw)
        return base + fw_idx

    for _, row in df.iterrows():
        x = get_x(row["sweep_value"], row["framework"], x_positions)
        if row["error"]:
            error_text = row["error_type"]
            # Use fixed position with proper alignment for rotated text
            # NOTE: 0.1 is like the 0 in log scale plotting.
            ax.text(x, 0.1, error_text, color="red", ha="center", va="bottom", rotation=90)
        else:
            framework_name = row["framework"]
            framework_color_ = COLORS[framework_name]
            framework_hatch_ = HATCHES[framework_name]
            bar = ax.bar(
                x,
                row["iter_mean"],
                color=framework_color_,
                hatch=framework_hatch_,
                zorder=3,
                label=PRETTY_NAMES[framework_name],
                alpha=0.99,
            )
            framework_labels[PRETTY_NAMES[framework_name]] = bar

            ratio_str = f"{row['iter_mean_ratio']:.1f}x"
            if ratio_str != "1.0x":
                y_pos = row["iter_mean"] * 1.25
                y_pos = max(y_pos, 0.12)
                ax.text(x, y_pos, ratio_str, ha="center", va="bottom", rotation=90, fontsize=18)

    # Remove x-axis tick labels if not showing
    if not show_x_labels:
        ax.set_xticks([])
        ax.set_xlabel("")  # no x label on top row

    ax.set_yscale("log")

    # Only put ylabel if requested
    if show_y_label:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("")

    # use thick lines for grid
    ax.grid(axis="y", which="major", zorder=0, linewidth=2)
    ax.set_ylim(y_lim)

    yticklabels = ax.get_yticklabels()
    if yticklabels:
        yticklabels[0].set_visible(False)  # Hide the label
        yticklabels[1].set_visible(False)  # Hide the label

    return framework_labels


def make_percentage_plot(
    ax: plt.Axes,
    df: pd.DataFrame,
    frameworks_ordered: List[str],
    sweep_values_ordered: List[str],
    x_positions: Dict[Any, float],
    metric_name: str,
    show_mean_peak_legend: bool = False,
    show_y_label: bool = False,
    y_label: str = "GPU Util. (%)",
    y_lim: tuple = (0, 100),
    y_ticks: tuple = (0, 25, 50, 75, 100),
    sweep_key_renamed: str = "",
) -> None:
    """
    Create a percentage-based subplot (e.g., GPU utilization, memory utilization).

    Args:
        ax: The matplotlib axes to plot on
        df: DataFrame containing the data
        frameworks_ordered: List of framework names in order
        sweep_values_ordered: List of sweep values in order
        x_positions: Dictionary mapping sweep values to x positions
        metric_name: Name of the metric column (e.g., 'gpu_util', 'cpu_mem')
        show_legend: Whether to show the Mean/Peak legend
        show_y_label: Whether to show y-axis label
        y_label: Y-axis label text
        y_lim: Y-axis limits as (min, max) tuple
        y_ticks: List of y-axis tick values
        sweep_key_renamed: Renamed sweep key for x-axis label
    """

    def get_x(rv: Any, fw: Any, x_pos: Dict[Any, float]) -> float:
        base = x_pos[rv]
        fw_idx = frameworks_ordered.index(fw)
        return base + fw_idx

    for _, row in df.iterrows():
        x = get_x(row["sweep_value"], row["framework"], x_positions)
        if row["error"]:
            error_text = row["error_type"]
            # Use fixed position with proper alignment for rotated text
            ax.text(x, 0, error_text, color="red", ha="center", va="bottom", rotation=90)
        else:
            framework_name = row["framework"]
            framework_color_ = COLORS[framework_name]
            framework_hatch_ = HATCHES[framework_name]
            mean_val = row[f"{metric_name}_mean"]
            peak_val = row[f"{metric_name}_peak"]
            ax.bar(
                x, mean_val, color=framework_color_, hatch=framework_hatch_, zorder=3, alpha=0.99
            )
            ax.bar(
                x,
                peak_val - mean_val,
                bottom=mean_val,
                alpha=0.5,
                color=framework_color_,
                hatch=framework_hatch_,
                zorder=3,
            )

    # Y-label only if requested
    if show_y_label:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("")
    # Add Peak/Mean legend if requested
    if show_mean_peak_legend:
        mean_patch = plt.Rectangle((0, 0), 1, 1, fc="black", edgecolor="k", alpha=1, label="Mean")
        peak_patch = plt.Rectangle((0, 0), 1, 1, fc="black", edgecolor="k", alpha=0.4, label="Peak")
        ax.legend([mean_patch, peak_patch], ["Mean", "Peak"], loc="upper left")

    ax.grid(axis="y", which="both", zorder=0, linewidth=2)
    ax.set_ylim(y_lim)
    ax.set_yticks(y_ticks)

    # Set x-axis tick labels
    ax.set_xticks(
        [x_positions[rv] + (len(frameworks_ordered) - 1) / 2 for rv in sweep_values_ordered]
    )
    ax.set_xticklabels(sweep_values_ordered)
    ax.set_xlabel(sweep_key_renamed)


def create_generic_metrics_panel(
    sweep_dfs: List[pd.DataFrame],
    frameworks_ordered: List[str],
    metrics: List[str],
    sweep_keys: List[str],
    sweep_values_ordered: List[List[str]] = None,
    figsize: tuple = None,
    share_y: str = "row",
    share_x: str = "col",
    bbox_to_anchor: tuple = (0.5, 1.05),
    gridspec_kw: dict = None,
    show_mean_peak_legend_on_percent_plot_number: int = 0,
    y_lim_iter_time: tuple = (pow(10, -1), pow(10, 1.6)),
    text_size: int = 22,
) -> plt.Figure:
    """
    Create a generic metrics panel that can work with different metric combinations.

    Args:
        dfs: List of dataframes for each sweep
        frameworks_ordered: List of framework names in order
        metrics: List of metrics to plot (e.g., ["iter_mean", "gpu_util"])
        sweep_keys: List of sweep keys
        sweep_values_ordered: Optional list of sweep values for each sweep. If None, will be inferred from dfs.
        figsize: Figure size as (width, height). If None, will be auto-calculated.
        share_y: How to share y-axis ('row', 'col', 'all', or None)
        share_x: How to share x-axis ('row', 'col', 'all', or None)

    Returns:
        matplotlib Figure object
    """
    num_sweeps = len(sweep_dfs)
    num_metrics = len(metrics)

    if num_sweeps != len(sweep_keys):
        raise ValueError(
            f"Number of dataframes ({num_sweeps}) must match number of sweep keys ({len(sweep_keys)})"
        )

    # If sweep_values_ordered is not provided, infer it from the dataframes
    if sweep_values_ordered is None:
        sweep_values_ordered = []
        for df in sweep_dfs:
            sweep_values = sorted(df["sweep_value"].unique())
            sweep_values_ordered.append(sweep_values)

    # Auto-calculate figsize if not provided
    if figsize is None:
        width = 7.5 * num_sweeps  # 7.5 units per sweep
        height = 3 * num_metrics  # 3 units per metric
        figsize = (width, height)

    # Set text size
    plt.rcParams.update({"font.size": text_size})
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    if gridspec_kw is None:
        gridspec_kw = {"hspace": 0.15, "wspace": 0.02}

    fig, axes = plt.subplots(
        num_metrics,
        num_sweeps,
        figsize=figsize,
        gridspec_kw=gridspec_kw,
        sharey=share_y,
        sharex=share_x,
    )

    # Handle single row/column case
    if num_metrics == 1:
        axes = axes.reshape(1, -1)
    elif num_sweeps == 1:
        axes = axes.reshape(-1, 1)

    framework_labels = {}

    for sweep_idx, (df, rv_order, rk) in enumerate(
        zip(sweep_dfs, sweep_values_ordered, sweep_keys, strict=False)
    ):
        sweep_key_renamed = SWEEP_RENAMINGS[rk]

        # Ensure frameworks in known order
        df["framework"] = pd.Categorical(
            df["framework"], categories=frameworks_ordered, ordered=True
        )

        # Reindex df to match the specified order
        df = (
            df.set_index(["sweep_value", "framework"])
            .reindex(
                pd.MultiIndex.from_product(
                    [rv_order, frameworks_ordered], names=["sweep_value", "framework"]
                )
            )
            .reset_index()
        )

        # Assign x positions for plotting
        x_positions = {rv: idx * (len(frameworks_ordered) + 1) for idx, rv in enumerate(rv_order)}

        percentage_plot_count = 0
        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx, sweep_idx]

            # Determine if this is the leftmost column (for y-labels)
            is_leftmost = sweep_idx == 0
            # Determine if this is the bottom row (for x-labels)
            is_bottom = metric_idx == num_metrics - 1

            if metric == "iter_mean":
                subplot_framework_labels = make_iter_time_subplot(
                    ax=ax,
                    df=df,
                    frameworks_ordered=frameworks_ordered,
                    sweep_values_ordered=rv_order,
                    x_positions=x_positions,
                    show_x_labels=is_bottom,
                    show_y_label=is_leftmost,
                    y_lim=y_lim_iter_time,
                )
                framework_labels.update(subplot_framework_labels)
            elif metric in ["gpu_util", "cpu_util", "gpu_mem", "cpu_mem"]:
                y_label = METRIC_PRETTY_NAMES[metric]

                make_percentage_plot(
                    ax=ax,
                    df=df,
                    frameworks_ordered=frameworks_ordered,
                    sweep_values_ordered=rv_order,
                    x_positions=x_positions,
                    metric_name=metric,
                    y_label=y_label,
                    show_mean_peak_legend=is_leftmost
                    and percentage_plot_count == show_mean_peak_legend_on_percent_plot_number,
                    show_y_label=is_leftmost,
                    sweep_key_renamed=sweep_key_renamed if is_bottom else "",
                )
                percentage_plot_count += 1
            else:
                raise ValueError(
                    f"Unknown metric: {metric}. Supported metrics: {METRIC_PRETTY_NAMES.keys()}"
                )

    # Add a single framework legend for all panels, centered above all plots
    unique_handles = list(framework_labels.values())
    unique_labels = list(framework_labels.keys())
    fig.legend(
        unique_handles,
        unique_labels,
        loc="upper center",
        ncol=len(frameworks_ordered),
        frameon=True,
        bbox_to_anchor=bbox_to_anchor,
    )

    return fig
