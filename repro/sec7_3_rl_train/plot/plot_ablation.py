from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from repro.data_loading import (
    DEFAULT_PLOTS_PATH,
    DEFAULT_RESULTS_PATH,
)
from repro.sec7_3_rl_train.shared import (
    RL_TRAIN_DIR,
)

""" Plot the ablation study results showing the impact of disabling optimizations in RL training.
"""

# --- CONFIG ---
METRIC = "iter_mean"
METRIC_DISPLAY_NAME = "Iteration Time (s)"
COLOR_MAP = {"in_order": "cadetblue"}

CFG_KEYS_IN_ORDER_TO_RENDER = [
    "enable_inplace_write",
    "enable_lazy_slice",
    "enable_donation_analysis",
    # NOTE: This must come before disable grouping, otherwise JAX will complain about
    # different devices.
    # "enable_device_assignment",
    # "enable_ast_promo",
    # "enable_isolate_loop_conditions",
    # "enable_custom_thunk_launchers",
    # "enable_pad_mask_removal",
    "enable_fold_pads_into_storage",
    "enable_hybrid_tensorstore",
    "enable_statifying_incrementalization",
    # "enable_vectorization",
    "enable_non_trivial_vectorization",
    # NOTE: JAX will always codegen even single ops...
    # "enable_codegen_dataflows",
    "enable_dataflow_grouping",
    ## NOTE: If we put these after disabling grouping, it will show more impact,
    ## because they will be not masked by JAX doing the same thing.
    # "enable_constant_folding",
    # "enable_algebraic_optimizer",
    # "enable_domain_reduction",
    # "enable_duplicate_code_elim",
    ## NOTE: We cannot afford to disable dead code elimination, as it will break other optimizations.
    ##"enable_dead_code_elim",
    ## NOTE: We cannot afford to disable broadcast elimination, due to OOM.
    # "enable_broadcast_elim",
    "enable_broadcast_elim",
    "enable_vectorization",
    # NOTE: I just don't care about this one.
    # "enable_symbol_prealloc_store",
]

# Short names for display
OPTIMIZATION_DISPLAY_NAMES = {
    # NOTE: Keep or throw away?
    "enable_non_trivial_vectorization": "Non-Trivial Vec.",
    "enable_inplace_write": "Inplace Writes",
    "enable_lazy_slice": "Lazy Slices",
    # NOTE: Leave this one alone. Explain it only helps memory
    "enable_donation_analysis": "Donations",
    # NOTE: Collapse these
    # "enable_device_assignment": "Device Assign",
    # "enable_ast_promo": "AST Promo",
    # "enable_isolate_loop_conditions": "Loop Cond",
    # "enable_custom_thunk_launchers": "Thunk Launch",
    "enable_custom_thunk_launchers": "AST Optimization",
    # NOTE: Collapse these
    # "enable_pad_mask_removal": "Pad Mask",
    # "enable_fold_pads_into_storage": "Fold Pads",
    "enable_fold_pads_into_storage": "Elim. Pad/Mask",
    "enable_hybrid_tensorstore": "Custom Stores",
    "enable_statifying_incrementalization": "Static Tiling",
    # NOTE: Collapse these
    # "enable_codegen_dataflows": "Codegen DF",
    # "enable_dataflow_grouping": "DF Group",
    "enable_dataflow_grouping": "Dataflow Fusion",
    # NOTE: Collapse these
    # "enable_constant_folding": "Const Fold",
    # "enable_algebraic_optimizer": "Alg Opt",
    # "enable_domain_reduction": "Domain Red",
    # "enable_duplicate_code_elim": "Dup Elim",
    # "enable_broadcast_elim": "Broadcast Elim",
    "enable_broadcast_elim": "Optimizer",
    "enable_vectorization": "Batch Vec.",
    # NOTE: Not used
    # "enable_symbol_prealloc_store": "Prealloc Store",
}
# CFG_KEYS_IN_ORDER_TO_RENDER = [
#    "enable_inplace_write",
#    "enable_lazy_slice",
#    "enable_donation_analysis",
#    "enable_device_assignment",
#    "enable_ast_promo",
#    "enable_isolate_loop_conditions",
#    "enable_custom_thunk_launchers",
#    "enable_pad_mask_removal",
#    "enable_fold_pads_into_storage",
#    "enable_hybrid_tensorstore",
#    "enable_statifying_incrementalization",
#    "enable_non_trivial_vectorization",
#    "enable_codegen_dataflows",
#    "enable_dataflow_grouping",
#    "enable_constant_folding",
#    "enable_algebraic_optimizer",
#    "enable_domain_reduction",
#    "enable_duplicate_code_elim",
#    "enable_broadcast_elim",
#    "enable_vectorization",
# ]
#
## Short names for display
# OPTIMIZATION_DISPLAY_NAMES = {
#    "enable_non_trivial_vectorization": "Non-Trivial Vec.",
#    "enable_inplace_write": "Inplace Writes",
#    "enable_lazy_slice": "Lazy Slices",
#    "enable_donation_analysis": "Donations",
#    "enable_device_assignment": "Device Assign",
#    "enable_ast_promo": "AST Promo",
#    "enable_isolate_loop_conditions": "Loop Cond",
#    "enable_custom_thunk_launchers": "Thunk Launch",
#    "enable_pad_mask_removal": "Pad Mask",
#    "enable_fold_pads_into_storage": "Fold Pads",
#    "enable_hybrid_tensorstore": "Custom Stores",
#    "enable_statifying_incrementalization": "Static Tiling",
#    "enable_codegen_dataflows": "Codegen DF",
#    "enable_dataflow_grouping": "DF Group",
#    "enable_constant_folding": "Const Fold",
#    "enable_algebraic_optimizer": "Alg Opt",
#    "enable_domain_reduction": "Domain Red",
#    "enable_duplicate_code_elim": "Dup Elim",
#    "enable_broadcast_elim": "Broadcast Elim",
#    "enable_vectorization": "Batch Vec.",
# }


def load_ablation_results(results_path: Path):
    """Load results from ablation experiments."""
    ablate_path = results_path / "ablate"
    if not ablate_path.exists():
        print(f"Warning: {ablate_path} does not exist")
        return None

    # Load data using the same mechanism as other RL training plots
    in_order_path = ablate_path / "in_order"
    if not in_order_path.exists():
        print(f"Warning: {in_order_path} does not exist")
        return None

    # Load data for each configuration by scanning the directory
    data = {}

    # First, look for the baseline "all_enabled" experiment
    all_enabled_path = in_order_path / "all_enabled"
    if all_enabled_path.exists():
        data["all_enabled"] = load_single_experiment_data(all_enabled_path, "all_enabled", [])

    # Then look for the in_order experiments
    for exp_dir in in_order_path.iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith("in_order_"):
            # Extract optimization name from directory name
            parts = exp_dir.name.split("_")
            if len(parts) >= 4:
                opt_name = "_".join(parts[3:-1])  # Get the optimization name
                if opt_name in CFG_KEYS_IN_ORDER_TO_RENDER:
                    # For in_order experiments, we need to determine which optimizations are disabled
                    # This is based on the order in CFG_KEYS_IN_ORDER_TO_RENDER
                    opt_index = CFG_KEYS_IN_ORDER_TO_RENDER.index(opt_name)
                    disabled_keys = CFG_KEYS_IN_ORDER_TO_RENDER[: opt_index + 1]

                    data[opt_name] = load_single_experiment_data(
                        exp_dir, exp_dir.name, disabled_keys
                    )

    return data


def load_single_experiment_data(exp_path: Path, name: str, disabled_keys: list):
    """Load data from a single experiment directory."""
    # Load monitor and log data
    monitor_file = exp_path / "monitor.csv"
    log_file = exp_path / "log.csv"

    if monitor_file.exists() and log_file.exists():
        try:
            monitor_df = pd.read_csv(monitor_file)
            log_df = pd.read_csv(log_file)

            # Calculate iteration time (similar to data_loading.py)
            if len(log_df) > 1:
                # Remove first and last iterations to avoid warmup/winddown
                iterations_from_start_to_remove = 1
                iterations_from_end_to_remove = 1

                if len(log_df) > iterations_from_start_to_remove + iterations_from_end_to_remove:
                    log_df_filtered = log_df.iloc[
                        iterations_from_start_to_remove:-iterations_from_end_to_remove
                    ]
                    if len(log_df_filtered) > 1:
                        iter_time = (
                            log_df_filtered["curr_time"].diff().mean() / 1e9
                        )  # Convert to seconds
                    else:
                        iter_time = 0
                else:
                    iter_time = 0
            else:
                iter_time = 0

            return {"iter_mean": float(iter_time), "name": name, "disabled_keys": disabled_keys}
        except Exception as e:
            print(f"Error loading data for {name}: {e}")
            return {"iter_mean": 0, "name": name, "disabled_keys": disabled_keys}
    else:
        print(f"Missing data files for {name}")
        return {"iter_mean": 0, "name": name, "disabled_keys": disabled_keys}


def organize_in_order_data(raw_data, metric=METRIC):
    """Organize data for cumulative disabled optimizations."""
    # Initialize data structure
    data = {}
    baseline_value = None
    print("raw_data keys:")
    print(raw_data.keys())

    # Find baseline (all_enabled)
    if "all_enabled" in raw_data:
        baseline_value = raw_data["all_enabled"].get(metric, 0)
        data["all_enabled"] = baseline_value

    # Organize in-order disabled optimizations
    for name, exp_data in raw_data.items():
        # Extract optimization name
        if name in CFG_KEYS_IN_ORDER_TO_RENDER:
            data[name] = exp_data.get(metric, 0)

    print("data keys:")
    print(data.keys())
    return data, baseline_value


def plot_in_order(data, baseline_value, out_pdf):
    """Plot results when optimizations are disabled cumulatively."""
    plt.rcParams.update({"font.size": 12})
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # Filter to only include optimizations that have data
    available_opts = [opt for opt in CFG_KEYS_IN_ORDER_TO_RENDER if opt in data]

    if not available_opts:
        print("Warning: No optimization data found for in-order plot")
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    # Prepare data for plotting
    opt_names = [OPTIMIZATION_DISPLAY_NAMES[opt] for opt in available_opts]
    opt_values = [data[opt] for opt in available_opts]

    # Calculate speedup relative to baseline
    if baseline_value and baseline_value > 0:
        speedups = [baseline_value / val if val > 0 else 0 for val in opt_values]
        y_values = speedups
        y_label = "Speedup over all enabled"
        baseline_line = 1.0
    else:
        y_values = opt_values
        y_label = METRIC_DISPLAY_NAME
        baseline_line = baseline_value if baseline_value else 0

    # Create bars
    x = np.arange(len(opt_names))
    bars = ax.bar(
        x, y_values, color=COLOR_MAP["in_order"], alpha=0.8, edgecolor="black", linewidth=0.5
    )

    # Customize plot
    ax.set_xticks(x)
    ax.set_xticklabels([])  # Remove x-tick labels
    ax.set_xlabel("Disabled Optimization", labelpad=1)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3, axis="y")

    ax.set_ylim(0, 1.1)

    # Add optimization names above bars with rotation
    for i, (bar, opt_name) in enumerate(zip(bars, opt_names, strict=False)):
        height = bar.get_height()
        x_pos = bar.get_x() + bar.get_width() * 0.5
        y_pos = height + 0.1

        # Adjust text position for better readability
        angle = 45

        ax.text(
            x_pos,
            y_pos,
            opt_name,
            ha="left",
            va="center",
            fontsize=9,
            rotation=angle,
            rotation_mode="anchor",
            fontweight="bold",
        )

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, y_values, strict=False)):
        if val > 0:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="normal",
            )

    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_ablation(results_path: str = DEFAULT_RESULTS_PATH, plots_path: str = DEFAULT_PLOTS_PATH):
    """
    Plot the ablation study results showing the impact of disabling optimizations.

    Args:
        results_path (str, optional): Path to the results directory. Defaults to "./results/".
        plots_path (str, optional): Path to the plots directory. Defaults to "./plots/".
    """
    # Setup paths
    results_path = Path(results_path) / RL_TRAIN_DIR
    plot_dir = Path(plots_path) / RL_TRAIN_DIR / "ablate"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_path}")
    print(f"Saving plots to: {plot_dir}")

    # Load data
    ablation_data = load_ablation_results(results_path)
    print(ablation_data)

    if not ablation_data:
        print("Error: No ablation data found. Please run the ablation experiments first.")
        return

    # Plot in-order disabled optimizations
    if ablation_data:
        print("Plotting in-order disabled optimizations...")
        in_order_plot_data, baseline = organize_in_order_data(ablation_data)
        if in_order_plot_data:
            out_pdf = plot_dir / "in_order_ablation.pdf"
            plot_in_order(in_order_plot_data, baseline, out_pdf)
            print(f"Saved: {out_pdf}")
        else:
            print("Warning: No valid data for in-order plot")

    print("Ablation plotting complete!")


if __name__ == "__main__":
    fire.Fire(plot_ablation)
