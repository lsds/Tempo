from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

from repro.data_loading import DEFAULT_PLOTS_PATH, DEFAULT_RESULTS_PATH
from repro.sec7_2_llama32_decode.plot.plot_shared import gather_summary_results
from repro.sec7_2_llama32_decode.shared import (
    ABLATE_DIR,
    LLAMA32_DECODE_DIR,
)

""" Plot the ablation study results showing the impact of disabling optimizations.
"""

# --- CONFIG ---
METRIC = "avg_iter_time"
METRIC_DISPLAY_NAME = "Time Per Token (s)"
COLOR_MAP = {
    "all_enabled": "mediumspringgreen",
    "torch_backend": "indianred",
    "single_disabled": "orange",
    "in_order": "cadetblue",
}
HATCH_PATTERNS = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]

CFG_KEYS_IN_ORDER_TO_RENDER = [
    # "enable_vectorization",
    "enable_non_trivial_vectorization",
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


def load_in_order_results(results_path: Path):
    """Load results from in_order experiments."""
    in_order_path = results_path / "in_order"
    if not in_order_path.exists():
        print(f"Warning: {in_order_path} does not exist")
        return []

    return gather_summary_results(in_order_path)


def organize_in_order_data(raw_data, metric=METRIC):
    """Organize data for cumulative disabled optimizations."""
    # Initialize data structure
    data = {}
    baseline_value = None

    # First pass: find baseline (all_enabled), torch_backend, and torch
    for d in raw_data:
        name = d["name"]
        if name == "all_enabled":
            baseline_value = d.get(metric, 0)
            data["all_enabled"] = baseline_value
        elif name == "torch_backend":
            data["torch_backend"] = d.get(metric, 0)
        elif name == "torch":
            data["torch"] = d.get(metric, 0)

    # Second pass: organize in-order disabled optimizations
    for d in raw_data:
        name = d["name"]
        if name.startswith("in_order_"):
            # Extract optimization name and index
            parts = name.split("_")
            if len(parts) >= 4:
                opt_name = "_".join(parts[3:-1])  # Get the optimization name
                if opt_name in CFG_KEYS_IN_ORDER_TO_RENDER:
                    data[opt_name] = d.get(metric, 0)

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

    fig, ax = plt.subplots(figsize=(8, 3))

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

    # Add baseline line
    # if baseline_value:
    #    ax.axhline(y=baseline_line, color='red', linestyle='--', alpha=0.7,
    #              label='Baseline (All Enabled)')

    # Add torch backend comparison if available
    # if "torch_backend" in data and data["torch_backend"] > 0:
    #    torch_speedup = baseline_value / data["torch_backend"] if baseline_value else 0
    #    ax.axhline(y=torch_speedup, color=COLOR_MAP["torch_backend"],
    #              linestyle=':', alpha=0.8, linewidth=2,
    #              label=f'PyTorch Backend ({torch_speedup:.2f}x)')

    # Add torch comparison if available
    # if "torch" in data and data["torch"] > 0:
    #    torch_speedup = baseline_value / data["torch"] if baseline_value else 0
    #    ax.axhline(y=torch_speedup, color='orange',
    #              linestyle='-.', alpha=0.8, linewidth=2,
    #              label=f'PyTorch ({torch_speedup:.2f}x)')

    # Customize plot
    ax.set_xticks(x)
    ax.set_xticklabels([])  # Remove x-tick labels
    ax.set_xlabel("Disabled Optimization", labelpad=1)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.1)

    # Add optimization names above bars with 30-degree rotation
    for i, (bar, opt_name) in enumerate(zip(bars, opt_names, strict=False)):
        height = bar.get_height()
        # For nontrivial vec and in-place writes, place text inside bar at fixed y=0.2
        x = bar.get_x() + bar.get_width() * 0.5
        y = height + 0.15
        angle = 55
        if opt_name in ["Non-Trivial Vec.", "Inplace Writes"]:
            x = bar.get_x() + bar.get_width() * 0.25
            y = 0.25
            angle = 75
        ax.text(
            x,
            y,  # Position above each bar
            opt_name,
            ha="left",
            va="center",
            fontsize=10,
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
                fontsize=12,
                fontweight="normal",
            )

    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)


# --- MAIN ---
def plot_ablation_both(
    results_path: str = DEFAULT_RESULTS_PATH, plots_path: str = DEFAULT_PLOTS_PATH
):
    """
    Plot the ablation study results showing two figures:
    1. Single disabled: impact of disabling each optimization individually
    2. In order: cumulative impact of disabling optimizations one by one

    Args:
        results_path (str, optional): Path to the results directory. Defaults to "./results/".
        plots_path (str, optional): Path to the plots directory. Defaults to "./plots/".
    """
    # Setup paths
    results_path = Path(results_path) / LLAMA32_DECODE_DIR / ABLATE_DIR
    plot_dir = Path(plots_path) / LLAMA32_DECODE_DIR / ABLATE_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_path}")
    print(f"Saving plots to: {plot_dir}")

    # Load data
    in_order_data = load_in_order_results(results_path)

    if not in_order_data:
        print("Error: No ablation data found. Please run the ablation experiments first.")
        return
    # Plot in-order disabled optimizations
    if in_order_data:
        print("Plotting in-order disabled optimizations...")
        in_order_plot_data, baseline = organize_in_order_data(in_order_data)
        if in_order_plot_data:
            out_pdf = plot_dir / "in_order_ablation.pdf"
            plot_in_order(in_order_plot_data, baseline, out_pdf)
            print(f"Saved: {out_pdf}")
        else:
            print("Warning: No valid data for in-order plot")

    print("Ablation plotting complete!")


if __name__ == "__main__":
    fire.Fire(plot_ablation_both)
