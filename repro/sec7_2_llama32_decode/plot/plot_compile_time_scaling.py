import json
from pathlib import Path

import colorcet as cc
import fire
import matplotlib.pyplot as plt
import numpy as np
import optree

from repro.data_loading import DEFAULT_PLOTS_PATH, DEFAULT_RESULTS_PATH
from repro.sec7_2_llama32_decode.run_compile_time_scaling import NUM_LAYERS, NUM_RUNS
from repro.sec7_2_llama32_decode.shared import (
    COMPILE_TIME_SCALING_DIR,
    LLAMA32_DECODE_DIR,
)

""" Plot the compile time scaling benchmark results.
"""

COLORS = cc.colormaps.get_cmap("cet_glasbey_category10").colors
COLORS = [
    "#AEC6CF",  # pastel blue
    "#FFB347",  # pastel orange
    "#77DD77",  # pastel green
    "#FF6961",  # pastel red
    "#F49AC2",  # pastel pink
    "#CFCFC4",  # pastel gray
    "#B39EB5",  # pastel purple
    "#FFD1DC",  # pastel light pink
    "#CB99C9",  # pastel lavender
    "#FFE5B4",  # pastel peach
]
import seaborn as sns

COLORS = sns.color_palette("deep", 10).as_hex()

# --- CONFIG ---
MAX_DEPTH = 2  # Maximum depth for nested stages
SUM_TOGETHER_THRESHOLD = 100  # Stages with time < 100ms will be summed into "Other"
SUM_OPTIMS = True  # Sum all Optim-related stages into a single "Optim" category
FORCE_DOMAIN_PROP_TO_OTHER = True  # Force Domain Propagation to be lumped with "Other"
BORDER_THICKNESS = 1.0  # Black border thickness around each sub-bar

# Stage order for consistent stacking
STAGE_ORDER = [
    "Other",
    "InsertMergeDataDependencies",
    "Optim",
    "IndividualizeLowCost",
    "Vectorize",
    "VecOptim",
    "StatifyingIncrementalize",
    "Incrementalize",
    "FinalOptim",
    "FoldPadsNMasksIntoStorage",
    "PropagateDomainConditions",
    "AnalyseDeviceAssignment",
    "GroupDataflowRegions",
    "AnalyseStorageMethods",
    "ScheduleExecution",
    "AnalyseDonations",
    "MergeCopyAnalysis",
    "DLBackendConfig",
    "TensorStore",
    "Codegen",
]

STAGE_DISPLAY_NAMES = {
    "FinalOptim": "Optim (w/ Const Folding)",
    "GroupDataflowRegions": "Dataflow Fusion",
    "ScheduleExecution": "Schedule",
    "AnalyseDonations": "Find Donations",
    "StatifyingIncrementalize": "Static Tiling",
    "FoldPadsNMasksIntoStorage": "Elim. Pad/Mask",
    "PropagateDomainConditions": "Domain Propagation",
    "TensorStore": "Storage Setup",
    "Optim": "Optimizer",
    "Codegen": "Code Generation",
}

HATCHES = [
    "**",
    "/",
    "",
    "oo",
    "x",
    ".",
    "xx",
    "\\",
    "..",
    "O",
    "o",
    "|",
    "-",
    "+",
    "O",
    "*",
    "//",
    "\\\\",
    "||",
    "--",
    "++",
]


# --- DATA LOADING ---
def load_results_from_runs(results_path: Path) -> dict[int, dict]:
    """Load and aggregate results from multiple runs for each layer configuration."""
    results = {}

    for num_layers in NUM_LAYERS:
        layer_dir = results_path / f"layers{num_layers}"
        if not layer_dir.exists():
            print(f"Warning: No results directory found for layers{num_layers}")
            continue

        comp_times = []
        compilation_profiles = []

        # Load data from each run
        for run_idx in range(NUM_RUNS):
            run_dir = layer_dir / f"run_{run_idx:02d}"
            summary_path = run_dir / "summary_results.json"
            profile_path = run_dir / "compilation_profile.json"

            # Load compilation time
            if summary_path.exists():
                try:
                    with open(summary_path) as f:
                        data = json.load(f)
                    comp_times.append(data.get("comp_time", 0))
                except Exception as e:
                    print(f"Error loading summary for layers{num_layers} run {run_idx}: {e}")
                    comp_times.append(0)
            else:
                print(f"Warning: No summary found for layers{num_layers} run {run_idx}")
                comp_times.append(0)

            # Load compilation profile
            if profile_path.exists():
                try:
                    with open(profile_path) as f:
                        profile = json.load(f)
                    compilation_profiles.append(profile)
                except Exception as e:
                    print(f"Error loading profile for layers{num_layers} run {run_idx}: {e}")
                    compilation_profiles.append({})
            else:
                print(f"Warning: No compilation profile found for layers{num_layers} run {run_idx}")
                compilation_profiles.append({})

        # Calculate statistics
        if comp_times:
            avg_comp_time = sum(comp_times) / len(comp_times)
            std_comp_time = calculate_std(comp_times)
            min_comp_time = min(comp_times)
            max_comp_time = max(comp_times)

            # Average compilation profiles using optree
            avg_profile = average_compilation_profiles(compilation_profiles)

            results[num_layers] = {
                "name": f"layers{num_layers}",
                "num_layers": num_layers,
                "num_runs": NUM_RUNS,
                "comp_times": comp_times,
                "avg_comp_time": round(avg_comp_time, 2),
                "std_comp_time": round(std_comp_time, 2),
                "min_comp_time": round(min_comp_time, 2),
                "max_comp_time": round(max_comp_time, 2),
                "compilation_profile": avg_profile,
            }

    return results


def calculate_std(values: list[float]) -> float:
    """Calculate standard deviation of a list of values."""
    if len(values) <= 1:
        return 0.0

    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance**0.5


def average_compilation_profiles(profiles: list[dict]) -> dict:
    """Average compilation profiles across multiple runs using optree."""
    if not profiles:
        return {}

    if len(profiles) == 1:
        return profiles[0]

    # Use optree to flatten all profiles and get their structure
    # Get the structure from the first profile
    reference_profile = profiles[0]
    reference_structure = optree.tree_structure(reference_profile)

    # Flatten all profiles to get their leaves
    all_leaves = []
    for profile in profiles:
        leaves, _ = optree.tree_flatten(profile)
        all_leaves.append(leaves)

    # Average the leaves
    averaged_leaves = [
        sum(leaf_values) / len(leaf_values) for leaf_values in zip(*all_leaves, strict=True)
    ]

    # Reconstruct the averaged profile using the reference structure
    averaged_profile = optree.tree_unflatten(reference_structure, averaged_leaves)

    return averaged_profile


def organize_data(
    raw_data: dict[int, dict], metric: str = "avg_comp_time"
) -> tuple[list[float], list[float]]:
    # Build: num_layers -> (metric value, std_dev)
    results = {}

    # Initialize with None for all expected layer counts
    for num_layers in NUM_LAYERS:
        results[num_layers] = (None, None)

    # Fill in actual data
    for num_layers, data in raw_data.items():
        if num_layers in NUM_LAYERS:
            v = data.get(metric, None)
            std_v = data.get("std_comp_time", 0.0)  # Get standard deviation
            if v is not None:
                results[num_layers] = (v, std_v)

    # Check for missing data and convert to list format
    missing_data = []
    data_list = []
    std_list = []
    for num_layers in NUM_LAYERS:
        val, std_val = results.get(num_layers, (None, None))
        if val is None:
            missing_data.append(f"layers{num_layers}")
            val = 0.0
            std_val = 0.0
        data_list.append(val)
        std_list.append(std_val)

    # Print warning for missing data
    if missing_data:
        print(f"Warning: Missing data for {len(missing_data)} configurations:")
        for missing in missing_data:
            print(f"  - {missing}")

    return data_list, std_list


def load_compilation_profiles(results_path: Path) -> dict[int, dict]:
    """Load averaged compilation profiles for all layer configurations."""
    profiles = {}

    aggregated_results = load_results_from_runs(results_path)
    for num_layers in NUM_LAYERS:
        # Try to load from the aggregated results first
        if num_layers in aggregated_results:
            profiles[num_layers] = aggregated_results[num_layers].get("compilation_profile", {})
        else:
            profiles[num_layers] = {}
    return profiles


def load_individual_run_profiles(results_path: Path) -> dict[int, list[dict]]:
    """Load compilation profiles from individual runs for statistical analysis."""
    profiles = {}

    for num_layers in NUM_LAYERS:
        layer_dir = results_path / f"layers{num_layers}"
        run_profiles = []

        for run_idx in range(NUM_RUNS):
            run_dir = layer_dir / f"run_{run_idx:02d}"
            profile_path = run_dir / "compilation_profile.json"
            if profile_path.exists():
                try:
                    with open(profile_path) as f:
                        profile = json.load(f)
                    run_profiles.append(profile)
                except Exception as e:
                    print(f"Error loading profile for layers{num_layers} run {run_idx}: {e}")
            else:
                print(f"Warning: No compilation profile found for layers{num_layers} run {run_idx}")

        profiles[num_layers] = run_profiles

    return profiles


def flatten_profile(
    profile: dict, max_depth: int = MAX_DEPTH, current_depth: int = 0
) -> list[tuple[str, float, str, int]]:
    """
    Flatten a nested compilation profile into a list of (stage_name, time, parent, depth) tuples.
    Returns stages in order they should be stacked, avoiding double-counting.
    """
    flattened = []

    if current_depth >= max_depth:
        return flattened

    # Handle top-level stages (Frontend, Backend)
    if "Frontend" in profile and current_depth == 0:
        frontend = profile["Frontend"]
        # Don't add Frontend total - we'll add its substages instead
        # flattened.append(("Frontend", frontend.get("Total", 0), "", current_depth))

        # Add Frontend substages
        for key, value in frontend.items():
            if key != "Total" and isinstance(value, dict):
                # This is a nested stage with substages
                if "Total" in value and current_depth + 1 < max_depth:
                    # Add the substage total
                    flattened.append((key, value["Total"], "Frontend", current_depth + 1))
                    # Add substages if we haven't reached max depth
                    if current_depth + 2 < max_depth:
                        for subkey, subvalue in value.items():
                            if subkey != "Total" and isinstance(subvalue, (int, float)):
                                flattened.append(
                                    (f"{key}.{subkey}", subvalue, key, current_depth + 2)
                                )
                else:
                    # This is a flat stage or we've reached max depth
                    flattened.append((key, value, "Frontend", current_depth + 1))
            elif key != "Total" and isinstance(value, (int, float)):
                # This is a flat stage
                flattened.append((key, value, "Frontend", current_depth + 1))

    if "Backend" in profile and current_depth == 0:
        backend = profile["Backend"]
        # Don't add Backend total - we'll add its substages instead
        # flattened.append(("Backend", backend.get("Total", 0), "", current_depth))

        # Add Backend substages
        for key, value in backend.items():
            if key != "Total" and isinstance(value, (int, float)):
                flattened.append((key, value, "Backend", current_depth + 1))

    return flattened


def aggregate_small_stages(
    stage_data: dict[str, list], threshold_ms: float = SUM_TOGETHER_THRESHOLD
) -> dict[str, list]:
    """Aggregate stages with time < threshold into 'Other' category."""
    aggregated = {}
    other_data = [0.0] * len(NUM_LAYERS)

    for stage_name, times in stage_data.items():
        max_time = max(times) if times else 0.0

        # Check if this stage should be forced into "Other"
        force_to_other = False
        if FORCE_DOMAIN_PROP_TO_OTHER and stage_name == "PropagateDomainConditions":
            force_to_other = True

        if force_to_other or max_time < threshold_ms / 1000.0:  # Convert threshold to seconds
            # Add to "Other" category
            for i, time_val in enumerate(times):
                other_data[i] += time_val
        else:
            aggregated[stage_name] = times

    if any(other_data):
        aggregated["Other"] = other_data

    return aggregated


def aggregate_optim_stages(stage_data: dict[str, list]) -> dict[str, list]:
    """Aggregate all Optim-related stages into a single 'Optim' category."""
    if not SUM_OPTIMS:
        return stage_data

    aggregated = {}
    optim_data = [0.0] * len(NUM_LAYERS)

    # List of stages to aggregate into "Optim"
    optim_stages = [
        "Optim",
        "VecOptim",
        "FinalOptim",
        "Optim.DeadCodeElimination",
        "Optim.DuplicateCodeElimination",
        "Optim.AlgebraicOptimizer",
        "Optim.DomainReduction",
        "Optim.Statify",
        "VecOptim.DeadCodeElimination",
        "VecOptim.DuplicateCodeElimination",
        "VecOptim.AlgebraicOptimizer",
        "VecOptim.DomainReduction",
        "VecOptim.Statify",
        "FinalOptim.CleanUpBroadcastingOps",
        "FinalOptim.DeadCodeElimination",
        "FinalOptim.ConstantFolding",
        "FinalOptim.DuplicateCodeElimination",
        "FinalOptim.AlgebraicOptimizer",
        "FinalOptim.DomainReduction",
        "FinalOptim.Statify",
    ]

    for stage_name, times in stage_data.items():
        if stage_name in optim_stages:
            # Add to "Optim" category
            for i, time_val in enumerate(times):
                optim_data[i] += time_val
        else:
            aggregated[stage_name] = times

    if any(optim_data):
        aggregated["Optim"] = optim_data

    return aggregated


def plot_compilation_breakdown(profiles: dict[int, dict], out_pdf: str):
    """Plot compilation time breakdown using stacked bar charts."""
    font_size = 16
    plt.rcParams.update({"font.size": font_size})
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # Get all unique stage names across all profiles
    all_stages = set()
    for profile in profiles.values():
        flattened = flatten_profile(profile)
        for stage_name, _, _, _ in flattened:
            all_stages.add(stage_name)

    # Use the constant stage order and filter to only include stages that exist in the data
    stage_order = [s for s in STAGE_ORDER if s in all_stages]

    # Add any remaining stages (excluding those that will be in "Other")
    for stage in all_stages:
        if stage not in stage_order and stage not in ["Other"]:
            stage_order.append(stage)

    # Prepare data for plotting
    x = np.arange(len(NUM_LAYERS))
    stage_data = {stage: [] for stage in stage_order}

    for num_layers in NUM_LAYERS:
        profile = profiles.get(num_layers, {})
        flattened = flatten_profile(profile)

        # Initialize all stages with 0
        for stage in stage_order:
            stage_data[stage].append(0)

        # Fill in actual values
        for stage_name, time_ms, _, _ in flattened:
            if stage_name in stage_order:
                stage_data[stage_name][-1] = time_ms / 1000.0  # Convert to seconds

    # Aggregate small stages
    stage_data = aggregate_small_stages(stage_data, SUM_TOGETHER_THRESHOLD)

    # Aggregate Optim stages if enabled
    stage_data = aggregate_optim_stages(stage_data)

    # Update stage order to include "Other" at the bottom and handle Optim consolidation
    if "Other" in stage_data:
        stage_order = ["Other"] + [s for s in stage_order if s in stage_data and s != "Other"]
    else:
        stage_order = [s for s in stage_order if s in stage_data]

    # If Optim stages are consolidated, make sure "Optim" is in the right place
    if SUM_OPTIMS and "Optim" in stage_data:
        # Remove individual Optim stages from order and ensure "Optim" is positioned correctly
        stage_order = [
            s for s in stage_order if not s.startswith(("Optim.", "VecOptim", "FinalOptim"))
        ]
        if "Optim" not in stage_order:
            # Insert "Optim" after "InsertMergeDataDependencies" for logical grouping
            try:
                insert_idx = stage_order.index("InsertMergeDataDependencies") + 1
                stage_order.insert(insert_idx, "Optim")
            except ValueError:
                stage_order.insert(0, "Optim")

    # Create the stacked bar chart - wider and shorter for double column paper
    # Increase figure width slightly to accommodate legend on the right
    fig, ax = plt.subplots(figsize=(8, 4))

    bottom = np.zeros(len(NUM_LAYERS))

    bar_width = 0.4  # Reduced from 0.5

    for i, stage in enumerate(stage_order):
        if stage in stage_data and any(stage_data[stage]):  # Only plot stages with data
            color = COLORS[i % len(COLORS)]
            hatch = HATCHES[i % len(HATCHES)]

            ax.bar(
                x,
                stage_data[stage],
                bottom=bottom,
                label=STAGE_DISPLAY_NAMES.get(stage, stage),
                color=color,
                alpha=0.9,
                width=bar_width,
                edgecolor="black",
                linewidth=BORDER_THICKNESS,
                hatch=hatch,
            )
            bottom += np.array(stage_data[stage])

    # Customize the plot
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Compilation Time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(NUM_LAYERS)

    # Set x-axis limits to center the bars nicely
    # ax.set_xlim(-0.2, len(NUM_LAYERS) - 0.8)

    # Create custom legend with better hatch visibility
    legend_elements = []
    for i, stage in enumerate(stage_order):
        if stage in stage_data and any(stage_data[stage]):
            color = COLORS[i % len(COLORS)]
            hatch = HATCHES[i % len(HATCHES)]
            label = STAGE_DISPLAY_NAMES.get(stage, stage)

            # Create a larger rectangle for better hatch visibility
            legend_elements.append(
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=BORDER_THICKNESS,
                    hatch=hatch,
                    alpha=0.9,
                    label=label,
                )
            )

    # Add custom legend with larger handles - place outside plot area on the right
    legend = ax.legend(
        handles=list(reversed(legend_elements)),
        loc="center right",
        fontsize=14,
        # framealpha=0.9,
        frameon=False,
        bbox_to_anchor=(1.5, 0.5),
    )

    # Add grid
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    # Adjust the right margin to make room for the legend
    plt.subplots_adjust(right=0.75)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_compilation_breakdown_results(
    results_path: str = DEFAULT_RESULTS_PATH, plots_path: str = DEFAULT_PLOTS_PATH
):
    """
    Plot the compilation time breakdown by stage.

    Args:
        results_path (str, optional): Path to the results directory. Defaults to "./results/".
        plots_path (str, optional): Path to the plots directory. Defaults to "./plots/".
    """
    results_path = Path(results_path) / LLAMA32_DECODE_DIR / COMPILE_TIME_SCALING_DIR
    profiles = load_compilation_profiles(results_path)

    plot_dir = Path(plots_path) / LLAMA32_DECODE_DIR / COMPILE_TIME_SCALING_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)

    out_pdf = plot_dir / "compilation_breakdown_multiple.pdf"
    plot_compilation_breakdown(profiles, out_pdf)
    print(f"Saved: {out_pdf}")


def plot_all_results(
    results_path: str = DEFAULT_RESULTS_PATH, plots_path: str = DEFAULT_PLOTS_PATH
):
    """Plot all results including the new run statistics."""
    plot_compilation_breakdown_results(results_path, plots_path)


if __name__ == "__main__":
    fire.Fire(plot_all_results)
