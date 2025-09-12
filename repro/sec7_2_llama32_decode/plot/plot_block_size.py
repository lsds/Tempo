import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

from repro.data_loading import DEFAULT_PLOTS_PATH, DEFAULT_RESULTS_PATH
from repro.sec7_2_llama32_decode.plot.plot_shared import gather_summary_results
from repro.sec7_2_llama32_decode.run_block_size_microbenchmark import (
    BACKENDS,
    BATCH_SIZES,
    SEQ_LEN,
    STATIFY_BLOCK_SIZES,
)
from repro.sec7_2_llama32_decode.shared import (
    BLOCK_SIZE_MICROBENCHMARK_DIR,
    LLAMA32_DECODE_DIR,
)

""" Plot the static tiling block size microbenchmark results from Figure 11.
"""

# --- CONFIG ---
STACK = "horizontal"  # "horizontal" or "vertical"
METRICS = ["avg_iter_time", "mean_gpu_util"]
METRIC_DISPLAY_NAMES = {"avg_iter_time": "MTBT (ms)", "mean_gpu_util": "GPU Util. (%)"}
COLOR_MAP = {4: "cadetblue", 8: "olivedrab", 16: "indianred", 32: "indianred"}
BS_MARKER = {4: "o", 8: "s", 16: "D", 32: "s"}
BS_LINE_STYLE = {4: "-", 8: "-.", 16: "--", 32: "--"}


def format_block_size_labels(block_sizes):
    """Format block sizes as 2^x format using LaTeX formatting for larger superscript."""
    labels = []
    for size in block_sizes:
        # Find the power of 2
        power = 0
        temp_size = size
        while temp_size > 1:
            temp_size //= 2
            power += 1

        # Use LaTeX formatting for proper superscript sizing
        labels.append(f"$2^{{{power}}}$")
    return labels


# --- DATA LOADING ---
def load_results(path):
    with open(path) as f:
        all_data = json.load(f)
    # Only keep 'tempo' framework, parse batch size and block size from name
    filtered = []
    for d in all_data:
        name = d["name"]
        if not name.startswith("bs"):  # skip any non-matching
            continue
        try:
            parts = name.split("_")
            bs = int(parts[0][2:])
            block = int(parts[1][5:])
            backend = parts[2][7:]
        except Exception:
            continue
        d["batch_size"] = bs
        d["block_size"] = block
        d["backend"] = backend
        print(f"Loaded {d}")
        filtered.append(d)
    return filtered


def organize_data(raw_data, metric="avg_iter_time") -> dict[str, dict[int, dict[int, float]]]:
    # Build: batch_size -> [metric for each block_size]
    results = {}
    for backend in BACKENDS:
        data_matrix = {}
        for bs in BATCH_SIZES:
            data_matrix[bs] = {}
            for block in STATIFY_BLOCK_SIZES:
                data_matrix[bs][block] = None

        # Iterate through raw data once
        for d in raw_data:
            name = d["name"]
            if backend not in name:
                continue
            # Extract batch_size and block_size from name format "bs16_block1024"
            try:
                if name.startswith("bs"):
                    parts = name.split("_")
                    if len(parts) >= 2 and parts[1].startswith("block"):
                        batch_size = int(parts[0][2:])  # Remove "bs" prefix
                        block_size = int(parts[1][5:])  # Remove "block" prefix

                        if batch_size in BATCH_SIZES and block_size in STATIFY_BLOCK_SIZES:
                            v = d.get(metric, None)
                            if v is not None and metric == "avg_iter_time":
                                v = (v / SEQ_LEN) * 1000  # time per token in milliseconds
                            data_matrix[batch_size][block_size] = v
            except (ValueError, IndexError):
                # Skip entries that don't match the expected format
                continue

        # Check for missing data and convert to list format
        missing_data = []
        for bs in BATCH_SIZES:
            temp_list = []
            for block in STATIFY_BLOCK_SIZES:
                val = data_matrix[bs].get(block, None)
                if val is None:
                    missing_data.append(f"bs{bs}_block{block}")
                    val = 0.0
                temp_list.append(val)
            data_matrix[bs] = temp_list

        # Print warning for missing data
        if missing_data:
            print(f"Warning: Missing data for {len(missing_data)} configurations:")
            for missing in missing_data:
                print(f"  - {missing}")

        results[backend] = data_matrix

    return results


def plot_block_size_combined(latency_data, gpu_util_data, out_pdf):
    """Plot both latency and GPU utilization in two subplots with configurable stacking direction."""
    font_size = 24
    plt.rcParams.update({"font.size": font_size})
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # Create figure with subplots based on STACK configuration
    if STACK == "vertical":
        font_size = 12
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(6, 4), sharex=True, gridspec_kw={"hspace": 0.1, "wspace": 0.1}
        )
    else:  # horizontal
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={"wspace": 0.3})

    x = np.arange(len(STATIFY_BLOCK_SIZES))
    bss = list(BATCH_SIZES)

    # Plot iteration time on first subplot (ax1)
    for j, bs in enumerate(bss):
        data = latency_data[bs]
        color = COLOR_MAP.get(bs, "gray")
        ax1.plot(
            x,
            data,
            marker=BS_MARKER[bs],
            linestyle=BS_LINE_STYLE[bs],
            color=color,
            label=f"B={bs}",
            linewidth=2,
            markersize=8,
            zorder=10,
        )

    # Plot GPU utilization on second subplot (ax2)
    for j, bs in enumerate(bss):
        data = gpu_util_data[bs]
        color = COLOR_MAP.get(bs, "gray")
        ax2.plot(
            x,
            data,
            marker=BS_MARKER[bs],
            linestyle=BS_LINE_STYLE[bs],
            color=color,
            label=f"BS={bs}",
            linewidth=2,
            markersize=8,
            zorder=10,
        )

    # Configure first subplot (iteration time)
    ax1.set_ylabel(METRIC_DISPLAY_NAMES["avg_iter_time"], fontsize=font_size)
    ax1.grid(True, alpha=0.7, zorder=1)
    # if STACK == "horizontal":
    #    ax1.set_yticks([0, 50, 100, 150])
    # else:
    #    ax1.set_yticks([0, 50, 100, 150, 200, 250])

    ax1.set_ylim(0, 250)
    if STACK == "vertical":
        ax1.legend(loc="upper center", ncol=2, fontsize=font_size, frameon=True)
        ax1.set_yticks([0, 100, 200])
        ax1.set_yticklabels([0, 100, 200], fontsize=font_size)
        ax2.set_yticks([0, 25, 50, 75, 100])
        ax2.set_yticklabels([0, 25, 50, 75, 100], fontsize=font_size)
    else:
        ax1.legend(
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.54, 1.05),
            fontsize=font_size - 2,
            columnspacing=0.5,
            frameon=True,
        )
        ax1.set_yticks(
            [
                0,
                50,
                100,
                150,
                200,
            ]
        )
        ax1.set_yticklabels(
            [
                0,
                50,
                100,
                150,
                200,
            ],
            fontsize=font_size,
        )
        ax2.set_yticks([0, 25, 50, 75, 100])
        ax2.set_yticklabels([0, 25, 50, 75, 100], fontsize=font_size)
    # Configure second subplot (GPU utilization)
    ax2.set_ylabel(METRIC_DISPLAY_NAMES["mean_gpu_util"], labelpad=1, fontsize=font_size)
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.7, zorder=1)

    axs = [ax2] if STACK == "vertical" else [ax1, ax2]
    # Configure x-axis for both subplots
    for ax in axs:
        ax.set_xticks(x)
        if STACK == "vertical":
            ax.set_xticklabels(STATIFY_BLOCK_SIZES, fontsize=font_size)
        else:
            ax.set_xticklabels(format_block_size_labels(STATIFY_BLOCK_SIZES), fontsize=font_size)
        ax.set_xlabel("Static Tile Size", labelpad=1, fontsize=font_size)

    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# --- MAIN ---
def plot_block_size_both(
    results_path: str = DEFAULT_RESULTS_PATH, plots_path: str = DEFAULT_PLOTS_PATH
):
    """
    Plot the block size microbenchmark results for both metrics.

    Args:
        results_path (str, optional): Path to the results directory. Defaults to "./results/".
        plots_path (str, optional): Path to the plots directory. Defaults to "./plots/".
    """
    summary_results_path = Path(results_path) / LLAMA32_DECODE_DIR / BLOCK_SIZE_MICROBENCHMARK_DIR
    summary_results = gather_summary_results(summary_results_path)
    plot_dir = Path(plots_path) / LLAMA32_DECODE_DIR / BLOCK_SIZE_MICROBENCHMARK_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load data for both metrics
    latency_data = organize_data(summary_results, "avg_iter_time")
    gpu_util_data = organize_data(summary_results, "mean_gpu_util")

    # Create combined plots for each backend
    for backend in BACKENDS:
        out_pdf = plot_dir / "block_size_lines_vert.pdf"
        plot_block_size_combined(latency_data[backend], gpu_util_data[backend], out_pdf)
        print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    fire.Fire(plot_block_size_both)
