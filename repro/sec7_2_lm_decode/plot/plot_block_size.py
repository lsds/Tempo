import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

from repro.data_loading import DEFAULT_PLOTS_PATH, DEFAULT_RESULTS_PATH
from repro.sec7_2_lm_decode.plot.plot_shared import gather_summary_results
from repro.sec7_2_lm_decode.shared import (
    BLOCK_SIZE_MICROBENCHMARK_DIR,
    GPT2_DECODE_DIR,
)

""" Plot the static tiling block size microbenchmark results from Figure 11.
"""

# --- CONFIG ---
METRIC = "avg_iter_time"
METRIC_DISPLAY_NAME = "Time Per Token (s)"
BATCH_SIZES = [16, 64]
BLOCK_SIZES = [128, 256, 512, 1024, 4096]
COLOR_MAP = {16: "cadetblue", 64: "indianred"}
HATCH_PATTERNS = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]


# --- DATA LOADING ---
def load_results(path):
    with open(path, "r") as f:
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
        except Exception:
            continue
        d["batch_size"] = bs
        d["block_size"] = block
        filtered.append(d)
    return filtered


def organize_data(raw_data, metric=METRIC):
    # Build: batch_size -> [metric for each block_size]
    data_matrix = {}
    for bs in BATCH_SIZES:
        data_matrix[bs] = {}
        for block in BLOCK_SIZES:
            data_matrix[bs][block] = None

    # Iterate through raw data once
    for d in raw_data:
        name = d["name"]
        # Extract batch_size and block_size from name format "bs16_block1024"
        try:
            if name.startswith("bs"):
                parts = name.split("_")
                if len(parts) >= 2 and parts[1].startswith("block"):
                    batch_size = int(parts[0][2:])  # Remove "bs" prefix
                    block_size = int(parts[1][5:])  # Remove "block" prefix

                    if batch_size in BATCH_SIZES and block_size in BLOCK_SIZES:
                        v = d.get(metric, None)
                        if v is not None and metric == "avg_iter_time":
                            v = v / 4096  # time per token
                        data_matrix[batch_size][block_size] = v
        except (ValueError, IndexError):
            # Skip entries that don't match the expected format
            continue

    # Check for missing data and convert to list format
    missing_data = []
    for bs in BATCH_SIZES:
        temp_list = []
        for block in BLOCK_SIZES:
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

    return data_matrix


# --- PLOTTING ---
def plot_block_size(data_matrix, out_pdf):
    plt.rcParams.update({"font.size": 11})
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(8, 2.5))
    x = np.arange(len(BLOCK_SIZES))
    width = 0.8 / len(BATCH_SIZES)
    for j, bs in enumerate(BATCH_SIZES):
        data = data_matrix[bs]
        color = COLOR_MAP.get(bs, "gray")
        hatch = HATCH_PATTERNS[j % len(HATCH_PATTERNS)]
        bars = ax.bar(
            x + j * width,
            data,
            width=width,
            color=color,
            hatch=hatch,
            label=f"Batch Size={bs}",
            alpha=0.99,
        )
        # OOM text
        for k, value in enumerate(data):
            if value == 0.0:
                x_pos = x[k] + j * width
                y_pos = ax.get_ylim()[1] * 0.03 if ax.get_ylim()[1] > 0 else 0.01
                ax.text(
                    x_pos,
                    y_pos,
                    "OOM",
                    rotation=90,
                    color="red",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )
    ax.set_xticks(x + width * (len(BATCH_SIZES) - 1) / 2)
    ax.set_xticklabels(BLOCK_SIZES)
    ax.set_xlabel("Static Block Size")
    ax.set_ylabel(METRIC_DISPLAY_NAME)
    ax.set_ylim(0, 0.08)
    ax.set_yticks([0, 0.02, 0.04, 0.06])
    ax.grid(True, alpha=0.7)
    ax.legend(loc="upper center", ncol=2)
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# --- MAIN ---
def plot_block_size_both(
    results_path: str = DEFAULT_RESULTS_PATH, plots_path: str = DEFAULT_PLOTS_PATH
):
    """
    Plot the block size microbenchmark results.

    Args:
        results_path (str, optional): Path to the results directory. Defaults to "./results/".
        plots_path (str, optional): Path to the plots directory. Defaults to "./plots/".
    """
    summary_results_path = Path(results_path) / GPT2_DECODE_DIR / BLOCK_SIZE_MICROBENCHMARK_DIR
    summary_results = gather_summary_results(summary_results_path)
    data_matrix = organize_data(summary_results)
    plot_dir = Path(plots_path) / GPT2_DECODE_DIR / BLOCK_SIZE_MICROBENCHMARK_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = plot_dir / "block_size_tpt.pdf"
    plot_block_size(data_matrix, out_pdf)
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    fire.Fire(plot_block_size_both)
