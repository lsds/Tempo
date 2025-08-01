import glob
import os
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from repro.data_loading import DEFAULT_PLOTS_PATH, DEFAULT_RESULTS_PATH
from repro.sec7_2_lm_decode.shared import (
    GPT2_DECODE_DIR,
    MEM_USAGE_DIR,
    MONITOR_CSV_FILE,
)

""" Plot the fine-grained memory usage results from Figure 12.
"""

# --- CONFIG ---
SYS = ["jax", "torchnaive", "torch", "tempo"]
SYS_NAMES = {
    "torchnaive": "Torch (Naive)",
    "torch": "Torch (Optimized)",
    "jax": "JAX",
    "tempo": "Tempo",
}
SYS_COLORS = {
    ("torch", "causal"): "darkorange",
    ("torch", "window"): "orange",
    ("torchnaive", "causal"): "red",
    ("torchnaive", "window"): "lightcoral",
    ("jax", "causal"): "royalblue",
    ("jax", "window"): "lightblue",
    ("tempo", "causal"): "cadetblue",
    ("tempo", "window"): "mediumspringgreen",
}
SYS_MARKERS = {
    ("torch", "causal"): "+",
    ("torch", "window"): "1",
    ("torchnaive", "causal"): "x",
    ("torchnaive", "window"): "2",
    ("jax", "causal"): "^",
    ("jax", "window"): "v",
    ("tempo", "causal"): "*",
    ("tempo", "window"): ".",
}

# Base framework names (without attention type suffix)
BASE_SYS = ["jax", "torchnaive", "torch"]

# Alignment percentages for each framework (can be tuned as needed to align runtimes)
ALIGN_TIME_PERCENTS_PER_SYS = {
    ("jax", "causal"): 80,
    ("jax", "window"): 80,
    ("tempo", "causal"): 97,
    ("tempo", "window"): 67,
    ("torch", "causal"): 62,
    ("torch", "window"): 62,
    ("torchnaive", "causal"): 91,
    ("torchnaive", "window"): 91,
}

ATTN_TYPES = ["causal", "window"]


# --- DATA LOADING ---
def parse_bench_dir_name(name):
    # Example: jax_attncausal_win0, torch_attnwindow_win512
    parts = name.split("_")
    framework = None
    attn_type = None

    # Extract framework
    framework = parts[0]

    # Extract attention type and window size
    for p in parts:
        if p.startswith("attn"):
            attn_type = p[4:]

    return framework, attn_type


def find_monitor_csvs(base_path):
    pattern = str(Path(base_path) / "*" / MONITOR_CSV_FILE)
    files = glob.glob(pattern)
    results = []
    for f in files:
        bench_dir = os.path.basename(os.path.dirname(f))
        framework, attn_type = parse_bench_dir_name(bench_dir)
        if framework is not None:
            results.append((framework, attn_type, f))
    return results


def get_aligned_dfs(monitor_csvs, n_points=100):
    # monitor_csvs: list of (framework, attn_type, path)
    dfs = {}
    for framework, attn_type, path in monitor_csvs:
        df = pd.read_csv(path)
        # Use elapsed_ns as percent of completion
        df["elapsed_ns"] = df["elapsed_ns"] / df["elapsed_ns"].max() * 100

        align_percent = ALIGN_TIME_PERCENTS_PER_SYS.get((framework, attn_type), 100)
        df = df[df["elapsed_ns"] <= align_percent]
        # Normalize to 100%
        df["elapsed_ns"] = df["elapsed_ns"] / df["elapsed_ns"].max() * 100
        # Find the correct GPU mem util column
        gpu_mem_col = None
        for col in df.columns:
            if "gpu" in col and "mem_util" in col:
                gpu_mem_col = col
                break
        if gpu_mem_col is None:
            continue
        # Only keep elapsed_ns and gpu_mem_util
        df = df[["elapsed_ns", gpu_mem_col]].rename(columns={gpu_mem_col: "gpu_mem_util"})
        # Sparsify: downsample to n_points
        df = df.sort_values("elapsed_ns")
        if len(df) > n_points:
            idxs = np.linspace(0, len(df) - 1, n_points).astype(int)
            df = df.iloc[idxs]
        # Interpolate to common grid
        time_grid = np.linspace(0, 100, n_points)
        interp = np.interp(time_grid, df["elapsed_ns"], df["gpu_mem_util"])
        dfs[(framework, attn_type)] = pd.DataFrame(
            {"elapsed_ns": time_grid, "gpu_mem_util": interp}
        )
    return dfs


# --- PLOTTING ---
def plot_memory_usage_lines(dfs, out_pdf, mode="paper"):
    plt.rcParams.update({"font.size": 12})
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["lines.markersize"] = 8
    fig, ax = plt.subplots(figsize=(8, 3))

    for framework in SYS:
        for attn_type in ATTN_TYPES:
            if mode == "paper" and attn_type == "window" and framework != "tempo":
                continue

            # Check if data exists for this framework-attention type combination
            if (framework, attn_type) not in dfs:
                print(f"Warning: No data found for {framework} with {attn_type} attention")
                continue

            df = dfs[(framework, attn_type)]

            label = SYS_NAMES.get(framework, framework)

            # Determine the label to use
            if (mode == "paper" and framework == "tempo") or mode == "all":
                label += f" ({attn_type.capitalize()})"

            ax.plot(
                df["elapsed_ns"],
                df["gpu_mem_util"],
                label=label,
                color=SYS_COLORS[(framework, attn_type)],
                marker=SYS_MARKERS[(framework, attn_type)],
            )

    ax.set_xlabel("Decode Completion (%)")
    ax.set_ylabel("GPU Mem. Util. (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 99)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    plt.grid(True)
    plt.tight_layout()
    ax.yaxis.set_label_coords(-0.05, 0.5)
    plt.legend(fontsize=12, loc="upper right", ncol=2)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")


# --- MAIN ---
def plot_memory_usage_both(
    results_path: str = DEFAULT_RESULTS_PATH,
    plots_path: str = DEFAULT_PLOTS_PATH,
    mode: str = "paper",
):
    """Plot memory usage lines for all frameworks and attn types.

    Args:
        results_path (str, optional): Path to the results directory. Defaults to "./results/".
        plots_path (str, optional): Path to the plots directory. Defaults to "./plots/".
        mode (str, optional): Mode to plot. Defaults to "paper". "paper" skips plotting
            both window and causal attention types for frameworks other than tempo since they are identical.
            "all" plots both window and causal attention lines for all frameworks.

            #NOTE: If mode=="paper", then the labels for frameworks other than tempo
            do not include "Window" or "Causal".
    """

    if mode not in ["paper", "all"]:
        raise ValueError(f"Invalid mode: {mode}. Expected 'paper' or 'all'.")

    base_path = Path(results_path) / GPT2_DECODE_DIR / MEM_USAGE_DIR

    monitor_csvs = find_monitor_csvs(base_path)
    dfs = get_aligned_dfs(monitor_csvs)

    # Output directory for plots
    plot_dir = Path(plots_path) / GPT2_DECODE_DIR / MEM_USAGE_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = plot_dir / "memory_usage_lines_gpt2_decode.pdf"
    plot_memory_usage_lines(dfs, out_pdf, mode=mode)
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    fire.Fire(plot_memory_usage_both)
