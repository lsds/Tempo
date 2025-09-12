from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

from repro.data_loading import DEFAULT_PLOTS_PATH, DEFAULT_RESULTS_PATH
from repro.sec7_2_llama32_decode.plot.plot_shared import gather_summary_results
from repro.sec7_2_llama32_decode.run_measure_tpt import BATCH_SIZES, SEQ_LENS, TPT_DIR
from repro.sec7_2_llama32_decode.shared import LLAMA32_DECODE_DIR

""" Plot the time per token decoding results from Figures 9 and 10.
"""

# --- CONFIG ---
FRAMEWORKS = ["torch", "jax", "tempo"]
FRAMEWORK_DISPLAY_NAMES = {
    "torch": "Torch",
    "jax": "JAX",
    "tempo": "Tempo",
}
FRAMEWORK_COLORS = {
    "torch": "darkorange",
    "jax": "mediumorchid",
    "tempo": "cadetblue",
}
METRIC = "avg_iter_time"
METRIC_DISPLAY_NAME = "MTBT (ms)"
HATCH_PATTERNS = ["/", "\\", "x", "o", "O", ".", "*"]

DRAW_SPEEDUPS = True
DRAW_LINES = False
# --- DATA LOADING ---

WINDOW_SIZE = 1024


def organize_data(raw_data, metric=METRIC):
    frameworks_found = set()
    organized = {}
    for dct in raw_data:
        name = dct["name"]
        # Parse framework, seq_len, batch_size from name
        # Handles both _trl_ and _torchprealloc_ etc.
        parts = name.split("_")
        # Find framework
        framework = None
        for fw in FRAMEWORKS:
            if fw in parts:
                framework = fw
                break
        if framework is None:
            continue  # skip unknown frameworks
        # Find seq_len
        seq_len = None
        for p in parts:
            if p.startswith("seq"):
                try:
                    seq_len = int(p[3:])
                except Exception:
                    ...
        # Find batch size
        bs = None
        for p in parts:
            if p.startswith("bs"):
                try:
                    bs = int(p[2:])
                except Exception:
                    ...
        # if bs != BATCH_SIZE or seq_len is None:
        #    continue
        frameworks_found.add(framework)
        val = dct.get(metric, None)
        if val is not None and metric == "avg_iter_time":
            val = (val / seq_len) * 1000  # time per token in milliseconds
        organized.setdefault(framework, {})[seq_len] = val
    # Always use all frameworks in FRAMEWORKS, even if not found in data
    frameworks = [fw for fw in FRAMEWORKS if fw in frameworks_found or fw in organized]
    # Build matrix: framework -> [val for each seq_len]
    data_matrix = {}
    for fw in frameworks:
        data_matrix[fw] = [organized.get(fw, {}).get(sl, 0.0) for sl in SEQ_LENS]
    return data_matrix, frameworks


# --- PLOTTING ---
def plot_tpt(data_matrix, frameworks, out_pdf, bs, attn_type):
    text_size = 14
    plt.rcParams.update({"font.size": text_size})
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(6, 2.5))
    x = np.arange(len(SEQ_LENS))
    width = 0.8 / len(frameworks)

    # Calculate speedups for each sequence length
    speedups = {}
    for k, seq_len in enumerate(SEQ_LENS):
        # Get valid values (non-zero) for this sequence length
        valid_data = [(fw, data_matrix[fw][k]) for fw in frameworks if data_matrix[fw][k] > 0.0]
        if len(valid_data) < 2:
            continue  # Need at least 2 systems to calculate speedup

        # Find fastest and slowest for this sequence length
        valid_data.sort(key=lambda x: x[1])  # Sort by time (ascending)
        fastest_fw, fastest_time = valid_data[0]
        slowest_fw, slowest_time = valid_data[-1]

        # Calculate speedup relative to slowest for all frameworks except the slowest
        if slowest_time > 0:
            framework_speedups = {}
            for fw, time in valid_data:
                if fw != slowest_fw:  # Skip the slowest framework
                    speedup = slowest_time / time
                    framework_speedups[fw] = speedup
            speedups[k] = framework_speedups

    # Plot bars
    for j, fw in enumerate(frameworks):
        data = data_matrix[fw]
        color = FRAMEWORK_COLORS.get(fw, "gray")
        hatch = HATCH_PATTERNS[j % len(HATCH_PATTERNS)]
        ax.bar(
            x + j * width,
            data,
            width=width,
            color=color,
            hatch=hatch,
            label=FRAMEWORK_DISPLAY_NAMES[fw],
            zorder=10,
            edgecolor="black",
        )

    if DRAW_LINES:
        # Add connecting lines between bars of the same framework across sequence lengths
        for j, fw in enumerate(frameworks):
            data = data_matrix[fw]
            color = FRAMEWORK_COLORS.get(fw, "gray")
            # Filter out zero values (OOM cases) for the line plot
            valid_indices = [k for k, val in enumerate(data) if val > 0.0]
            if len(valid_indices) > 1:
                # Get x positions for valid bars
                x_positions = x[valid_indices] + j * width
                y_values = [data[k] for k in valid_indices]
                # Plot line connecting the bars
                ax.plot(
                    x_positions,
                    y_values,
                    color=color,
                    linewidth=2,  # marker='o', markersize=4,
                    zorder=15,
                    alpha=1,
                )

    # Add speedup labels above each framework's bar (except the slowest)
    if DRAW_SPEEDUPS:
        for k, seq_len in enumerate(SEQ_LENS):
            if k in speedups:
                framework_speedups = speedups[k]
                for fw, speedup in framework_speedups.items():
                    # Find the index of this framework
                    fw_idx = frameworks.index(fw)
                    x_pos = x[k] + fw_idx * width + 0.01
                    y_pos = data_matrix[fw][k] * 1.1

                    # Format speedup label
                    # speedup_label = f"{round(speedup)}x"
                    speedup_label = f"{speedup:.1f}x"

                    if speedup_label != "1.0x":
                        ax.text(
                            x_pos,
                            y_pos,
                            speedup_label,
                            ha="center",
                            va="bottom",
                            fontsize=text_size,
                            fontweight="normal",
                            # No edge, white background
                            bbox=dict(
                                boxstyle="square,pad=-0.05", edgecolor="none", facecolor="white"
                            ),
                            zorder=10,
                            rotation=90,
                        )

    # OOM text
    for j, fw in enumerate(frameworks):
        data = data_matrix[fw]
        for k, value in enumerate(data):
            if value == 0.0:
                x_pos = x[k] + j * width
                y_pos = pow(10, 1)
                ax.text(
                    x_pos,
                    y_pos,
                    "OOM",
                    rotation=90,
                    color="black",
                    ha="center",
                    va="bottom",
                    fontweight="normal",
                    fontsize=text_size,
                    zorder=10,
                )
    ax.set_xticks(x + width * (len(frameworks) - 1) / 2)
    ax.set_xticklabels(SEQ_LENS)
    ax.set_xlabel("Number of Decoded Tokens", labelpad=1)
    ax.set_ylabel(METRIC_DISPLAY_NAME, labelpad=1)
    ax.set_ylim(pow(10, 1), pow(10, 3.2))

    # Log scale
    ax.set_yscale("log")
    # ax.set_yticks([0, 50, 100, 500])
    # ax.set_yticklabels(["0", "50", "100", "500"])
    yticklabels = ax.get_yticklabels()
    if yticklabels:
        yticklabels[0].set_visible(False)  # Hide the label
        yticklabels[1].set_visible(False)  # Hide the label

    # Grid on every subtick
    ax.grid(True, alpha=0.9, zorder=0, which="major", axis="y", linewidth=1)
    loc = "upper left"
    ncols = 3
    bbox_to_anchor = (0, 1)
    if bs == 16 and attn_type == "causal":
        loc = "upper right"
        ncols = 1
        bbox_to_anchor = (1, 1)
    ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, ncols=ncols, columnspacing=0.5)
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return ax


# Removed plot_legend function as legends are now included in the main plots


# --- MAIN ---
def plot_tpt_both(results_path: str = DEFAULT_RESULTS_PATH, plots_path: str = DEFAULT_PLOTS_PATH):
    """
    Plot the time per token results.

    Args:
        results_path (str, optional): Path to the results directory. Defaults to "./results/".
        plots_path (str, optional): Path to the plots directory. Defaults to "./plots/".
    """
    summary_results_path = Path(results_path) / LLAMA32_DECODE_DIR / TPT_DIR
    summary_results = gather_summary_results(summary_results_path)
    for BATCH_SIZE in BATCH_SIZES:
        try:
            # Full attention
            bs_summary_results = [d for d in summary_results if f"bs{BATCH_SIZE}" in d["name"]]

            # Load and organize
            causal_results = [d for d in bs_summary_results if "attncausal" in d["name"]]

            causal_data, frameworks = organize_data(causal_results)

            # Output directory for plots
            plot_dir = Path(plots_path) / LLAMA32_DECODE_DIR / TPT_DIR
            plot_dir.mkdir(parents=True, exist_ok=True)
            f_path = plot_dir / f"causal_tpt_bs{BATCH_SIZE}.pdf"
            # Plot causal attention
            plot_tpt(
                causal_data,
                frameworks,
                f_path,
                BATCH_SIZE,
                "causal",
            )

            window_results = [
                d for d in bs_summary_results if f"attnwindow_win{WINDOW_SIZE}" in d["name"]
            ]
            window_data, _ = organize_data(window_results)

            w_path = plot_dir / f"window_tpt_bs{BATCH_SIZE}.pdf"
            # Plot window attention
            plot_tpt(
                window_data,
                frameworks,
                w_path,
                BATCH_SIZE,
                "window",
            )
            print(f"Saved: {f_path}, {w_path}")
        except Exception as e:
            print(f"Error processing BATCH_SIZE {BATCH_SIZE}: {e}")
            continue


if __name__ == "__main__":
    fire.Fire(plot_tpt_both)
