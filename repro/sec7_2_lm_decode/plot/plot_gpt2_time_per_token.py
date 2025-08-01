from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

from repro.data_loading import DEFAULT_PLOTS_PATH, DEFAULT_RESULTS_PATH
from repro.sec7_2_lm_decode.plot.plot_shared import gather_summary_results
from repro.sec7_2_lm_decode.run_measure_tpt import TPT_DIR
from repro.sec7_2_lm_decode.shared import GPT2_DECODE_DIR

""" Plot the time per token decoding results from Figures 9 and 10.
"""

# --- CONFIG ---
FRAMEWORKS = ["jax", "torchnaive", "torch", "tempo"]
FRAMEWORK_DISPLAY_NAMES = {
    "torchnaive": "Torch (Naive)",
    "torch": "Torch (Optimized)",
    "jax": "JAX",
    "tempo": "Tempo (Ours)",
}
FRAMEWORK_COLORS = {
    "torch": "darkorange",
    "torchnaive": "red",
    "jax": "royalblue",
    "tempo": "cadetblue",
}
METRIC = "avg_iter_time"
METRIC_DISPLAY_NAME = "Time Per Token (s)"
BATCH_SIZE = 64
HATCH_PATTERNS = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]


# --- DATA LOADING ---


def organize_data(raw_data, metric=METRIC):
    seq_lens = set()
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
                    pass
        # Find batch size
        bs = None
        for p in parts:
            if p.startswith("bs"):
                try:
                    bs = int(p[2:])
                except Exception:
                    pass
        if bs != BATCH_SIZE or seq_len is None:
            continue
        seq_lens.add(seq_len)
        frameworks_found.add(framework)
        val = dct.get(metric, None)
        if val is not None and metric == "avg_iter_time":
            val = val / seq_len  # time per token
        organized.setdefault(framework, {})[seq_len] = val
    seq_lens = sorted(seq_lens)
    # Always use all frameworks in FRAMEWORKS, even if not found in data
    frameworks = [fw for fw in FRAMEWORKS if fw in frameworks_found or fw in organized]
    # Build matrix: framework -> [val for each seq_len]
    data_matrix = {}
    for fw in frameworks:
        data_matrix[fw] = [organized.get(fw, {}).get(sl, 0.0) for sl in seq_lens]
    return data_matrix, seq_lens, frameworks


# --- PLOTTING ---
def plot_tpt(data_matrix, seq_lens, frameworks, out_pdf):
    plt.rcParams.update({"font.size": 11})
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(8, 2.5))
    x = np.arange(len(seq_lens))
    width = 0.8 / len(frameworks)
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
            alpha=0.99,
        )
    # OOM text
    for j, fw in enumerate(frameworks):
        data = data_matrix[fw]
        for k, value in enumerate(data):
            if value == 0.0:
                x_pos = x[k] + j * width
                y_pos = 0.0
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
    ax.set_xticks(x + width * (len(frameworks) - 1) / 2)
    ax.set_xticklabels(seq_lens)
    ax.set_xlabel("Number of Decoded Tokens")
    ax.set_ylabel(METRIC_DISPLAY_NAME)
    ax.set_ylim(0, 0.08)
    ax.set_yticks([0, 0.02, 0.04, 0.06])
    ax.grid(True, alpha=0.7)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1))
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
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
    # Full attention
    summary_results_path = Path(results_path) / GPT2_DECODE_DIR / TPT_DIR
    summary_results = gather_summary_results(summary_results_path)

    # Load and organize
    causal_results = [d for d in summary_results if "attncausal" in d["name"]]
    window_results = [d for d in summary_results if "attnwindow" in d["name"]]
    causal_data, causal_seq_lens, frameworks = organize_data(causal_results)
    window_data, window_seq_lens, _ = organize_data(window_results)

    # Output directory for plots
    plot_dir = Path(plots_path) / GPT2_DECODE_DIR / TPT_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)
    f_path = plot_dir / "causal_tpt.pdf"
    w_path = plot_dir / "window_tpt.pdf"
    # Plot causal attention
    plot_tpt(
        causal_data,
        causal_seq_lens,
        frameworks,
        f_path,
    )
    # Plot window attention
    plot_tpt(
        window_data,
        window_seq_lens,
        frameworks,
        w_path,
    )
    print(f"Saved: {f_path}, {w_path}")


if __name__ == "__main__":
    fire.Fire(plot_tpt_both)
