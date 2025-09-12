from pathlib import Path

import fire

from repro.data_loading import DEFAULT_RESULTS_PATH
from repro.launch_lib import launch_par
from repro.sec7_2_llama32_decode.shared import (
    LLAMA32_DECODE_DIR,
    MEM_USAGE_DIR,
    get_prompts,
    run_bench,
)

""" Run the fine-grained memory usage experiments from Figure 12.
"""

BASE_RESULTS_PATH = "./results/"

MAX_BENCH_TIME_SECS = 60

# NOTE: We increase the batch size and lower the SEQ_LEN to speed up the experiments
# As using non-caching allocators will lead to much slower runtime.
# Bigger batch size leads to same behaviour, but faster as it requires fewer allocations

# Only these batch sizes and block sizes are varied
BATCH_SIZE = 64

# NOTE: 1024 as obtained from the original block size microbenchmark
STATIFY_BLOCK_SIZE = 256

SEQ_LEN = 2048

FRAMEWORKS = ["jax", "torch", "tempo"]

ATTN_CONFIGS = [
    ("causal", 0),
    ("window", 256),
]


def generate_configs(base_results_path: str, mode: str):
    configs = []

    base_cfg = {
        "runner": run_bench,
        "use_caching_allocators": False,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "statify_block_size": STATIFY_BLOCK_SIZE,
        "checkpoint_dir": "~/.llama/checkpoints/Llama3.2-3B/",
        # Always use these settings for this microbenchmark
        "dev": "gpu",
        "backend": "jax",
        "max_bench_time_secs": MAX_BENCH_TIME_SECS,
        "monitor_fps": 1,
        "fine_grained_mem": mode == "fine_grained",
        "temperature": 0.6,  # NOTE: top-p sampling
        "top_p": 0.9,
        "max_prompt_len": 32,
    }

    prompts = get_prompts(**base_cfg)
    base_cfg["prompts"] = prompts
    for framework in FRAMEWORKS:
        for attn_type, window_size in ATTN_CONFIGS:
            name = f"{framework}_attn{attn_type}_win{window_size}"
            cfg = {
                "name": name,
                "results_path": str(Path(base_results_path) / name),
                "attn_type": attn_type,
                "window_size": window_size,
                "framework_name": framework,
                "is_jax": framework == "tempo",
                **base_cfg,
            }
            configs.append(cfg)
    return configs


def run_mem_usage(
    gpus: str = "0,1,2,3",
    phbgpu: int = None,
    results_path: str = DEFAULT_RESULTS_PATH,
    mode: str = "fine_grained",
):
    """
    Usage Example:
    python repro/sec7_2_lamma_decode/run_mem_usage.py --gpus "0,1,2,3"

    Args:
        gpus (str, optional): Comma-separated list of GPU IDs to use. Defaults to "0,1,2,3".
        phbgpu (int, optional): Specific GPU ID to use for sequential execution. Ignored.
        results_path (str, optional): Path to the results directory. Defaults to "./results/".
        mode (str, optional): "fast" or "fine_grained". Defaults to "fine_grained".
        "fast" disables preallocation, but uses caching allocators, leading to much faster runtime (minutes vs hours).
        "fine_grained" additionally disables caching allocators, leading to much slower runtime (hours vs days),
        but more accurate memory usage tracking. To get plots exactly like in the paper, use "fine_grained",
        but beware this may take up to 10 hours.
    """
    if isinstance(gpus, tuple):
        assert all(isinstance(gpu, int) for gpu in gpus), (
            f"gpus must be a string '0,1,2,3', got {type(gpus)}"
        )
        visible_gpus = gpus
    else:
        assert isinstance(gpus, str), f"gpus must be a string '0,1,2,3', got {type(gpus)}"
        visible_gpus = tuple(int(gpu) for gpu in gpus.split(","))

    assert mode.lower().strip() in ["fast", "fine_grained"], (
        f"Invalid mode: {mode}. Valid modes are 'fast' and 'fine_grained'."
    )

    results_path = Path(results_path) / LLAMA32_DECODE_DIR / MEM_USAGE_DIR
    results_path.mkdir(parents=True, exist_ok=True)
    configs = generate_configs(results_path, mode)
    print(f"Generated {len(configs)} configs")
    launch_par(configs, visible_gpus=visible_gpus, timeout_minutes=4 * 60)


if __name__ == "__main__":
    fire.Fire(run_mem_usage)
