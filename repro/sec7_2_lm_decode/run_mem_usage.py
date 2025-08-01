from pathlib import Path

import fire

from repro.data_loading import DEFAULT_RESULTS_PATH
from repro.launch_lib import launch_par
from repro.sec7_2_lm_decode.shared import (
    GPT2_DECODE_DIR,
    MEM_USAGE_DIR,
    run_bench,
)

""" Run the fine-grained memory usage experiments from Figure 12.
"""

BASE_RESULTS_PATH = "./results/"

MAX_BENCH_TIME_SECS = 60

# Only these batch sizes and block sizes are varied
BATCH_SIZE = 64

# NOTE: 512 as obtained from the original block size microbenchmark
STATIFY_BLOCK_SIZE = 512

GPT2_PARAMS = {
    "num_blocks": 12,
    "num_heads": 12,
    "embed_size": 768,
}
SEQ_LEN = 4096

FRAMEWORKS = ["jax", "torchnaive", "torch", "tempo"]

ATTN_CONFIGS = [
    ("causal", 0),
    ("window", 512),
]


def generate_configs(base_results_path: str):
    configs = []
    for framework in FRAMEWORKS:
        for attn_type, window_size in ATTN_CONFIGS:
            name = f"{framework}_attn{attn_type}_win{window_size}"
            cfg = {
                "runner": run_bench,
                "name": name,
                "results_path": str(Path(base_results_path) / name),
                "use_caching_allocators": False,
                "attn_type": attn_type,
                "window_size": window_size,
                "framework_name": framework,
                "batch_size": BATCH_SIZE,
                "seq_len": SEQ_LEN,
                **GPT2_PARAMS,
                "statify_block_size": STATIFY_BLOCK_SIZE,
                # Always use these settings for this microbenchmark
                "dev": "gpu",
                "backend": "jax",
                "max_bench_time_secs": MAX_BENCH_TIME_SECS,
                "monitor_fps": 5,
                "is_jax": framework == "tempo",
            }
            configs.append(cfg)
    return configs


def run_mem_usage(
    gpus: str = "0,1,2,3", phbgpu: int = None, results_path: str = DEFAULT_RESULTS_PATH
):
    """
    Usage Example:
    python repro/sec7_2_lm_decode/run_mem_usage.py --gpus "0,1,2,3"

    Args:
        gpus (str, optional): Comma-separated list of GPU IDs to use. Defaults to "0,1,2,3".
        phbgpu (int, optional): Specific GPU ID to use for sequential execution. Ignored.
        results_path (str, optional): Path to the results directory. Defaults to "./results/".
    """
    if isinstance(gpus, tuple):
        assert all(isinstance(gpu, int) for gpu in gpus), (
            f"gpus must be a string '0,1,2,3', got {type(gpus)}"
        )
        visible_gpus = gpus
    else:
        assert isinstance(gpus, str), f"gpus must be a string '0,1,2,3', got {type(gpus)}"
        visible_gpus = tuple(int(gpu) for gpu in gpus.split(","))
    results_path = Path(results_path) / GPT2_DECODE_DIR / MEM_USAGE_DIR
    results_path.mkdir(parents=True, exist_ok=True)
    configs = generate_configs(results_path)
    print(f"Generated {len(configs)} configs")
    launch_par(configs, visible_gpus=visible_gpus)


if __name__ == "__main__":
    fire.Fire(run_mem_usage)
