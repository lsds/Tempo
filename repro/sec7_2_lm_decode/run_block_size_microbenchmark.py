from pathlib import Path

import fire

from repro.data_loading import DEFAULT_RESULTS_PATH
from repro.launch_lib import launch_par
from repro.sec7_2_lm_decode.shared import (
    BLOCK_SIZE_MICROBENCHMARK_DIR,
    GPT2_DECODE_DIR,
    run_bench,
)

""" Run the static tiling block size microbenchmark from Figure 11.
"""

BASE_RESULTS_PATH = "./results/"

MAX_BENCH_TIME_SECS = 60

# Only these batch sizes and block sizes are varied
BATCH_SIZES = [16, 64]
STATIFY_BLOCK_SIZES = [128, 256, 512, 1024, 4096]

GPT2_PARAMS = {
    "num_blocks": 12,
    "num_heads": 12,
    "embed_size": 768,
}
SEQ_LEN = 4096


def generate_configs(base_results_path: str):
    configs = []
    for batch_size in BATCH_SIZES:
        for statify_block_size in STATIFY_BLOCK_SIZES:
            name = f"bs{batch_size}_block{statify_block_size}"
            cfg = {
                "runner": run_bench,
                "name": name,
                "results_path": str(Path(base_results_path) / name),
                "use_caching_allocators": True,
                "attn_type": "causal",
                "window_size": 0,
                "framework_name": "tempo",
                "batch_size": batch_size,
                "seq_len": SEQ_LEN,
                **GPT2_PARAMS,
                "statify_block_size": statify_block_size,
                # Always use these settings for this microbenchmark
                "dev": "gpu",
                "backend": "jax",
                "max_bench_time_secs": MAX_BENCH_TIME_SECS,
                "is_jax": True,
            }
            configs.append(cfg)
    return configs


def main_all_cfgs(
    gpus: str = "0,1,2,3", phbgpu: int = None, results_path: str = DEFAULT_RESULTS_PATH
):
    """
    Usage Example:
    python repro/sec7_2_lm_decode/run_block_size_microbenchmark.py --gpus "0,1,2,3"

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
    results_path = Path(results_path) / GPT2_DECODE_DIR / BLOCK_SIZE_MICROBENCHMARK_DIR
    results_path.mkdir(parents=True, exist_ok=True)
    configs = generate_configs(results_path)
    print(f"Generated {len(configs)} configs")
    launch_par(configs, visible_gpus=visible_gpus)


if __name__ == "__main__":
    fire.Fire(main_all_cfgs)
