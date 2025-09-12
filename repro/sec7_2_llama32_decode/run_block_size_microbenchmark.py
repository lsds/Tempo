from pathlib import Path

import fire

from repro.data_loading import DEFAULT_RESULTS_PATH
from repro.launch_lib import launch_par
from repro.sec7_2_llama32_decode.shared import (
    BLOCK_SIZE_MICROBENCHMARK_DIR,
    LLAMA32_DECODE_DIR,
    get_prompts,
    run_bench,
)

""" Run the static tiling block size microbenchmark from Figure 11.
"""

BASE_RESULTS_PATH = "./results/"

MAX_BENCH_TIME_SECS = 60

BACKENDS = ["jax"]  # , "torch"]

# Only these batch sizes and block sizes are varied
BATCH_SIZES = [4, 16]
STATIFY_BLOCK_SIZES = [512, 1024, 2048, 4096, 8192, 16384]
SEQ_LEN = 16384


def generate_configs(base_results_path: str):
    configs = []
    for batch_size in BATCH_SIZES:
        prompts = get_prompts(batch_size=batch_size, max_prompt_len=32)
        for statify_block_size in STATIFY_BLOCK_SIZES:
            for backend in BACKENDS:
                name = f"bs{batch_size}_block{statify_block_size}_backend{backend}"
                cfg = {
                    "runner": run_bench,
                    "name": name,
                    "results_path": str(Path(base_results_path) / name),
                    "use_caching_allocators": True,
                    "attn_type": "causal",
                    "window_size": 0,
                    "temperature": 0.0,  # NOTE: top-p sampling
                    "framework_name": "tempo",
                    "batch_size": batch_size,
                    "seq_len": SEQ_LEN,
                    "checkpoint_dir": "~/.llama/checkpoints/Llama3.2-3B/",
                    "statify_block_size": statify_block_size,
                    # Always use these settings for this microbenchmark
                    "dev": "gpu",
                    "backend": backend,
                    "max_bench_time_secs": MAX_BENCH_TIME_SECS,
                    "is_jax": backend == "jax",
                    "prompts": prompts,
                    "is_block_microbenchmark": True,
                }
                configs.append(cfg)
    return configs


def main_all_cfgs(
    gpus: str = "0,1,2,3", phbgpu: int = None, results_path: str = DEFAULT_RESULTS_PATH
):
    """
    Usage Example:
    python repro/sec7_2_lamma_decode/run_block_size_microbenchmark.py --gpus "0,1,2,3"

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
    results_path = Path(results_path) / LLAMA32_DECODE_DIR / BLOCK_SIZE_MICROBENCHMARK_DIR
    results_path.mkdir(parents=True, exist_ok=True)
    configs = generate_configs(results_path)
    print(f"Generated {len(configs)} configs")
    launch_par(configs, visible_gpus=visible_gpus, timeout_minutes=240)


if __name__ == "__main__":
    fire.Fire(main_all_cfgs)
