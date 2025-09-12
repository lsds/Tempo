from pathlib import Path

import fire

from repro.data_loading import DEFAULT_RESULTS_PATH
from repro.launch_lib import launch_par
from repro.sec7_2_llama32_decode.shared import LLAMA32_DECODE_DIR, get_prompts, run_bench

""" Run the time per token decoding experiments from Figures 9 and 10.
"""

BASE_PATH = "./results"
TPT_DIR = "tpt"

MAX_BENCH_TIME_SECS = 60


BATCH_SIZES = [4, 16]
SEQ_LENS = [2048, 4096, 8192, 16384, 32768, 65536]

# NOTE: as obtained from the block size microbenchmark
BS_TO_STATIFY_BLOCK_SIZE = {4: 8192, 16: 4096}


def generate_configs(base_results_path: Path) -> list[dict]:
    """Create configurations for all benchmarks to run."""
    configs = []

    # Test both causal attention and windowed with size 512
    attention_configs = [
        ("causal", 0),
        ("window", 1024),
    ]

    systems = [
        "jax",
        "torch",
        "tempo",
    ]

    for batch_size in BATCH_SIZES:
        for seq_len in SEQ_LENS:
            tile_size = BS_TO_STATIFY_BLOCK_SIZE[batch_size]
            prompts = get_prompts(batch_size=batch_size, max_prompt_len=16)
            for attn_type, window_size in attention_configs:
                for sys in systems:
                    name = f"{sys}_seq{seq_len}_attn{attn_type}_win{window_size}_bs{batch_size}"

                    base_cfg = {
                        "runner": run_bench,
                        "name": name,
                        "results_path": str(base_results_path / name),
                        "use_caching_allocators": True,
                        "attn_type": attn_type,
                        "window_size": window_size,
                        "framework_name": sys,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "checkpoint_dir": "~/.llama/checkpoints/Llama3.2-3B/",
                        "dev": "cuda",
                        "prompts": prompts,
                        "temperature": 0.0,  # NOTE: top-p sampling
                        "statify_block_size": tile_size,
                        "max_bench_time_secs": MAX_BENCH_TIME_SECS,
                        "is_jax": sys == "tempo" or sys == "jax",
                    }
                    configs.append(base_cfg)

    return configs


def main_all_cfgs(
    gpus: str = "0,1,2,3", phbgpu: int = None, results_path: str = DEFAULT_RESULTS_PATH
) -> None:
    """
    Usage Example:
    python repro/sec7_2_lamma_decode/run_measure_tpt.py --gpus "0,1,2,3"

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
    # Create results directory
    results_path = Path(results_path) / LLAMA32_DECODE_DIR / TPT_DIR
    results_path.mkdir(parents=True, exist_ok=True)

    # Generate configs
    configs = generate_configs(results_path)
    print(f"Generated {len(configs)} configs")

    # Launch experiments in parallel
    launch_par(configs, visible_gpus=visible_gpus, timeout_minutes=480 * 4)


if __name__ == "__main__":
    fire.Fire(main_all_cfgs)
