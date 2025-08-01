from pathlib import Path

import fire

from repro.data_loading import DEFAULT_RESULTS_PATH
from repro.launch_lib import launch_par
from repro.sec7_2_lm_decode.shared import GPT2_DECODE_DIR, run_bench

""" Run the time per token decoding experiments from Figures 9 and 10.
"""

BASE_PATH = "./results"
TPT_DIR = "tpt"

MAX_BENCH_TIME_SECS = 60

GPT2_SMALL_PARAMS = {
    "num_blocks": 12,
    "num_heads": 12,
    "embed_size": 768,
}
BATCH_SIZE = 64

SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384]

# NOTE: 512 as obtained from the block size microbenchmark
STATIFY_BLOCK_SIZE = 1024


def generate_configs(base_results_path: Path) -> list[dict]:
    """Create configurations for all benchmarks to run."""
    configs = []

    # Test both causal attention and windowed with size 512
    attention_configs = [
        ("causal", 0),
        ("window", 512),
    ]

    systems = [
        "torch",
        "torchnaive",
        "jax",
        "tempo",
    ]

    for seq_len in SEQ_LENS:
        for attn_type, window_size in attention_configs:
            for sys in systems:
                name = f"{sys}_seq{seq_len}_attn{attn_type}_win{window_size}_bs{BATCH_SIZE}"

                base_cfg = {
                    "runner": run_bench,
                    "name": name,
                    "results_path": str(base_results_path / name),
                    "use_caching_allocators": True,
                    "attn_type": attn_type,
                    "window_size": window_size,
                    "framework_name": sys,
                    "batch_size": BATCH_SIZE,
                    "seq_len": seq_len,
                    **GPT2_SMALL_PARAMS,
                    "max_bench_time_secs": MAX_BENCH_TIME_SECS,
                    "is_jax": sys == "tempo",
                }
                configs.append(base_cfg)

    return configs


def main_all_cfgs(
    gpus: str = "0,1,2,3", phbgpu: int = None, results_path: str = DEFAULT_RESULTS_PATH
) -> None:
    """
    Usage Example:
    python repro/sec7_2_lm_decode/run_measure_tpt.py --gpus "0,1,2,3"

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
    results_path = Path(results_path) / GPT2_DECODE_DIR / TPT_DIR
    results_path.mkdir(parents=True, exist_ok=True)

    # Generate configs
    configs = generate_configs(results_path)
    print(f"Generated {len(configs)} configs")

    # Launch experiments in parallel
    launch_par(configs, visible_gpus=visible_gpus)


if __name__ == "__main__":
    fire.Fire(main_all_cfgs)
