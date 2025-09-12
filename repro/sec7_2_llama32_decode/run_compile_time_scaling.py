import json
import random
import time
from pathlib import Path

import fire

from repro.data_loading import DEFAULT_RESULTS_PATH
from repro.launch_lib import launch_par
from repro.sec7_2_llama32_decode.shared import (
    COMPILE_TIME_SCALING_DIR,
    LLAMA32_DECODE_DIR,
    SUMMARY_RESULTS_FILE,
    get_prompts,
)

""" Run the compile time scaling benchmark.
"""

BASE_RESULTS_PATH = "./results/"

MAX_BENCH_TIME_SECS = 60

BACKEND = "jax"

BATCH_SIZE = 4
STATIFY_BLOCK_SIZE = 512

SEQ_LEN = 1024

NUM_LAYERS = [4, 8, 16, 32, 64]
NUM_RUNS = 5  # Number of times to run each experiment


def run_bench_compile_time(**kwargs):
    """Run a single benchmark with the given runner class and configuration."""
    from repro.sec7_2_llama32_decode.impls.tempo_llama32 import TempoLlama32InferenceRunner

    runner_cls = TempoLlama32InferenceRunner

    name = kwargs["name"]
    run_dir = Path(kwargs["results_path"])
    run_idx = kwargs["run_idx"]

    # Create run-specific directory
    run_dir.mkdir(parents=True, exist_ok=True)

    # Update kwargs to use run-specific directory
    kwargs["results_path"] = str(run_dir)

    runner = runner_cls(**kwargs)

    # One warm-up call to trigger compilation
    start = time.perf_counter()
    runner.compile()
    end = time.perf_counter()
    comp_time = end - start

    # Save individual run results
    summary_results = {
        "name": name,
        "run_idx": run_idx,
        "comp_time": round(comp_time, 2),
    }

    summary_results_path = run_dir / SUMMARY_RESULTS_FILE
    with open(summary_results_path, "w") as f:
        json.dump(summary_results, f)


def generate_configs(base_results_path: str):
    configs = []
    prompts = get_prompts(batch_size=BATCH_SIZE, max_prompt_len=32)
    for num_layers in NUM_LAYERS:
        for run_idx in range(NUM_RUNS):
            name = f"layers{num_layers}_run_{run_idx:02d}"
            cfg = {
                "runner": run_bench_compile_time,
                "name": name,
                "results_path": str(
                    Path(base_results_path) / f"layers{num_layers}" / f"run_{run_idx:02d}"
                ),
                "use_caching_allocators": True,
                "attn_type": "causal",
                "window_size": 0,
                "temperature": 0.6,  # NOTE: top-p sampling
                "framework_name": "tempo",
                "batch_size": BATCH_SIZE,
                "seq_len": SEQ_LEN,
                "num_layers": num_layers,
                "checkpoint_dir": "~/.llama/checkpoints/Llama3.2-3B/",
                "statify_block_size": STATIFY_BLOCK_SIZE,
                # Always use these settings for this microbenchmark
                "dev": "gpu",
                "backend": BACKEND,
                "max_bench_time_secs": MAX_BENCH_TIME_SECS,
                "is_jax": BACKEND == "jax",
                "prompts": prompts,
                "compile_only": True,
                "run_idx": run_idx,
            }
            configs.append(cfg)
    return configs


def main_all_cfgs(
    gpus: str = "0,1,2,3", phbgpu: int = None, results_path: str = DEFAULT_RESULTS_PATH
):
    """
    Usage Example:
    python repro/sec7_2_llama32_decode/run_compile_time_scaling.py --gpus "0,1,2,3"

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
    results_path = Path(results_path) / LLAMA32_DECODE_DIR / COMPILE_TIME_SCALING_DIR
    results_path.mkdir(parents=True, exist_ok=True)
    configs = generate_configs(results_path)
    print(f"Generated {len(configs)} configs, each will run {NUM_RUNS} times")

    # NOTE: Shuffle configs to avoid last configurations having an advantage
    random.shuffle(configs)

    launch_par(configs, visible_gpus=visible_gpus)


if __name__ == "__main__":
    fire.Fire(main_all_cfgs)
