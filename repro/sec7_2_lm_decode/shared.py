import json
import time
from pathlib import Path

from tempo.utils.resource_monitor import ResourceMonitorManager

""" Shared constants for the LM decode experiments.
"""

# Constants for directory and file names
GPT2_DECODE_DIR = "gpt2_decode"
MEM_USAGE_DIR = "mem_usage"
BLOCK_SIZE_MICROBENCHMARK_DIR = "block_size_microbenchmark"

MONITOR_CSV_FILE = "monitor.csv"
SUMMARY_RESULTS_FILE = "summary_results.json"
LOG_CSV_FILE = "log.csv"
LOG_CONFIG_FILE = "log.config"
ERROR_TXT_FILE = "error.txt"


def run_bench(**kwargs):
    """Run a single benchmark with the given runner class and configuration."""
    runner_cls = None
    if kwargs["framework_name"] == "torch":
        from repro.sec7_2_lm_decode.impls.torch_gpt2 import TorchBenchRunner

        runner_cls = TorchBenchRunner
    elif kwargs["framework_name"] == "torchnaive":
        from repro.sec7_2_lm_decode.impls.torchnaive_gpt2 import TorchBenchRunnerNaive

        runner_cls = TorchBenchRunnerNaive
    elif kwargs["framework_name"] == "jax":
        from repro.sec7_2_lm_decode.impls.jax_gpt2 import JaxBenchRunner

        runner_cls = JaxBenchRunner
    elif kwargs["framework_name"] == "tempo":
        from repro.sec7_2_lm_decode.impls.tempo_gpt2 import TempoBenchRunner

        runner_cls = TempoBenchRunner

    else:
        raise ValueError(f"Unknown framework: {kwargs['framework_name']}")

    name = kwargs["name"]
    runner = runner_cls(**kwargs)

    # One warm-up call to trigger compilation
    start = time.perf_counter()
    runner.compile()
    end = time.perf_counter()
    comp_time = end - start

    base_path = Path(kwargs["results_path"])
    base_path.mkdir(parents=True, exist_ok=True)

    mon_path = base_path / MONITOR_CSV_FILE
    summary_results_path = base_path / SUMMARY_RESULTS_FILE

    mon_fps = kwargs.get("monitor_fps", 10)
    mon = ResourceMonitorManager(mon_path, fps=mon_fps, gpu_ids=[kwargs["gpu_id"]])

    max_bench_time_secs = kwargs.get("max_bench_time_secs", None)

    with mon:
        start_total = time.perf_counter()
        n_iterations = 0
        while (max_bench_time_secs is None and n_iterations < 1) or (
            max_bench_time_secs is not None
            and time.perf_counter() - start_total < max_bench_time_secs
        ):
            runner.run()
            n_iterations += 1
        end_total = time.perf_counter()

    elapsed = end_total - start_total
    avg_time = elapsed / n_iterations

    res = mon.get_results()
    gpu_id = int(kwargs["gpu_id"])

    has_gpu = len(res.mean_gpu_util) > 0

    summary_results = {
        "name": name,
        "elapsed_time": elapsed,
        "comp_time": comp_time,
        "avg_iter_time": avg_time,
        "n_iterations": n_iterations,
        "mean_cpu_util": res.mean_cpu_util,
        "mean_cpu_mem_util": res.mean_cpu_mem_util,
    }

    if has_gpu:
        summary_results["mean_gpu_util"] = res.mean_gpu_util[gpu_id]
        summary_results["peak_gpu_util"] = res.peak_gpu_util[gpu_id]
        summary_results["mean_gpu_mem_util"] = res.mean_gpu_mem_util[gpu_id]
        summary_results["peak_gpu_mem_util"] = res.peak_gpu_mem_util[gpu_id]

    # Round everything to 2 decimal places
    summary_results = {
        k: round(v, 2) if isinstance(v, float) else v for k, v in summary_results.items()
    }

    with open(summary_results_path, "w") as f:
        json.dump(summary_results, f)

    mon._save_results_as_csv(str(mon_path))
