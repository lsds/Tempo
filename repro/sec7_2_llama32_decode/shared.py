import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Protocol

from tempo.utils.resource_monitor import ResourceMonitorManager

""" Shared constants for the LM decode experiments.
"""

# Constants for directory and file names
LLAMA32_DECODE_DIR = "llama32_decode"
MEM_USAGE_DIR = "mem_usage"
BLOCK_SIZE_MICROBENCHMARK_DIR = "block_size_microbenchmark"
ABLATE_DIR = "ablate"
COMPILE_TIME_SCALING_DIR = "compile_time_scaling"

MONITOR_CSV_FILE = "monitor.csv"
SUMMARY_RESULTS_FILE = "summary_results.json"
LOG_CSV_FILE = "log.csv"
LOG_CONFIG_FILE = "log.config"
ERROR_TXT_FILE = "error.txt"

SAMPLE_PROMPTS = [
    "I believe the meaning of life is",
    "In the heart of the ancient forest, I stumbled upon",
    "My favorite book from childhood is",
    "Nothing beats a good cup of coffee when",
    "If I could travel anywhere in the world right now, I'd go to",
    "The secret ingredient in the perfect chocolate cake is",
    "People often forget the importance of",
    "Growing a small herb garden can",
    "The invention that changed the world the most was",
    "My grandfather always used to say",
    "On a misty autumn morning, the old lighthouse keeper discovered",
    "The one song that never fails to lift my spirits is",
    "Whenever I walk past an art gallery, I imagine",
    "If kindness were a currency, the wealthiest person I know would be",
    "The most unforgettable smell from my childhood kitchen is",
    "Late at night, when the city finally quiets down, you can almost hear",
    "The first star I ever wished upon was",
    "When the clock struck midnight, the abandoned carnival suddenly",
    "If walls could talk, the one in my childhood bedroom would whisper",
    "On rainy Sundays, I always find myself",
    "The greatest lesson my favorite teacher taught me was",
    "If dreams were postcards, last night's would have shown",
    "The smell of fresh paint always reminds me of",
    "When I open an old photo album, I'm transported to",
    "Every winter, the frozen lake hides a secret beneath",
    "The one tradition I hope never disappears is",
    "If time travel were possible, I'd first visit",
    "The taste of summer can be found in",
    "My favorite place to watch the sunset is",
    "Whenever I hear distant thunder, I remember",
    "The silent hero in my everyday life is",
    "If I wrote a letter to my future self, I'd start with",
    "The city lights at dawn reveal",
    "In the quiet corner of the library, I discovered",
    "My most cherished family heirloom is more than an object; it's",
    "If laughter were a color, mine would be",
    "The shadow that follows me on late-night walks feels like",
    "When I smell fresh bread baking, I imagine",
    "The moment I realized I'd grown up was",
    "The first star I ever wished upon was",
    "When the clock struck midnight, the abandoned carnival suddenly",
    "If walls could talk, the one in my childhood bedroom would whisper",
    "On rainy Sundays, I always find myself",
    "The greatest lesson my favorite teacher taught me was",
    "If dreams were postcards, last night's would have shown",
    "The most courageous thing I ever witnessed was",
    "If animals could speak, my pet would tell me",
    "Every winter, the frozen lake hides a secret beneath",
    "The one tradition I hope never disappears is",
    "If time travel were possible, I'd first visit",
    "The taste of summer can be found in",
    "My favorite place to watch the sunset is",
    "Whenever I hear distant thunder, I remember",
    "The silent hero in my everyday life is",
    "If I wrote a letter to my future self, I'd start with",
    "The city lights at dawn reveal",
    "In the quiet corner of the library, I discovered",
    "My most cherished family heirloom is more than an object; it's",
    "If laughter were a color, mine would be",
    "The shadow that follows me on late-night walks feels like",
    "When I smell fresh bread baking, I imagine",
    "The moment I realized I'd grown up was",
    "Deep down, I truly believe",
    "The most important thing in life is",
    "The one thing I've always wanted to do is",
    "I know this is unpopular, but",
]


class BenchRunner(Protocol):
    def __init__(self, *args, **kwargs) -> None: ...

    def reset(self, **kwargs) -> None: ...

    def compile(self, **kwargs) -> None: ...

    def warmup(self, **kwargs) -> None: ...

    def run(self, **kwargs) -> None: ...

    def get_decoded_outputs(self, **kwargs) -> list[str]: ...


def run_bench(**kwargs):
    """Run a single benchmark with the given runner class and configuration."""
    runner_cls = None
    if kwargs["framework_name"] == "torch":
        from repro.sec7_2_llama32_decode.impls.torch_llama32 import TorchLlama32InferenceRunner

        runner_cls = TorchLlama32InferenceRunner
    elif kwargs["framework_name"] == "jax":
        from repro.sec7_2_llama32_decode.impls.jax_llama32 import JAXLlama32InferenceRunner

        runner_cls = JAXLlama32InferenceRunner
    elif kwargs["framework_name"] == "tempo":
        from repro.sec7_2_llama32_decode.impls.tempo_llama32 import TempoLlama32InferenceRunner

        runner_cls = TempoLlama32InferenceRunner

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

    # if kwargs.get("is_block_microbenchmark", False) and (not kwargs.get("is_ablation", False)):
    runner.warmup()
    runner.reset()

    with mon:
        start_total = time.perf_counter()
        n_iterations = 0
        while (max_bench_time_secs is None and n_iterations < 1) or (
            max_bench_time_secs is not None
            and time.perf_counter() - start_total < max_bench_time_secs
        ):
            runner.run()
            runner.reset()
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


def get_prompts(**kwargs: Any) -> list[str]:
    batch_size = kwargs["batch_size"]

    sampled_prompts = random.choices(SAMPLE_PROMPTS, k=batch_size)

    max_prompt_len = kwargs.get("max_prompt_len", sys.maxsize)

    sampled_prompts = [p[:max_prompt_len] for p in sampled_prompts]

    return sampled_prompts
