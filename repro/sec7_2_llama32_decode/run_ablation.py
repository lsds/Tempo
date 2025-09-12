from pathlib import Path

import fire

from repro.data_loading import DEFAULT_RESULTS_PATH
from repro.launch_lib import launch_par
from repro.sec7_2_llama32_decode.shared import (
    ABLATE_DIR,
    LLAMA32_DECODE_DIR,
    get_prompts,
    run_bench,
)

""" Ablate the optimizations in the tempo compiler.
"""

MAX_BENCH_TIME_SECS = 120

# Only these batch sizes and block sizes are varied
# BATCH_SIZE = 64
BATCH_SIZE = 4
STATIFY_BLOCK_SIZE = 256
SEQ_LEN = 2048
BACKEND = "jax"

ATTN_TYPE = "window"
ATTN_WINDOW_SIZE = 128

MODEL = "Llama3.2-3B"
# NOTE: We will be disabling these in order, top down
CFG_KEYS_IN_ORDER = [
    # "enable_vectorization",
    # NOTE This computation does not use lazy slicing.
    "enable_inplace_write",
    "enable_lazy_slice",
    "enable_donation_analysis",
    "enable_pad_mask_removal",
    "enable_fold_pads_into_storage",
    "enable_hybrid_tensorstore",
    "enable_statifying_incrementalization",
    "enable_non_trivial_vectorization",
    # NOTE: JAX will always codegen even single ops...
    "enable_codegen_dataflows",
    "enable_dataflow_grouping",
    # NOTE: If we put these after disabling grouping, it will show more impact,
    # because they will be not masked by JAX doing the same thing.
    "enable_constant_folding",
    "enable_algebraic_optimizer",
    "enable_domain_reduction",
    "enable_duplicate_code_elim",
    # NOTE: We cannot afford to disable dead code elimination, as it will break other optimizations.
    # "enable_dead_code_elim",
    # NOTE: We cannot afford to disable broadcast elimination, due to OOM.
    "enable_broadcast_elim",
    ## NOTE: This must come before disable grouping, otherwise JAX will complain about
    ## different devices.
    # "enable_device_assignment",
    # "enable_ast_promo",
    # "enable_isolate_loop_conditions",
    # "enable_custom_thunk_launchers",
    "enable_vectorization",
    # NOTE: I just don't care about this one.
    # "enable_symbol_prealloc_store",
]

SKIP_KEYS = [
    "enable_device_assignment",
    "enable_ast_promo",
    "enable_isolate_loop_conditions",
    "enable_pad_mask_removal",
    "enable_codegen_dataflows",
    "enable_constant_folding",
    "enable_algebraic_optimizer",
    "enable_domain_reduction",
    "enable_duplicate_code_elim",
]


def add_baseline_configs(configs: list[dict], base_results_path: str, base_cfg: dict):
    """Add baseline configs to the list of configs."""

    # Add a config with all optimizations enabled
    name = "all_enabled"
    cfg = {
        "name": name,
        "results_path": str(Path(base_results_path) / name),
        **base_cfg,
    }
    configs.append(cfg)

    # TODO: understand why torch backend is so slow here...
    ## Add a config with torch backend
    # name = "torch_backend"
    # cfg = base_cfg.copy()
    # cfg["name"] = name
    # cfg["results_path"] = str(Path(base_results_path) / name)
    # cfg["backend"] = "torch"
    # cfg["is_jax"] = False
    # configs.append(cfg)

    # Add torch executor
    name = "torch"
    cfg = base_cfg.copy()
    cfg["name"] = name
    cfg["results_path"] = str(Path(base_results_path) / name)
    cfg["is_jax"] = False
    cfg["framework_name"] = "torch"
    configs.append(cfg)

    return configs


def generate_configs_single_disabled(base_results_path: str):
    configs = []
    prompts = get_prompts(batch_size=BATCH_SIZE, max_prompt_len=32)
    base_cfg = {
        "runner": run_bench,
        "use_caching_allocators": True,
        "attn_type": ATTN_TYPE,
        "window_size": ATTN_WINDOW_SIZE,
        "temperature": 0.6,  # NOTE: top-p sampling
        "framework_name": "tempo",
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "checkpoint_dir": f"~/.llama/checkpoints/{MODEL}/",
        "statify_block_size": STATIFY_BLOCK_SIZE,
        # Always use these settings for this microbenchmark
        "dev": "gpu",
        "max_bench_time_secs": MAX_BENCH_TIME_SECS,
        "prompts": prompts,
        "backend": BACKEND,
        "is_jax": True,
        "is_ablation": True,
    }
    for cfg_key in CFG_KEYS_IN_ORDER:
        name = f"{cfg_key}_disabled"

        cfg = {
            "name": name,
            "results_path": str(Path(base_results_path) / name),
            "disable_cfg_keys": [cfg_key],
            **base_cfg,
        }
        if cfg_key not in SKIP_KEYS:
            configs.append(cfg)
    configs = add_baseline_configs(configs, base_results_path, base_cfg)
    return configs


def generate_configs_disable_in_order(base_results_path: str):
    """Generate configs that disable optimizations in order, accumulating keys to disable."""
    configs = []
    prompts = get_prompts(batch_size=BATCH_SIZE, max_prompt_len=32)
    base_cfg = {
        "runner": run_bench,
        "use_caching_allocators": True,
        "attn_type": ATTN_TYPE,
        "window_size": ATTN_WINDOW_SIZE,
        "temperature": 0.6,  # NOTE: top-p sampling
        "framework_name": "tempo",
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "checkpoint_dir": f"~/.llama/checkpoints/{MODEL}/",
        "statify_block_size": STATIFY_BLOCK_SIZE,
        # Always use these settings for this microbenchmark
        "dev": "gpu",
        "max_bench_time_secs": MAX_BENCH_TIME_SECS,
        "prompts": prompts,
        "backend": BACKEND,
        "is_jax": True,
        "is_ablation": True,
    }

    # Start with all optimizations enabled
    disabled_keys = []

    # Add configs that disable optimizations one by one, accumulating the disabled keys
    for i, cfg_key in enumerate(CFG_KEYS_IN_ORDER):
        disabled_keys.append(cfg_key)
        name = f"in_order_{i + 1}_{cfg_key}_disabled"

        cfg = {
            "name": name,
            "results_path": str(Path(base_results_path) / name),
            "disable_cfg_keys": disabled_keys.copy(),
            **base_cfg,
        }
        configs.append(cfg)

    configs = add_baseline_configs(configs, base_results_path, base_cfg)

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
    results_path = Path(results_path) / LLAMA32_DECODE_DIR / ABLATE_DIR

    results_path_single = results_path / "single_disabled"
    results_path_in_order = results_path / "in_order"
    results_path_single.mkdir(parents=True, exist_ok=True)
    results_path_in_order.mkdir(parents=True, exist_ok=True)

    # Generate both types of configs
    # configs_single = generate_configs_single_disabled(results_path_single)
    configs_in_order = generate_configs_disable_in_order(results_path_in_order)
    all_configs = configs_in_order

    # Concatenate both lists
    # all_configs = configs_single + configs_in_order

    launch_par(all_configs, visible_gpus=visible_gpus, timeout_minutes=240)


if __name__ == "__main__":
    fire.Fire(main_all_cfgs)
