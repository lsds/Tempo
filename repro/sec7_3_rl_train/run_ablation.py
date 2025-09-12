from pathlib import Path

import fire

from repro.data_loading import DEFAULT_RESULTS_PATH
from repro.launch_lib import launch_par
from repro.sec7_3_rl_train.shared import (
    RL_TRAIN_DIR,
    SHARED_PPO_HYPERPARAMS,
    run_experiment,
)

""" Ablate the optimizations in the tempo compiler.
"""

ABLATE_DIR = "ablate"

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


def generate_configs(base_results_path: str):
    configs = []
    base_cfg = {
        "runner": run_experiment,
        **SHARED_PPO_HYPERPARAMS,
        "is_ablation": True,
        "sys_cfg": "tempo-jax",
    }

    base_cfg["obs_shape"] = (3, 64, 64)
    base_cfg["ep_len"] = 100
    base_cfg["batch_size"] = 512
    base_cfg["params_per_layer"] = 256
    base_cfg["num_layers"] = 2
    # base_cfg["iterations"] = 100

    # Start with all optimizations enabled
    configs.append(
        {
            "name": "all_enabled",
            "results_path": str(Path(base_results_path) / "all_enabled"),
            "disable_cfg_keys": [],
            **base_cfg,
        }
    )

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
        if cfg_key not in SKIP_KEYS:
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
    results_path = Path(results_path) / RL_TRAIN_DIR / ABLATE_DIR

    results_path_in_order = results_path / "in_order"
    results_path_in_order.mkdir(parents=True, exist_ok=True)

    # Generate both types of configs
    configs_in_order = generate_configs(results_path_in_order)

    launch_par(configs_in_order, visible_gpus=visible_gpus, timeout_minutes=240)


if __name__ == "__main__":
    fire.Fire(main_all_cfgs)
