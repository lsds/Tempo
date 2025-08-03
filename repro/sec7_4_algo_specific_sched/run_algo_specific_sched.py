import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fire

from repro.data_loading import DEFAULT_RESULTS_PATH
from repro.launch_lib import FakeWandBLogger, launch_par, launch_seq
from repro.sec7_3_rl_train.shared import sweep_param
from repro.sec7_4_algo_specific_sched.shared import (
    ALGO_SPECIFIC_SCHED_DIR,
    ERROR_TXT_FILE,
    LOG_CSV_FILE,
    MONITOR_CSV_FILE,
    OBJECTIVE_SWEEP,
    SHARED_REINFORCE_HYPERPARAMS,
    SYS,
    get_experiment_name_and_results_path,
)
from tempo.utils.resource_monitor import ResourceMonitorManager

""" Run the algorithm-specific scheduling experiments from Figure 15 in Section 7.4.
"""

CACHING_ALLOC_TO_ITERS = {
    True: 20,
    False: 5,
}

def run_experiment(  # noqa: C901
    **kwargs,
):
    sys_cfg = kwargs["sys_cfg"]
    results_path = Path(kwargs["results_path"])
    results_path.mkdir(parents=True, exist_ok=True)

    wandb_run = FakeWandBLogger(str(results_path / LOG_CSV_FILE))
    wandb_run.set_config(kwargs)

    params = dict(kwargs)
    params["wandb_run"] = wandb_run

    try:
        from repro.sec7_4_algo_specific_sched.impls.tempo_reinforce import (
            get_tempo_reinforce_executor,
        )

        executor = get_tempo_reinforce_executor(**params)
        compilation_time_breakdown = executor.get_compilation_time_breakdown()
        with open(results_path / "compilation_time_breakdown.txt", "w") as f:
            f.write(str(compilation_time_breakdown))
        execute_fn = executor.execute

        gpu_id = kwargs["gpu_id"]
        mon = ResourceMonitorManager(results_path / MONITOR_CSV_FILE, fps=10, gpu_ids=(gpu_id,))
        with mon:
            execute_fn()

    except Exception as e:
        print(f"Failed to run {sys_cfg}: {e}")

        # Log the exception to a file in results_path
        with open(results_path / ERROR_TXT_FILE, "w") as f:
            f.write(f"Exception: {e}\n")
            traceback.print_exc(file=f)  # Writes the stack trace to the file
            f.write("\n")
    finally:
        wandb_run.finish(quiet=True)


# NOTE: Section 7.3: Fig 14
def build_algo_specific_sched_obs_configs(
    results_path: str = DEFAULT_RESULTS_PATH,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_path = str(Path(results_path) / ALGO_SPECIFIC_SCHED_DIR)

    params_base = {
        "runner": run_experiment,
        **SHARED_REINFORCE_HYPERPARAMS,
    }


    par, seq = [], []
    for use_caching_allocators in [True, False]:
        params_base["use_caching_allocators"] = use_caching_allocators
        params_base["iterations"] = CACHING_ALLOC_TO_ITERS[use_caching_allocators]

        # This experiment uses only Tempo-JAX
        exclude_sys = list(SYS)
        exclude_sys.remove("tempo-jax")

        par_, seq_ = sweep_param(
            "objective",
            OBJECTIVE_SWEEP,
            params_base,
            base_path,
            exclude_sys=exclude_sys,
            name_function=get_experiment_name_and_results_path,
        )

        par.extend(par_)
        seq.extend(seq_)

    return seq, par


def launch(
    gpus: str = "0,1,2,3", phbgpu: int = None, results_path: str = DEFAULT_RESULTS_PATH
) -> None:
    """
    Usage Example:
    python repro/sec7_4_algo_specific_sched/run_algo_specific_sched.py launch --gpus "0,1,2,3" --phbgpu 1

    Args:
        gpus (str, optional): Comma-separated list of GPU IDs to use. Defaults to "0,1,2,3".
        phbgpu (int, optional): Specific GPU ID to use for sequential execution. Defaults to None.
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

    seq, par = build_algo_specific_sched_obs_configs(results_path=results_path)

    launch_par(par, visible_gpus=visible_gpus)
    launch_seq(seq, timeout_minutes=40, visible_gpus=visible_gpus, phbgpu=phbgpu)


if __name__ == "__main__":
    fire.Fire(launch)
