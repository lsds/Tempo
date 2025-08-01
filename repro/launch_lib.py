import csv
import os
import shutil
import signal
import time
from multiprocessing import current_process, get_context
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

"""
This file offers utilities for launching experiments in parallel, helping to speed-up the
reproducibility efforts.
Some configurations must be run in sequence for correctness (RLLib or Tempo with swapping) .
"""


class FakeWandBLogger:
    def __init__(self, csv_file_path: Union[Path, str]) -> None:
        self.csv_file = Path(csv_file_path)
        self.config_file = self.csv_file.with_suffix(".config")
        self.fields = set()
        self.rows: List[Dict[str, Any]] = [{} for _ in range(100_000)]
        self.num_rows = 0

        self.start_time = time.perf_counter_ns()

    def set_config(self, config: Dict[str, Any]) -> None:
        # Make dirs if they don't exist
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config_file, mode="w+") as file:
            writer = csv.DictWriter(file, fieldnames=config.keys())
            writer.writeheader()
            writer.writerow(config)

    def log(self, data: Dict[str, Any]) -> None:
        ts = time.perf_counter_ns()
        elapsed_ns = ts - self.start_time
        print(data)

        data["elapsed_ns"] = elapsed_ns
        data["curr_time"] = ts

        iteration = data["iteration"]
        self.num_rows = max(self.num_rows, iteration + 1)
        # Update fields
        self.fields.update(data.keys())

        # Add the row of data
        self.rows[iteration].update(data)

    def finish(self, quiet: bool = False) -> None:
        self._write_csv()

    def _write_csv(self) -> None:
        self.csv_file.parent.mkdir(parents=True, exist_ok=True)
        # Ensure the order of fields remains consistent
        sorted_fields = sorted(self.fields)
        with open(self.csv_file, mode="w+", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=sorted_fields)
            writer.writeheader()
            for row in range(self.num_rows):
                writer.writerow(self.rows[row])


class StrictSemaphore:
    """
    A multiprocessing-aware semaphore with strict ownership tracking.

    Unlike a standard multiprocessing.Semaphore, this class maintains an internal
    record of which processes have successfully acquired the semaphore, and prevents
    erroneous releases from processes that do not currently hold it.
    """

    def __init__(self, value: int = 1, ctx=None):
        if value < 0:
            raise ValueError("Semaphore initial value must be >= 0")
        if ctx is None:
            ctx = get_context("spawn")
        self._semaphore = ctx.Semaphore(value)
        self._lock = ctx.Lock()  # To protect shared resources
        manager = ctx.Manager()
        self._owners = manager.list()  # List of process IDs that currently hold the semaphore

    def acquire(self, blocking: bool = True, timeout: float = None) -> bool:
        result = self._semaphore.acquire(blocking, timeout)
        if result:
            with self._lock:
                self._owners.append(current_process().pid)
        return result

    def release(self):
        with self._lock:
            if current_process().pid not in self._owners:
                return  # Do nothing
            self._owners.remove(current_process().pid)
        self._semaphore.release()

    def __enter__(self):
        """Acquire the semaphore when entering the context."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Release the semaphore when exiting the context."""
        self.release()

    def current_owners(self):
        """Return a list of process IDs that currently hold the semaphore."""
        with self._lock:
            return list(self._owners)


def create_timeout_file(
    results_path: Union[Path, str],
    timeout_minutes: int,
    process_pid: int,
    experiment_name: str,
    gpu_id: int,
    start_time: float = None,
) -> None:
    """Create a timeout file in the results path with experiment details.

    Args:
        results_path: Path where the timeout file should be created
        timeout_minutes: Timeout duration in minutes
        process_pid: Process ID that timed out
        experiment_name: Name of the experiment
        gpu_id: GPU ID used by the experiment
        start_time: Start time of the process (optional)
    """
    results_path = Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / "timed_out.txt", "w") as f:
        f.write(f"Experiment timed out after {timeout_minutes} minutes.\n")
        f.write(f"Process PID: {process_pid}\n")
        f.write(f"Experiment name: {experiment_name}\n")
        f.write(f"GPU ID: {gpu_id}\n")
        if start_time is not None:
            f.write(f"Start time: {start_time}\n")
        f.write(f"Timeout time: {timeout_minutes} minutes\n")


def terminate_process_and_children(pid: int):
    """Terminates a process and its child processes."""
    try:
        os.kill(pid, signal.SIGTERM)  # Terminate the process
    except OSError:
        pass  # Process might have already exited

    time.sleep(2)  # Wait for the process to terminate

    try:
        # Killing children if any remain
        for child_pid in os.popen(f"pgrep -P {pid}").read().split():
            os.kill(int(child_pid), signal.SIGTERM)
    except OSError:
        pass


def set_flags(gpu_id: int, use_caching_allocators: bool, is_jax: bool) -> None:
    # Set the GPU for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # if is_jax:
    #    os.environ["XLA_FLAGS"] = (
    #        "--xla_gpu_enable_custom_fusions=true "
    #        # https://github.com/NVIDIA/JAX-Toolbox/issues/1098
    #        "--xla_gpu_enable_dynamic_slice_fusion=true "
    #        "--xla_gpu_enable_triton_gemm=true "
    #    )

    # Clear any existing environment variables
    os.environ.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
    os.environ.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
    os.environ.pop("PYTORCH_NO_CUDA_MEMORY_CACHING", None)

    if use_caching_allocators:
        if not is_jax:
            # For non-JAX systems, disable JAX's aggressive memory preallocation
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        else:
            # For JAX-based systems, allow them to preallocate memory
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
    else:
        # Disable caching allocators so we can measure real runtime memory usage
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"


def worker(
    gpu_id: int,
    experiment_runner: Callable,
    **kwargs: Any,
) -> None:
    semaphore = kwargs["semaphore"]
    name = kwargs.get("name", None)

    if name is None:
        raise ValueError("name must be provided")

    results_path = Path(kwargs["results_path"])
    results_path.mkdir(parents=True, exist_ok=True)

    def release_semaphore_and_write_results_on_termination(signum, frame):
        print(f"Worker {name} received termination signal. Releasing semaphore.")
        with open(results_path / "error.txt", "w") as f:
            f.write(f"Worker {name} received termination signal. Likely timeout due to memory.")

        try:
            semaphore.release()
        except Exception:
            pass
        exit(1)

    # Register the signal handler for termination signals
    signal.signal(signal.SIGTERM, release_semaphore_and_write_results_on_termination)
    signal.signal(signal.SIGINT, release_semaphore_and_write_results_on_termination)

    is_jax = (
        "jax" in kwargs["name"]
        or ("tempo_cfg" in kwargs and kwargs["tempo_cfg"].backend == "jax")
        or kwargs.get("is_jax", False)
    )
    # Set the flags for this process
    set_flags(gpu_id, kwargs["use_caching_allocators"], is_jax)

    # add gpu_id to kwargs
    kwargs["gpu_id"] = gpu_id

    try:
        with semaphore:
            # Run the experiment
            experiment_runner(**kwargs)
    except Exception as e:
        # Write error to results path
        with open(results_path / "error.txt", "w") as f:
            f.write(f"Error occurred during experiment:\n{str(e)}\n")
            f.write("\nTraceback:\n")
            import traceback

            traceback.print_exc(file=f)
        print(f"Error in worker {name}. Error written to {results_path}/error.txt")
        raise e
    finally:
        try:
            semaphore.release()
        except Exception:
            pass


def launch_par(
    configs: List[Dict[str, Any]],
    visible_gpus: Tuple[int, ...] = (0, 1, 2, 3),
    timeout_minutes: int = 60,
) -> None:
    """Launch a sequence of experiments in parallel.

    Args:
        configs (List[Dict[str, Any]]): List of configurations to run. Must contain "runner" callable accepting kwargs.
        Particular kwargs of interest are "name" string, "results_path" string, "use_caching_allocators" bool
        visible_gpus (Tuple[int, ...], optional): List of GPU IDs to use. Defaults to (0, 1, 2, 3).
        timeout_minutes (int, optional): Timeout in minutes for each experiment. Defaults to 60.
    """
    # Shuffle the available_gpus to avoid the same backend being assigned to the same GPU
    import random

    available_gpus = list(visible_gpus)
    concurrent_processes = len(available_gpus)

    random.shuffle(available_gpus)

    processes = []
    active_sys_cfgs = {}  # Map to track active sys_cfgs
    process_start_times = {}  # Map to track start times of processes

    configs_kwargs = configs.copy()
    ctx = get_context("spawn")
    semaphore = StrictSemaphore(concurrent_processes, ctx)

    total = len(configs_kwargs)
    finished = 0
    is_par = concurrent_processes > 1
    timeout_seconds = timeout_minutes * 60

    while configs_kwargs or processes:
        print(f"Progress (par={is_par}): {finished}/{total}")

        # Print active sys_cfgs
        active_cfgs = ", ".join(active_sys_cfgs.values())
        print(f"Active sys_cfgs: {active_cfgs if active_cfgs else 'None'}")

        # Start new processes if GPUs are available
        while available_gpus and configs_kwargs:
            kwargs = configs_kwargs.pop(0)
            kwargs = kwargs.copy()
            kwargs["semaphore"] = semaphore
            runner = kwargs.pop("runner")
            name = kwargs["name"]
            path = kwargs["results_path"]

            if os.path.exists(path):
                print(f"Skipping {path} because it already exists")
                finished += 1
                continue

            gpu_id = available_gpus.pop(0)

            p = ctx.Process(
                target=worker,
                args=(gpu_id, runner),
                kwargs=kwargs,
            )
            p.start()
            processes.append((p, gpu_id, name, kwargs["results_path"]))
            active_sys_cfgs[p.pid] = name
            process_start_times[p.pid] = time.time()

        for p, gpu_id, name, results_path in processes.copy():
            if not p.is_alive():
                p.join()
                processes.remove((p, gpu_id, name, results_path))
                available_gpus.append(gpu_id)
                finished += 1
                active_sys_cfgs.pop(p.pid, None)
                process_start_times.pop(p.pid, None)
            else:
                # Check for timeout
                elapsed_time = time.time() - process_start_times[p.pid]
                if elapsed_time > timeout_seconds:
                    print(
                        f"Terminating process {p.pid} ({name}) due to timeout after {timeout_minutes} minutes."
                    )
                    terminate_process_and_children(p.pid)
                    p.join()
                    processes.remove((p, gpu_id, name, results_path))
                    available_gpus.append(gpu_id)
                    finished += 1
                    active_sys_cfgs.pop(p.pid, None)
                    process_start_times.pop(p.pid, None)

                    # Create timeout file in results path
                    create_timeout_file(
                        results_path,
                        timeout_minutes,
                        p.pid,
                        name,
                        gpu_id,
                        process_start_times.get(p.pid),
                    )
        time.sleep(1)  # Short sleep to prevent busy waiting


def launch_seq(
    configs: List[Dict[str, Any]],
    timeout_minutes: int = 60,
    retry_attempts: int = 0,
    visible_gpus: Tuple[int, ...] = (0, 1, 2, 3),
    phbgpu: int = None,
) -> None:
    """Launch a sequence of experiments in sequence.

    Args:
        configs (List[Dict[str, Any]]): List of configurations to run.
        Must contain "runner" callable accepting kwargs, "name" string, "results_path" string,
          "use_caching_allocators" bool, "gpu_id" int
        timeout_minutes (int, optional): Timeout in minutes. Defaults to 60.
        retry_attempts (int, optional): Number of times to retry each individual config if it fails. Defaults to 0.
        visible_gpus (Tuple[int, ...], optional): List of GPU IDs to use. Defaults to (0, 1, 2, 3).
        phbgpu (int, optional): Specific GPU ID to use for sequential execution. If provided, this GPU will be used for all sequential experiments.
    """
    if phbgpu is not None:
        # Use the specified PHB GPU for all sequential experiments
        available_gpus = [phbgpu]
    else:
        # Fall back to the original behavior
        available_gpus = list(visible_gpus)
        # Shuffle the available_gpus to avoid the same backend being assigned to the same GPU
        import random

        random.shuffle(available_gpus)

    active_sys_cfgs = {}  # Map to track active sys_cfgs
    ctx = get_context("spawn")
    semaphore = StrictSemaphore(1, ctx)
    timeout_seconds = timeout_minutes * 60

    # Track retry counts for each config by results_path
    config_retry_counts = {}

    # Initialize retry counts for all configs
    for config in configs:
        config_retry_counts[config["results_path"]] = 0

    # Main execution loop
    configs_to_run = configs.copy()

    total = len(configs)
    finished = 0

    while configs_to_run:
        kwargs_ = configs_to_run.pop(0)
        kwargs = kwargs_.copy()
        kwargs["semaphore"] = semaphore
        name = kwargs["name"]
        path_full = kwargs["results_path"]
        runner = kwargs.pop("runner")
        current_retry_count = config_retry_counts[path_full]

        if os.path.exists(path_full):
            print(f"Skipping {path_full} because it already exists")
            finished += 1
            continue

        gpu_id = available_gpus.pop(0)

        p = ctx.Process(
            target=worker,
            args=(gpu_id, runner),
            kwargs=kwargs,
        )
        p.start()
        start_time = time.time()
        active_sys_cfgs[p.pid] = name

        timeout = False
        while p.is_alive():
            print(f"Progress (par=False): {finished}/{total} - {name}")
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                print(f"Terminating process {p.pid} due to timeout.")
                terminate_process_and_children(p.pid)
                p.join()
                available_gpus.append(gpu_id)
                active_sys_cfgs.pop(p.pid, None)
                timeout = True

                # Check if we should retry this config
                if current_retry_count < retry_attempts:
                    print(
                        f"Retrying {name} (attempt {current_retry_count + 1}/{retry_attempts + 1})"
                    )
                    # Clean up the failed results directory
                    if os.path.exists(path_full):
                        shutil.rmtree(path_full)
                    # Increment retry count and add back to the queue
                    config_retry_counts[path_full] = current_retry_count + 1
                    configs_to_run.append(kwargs_)
                else:
                    # No more retries for this config, create timeout file
                    print(f"No more retries left for {name}. Creating timeout file.")
                    create_timeout_file(path_full, timeout_minutes, p.pid, name, gpu_id, start_time)

            time.sleep(5)  # Short sleep to prevent busy waiting

        if not timeout:
            # Check for exceptions
            p.join()
            available_gpus.append(gpu_id)
            active_sys_cfgs.pop(p.pid, None)
            finished += 1
            print(f"Successfully completed {name}")
