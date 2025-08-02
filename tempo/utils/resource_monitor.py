import contextlib
import multiprocessing as mp
import random
import sys
import time
from dataclasses import dataclass
from multiprocessing.managers import DictProxy, ListProxy
from multiprocessing.synchronize import Event
from types import TracebackType
from typing import Any, Callable, ContextManager, Dict, Optional, Sequence, Tuple, Type

import numpy as np
import psutil
import pynvml as nvml

try:
    import dcgm_fields
    import dcgm_structs
    import pydcgm

    HAS_DCGM = True
except ImportError:
    HAS_DCGM = False


NS_IN_A_SEC = 1e9
NS_IN_A_MILLI_SEC = 1e6

pci_bw = {
    # Keys = PCIe-Generation, Values = Max PCIe Lane BW (per direction)
    # [Note: Using specs at https://en.wikipedia.org/wiki/PCI_Express]
    1: (250.0 * 1e6),
    2: (500.0 * 1e6),
    3: (985.0 * 1e6),
    4: (1969.0 * 1e6),
    5: (3938.0 * 1e6),
    6: (7877.0 * 1e6),
}


@dataclass
class ResourceMonitorResults:
    elapsed_ns: ListProxy  # [int]
    curr_time: ListProxy  # [int]
    cpu_util: ListProxy  # [float]
    cpu_mem_util: ListProxy  # [float]
    cpu_mem_total_bytes: int
    gpu_util: DictProxy  # [int,ListProxy[float]]
    smact: DictProxy  # [int,ListProxy[float]]
    gpu_mem_util: DictProxy  # [int,ListProxy[float]]
    per_gpu_total_memory_bytes: DictProxy  # [int, int]
    per_gpu_max_pcie_bw: DictProxy  # [int, float]
    pcie_tx_bytes: DictProxy  # [int, ListProxy[int]]
    pcie_rx_bytes: DictProxy  # [int, ListProxy[int]]
    pcie_rx_util: DictProxy  # [int, ListProxy[float]]
    pcie_tx_util: DictProxy  # [int, ListProxy[float]]
    # net_rcv: List[int]

    @property
    def has_gpu_data(self) -> bool:
        return len(self.gpu_util) > 0

    @property
    def mean_cpu_util(self) -> float:
        return float(np.mean(self.cpu_util))

    @property
    def mean_cpu_mem_util(self) -> float:
        return float(np.mean(self.cpu_mem_util))

    @property
    def mean_gpu_util(self) -> Dict[int, float]:
        return {k: float(np.mean(v)) for k, v in self.gpu_util.items()}

    @property
    def peak_gpu_util(self) -> Dict[int, float]:
        return {k: float(np.max(v)) for k, v in self.gpu_util.items()}

    @property
    def mean_smact(self) -> Dict[int, float]:
        return {k: float(np.mean(v)) for k, v in self.smact.items()}

    @property
    def mean_gpu_mem_util(self) -> Dict[int, float]:
        return {k: float(np.mean(v)) for k, v in self.gpu_mem_util.items()}

    @property
    def peak_gpu_mem_util(self) -> Dict[int, float]:
        return {k: float(np.max(v)) for k, v in self.gpu_mem_util.items()}

    # @property
    # def mean_net_rcv(self) -> int:
    #    return round(np.mean(self.net_rcv), 2)

    def __str__(self) -> str:
        # print only the means
        return (
            f"CPU Util: {self.mean_cpu_util}%\n"
            f"CPU Mem Util: {self.mean_cpu_mem_util}%\n"
            # f"Net RCV: {self.mean_net_rcv}%\n"
            f"GPU Util: {self.mean_gpu_util}%\n"
            f"SMACT: {self.mean_smact}%\n"
            if HAS_DCGM
            else f"GPU Mem Util: {self.mean_gpu_mem_util}%\n"
        )


def regular_interval_loop(
    callback: Callable[[], None],
    stop_event: Event,
    fps: int = 5,  # type: ignore
) -> None:
    start_ts = time.perf_counter_ns()
    last_time = start_ts

    lag = 0.0

    ns_per_update = NS_IN_A_SEC / fps

    while not stop_event.is_set():
        current_time = time.perf_counter_ns()
        elapsed = current_time - last_time
        last_time = current_time
        lag += elapsed
        while lag >= ns_per_update:
            callback()
            lag -= ns_per_update
        time.sleep((ns_per_update / NS_IN_A_SEC) / 2)


def _monitor(
    fps: int,
    path: str,
    stop_event: Event,
    results: ResourceMonitorResults,
    gpu_ids: Sequence[int],
) -> None:
    handles = []
    try:
        nvml.nvmlInit()
        for i in gpu_ids:
            gpu_handle = nvml.nvmlDeviceGetHandleByIndex(i)
            handles.append(gpu_handle)
    except Exception as e:
        print(f"Got exception: {e}")
        print("Most likely this deployment does not have GPU, skipping GPU measurements.")
    dcgm_data = None
    if HAS_DCGM:
        opMode = dcgm_structs.DCGM_OPERATION_MODE_AUTO
        dcgm_handle = pydcgm.DcgmHandle(ipAddress="127.0.0.1", opMode=opMode)
        gpu_group = pydcgm.DcgmGroup(
            dcgm_handle, groupName="group", groupType=dcgm_structs.DCGM_GROUP_EMPTY
        )
        gpu_ids_str = "_".join([str(i) for i in gpu_ids]) + f"_{random.randint(0, sys.maxsize)}"
        smact_group = pydcgm.DcgmFieldGroup(
            dcgm_handle,
            f"smact_group_{gpu_ids_str}",
            [dcgm_fields.DCGM_FI_PROF_SM_ACTIVE],
        )

        for gpu_id in gpu_ids:
            gpu_group.AddGpu(gpu_id)

        # update all fields
        dcgm_handle.GetSystem().UpdateAllFields(True)

        # Start sampling
        # update frequency is 10 seconds
        # max keep age is 1 hour
        # max keep count is 0, meaning all
        assert gpu_group.samples is not None

        # updateFreq: How often to update these fields in usec
        # maxKeepAge: How long to keep data for these fields in seconds
        # maxKeepSamples: Maximum number of samples to keep per field. 0=no limit
        gpu_group.samples.WatchFields(smact_group, 10000, 100.0, 1)

        dcgm_data = (gpu_group, smact_group, dcgm_handle)

    start_ts = time.perf_counter_ns()

    # Open file to write results
    # file = open(f"{path}/resource_monitor.csv", "w")
    # file = None

    def callback() -> None:
        # TODO make this observer based
        take_reading(fps, int(start_ts), gpu_ids, handles, results, dcgm_data)  # file,

    regular_interval_loop(callback, stop_event, fps)

    try:
        nvml.nvmlShutdown()
        if HAS_DCGM:
            # Explicitly delete the field group and GPU group
            smact_group.Delete()
            gpu_group.Delete()
            dcgm_handle.Shutdown()
    except Exception as e:
        print(f"Got exception: {e}")
        print("Most likely this deployment does not have GPU, skipping nvml shutdown.")


def take_reading(
    fps: int,
    start_ts: int,
    gpu_ids: Sequence[int],
    gpu_handles: Sequence[Any],
    results: ResourceMonitorResults,
    dcgm_data: Optional[Tuple[Any, Any, Any]],
    # file: TextIOWrapper,
) -> None:
    now = time.perf_counter_ns()
    elapsed_total_ns = now - start_ts

    vm = psutil.virtual_memory()
    cpu_util = psutil.cpu_percent()
    cpu_util = round(cpu_util, 2)
    cpu_mem_util = vm.percent
    cpu_mem_util = round(cpu_mem_util, 2)
    results.curr_time.append(now)
    results.elapsed_ns.append(elapsed_total_ns)
    results.cpu_util.append(cpu_util)
    results.cpu_mem_util.append(cpu_mem_util)

    ## HACK: push the new absolute value before the delta
    # new_net_rcv_value = psutil.net_io_counters().bytes_recv
    # prev_net_rcv_val = l_net_rcv.pop() if len(l_net_rcv) > 0 else new_net_rcv_value
    # net_rcv = (new_net_rcv_value - prev_net_rcv_val) * 8e-9  # Convert bytes to gigabit
    # net_rcv = net_rcv / (1 / fps)  # Convert to gigabit/s

    # l_net_rcv.append(net_rcv)
    # l_net_rcv.append(new_net_rcv_value)

    # file.write(f"{elapsed_total_ns},{cpu_util},{cpu_mem_util}")
    for i, gpu_handle in zip(gpu_ids, gpu_handles, strict=False):
        gpu_util = float(nvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu)
        gpu_util = round(gpu_util, 2)

        memory_info = nvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        gpu_memory_used = int(memory_info.used)
        gpu_mem_util = (gpu_memory_used / results.per_gpu_total_memory_bytes[i]) * 100
        gpu_mem_util = round(gpu_mem_util, 2)

        max_rxtx_tp = results.per_gpu_max_pcie_bw[i]
        pcie_tx_bytes = (
            nvml.nvmlDeviceGetPcieThroughput(gpu_handle, nvml.NVML_PCIE_UTIL_TX_BYTES) * 1e3
        )
        pcie_tx_percentage = round((pcie_tx_bytes / max_rxtx_tp) * 100, 2)
        pcie_rx_bytes = (
            nvml.nvmlDeviceGetPcieThroughput(gpu_handle, nvml.NVML_PCIE_UTIL_RX_BYTES) * 1e3
        )
        pcie_rx_percentage = round((pcie_rx_bytes / max_rxtx_tp) * 100, 2)
        results.pcie_tx_bytes[i].append(pcie_tx_bytes)
        results.pcie_rx_bytes[i].append(pcie_rx_bytes)
        results.pcie_tx_util[i].append(pcie_tx_percentage)
        results.pcie_rx_util[i].append(pcie_rx_percentage)

        results.gpu_util[i].append(gpu_util)
        results.gpu_mem_util[i].append(gpu_mem_util)
        # file.write(f",{gpu_util},{gpu_mem_util}")

        if HAS_DCGM:
            assert dcgm_data is not None
            # pydcgm.DcgmGroup, pydcgm.DcgmFieldGroup, pydcgm.DcgmHandle
            gpu_group, smact_group, dcgm_handle = dcgm_data
            field_value = gpu_group.samples.GetLatest_v2(smact_group)

            smact = (
                float(
                    field_value.values[dcgm_fields.DCGM_FE_GPU][i][smact_group.fieldIds[0]]
                    .values[-1]
                    .value
                )
                * 100
            )
            smact = round(smact, 2)
            results.smact[i].append(smact)
        else:
            results.smact[i].append(0.0)

    # file.write("\n")


class ResourceUsageMonitor:
    def __init__(self, path: str, fps: int, gpus_to_monitor: Optional[Sequence[int]] = None):
        self.fps = fps
        self.path = path
        self.ctx = mp.get_context("spawn")
        self.stop_event = self.ctx.Event()
        manager = self.ctx.Manager()
        self.manager = manager

        self.results = ResourceMonitorResults(
            curr_time=manager.list(),
            elapsed_ns=manager.list(),
            cpu_util=manager.list(),
            cpu_mem_util=manager.list(),
            cpu_mem_total_bytes=psutil.virtual_memory().total,
            gpu_util=manager.dict(),
            smact=manager.dict(),
            gpu_mem_util=manager.dict(),
            per_gpu_total_memory_bytes=manager.dict(),
            per_gpu_max_pcie_bw=manager.dict(),
            pcie_tx_bytes=manager.dict(),
            pcie_rx_bytes=manager.dict(),
            pcie_rx_util=manager.dict(),
            pcie_tx_util=manager.dict(),
        )
        try:
            nvml.nvmlInit()

            if gpus_to_monitor is None:
                gpus_to_monitor = list(range(nvml.nvmlDeviceGetCount()))
            for i in gpus_to_monitor:
                self.results.gpu_util[i] = manager.list()
                self.results.smact[i] = manager.list()
                self.results.gpu_mem_util[i] = manager.list()
                self.results.pcie_tx_bytes[i] = manager.list()
                self.results.pcie_rx_bytes[i] = manager.list()
                self.results.pcie_rx_util[i] = manager.list()
                self.results.pcie_tx_util[i] = manager.list()

                gpu_handle = nvml.nvmlDeviceGetHandleByIndex(i)
                total_memory_bytes = int(nvml.nvmlDeviceGetMemoryInfo(gpu_handle).total)
                self.results.per_gpu_total_memory_bytes[i] = total_memory_bytes
                pci_gen = nvml.nvmlDeviceGetMaxPcieLinkGeneration(gpu_handle)
                # Enable Counters for PCIe measurements
                pci_width = nvml.nvmlDeviceGetMaxPcieLinkWidth(gpu_handle)

                # Max PCIe Throughput = BW-per-lane * Width
                max_rxtx_tp = pci_width * pci_bw[pci_gen]
                self.results.per_gpu_max_pcie_bw[i] = max_rxtx_tp
            nvml.nvmlShutdown()
        except Exception as e:
            print(f"Got exception: {e}")
            print("Most likely this deployment does not have GPU, skipping GPU measurements.")

        self.process = self.ctx.Process(
            target=_monitor,
            args=(self.fps, self.path, self.stop_event, self.results, gpus_to_monitor),
            daemon=True,
        )
        # self.net_rcv = manager.list()

    def start(self) -> None:
        self.stop_event.clear()
        self.process.start()

    def stop(self) -> None:
        self.stop_event.set()
        count = 0
        while self.process.is_alive():
            time.sleep(0.1)
            if count > 20:
                self.process.terminate()
                break
        self.process.join()

        # self.net_rcv.pop() # remove the last value which is not a delta


class ResourceMonitorManager:
    def __init__(
        self,
        path: str = "./monitor.csv",
        fps: int = 1,
        gpu_ids: Optional[Sequence[int]] = None,
    ):
        self.monitor = ResourceUsageMonitor(path, fps, gpu_ids)
        self.path = path

    def __enter__(self) -> ResourceUsageMonitor:
        self.monitor.start()
        return self.monitor

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        try:
            self.monitor.stop()
        except Exception as e:
            print(f"Got exception: {e}")
        finally:
            self._save_results_as_csv(self.path)
        return None

    def get_results(self) -> ResourceMonitorResults:
        return self.monitor.results

    def _save_results_as_csv(self, path: str) -> None:
        results = self.monitor.results

        with open(path, "w") as f:
            # Write header
            f.write("elapsed_ns,curr_time,cpu_util,cpu_mem_util")

            for g in results.gpu_util.keys():
                f.write(
                    f",gpu{g}_util,gpu{g}_smact,gpu{g}_mem_util,gpu{g}_pcie_tx_bytes,"
                    + f"gpu{g}_pcie_rx_bytes,gpu{g}_pcie_rx_util,gpu{g}_pcie_tx_util"
                )

            f.write("\n")

            # Write data
            max_index = min(
                len(results.elapsed_ns),
                len(results.curr_time),
                len(results.cpu_util),
                len(results.cpu_mem_util),
                *[len(results.gpu_util[g]) for g in results.gpu_util.keys()],
                *[len(results.smact[g]) for g in results.smact.keys()],
                *[len(results.gpu_mem_util[g]) for g in results.gpu_mem_util.keys()],
                *[len(results.pcie_tx_bytes[g]) for g in results.pcie_tx_bytes.keys()],
                *[len(results.pcie_rx_bytes[g]) for g in results.pcie_rx_bytes.keys()],
                *[len(results.pcie_rx_util[g]) for g in results.pcie_rx_util.keys()],
                *[len(results.pcie_tx_util[g]) for g in results.pcie_tx_util.keys()],
            )
            for i in range(max_index):
                f.write(
                    f"{results.elapsed_ns[i]},{results.curr_time[i]},"
                    + f"{results.cpu_util[i]},{results.cpu_mem_util[i]}"
                )
                for g in results.gpu_util.keys():
                    f.write(
                        f",{results.gpu_util[g][i]},"
                        + f"{results.smact[g][i]},{results.gpu_mem_util[g][i]}"
                        + f",{results.pcie_tx_bytes[g][i]},{results.pcie_rx_bytes[g][i]},"
                        + f"{results.pcie_rx_util[g][i]},{results.pcie_tx_util[g][i]}"
                    )
                f.write("\n")

    @staticmethod
    def get(enabled: bool, path: str, fps: int = 1) -> ContextManager:
        if enabled:
            return ResourceMonitorManager(path, fps)
        else:
            return contextlib.nullcontext()
