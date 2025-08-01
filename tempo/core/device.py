from __future__ import annotations

from dataclasses import dataclass
from typing import Final, List, Union

import psutil

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

cores_per_sm = {
    2: 32,  # Fermi
    3: 192,  # Kepler
    5: 128,  # Maxwell
    6: 64,  # Pascal GP100
    7: 64,  # Volta
    7.5: 64,  # Turing
    8: 128,  # Ampere
    8.9: 128,  # Ada Lovelace
}


def _get_cpu_metrics() -> tuple[int, float, float]:
    # Get CPU memory in GB
    memory = psutil.virtual_memory().total / (1024**3)

    # Estimate CPU FLOPS (very rough estimate)
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq().max if psutil.cpu_freq() else 3.0  # Default to 3GHz if unknown
    # Assume 8 FLOPS per cycle per core (typical for modern CPUs)
    flops = cpu_count * cpu_freq * 8 / 1000  # Convert to TFLOPS

    return cpu_count, memory, flops


def _get_gpu_metrics() -> tuple[int, List[float], List[float]]:
    if not CUPY_AVAILABLE:
        return 0, [], []

    num_gpus = cp.cuda.runtime.getDeviceCount()
    memory_list = []
    flops_list = []

    for i in range(num_gpus):
        device = cp.cuda.Device(i)
        prop = device.attributes

        # Memory in GB
        memory = prop["Total Global Memory"] / (1024**3)
        memory_list.append(memory)

        # Calculate FLOPS
        major = prop["Compute Capability Major"]
        _cores_per_sm = cores_per_sm[major]
        flops = 2 * prop["Multiprocessor Count"] * _cores_per_sm * prop["Clock Rate"] * 1e3
        flops_list.append(flops / 1e12)  # Convert to TFLOPS

    return num_gpus, memory_list, flops_list


@dataclass(frozen=True)
class Device:
    device_id: int  # Unique identifier for this specific device
    name: str
    priority: int
    memory_capacity: float  # Memory in GB
    flops: float  # Theoretical peak FLOPS in TFLOPS

    def __str__(self) -> str:
        return f"{self.name}[{self.device_id}]{{{self.memory_capacity}GB/{self.flops}TFLOPs}}"

    def __hash__(self) -> int:
        return hash(self.name)

    __repr__ = __str__


@dataclass(frozen=True, repr=False)
class DeviceGroup:
    name: str
    priority: int
    devices: tuple[Device, ...]

    # TODO: This is not correct
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (DeviceGroup, Device)):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    @property
    def backing_device(self) -> Device:
        return self.devices[0]

    @property
    def count(self) -> int:
        return len(self.devices)

    @property
    def agg_memory_capacity(self) -> float:
        return sum(d.memory_capacity for d in self.devices)

    @property
    def agg_flops(self) -> float:
        return sum(d.flops for d in self.devices)

    def split(self, n: int) -> tuple[DeviceGroup, DeviceGroup]:
        """Split this device into two devices with n and (count-n) devices respectively."""
        if n <= 0 or n >= self.count:
            raise ValueError(f"Invalid split point {n} for device with {self.count} units")
        return (
            DeviceGroup(self.name, self.priority, self.devices[:n]),
            DeviceGroup(self.name, self.priority, self.devices[n:]),
        )

    def select_devices(self, indices: list[int]) -> DeviceGroup:
        """Create a new DeviceGroup containing only the devices at the specified indices."""
        if not indices:
            raise ValueError("No device indices specified")
        if min(indices) < 0 or max(indices) >= self.count:
            raise ValueError(f"Device indices must be between 0 and {self.count - 1}")
        return DeviceGroup(self.name, self.priority, tuple(self.devices[i] for i in indices))

    def __str__(self) -> str:
        if self.count > 1:
            return f"{self.name}[{self.count}]"
        return self.name

    __repr__ = __str__


DeviceLike = Union[str, DeviceGroup]

_cpu_count, _cpu_memory, _cpu_flops = _get_cpu_metrics()
_gpu_count, _gpu_memory, _gpu_flops = _get_gpu_metrics()


class device:  # noqa: N801
    cpu: Final[DeviceGroup] = DeviceGroup(
        "cpu",
        0,
        tuple(
            Device(i, "cpu", 0, _cpu_memory / _cpu_count, _cpu_flops / _cpu_count)
            for i in range(_cpu_count)
        ),
    )

    gpu: Final[DeviceGroup] = DeviceGroup(
        "gpu",
        1,
        tuple(Device(i, "gpu", 1, _gpu_memory[i], _gpu_flops[i]) for i in range(_gpu_count))
        if _gpu_count > 0
        else (),
    )

    fake_gpu: Final[DeviceGroup] = DeviceGroup(
        "fake-gpu", 1, (Device(0, "fake-gpu", 1, _cpu_memory, _cpu_flops),)
    )

    @staticmethod
    def _parse_index_spec(spec: str, max_idx: int) -> list[int]:
        """Parse an index specification like '0', '0,1,2' or '0-2'."""
        indices: list[int] = []
        for part in spec.split(","):
            if "-" in part:
                start_str, end_str = part.split("-")
                try:
                    start = int(start_str)
                    end = int(end_str)
                    if start > end:
                        start, end = end, start
                    indices.extend(range(start, end + 1))
                except ValueError as e:
                    raise ValueError(f"Invalid range specification: {part}") from e
            else:
                try:
                    indices.append(int(part))
                except ValueError as e:
                    raise ValueError(f"Invalid index: {part}") from e

        # Remove duplicates while preserving order
        indices = list(dict.fromkeys(indices))
        return indices

    @staticmethod
    def from_(s: DeviceLike) -> DeviceGroup:
        if isinstance(s, DeviceGroup):
            return s
        if isinstance(s, Device):
            return DeviceGroup(s.name, s.priority, (s,))

        assert isinstance(s, str), f"Unknown device type {s}"
        s = s.lower()

        # Check for device index specification (e.g., "gpu:0" or "gpu:0,1" or "gpu:0-2")
        if ":" in s:
            device_name, idx_spec = s.split(":")
            # Get the full device group first
            device_group = device.from_(device_name)
            indices = device._parse_index_spec(idx_spec, device_group.count - 1)
            return device_group.select_devices(indices)

        # Original device parsing logic
        if "cpu" in s:
            return device.cpu
        elif "fake-cuda" in s or "fake-gpu" in s:
            return device.fake_gpu
        elif "gpu" in s or "cuda" in s:
            return device.gpu
        else:
            raise ValueError(f"Unknown device {s}")
