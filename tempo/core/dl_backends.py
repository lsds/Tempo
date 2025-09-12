from __future__ import annotations

from enum import IntEnum


class DLBackendName(IntEnum):
    TORCH = 0
    JAX = 1
    NUMPY = 2

    @classmethod
    def str_to_enum(cls, name: str) -> DLBackendName:
        lower_name = name.lower().strip()

        torch_backend_names = ["pytorch", "torch"]
        jax_backend_names = ["jax", "xla"]
        numpy_backend_names = ["numpy", "np"]

        if lower_name in torch_backend_names:
            return DLBackendName.TORCH
        elif lower_name in jax_backend_names:
            return DLBackendName.JAX
        elif lower_name in numpy_backend_names:
            return DLBackendName.NUMPY
        else:
            raise ValueError(f"Invalid backend name: {name}")

    def __str__(self) -> str:
        return self.name.lower()
