from __future__ import annotations

import contextlib
from types import TracebackType
from typing import ContextManager

import torch


class TorchProfiler:
    def __init__(self, results_path: str) -> None:
        self.results_path = results_path[:-1] if results_path.endswith("/") else results_path

        self.torch_prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=True,
            record_shapes=False,
            profile_memory=False,
            with_flops=False,
            use_cuda=True,
        )

    def __enter__(self) -> TorchProfiler:
        self.torch_prof.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        self.torch_prof.__exit__(exc_type, exc_val, exc_tb)

        self.torch_prof.export_chrome_trace(f"{self.results_path}/torch_profile.json")

        return None

    @staticmethod
    def get(enabled: bool, results_path: str) -> ContextManager:
        if enabled:
            return TorchProfiler(results_path)
        else:
            return contextlib.nullcontext()
