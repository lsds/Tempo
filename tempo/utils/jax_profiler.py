from __future__ import annotations

import contextlib
from types import TracebackType
from typing import ContextManager, Optional, Type

import jax


class JaxProfiler:
    def __init__(self, res_dir: str) -> None:
        self.results_path = res_dir

    def __enter__(self) -> JaxProfiler:
        jax.profiler.start_trace(self.results_path)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        jax.profiler.stop_trace()
        return None

    @staticmethod
    def get(enabled: bool, results_path: str) -> ContextManager:
        if enabled:
            return JaxProfiler(results_path)
        else:
            return contextlib.nullcontext()
