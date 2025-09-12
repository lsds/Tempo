from __future__ import annotations

import contextlib
import cProfile
from pathlib import Path
from types import TracebackType
from typing import ContextManager


class Profiler:
    def __init__(self, results_path: str) -> None:
        self.results_path = Path(results_path)

    def __enter__(self) -> Profiler:
        self.pr = cProfile.Profile()
        self.pr.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        self.pr.__exit__(exc_type, exc_val, exc_tb)
        self.pr.dump_stats(self.results_path / "profile.prof")
        # self.pr.print_stats()
        return None

    @staticmethod
    def get(enabled: bool, results_path: str) -> ContextManager:
        if enabled:
            return Profiler(results_path)
        else:
            return contextlib.nullcontext()
