from abc import ABC

""" Abstract base class for benchmark runners.
"""


class BenchRunner(ABC):
    def __init__(self, *args, **kwargs): ...

    def compile(self, **kwargs): ...

    def warmup(self, **kwargs): ...

    def run(self, **kwargs): ...
