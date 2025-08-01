from abc import ABC

""" Abstract base class for benchmark runners.
"""


class BenchRunner(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def compile(self, **kwargs):
        pass

    def warmup(self, **kwargs):
        pass

    def run(self, **kwargs):
        pass
