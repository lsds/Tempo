import random
from collections import deque
from typing import Any, Deque, List, Sequence

import numpy as np

from tempo.api.rl.replay_buffer.replay_buffer_registry import ReplayBufferRegistry
from tempo.api.rl.replay_buffer.runtime_replay_buffer_interface import (
    ReplayBufferCtx,
    RuntimeReplayBufferInterface,
)
from tempo.runtime.backends.backend import DLBackend


class DequeReplayBuffer(RuntimeReplayBufferInterface):
    def __init__(self, ctx: ReplayBufferCtx):
        super().__init__()
        self.ctx = ctx

        self.backend = DLBackend.get_backend(self.ctx.exec_cfg.backend)

        self.random = random.Random(self.ctx.exec_cfg.seed)
        self.num_items = len(self.ctx.item_shapes)
        self.capacity = 0

        # NOTE: Doing this to pre-allocate large lists inside the deque
        def mk_deque() -> Deque:
            dq: Deque = deque(maxlen=self.ctx.max_size)
            dq.extend(list(range(self.ctx.max_size)))
            dq.clear()
            return dq

        self._buffers: List[Deque] = [mk_deque() for _ in range(self.num_items)]

    def clear(self) -> None:
        for b in self._buffers:
            b.clear()
        self.capacity = 0

    def insert(self, data: Sequence[Any]) -> None:
        for i, d in enumerate(data):
            self._buffers[i].append(d)
        self.capacity += 1

    def insert_batched(self, data: Sequence[Any]) -> None:
        for i, d in enumerate(data):
            unbound = self.backend.unbind(d, axis=0)
            self._buffers[i].extend(unbound)

        self.capacity = min(self.capacity + len(unbound), self.ctx.max_size)

    def sample(self) -> Sequence[Any]:
        idx = self.random.randint(0, self.capacity - 1)
        return tuple(b[idx] for b in self._buffers)

    def sample_batched(self, num_samples: int) -> Sequence[Any]:
        replacement = num_samples > self.capacity

        # TODO -1 needed?
        idx = np.random.choice(self.capacity, size=num_samples, replace=replacement).tolist()

        ret = []
        for b in self._buffers:
            samples = [b[i] for i in idx]
            # samples = []
            # for i in idx:
            #    try:
            #        samples.append(b[i])
            #    except Exception as e:
            #        print(f"{self.capacity=}, {num_samples=}, {i=}, {len(b)=}")
            #        print(f"{idx=}")
            #        raise e
            batch_sample = self.backend.stack(samples, axis=0)
            ret.append(batch_sample)

        return ret


ReplayBufferRegistry.register_replay_buffer("deque", DequeReplayBuffer)
