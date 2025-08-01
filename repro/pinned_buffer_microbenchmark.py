import time

import torch


def time_alloc(n: int, pin: bool = False) -> None:
    print(f"Allocating {n * 4 / 1e9:.2f} GB, pin={pin}")
    start = time.time()
    x = torch.empty(n, dtype=torch.float32, pin_memory=pin)
    print(f"Elapsed: {time.time() - start:.2f} seconds")


time_alloc(1_000_000_000, pin=False)  # ~4GB, unpinned
time_alloc(1_000_000_000, pin=True)  # ~4GB, pinned
