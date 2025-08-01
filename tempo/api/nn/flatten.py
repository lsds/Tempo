from tempo.api.nn.module import Module
from tempo.api.recurrent_tensor import RecurrentTensor


class Flatten(Module):
    def __init__(self, start_dim: int = 0, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: RecurrentTensor) -> RecurrentTensor:
        result = x.flatten(self.start_dim, self.end_dim)

        return result
