from dataclasses import replace
from typing import Any, List

from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.api.rl.replay_buffer.replay_buffer import ReplayBuffer
from tempo.api.tempo_context_manager import TempoContext
from tempo.core.configs import ExecutionConfig
from tempo.core.dtype import dtypes
from tempo.core.shape import Shape


def log_to_arr(
    x: Any,
    arr: list[Any],
) -> None:
    arr.append(x)


def test_replay(exec_cfg: ExecutionConfig) -> None:
    ctx = TempoContext(num_dims=2)
    ctx.execution_config = exec_cfg

    item_shapes = [
        Shape((3,)),
    ]
    item_dtypes = [dtypes.float32]

    inserted_data: list[float] = []
    sampled_data: list[float] = []

    with ctx:
        s, i = ctx.variables
        S, I = ctx.upper_bounds

        replay_memory = ReplayBuffer.make(64, item_shapes, item_dtypes)

        # ==== Insert
        data = RecurrentTensor.placeholder(Shape((3,)), dtypes.float32, domain=(i,))
        data[i] = RecurrentTensor.const(5.1, Shape((3,)))  # data[i - 1] * 2
        data.sink_udf(lambda x: log_to_arr(x, inserted_data))
        token = replay_memory.insert((data[i],), domain=(i,))

        # # ==== Sample
        sample = replay_memory.sample((s,), token[0:I])
        a = sample[0]
        a.sink_udf(lambda x: log_to_arr(x, sampled_data))

        # ==== Execute
        executor = ctx.compile(bounds={I: 10, S: 5})
        executor.execute()

    # print(f"Inserted data: {inserted_data}")
    # print(f"Sampled data: {sampled_data}")
    for d1, d2 in zip(inserted_data, sampled_data):
        assert all(d1 == d2)  # type: ignore


if __name__ == "__main__":
    cfg = ExecutionConfig.test_cfg()
    cfg = replace(cfg, visualize_pipeline_stages=True)
    test_replay(cfg)
