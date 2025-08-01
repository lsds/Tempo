import pytest
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.api.tempo_context_manager import TempoContext
from tempo.core.configs import ExecutionConfig


@pytest.mark.skip
def test_merge_const(exec_cfg: ExecutionConfig) -> None:
    ctx = TempoContext(num_dims=1)
    ((t, T),) = ctx.variables_and_bounds
    ctx.execution_config = exec_cfg

    with ctx:
        x = RecurrentTensor.ones(shape=(3,), requires_grad=True)
        y = RecurrentTensor.ones(shape=(3,), requires_grad=True) * 2
        x = -x
        x = -x

        x2 = RecurrentTensor.placeholder(shape=(3,), requires_grad=True, domain=(t,))

        x2[1] = x
        x2[3] = x + y
        x2[t] = RecurrentTensor.zeros(shape=(3,), requires_grad=True)

        w = (x * 3).sqrt()
        z = x.sigmoid()
        y = (w + z).sum()
        y2 = (z + y).sum()
        y.backward()
        y2.backward()

        executor = ctx.compile(bounds={T: 10})
        executor.execute()
        # print(executor.get_spatial_tensor(x2.tensor_id,(1, 3)))


#
# def test_merge_const_boolean() -> None:
#    ctx = TempoContext(num_dims=1)
#    ((b, B),) = ctx.variables_and_bounds
#
#    with ctx:
#        # ctx.execution_config.visualize_pipeline_stages = True
#        x = RecurrentTensor.ones(shape=(3,), dtype=dtypes.bool_)
#        y = RecurrentTensor.zeros(shape=(3,), dtype=dtypes.bool_)
#        z = RecurrentTensor.placeholder(shape=(3,), dtype=dtypes.bool_)
#        z[b % 2] = x[b] & y[b]
#        z[b] = x[b] | y[b]
#
#        exec = ctx.compile(bounds={B: 6})
#        exec.execute()
#
#
# def test_merge_const_with_multi_output() -> None:
#    ctx = TempoContext()
#    with ctx:
#        # ctx.execution_config.visualize_pipeline_stages = True
#        x = RecurrentTensor.arange(10, dtype=dtypes.float32)
#        y = RecurrentTensor.arange(10, dtype=dtypes.float32)
#
#        z = RecurrentTensor.max(x, y)
#        z1 = z[0] + 1
#        z2 = z[1] + 2
#
#        exec = ctx.compile({})
#        exec.execute()
#        # z1_computed = exec.get_spatial_tensor(z1.tensor_id, ())
#        # z2_computed = exec.get_spatial_tensor(z2.tensor_id, ())
#
#        # print(z1_computed)
#        # print(z2_computed)


if __name__ == "__main__":
    test_merge_const()
    # test_merge_const_boolean()
    # test_merge_const_with_multi_output()
