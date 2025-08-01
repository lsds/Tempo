from tempo.api import RecurrentTensor, TempoContext, min
from tempo.core.configs import ExecutionConfig


def test_no_grad_small(exec_cfg: ExecutionConfig) -> None:
    ctx = TempoContext(num_dims=2)
    (b, B), (t, T) = ctx.variables_and_bounds

    with ctx:

        x = RecurrentTensor.ones(requires_grad=False) * t
        y = RecurrentTensor.ones(requires_grad=False) * b
        z = x + y

        assert not z.requires_grad
        assert z._ctx is None
        assert z.grad is None

        z = z.sqrt()

        w = z[b, t : min(t + 3, T)].sum(dims=0)
        w = w.sigmoid()

        a = w[b, t:T].sum()

    ctx.execution_config = exec_cfg
    ctx.compile({B: 2, T: 5}).execute()


def test_grad_small(exec_cfg: ExecutionConfig) -> None:
    ctx = TempoContext(num_dims=2)
    ctx.execution_config = exec_cfg
    (b, B), (t, T) = ctx.variables_and_bounds

    with ctx:

        x = RecurrentTensor.ones(requires_grad=True)
        y = RecurrentTensor.ones(requires_grad=False)
        z1 = x + y

        z2 = z1.sqrt()

        z3 = z2.sigmoid()

        assert z2._ctx is not None

        z3.backward()

        assert x._ctx is None

        assert z2._ctx is None
        assert z2.requires_grad
        assert z2.grad is not None

        ctx.compile({B: 2, T: 5}).execute()


def test_grad_0d_small(exec_cfg: ExecutionConfig) -> None:
    ctx = TempoContext()
    ctx.execution_config = exec_cfg

    with ctx:

        x = RecurrentTensor.ones(requires_grad=True)
        y = RecurrentTensor.ones(requires_grad=False)
        z = x + y

        z = z.sqrt()

        z = z.sigmoid()
        assert z._ctx is not None

        z.backward()

        assert x._ctx is None
        assert x.grad is not None

        #TODO this is not working likely due to copies induced by ident
        #assert z.requires_grad
        #assert z.grad is not None
        #assert z._ctx is None

    ctx.compile({}).execute()


if __name__ == "__main__":
    exec_cfg = ExecutionConfig(
        path="./",
        visualize_pipeline_stages=False,
        dev="cpu",
        backend="torch+jax",
        deterministic=True,
        seed=0,
        gc_bytes_min=1024 * 1024,  # 1MiB
        enable_dataflow_grouping=False,
        enable_constant_folding=False,
        enable_dead_code_elim=False,
        enable_gc=False,
        enable_swap=False,
        enable_parallel_block_detection=False,
    )

    exec_cfg = ExecutionConfig(
        path="./",
        visualize_pipeline_stages=False,
        dev="cpu",
        backend="torch",
        deterministic=True,
        seed=0,
        gc_bytes_min=1024 * 1024,  # 1MiB
        enable_dataflow_grouping=False,
        enable_constant_folding=False,
        enable_dead_code_elim=False,
        enable_duplicate_code_elim=False,
        enable_incrementalization=False,
        enable_vectorization=False,
        enable_gc=False,
        enable_swap=False,
        enable_parallel_block_detection=False,
    )

    test_grad_small(exec_cfg)
