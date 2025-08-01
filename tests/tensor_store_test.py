import torch

from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import OpId, OpOutId, TensorId
from tempo.core.domain import Domain
from tempo.core.dtype import dtypes
from tempo.core.shape import Shape
from tempo.runtime.tensor_store.point_tensor_store import PointRuntimeTensor
from tempo.core.device import device


def test_scalar_single_write(exec_cfg: ExecutionConfig, domain_3d: Domain):
    tensor_id = TensorId(OpId(0), OpOutId(0))
    dev = device.from_(exec_cfg.dev)
    tensor = PointRuntimeTensor[torch.Tensor](
        exec_cfg, tensor_id, Shape(()), dtypes.float32, dev, domain_3d
    )

    inp = torch.ones(())
    tensor[0, 0, 0] = inp
    out = tensor[0, 0, 0]
    assert torch.allclose(out, inp)


def test_scalar_batch_write(exec_cfg: ExecutionConfig, domain_3d: Domain):
    tensor_id = TensorId(OpId(0), OpOutId(0))
    dev = device.from_(exec_cfg.dev)
    tensor = PointRuntimeTensor[torch.Tensor](
        exec_cfg, tensor_id, Shape(()), dtypes.float32, dev, domain_3d
    )

    D0 = 5
    inp = torch.ones((D0))

    tensor[0:5, 0, 0] = inp
    out = tensor[0:5, 0, 0]
    assert torch.allclose(out, inp)


def test_scalar_batch_multi_write(exec_cfg: ExecutionConfig, domain_3d: Domain):
    tensor_id = TensorId(OpId(0), OpOutId(0))
    dev = device.from_(exec_cfg.dev)
    tensor = PointRuntimeTensor[torch.Tensor](
        exec_cfg, tensor_id, Shape(()), dtypes.float32, dev, domain_3d
    )

    D0 = 5
    inp = torch.ones((D0))

    # Write
    tensor[0:5, 0, 0] = inp
    tensor[0:5, 0, 1] = inp * 2

    # Read
    out = tensor[0:5, 0, 0]
    assert torch.allclose(out, inp)

    out = tensor[0:5, 0, 1]
    assert torch.allclose(out, inp * 2)


def test_scalar_batch_multi_write_batch_read(
    exec_cfg: ExecutionConfig, domain_3d: Domain
):
    tensor_id = TensorId(OpId(0), OpOutId(0))

    dev = device.from_(exec_cfg.dev)
    tensor = PointRuntimeTensor[torch.Tensor](
        exec_cfg, tensor_id, Shape(()), dtypes.float32, dev, domain_3d
    )

    D0 = 5
    inp = torch.ones((D0))

    # Write
    tensor[0:5, 0, 0] = inp
    tensor[0:5, 0, 1] = inp * 2

    out = tensor[0:5, 0, 0:2]
    expected = torch.stack([inp, inp * 2], dim=1)
    assert torch.allclose(out, expected)


#def test_circular_buffer_single_write(exec_cfg: ExecutionConfig):
#    tensor_id = TensorId(OpId(0), OpOutId(0))
#    dev = device.from_("cpu")
#
#
#    ctx = TempoContext(exec_cfg, 2)
#
#    (b, B), (t, T) = ctx.variables_and_bounds
#    domain = ctx.universe
#
#    with ctx:
#
#        bounds = {B: 5, T: 20}
#
#        store = PreallocCircularRuntimeTensor[torch.Tensor](
#            exec_cfg, tensor_id, Shape.from_((3,)), dtypes.float32, dev, domain, [(t, 4)], bounds, 2
#        )
#
#        inp = torch.zeros((3,))
#        store[0, 0] = inp
#        #out = store[0, 0]
#        #assert torch.allclose(out, inp), f"out: {out}, inp: {inp}, tensor: {store._buffers}"
#
#        inps = []
#
#        for i in range(1,5):
#            inp = torch.ones((3,)) * i # 2, 3, 4, 5
#            inps.append(inp)
#            store[0, i] = inp
#
#        out = store[0, 1:5]
#        expected = torch.stack(inps, dim=0)
#
#        assert out.shape == expected.shape, f"out: {out}, expected: {expected}, tensor: {store._buffers}"
#        assert torch.allclose(out, expected), f"out: {out}, expected: {expected}, tensor: {store._buffers}"
#
#        inps = []
#        for i in range(5, 18):
#            inp = torch.ones((3,)) * i
#            inps.append(inp)
#            store[0, i] = inp
#
#        out = store[0, 14:18]
#        inps = inps[-4:]
#        expected = torch.stack(inps, dim=0)
#        assert torch.allclose(out, expected), f"out: {out}, expected: {expected}, tensor: {store._buffers}"
#
