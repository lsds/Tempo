from tempo.core import tensor_ops as top
from tempo.core.datatypes import OpId
from tempo.core.dependence_graph import PDG
from tempo.core.domain import Domain
from tempo.core.shape import Shape


def test_affected_dims_static(domain_3d: Domain):
    from tempo.core.global_objects import set_active_dg

    set_active_dg(PDG(domain_3d))

    input_shape = Shape((256, 512, 32, 64))
    output_shape = Shape((256, 512, 32 * 64))
    reshape = top.ReshapeOp(OpId(0), domain_3d, {}, output_shape)
    assert set(reshape.dims_affected((input_shape,))) == {2, 3}

    input_shape = Shape((256, 512, 1, 32, 64))
    output_shape = Shape((256, 512, 32 * 64))
    reshape = top.ReshapeOp(OpId(0), domain_3d, {}, output_shape)
    assert set(reshape.dims_affected((input_shape,))) == {2, 3, 4}

    input_shape = Shape((256, 512, 32, 64))
    output_shape = Shape((256 * 512, 32 * 64))
    reshape = top.ReshapeOp(OpId(0), domain_3d, {}, output_shape)
    assert set(reshape.dims_affected((input_shape,))) == {0, 1, 2, 3}

    input_shape = Shape((256, 512, 32, 64))
    output_shape = Shape((1, 256, 512 * 32 * 64))
    reshape = top.ReshapeOp(OpId(0), domain_3d, {}, output_shape)
    assert set(reshape.dims_affected((input_shape,))) == {0, 1, 2, 3}

    input_shape = Shape((256, 512))
    output_shape = Shape((256 // 4, 4, 512))
    reshape = top.ReshapeOp(OpId(0), domain_3d, {}, output_shape)
    assert set(reshape.dims_affected((input_shape,))) == {0}

    input_shape = Shape((256, 512))
    output_shape = Shape((256, 4, 512 // 4))
    reshape = top.ReshapeOp(OpId(0), domain_3d, {}, output_shape)
    assert set(reshape.dims_affected((input_shape,))) == {1}
