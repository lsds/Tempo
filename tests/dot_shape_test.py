from dataclasses import replace
import pytest
import numpy as np

from tempo.api import RecurrentTensor
from tempo.api.tempo_context_manager import TempoContext
from tempo.core.configs import ExecutionConfig
from tempo.core.dtype import dtypes
from tempo.core.shape import Shape


@pytest.mark.parametrize(
    "x_shape,y_shape,expected_shape",
    [
        # Case 1: Both 1D tensors
        ((3,), (3,), ()),  # Vector dot product -> scalar

        # Case 2: 1D x 2D cases
        ((3,), (3, 4), (4,)),  # Vector x Matrix -> Vector
        ((4,), (4, 3), (3,)),  # Vector x Matrix.T -> Vector

        # Case 3: 2D x 1D cases
        ((2, 3), (3,), (2,)),  # Matrix x Vector -> Vector
        ((3, 2), (2,), (3,)),  # Matrix x Vector -> Vector

        # Case 4: 2D x 2D cases
        ((2, 3), (3, 4), (2, 4)),  # Matrix multiplication
        ((3, 2), (2, 4), (3, 4)),  # Matrix multiplication

        # Case 5: Batched cases
        ((5, 2, 3), (5, 3, 4), (5, 2, 4)),  # Batched matrix multiplication
        ((2, 3), (5, 3, 4), (5, 2, 4)),  # Broadcasting batch dimension
        ((5, 2, 3), (3, 4), (5, 2, 4)),  # Broadcasting batch dimension

        # Case 6: Multiple batch dimensions
        ((5, 6, 2, 3), (5, 6, 3, 4), (5, 6, 2, 4)),  # Multiple batch dims
        ((5, 6, 2, 3), (6, 3, 4), (5, 6, 2, 4)),  # Broadcasting multiple batch dims
        ((2, 3), (5, 6, 3, 4), (5, 6, 2, 4)),  # Broadcasting multiple batch dims
    ]
)
def test_dot_shapes(x_shape: tuple, y_shape: tuple, expected_shape: tuple, exec_cfg: ExecutionConfig):
    """Test shape propagation for dot product between tensors of various dimensions"""

    # Create random tensors with the specified shapes
    x_data = np.random.rand(*x_shape).astype(np.float32)
    y_data = np.random.rand(*y_shape).astype(np.float32)

    # Create RecurrentTensors
    ctx = TempoContext(exec_cfg)
    with ctx:
        x = RecurrentTensor.const(x_data, dtype=dtypes.float32)
        y = RecurrentTensor.const(y_data, dtype=dtypes.float32)
        z = x.dot(y)
        assert z.shape == Shape.from_(expected_shape)

@pytest.mark.parametrize(
    "x_shape,y_shape,expected_shape",
    [
        # Case 1: Both 1D tensors
        ((3,), (3,), ()),  # Vector dot product -> scalar

        # Case 2: 1D x 2D cases
        ((3,), (3, 4), (4,)),  # Vector x Matrix -> Vector
        ((4,), (4, 3), (3,)),  # Vector x Matrix.T -> Vector

        # Case 3: 2D x 1D cases
        ((2, 3), (3,), (2,)),  # Matrix x Vector -> Vector
        ((3, 2), (2,), (3,)),  # Matrix x Vector -> Vector

        # Case 4: 2D x 2D cases
        ((2, 3), (3, 4), (2, 4)),  # Matrix multiplication
        ((3, 2), (2, 4), (3, 4)),  # Matrix multiplication

        # Case 5: Batched cases
        ((5, 2, 3), (5, 3, 4), (5, 2, 4)),  # Batched matrix multiplication
        ((2, 3), (5, 3, 4), (5, 2, 4)),  # Broadcasting batch dimension
        ((5, 2, 3), (3, 4), (5, 2, 4)),  # Broadcasting batch dimension

        # Case 6: Multiple batch dimensions
        ((5, 6, 2, 3), (5, 6, 3, 4), (5, 6, 2, 4)),  # Multiple batch dims
        ((5, 6, 2, 3), (6, 3, 4), (5, 6, 2, 4)),  # Broadcasting multiple batch dims
        ((2, 3), (5, 6, 3, 4), (5, 6, 2, 4)),  # Broadcasting multiple batch dims
    ]
)
def test_dot_shapes_no_matmul_op(x_shape: tuple, y_shape: tuple, expected_shape: tuple, exec_cfg: ExecutionConfig):
    """Test shape propagation for dot product between tensors of various dimensions"""

    # Create random tensors with the specified shapes
    x_data = np.random.rand(*x_shape).astype(np.float32)
    y_data = np.random.rand(*y_shape).astype(np.float32)

    # Create RecurrentTensors
    exec_cfg = replace(exec_cfg, enable_matmul_ops=False)
    ctx = TempoContext(exec_cfg)
    with ctx:
        x = RecurrentTensor.const(x_data, dtype=dtypes.float32)
        y = RecurrentTensor.const(y_data, dtype=dtypes.float32)
        z = x.dot(y)
        assert z.shape == Shape.from_(expected_shape)

#def test_dot_shape_errors(exec_cfg: ExecutionConfig):
#    """Test that invalid shape combinations raise appropriate errors"""
#
#    ctx = TempoContext(exec_cfg)
#    with ctx:
#        # Incompatible dimensions
#        x = RecurrentTensor.const(np.random.rand(2, 3))
#        y = RecurrentTensor.const(np.random.rand(4, 5))
#
#        with pytest.raises(AssertionError):
#            _ = x.dot(y)
#
#        # Invalid number of dimensions
#        x = RecurrentTensor.const(np.random.rand(2, 3, 4, 5))
#        y = RecurrentTensor.const(np.random.rand(3, 4))
#
#        with pytest.raises(ValueError):
#            _ = x.dot(y)
#
#
#def test_dot_singleton_dims(exec_cfg: ExecutionConfig):
#    """Test dot product with singleton dimensions that should be squeezed"""
#
#    ctx = TempoContext(exec_cfg)
#    with ctx:
#        # 2D x 2D with singleton dimensions
#        x = RecurrentTensor.const(np.random.rand(1, 3))
#        y = RecurrentTensor.const(np.random.rand(3, 1))
#        _check_dot_result(x, y, (), exec_cfg)  # Should squeeze to scalar
#
#        # Batched case with singleton dimensions
#        x = RecurrentTensor.const(np.random.rand(5, 1, 3))
#        y = RecurrentTensor.const(np.random.rand(5, 3, 1))
#        _check_dot_result(x, y, (5,), exec_cfg)  # Should squeeze to 1D
