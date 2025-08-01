import pytest

from tempo.core.configs import ExecutionConfig
from tempo.core.domain import Domain
from tempo.core.utils import make_symbols


@pytest.fixture
def exec_cfg():
    return ExecutionConfig.test_cfg()
    #return ExecutionConfig.test_debug_cfg()


@pytest.fixture
def domain_3d():
    (d0, D0), (d1, D1), (d2, D2) = make_symbols(("d0", "d1", "d2"))
    return Domain.from_vars_and_bounds((d0, d1, d2), (D0, D1, D2))


@pytest.fixture
def domain_0d():
    return Domain.from_vars_and_bounds((), ())
