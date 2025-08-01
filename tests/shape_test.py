from tempo.core.dependence_graph import PDG
from tempo.core.domain import Domain
from tempo.core.shape import Shape
from tempo.core.utils import make_symbols


def test_shape_resolution_static(domain_3d: Domain):
    from tempo.core.global_objects import set_active_dg

    set_active_dg(PDG(domain_3d))

    ((t, T),) = make_symbols(("t",))

    shape = Shape((3, 3))
    assert shape.is_static()
    shape = shape.try_resolve({})
    assert shape == Shape((3, 3))


def test_shape_resolution_dynamic(domain_3d: Domain):
    from tempo.core.global_objects import set_active_dg

    set_active_dg(PDG(domain_3d))

    ((d0, D0),) = make_symbols(("d0",))
    shape = Shape((D0 - d0, 5, 2))
    assert not shape.is_static()
    assert shape.is_dynamic()

    shape = shape.try_resolve({})
    assert not shape.is_static()
    assert shape.is_dynamic()

    shape = shape.try_resolve({D0: 10})
    assert not shape.is_static()
    assert shape.is_dynamic()

    assert shape.evaluate({d0: 5, D0: 10}) == (5, 5, 2)

    shape2 = shape.try_resolve({d0: 5, D0: 10})

    assert shape2.is_static()
    assert not shape2.is_dynamic()
    assert shape2 == Shape((5, 5, 2))
