from tempo.core.dependence_graph import PDG
from tempo.core.domain import Domain
from tempo.core import global_objects as glob
from tempo.core import index_expr as ie
from typing import Dict

def mock_dg(domain: Domain, bounds: dict[ie.Symbol, int]) -> PDG:
    class MockDG:
        universe = domain

        static_bounds = bounds
        dynamic_bounds = {}

    return MockDG()

def test_domain_count(domain_3d: Domain) -> None:
    d0, d1, d2 = domain_3d.variables
    D0, D1, D2 = domain_3d.ubounds
    glob.set_active_dg(mock_dg(domain_3d, {D0: 5, D1: 6, D2: 7}))

    eval_val = domain_3d.linearized_count_expr.evaluate(
        {d0: 2, d1: 3, d2: 5, D0: 5, D1: 6, D2: 7}
    )
    assert eval_val == 110, f"Expected 110, got {eval_val}"


def test_domain_lex_prev(domain_3d: Domain) -> None:
    d0, d1, d2 = domain_3d.variables
    D0, D1, D2 = domain_3d.ubounds

    from tempo.core import index_expr as ie

    ie.IntIndexValue.__eq__ = ie.IntIndexValue.symb_eq  # type: ignore
    ie.IntIndexValue.__ne__ = ie.IntIndexValue.symb_ne  # type: ignore
    expr = domain_3d.lex_prev_expr

    assert expr.evaluate({d0: 1, d1: 1, d2: 1, D0: 5, D1: 5, D2: 5}) == (1, 1, 0)
    assert expr.evaluate({d0: 1, d1: 1, d2: 0, D0: 5, D1: 5, D2: 5}) == (1, 0, 4)
    assert expr.evaluate({d0: 1, d1: 0, d2: 0, D0: 5, D1: 5, D2: 5}) == (0, 4, 4)
    assert expr.evaluate({d0: 1, d1: 0, d2: 1, D0: 5, D1: 5, D2: 5}) == (1, 0, 0)

    # assert expr.evaluate({d0: 0, d1: 0, d2: 0, D0: 5, D1: 5, D2: 5}) == (4, 4, 4)


def test_domain_lex_next(domain_3d: Domain) -> None:
    d0, d1, d2 = domain_3d.variables
    D0, D1, D2 = domain_3d.ubounds

    from tempo.core import index_expr as ie

    ie.IntIndexValue.__eq__ = ie.IntIndexValue.symb_eq  # type: ignore
    ie.IntIndexValue.__ne__ = ie.IntIndexValue.symb_ne  # type: ignore

    expr = domain_3d.lex_next_expr

    # Test incrementing the last value
    assert expr.evaluate({d0: 1, d1: 1, d2: 1, D0: 5, D1: 5, D2: 5}) == (1, 1, 2)

    # Test wrap-around and increment of the previous value
    assert expr.evaluate({d0: 1, d1: 1, d2: 4, D0: 5, D1: 5, D2: 5}) == (1, 2, 0)

    # Test sequential wrap-around affecting all values
    assert expr.evaluate({d0: 2, d1: 4, d2: 4, D0: 5, D1: 5, D2: 5}) == (3, 0, 0)

    # Test increment with no wrap-around needed
    assert expr.evaluate({d0: 1, d1: 0, d2: 1, D0: 5, D1: 5, D2: 5}) == (1, 0, 2)

    # with pytest.raises(Exception):
    #    expr.evaluate({d0: 4, d1: 4, d2: 4, D0: 5, D1: 5, D2: 5})
