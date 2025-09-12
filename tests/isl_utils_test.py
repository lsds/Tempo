import tempo.core.global_objects as glob
from tempo.core import index_expr as ie
from tempo.core.dependence_graph import PDG
from tempo.core.domain import Domain
from tempo.core.shape import Shape
from tempo.utils import isl as isl_utils


def mock_dg(domain: Domain) -> PDG:
    dg = PDG(domain)

    dg.bound_defs = {d: 100 for d in domain.ubounds}

    return dg


def test_simplify_cond(domain_3d: Domain) -> None:
    glob.set_active_dg(mock_dg(domain_3d))
    ie.IntIndexValue.__eq__ = ie.IntIndexValue.symb_eq  # type: ignore
    ie.IntIndexValue.__ne__ = ie.IntIndexValue.symb_ne  # type: ignore
    d0, d1, d2 = domain_3d.variables
    D0, D1, D2 = domain_3d.ubounds

    cond = (d0 % 5 == 0) & (d0 % 3 == 0)
    simplified = isl_utils.simplify_boolean_index_expr(domain_3d, cond)

    assert simplified.struct_eq(d0 % 15 == 0)

    cond = (d0 == 0) & ~((d0 == 0) & (d1 == 0))
    simplified = isl_utils.simplify_boolean_index_expr(domain_3d, cond)

    print(simplified)

    assert simplified.struct_eq((d0 == 0) & (d1 >= 1))


def test_simplify_dependence_expr(domain_3d: Domain) -> None:
    glob.set_active_dg(mock_dg(domain_3d))

    d0, d1, d2 = domain_3d.variables
    D0, D1, D2 = domain_3d.ubounds

    expr = ie.IndexSequence((d0, d1, d2))
    simplified_expr = isl_utils.simplify_dependence_expr(expr, domain_3d, domain_3d)
    print(expr)
    print(simplified_expr)
    assert simplified_expr.struct_eq(expr)

    expr = ie.IndexSequence((d0 - 1 + 2 - 1, d1, d2))
    simplified_expr = isl_utils.simplify_dependence_expr(expr, domain_3d, domain_3d)
    print(expr)
    print(simplified_expr)
    assert simplified_expr.struct_eq(ie.IndexSequence((d0, d1, d2)))


def test_reverse_dependence_expr(domain_3d: Domain) -> None:
    glob.set_active_dg(mock_dg(domain_3d))

    d0, d1, d2 = domain_3d.variables
    D0, D1, D2 = domain_3d.ubounds

    # d0 + 2, d1 - 2, d2 + 4 -> d0 - 2, d1 + 2, d2 - 4
    e = ie.IndexSequence((d0 + 2, d1 - 2, d2 + 4))
    reversed_expr = isl_utils.reverse_dependence_expr(e, e.vars_used(), domain_3d)
    assert reversed_expr.struct_eq(ie.IndexSequence((d0 - 2, d1 + 2, d2 - 4)))

    # 0:D0, d1 - 2, d2:D2 -> d1 + 2, 0:d2 + 1
    e = ie.IndexSequence((ie.Slice(ie.ConstInt(0), D0), d1 - 2, ie.Slice(d2, D2)))
    reversed_expr = isl_utils.reverse_dependence_expr(e, e.vars_used(), domain_3d)
    assert reversed_expr.struct_eq(
        ie.IndexSequence((d1 + 2, ie.Slice(ie.ConstInt(0), d2 + 1)))
    )

    # Constant exprs invert into nothing
    e = ie.IndexSequence((ie.ConstInt(5), ie.ConstInt(2), ie.ConstInt(1)))
    reversed_expr = isl_utils.reverse_dependence_expr(e, e.vars_used(), domain_3d)
    assert reversed_expr.struct_eq(ie.IndexSequence(()))

    domain_1d = Domain.from_vars((d1,))
    domain_2d = Domain.from_vars((d0, d1))

    e = ie.IndexSequence((ie.slice_(ie.max(0, d1 - 3), d1+1),))
    reversed_expr = isl_utils.reverse_dependence_expr(e, domain_1d, domain_1d)
    expected = ie.IndexSequence((ie.slice_(d1, ie.min(glob.get_active_dg().static_bounds[D1], d1 + 4 )),))
    assert ie.logical_eq(reversed_expr,expected), f"For inverting {e}, expected {expected} got {reversed_expr}"

    # Sink[d0,d1] = f(Source[d1])
    e = ie.IndexSequence((d1,))
    reversed_expr = isl_utils.reverse_dependence_expr(e, domain_2d, domain_1d)
    expected1 = ie.IndexSequence(
        (
            ie.Slice(ie.ConstInt(0), D0),
            d1,
        )
    )
    from tempo.core.global_objects import get_active_dg
    expected2 = ie.IndexSequence(
        (
            ie.Slice(ie.ConstInt(0), get_active_dg().static_bounds[D0]),
            d1,
        )
    )
    assert str(reversed_expr) == str(
        expected1
    ) or str(reversed_expr) == str(
        expected2
    ), f"Expected {expected1} or {expected2}, got {reversed_expr}"

    # Sink[d1] = f(Source[5,d1])
    e = ie.IndexSequence(
        (
            ie.ConstInt(5),
            d1,
        )
    )
    reversed_expr = isl_utils.reverse_dependence_expr(e, domain_1d, domain_2d)
    expected = ie.IndexSequence((d1,))
    assert str(reversed_expr) == str(
        expected
    ), f"Expected {expected}, got {reversed_expr}"


def simplify_shape_test(domain_3d: Domain) -> None:
    glob.set_active_dg(mock_dg(domain_3d))

    d0, d1, d2 = domain_3d.variables
    D0, D1, D2 = domain_3d.ubounds

    shape = Shape.from_((200, ((d1 + 1) * 20) - (d1 * 20), (d2) * 4 + 5))
    print(shape)
    print(isl_utils.simplify_shape(shape))


def test_isl_expr_eq(domain_3d: Domain) -> None:
    glob.set_active_dg(mock_dg(domain_3d))

    d0, d1, d2 = domain_3d.variables
    D0, D1, D2 = domain_3d.ubounds

    assert isl_utils.expr_eq(5, 5)
    assert not isl_utils.expr_eq(5, 3)

    assert isl_utils.expr_eq(ie.ConstInt(5), 5)
    assert isl_utils.expr_eq(5, ie.ConstInt(5))
    assert isl_utils.expr_eq(ie.ConstInt(5), ie.ConstInt(5))

    assert isl_utils.expr_eq((5,), (5,))
    assert isl_utils.expr_eq(ie.IndexSequence((ie.ConstInt(5),)), (5,))
    assert isl_utils.expr_eq((5,), ie.IndexSequence((ie.ConstInt(5),)))
    assert not isl_utils.expr_eq(ie.IndexSequence((ie.ConstInt(5),)), 5)

    assert isl_utils.expr_eq(d1, d1)
    assert not isl_utils.expr_eq(d1, d1 + 1)
    assert not isl_utils.expr_eq(d1, d2)

    assert isl_utils.expr_eq((d1,), (d1,))
    assert not isl_utils.expr_eq((d1,), (d1+1,))
    assert not isl_utils.expr_eq((d1,), (d2+1,))
    assert not isl_utils.expr_eq((d1,), (d1, d2))

    # d0 + 2, d1 - 2, d2 + 4
    e1 = ie.IndexSequence((d0 + 2, d1 - 2, d2 + 4))
    e2 = ie.IndexSequence((d0 - 2, d1 + 2, d2 - 4))
    assert not isl_utils.expr_eq(e1, e2), f" should not be equal"
    e2 = ie.IndexSequence((d0 + 2, d1 - 2, d2 + 4))
    assert isl_utils.expr_eq(e1, e2), f" should be equal"
    e2 = ie.IndexSequence((2 + d0, d1 - 2 , 4 + d2 ))
    assert isl_utils.expr_eq(e1, e2), f" Does not commute"



    # 0:D0, d1 - 2, d2:D2
    e1 = ie.IndexSequence((ie.Slice(ie.ConstInt(0), D0), d1 - 2, ie.Slice(d2, D2)))
    e2 = ie.IndexSequence((ie.Slice(ie.ConstInt(0), D0), d1 - 2, ie.Slice(d2, D2)))
    assert isl_utils.expr_eq(e1, e2), f" should be equal"
    e2 = ie.IndexSequence((ie.Slice(ie.ConstInt(0) - 1 +2 -1, D0 * 2 // 2), d1 - 2, ie.Slice(d2, D2)))
    assert isl_utils.expr_eq(e1, e2), f" should be equal"
    e2 = ie.IndexSequence((ie.Slice(ie.ConstInt(0) - 1 +2 -1, D0 * 2 // 2), d1 - 2, ie.Slice(d2, D2-1)))
    assert not isl_utils.expr_eq(e1, e2), f" should not be equal"


    # 0:D0, d1 - 2, d2:D2
    e1 = ie.IndexSequence((ie.ConstInt(5), ie.ConstInt(2), d2))
    e2 = ie.IndexSequence((ie.ConstInt(5), ie.ConstInt(2), d2))
    assert isl_utils.expr_eq(e1, e2), f" should be equal"
    e2 = ie.IndexSequence((ie.ConstInt(5), ie.ConstInt(2), ie.ConstInt(1)+d2 - 1))
    assert isl_utils.expr_eq(e1, e2), f" should be equal"
    e2 = ie.IndexSequence((ie.ConstInt(3), ie.ConstInt(2), ie.ConstInt(1)+d2 - 1))
    assert not isl_utils.expr_eq(e1, e2), f" should not be equal"


def test_expr_eq_logical_and_struct(domain_3d: Domain) -> None:
    glob.set_active_dg(mock_dg(domain_3d))

    d0, d1, d2 = domain_3d.variables
    D0, D1, D2 = domain_3d.ubounds

    assert ie.struct_eq(5, 5)
    assert ie.logical_eq(5, 5)

    assert not ie.struct_eq(5, 3)
    assert not ie.logical_eq(5, 3)

    assert ie.struct_eq(ie.ConstInt(5), 5)
    assert ie.logical_eq(ie.ConstInt(5), 5)

    assert not ie.struct_eq(ie.ConstInt(5), 3)
    assert not ie.logical_eq(ie.ConstInt(5), 3)



    assert ie.struct_eq((5,), (5,))
    assert ie.logical_eq((5,), (5,))

    assert ie.struct_eq(ie.IndexSequence((ie.ConstInt(5),)), (5,))
    assert ie.logical_eq(ie.IndexSequence((ie.ConstInt(5),)), (5,))

    assert ie.struct_eq((5,), ie.IndexSequence((ie.ConstInt(5),)))
    assert ie.logical_eq((5,), ie.IndexSequence((ie.ConstInt(5),)))

    assert not ie.struct_eq(ie.IndexSequence((ie.ConstInt(5),)), 5)
    assert not ie.logical_eq(ie.IndexSequence((ie.ConstInt(5),)), 5)

    assert ie.struct_eq(d1, d1)
    assert not ie.struct_eq(d1, d1 + 1)
    assert not ie.struct_eq(d1, d2 + 1)
    assert not ie.struct_eq(d1, d2)

    assert ie.logical_eq(d1, d1)
    assert not ie.logical_eq(d1, d1 + 1)
    assert not ie.logical_eq(d1, d2 + 1)
    assert not ie.logical_eq(d1, d2)


    ## d0 + 2, d1 - 2, d2 + 4
    e1 = ie.IndexSequence((d0 + 2, d1 - 2, d2 + 4))
    e2 = ie.IndexSequence((d0 - 2, d1 + 2, d2 - 4))
    assert not ie.struct_eq(e1, e2), f" should not be equal"
    assert not ie.logical_eq(e1, e2), f" should not be equal"
    e2 = ie.IndexSequence((d0 + 2, d1 - 2, d2 + 4))
    assert ie.struct_eq(e1, e2), f" should be equal"
    assert ie.logical_eq(e1, e2), f" should be equal"
    e2 = ie.IndexSequence((2 + d0, d1 - 2 , 4 + d2 ))
    assert ie.struct_eq(e1, e2), f" Reflexivity should be allowed in struct_eq"
    assert ie.logical_eq(e1, e2), f" Reflexivity should be allowed in logical_eq"



    ## 0:D0, d1 - 2, d2:D2
    e1 = ie.IndexSequence((ie.Slice(ie.ConstInt(0), D0), d1 - 2, ie.Slice(d2, D2)))
    e2 = ie.IndexSequence((ie.Slice(ie.ConstInt(0), D0), d1 - 2, ie.Slice(d2, D2)))
    assert ie.struct_eq(e1, e2), f" should be equal"
    assert ie.logical_eq(e1, e2), f" should be equal"
    e2 = ie.IndexSequence((ie.Slice(ie.ConstInt(0) - 1 +2 -1, D0 * 2 // 2), d1 - 2, ie.Slice(d2, D2)))
    assert not ie.struct_eq(e1, e2), f"Expected struct_eq to not be powerful enough"
    assert ie.logical_eq(e1, e2), f"Logical eq should be powerful enough"
    e2 = ie.IndexSequence((ie.Slice(ie.ConstInt(0) - 1 +2 -1, D0 * 2 // 2), d1 - 2, ie.Slice(d2, D2-1)))
    assert not ie.struct_eq(e1, e2), f" should not be equal"
    assert not ie.logical_eq(e1, e2), f" should not be equal"


    ## 0:D0, d1 - 2, d2:D2
    e1 = ie.IndexSequence((ie.ConstInt(5), ie.ConstInt(2), d2))
    e2 = ie.IndexSequence((ie.ConstInt(5), ie.ConstInt(2), d2))
    assert ie.struct_eq(e1, e2), f" should be equal"
    assert ie.logical_eq(e1, e2), f" should be equal"
    e2 = ie.IndexSequence((ie.ConstInt(5), ie.ConstInt(2), ie.ConstInt(1)+d2 - 1))
    assert not ie.struct_eq(e1, e2), f" should be equal"
    assert ie.logical_eq(e1, e2), f" should be equal"
    e2 = ie.IndexSequence((ie.ConstInt(3), ie.ConstInt(2), ie.ConstInt(1)+d2 - 1))
    assert not ie.struct_eq(e1, e2), f" should not be equal"
    assert not ie.logical_eq(e1, e2), f" should not be equal"

    e1 = ie.Min((ie.ConstInt(5), ie.ConstInt(2), d2))
    e2 = ie.Min((ie.ConstInt(2), d2,ie.ConstInt(5)))
    assert ie.struct_eq(e1, e2), f" should be equal"
    assert ie.logical_eq(e1, e2), f" should be equal"

    e1 = ie.Min((ie.ConstInt(5), ie.ConstInt(2), d2))
    e2 = ie.Min((ie.ConstInt(2), d2,ie.ConstInt(6)))
    assert not ie.struct_eq(e1, e2), f"should not be equal, struct_eq is not powerful enough"
    assert ie.logical_eq(e1, e2), f" should be equal, logical_eq should eliminate 2"

from tempo.core.utils import make_symbols

(d0, D0), (d1, D1), (d2, D2) = make_symbols(("d0", "d1", "d2"))
domain_3d = Domain.from_vars_and_bounds((d0, d1, d2), (D0, D1, D2))
simplify_shape_test(domain_3d)
