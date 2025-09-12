import time

from tempo.core import index_expr as ie


def test_simplifications():
    t = ie.Symbol("t", idx=0)
    T = ie.Symbol("T", is_bound=True, idx=1)
    e = t * 1 + 3 - 3 + T - T
    assert e.struct_eq(e)


def test_index_expr_api():
    ie.IntIndexValue.__eq__ = ie.IntIndexValue.symb_eq  # type: ignore
    ie.IntIndexValue.__ne__ = ie.IntIndexValue.symb_ne  # type: ignore

    t = ie.Symbol("t", idx=0)
    T = ie.Symbol("T", is_bound=True, idx=1)
    e = (t + t * 2 == t - 3) & (T <= 3)

    e2 = ie.And(
        ie.Equal(
            ie.Add(t, ie.Mul(t, ie.ConstInt(2))),
            ie.Sub(t, ie.ConstInt(3)),
        ),
        ie.LessThanOrEqual(T, ie.ConstInt(3)),
    )

    assert e.struct_eq(e2)

    assert e.evaluate({t: 1, T: 3}) == False


def test_piecewise():
    i = ie.Symbol("i", idx=0)
    I = ie.Symbol("I", is_bound=True, idx=1)
    t = ie.Symbol("t", idx=2)
    T = ie.Symbol("T", is_bound=True, idx=3)
    e0 = t + 1
    branches = e0.enumerate_all_cond_branches()
    assert len(branches) == 1
    assert branches[0][0] is None
    assert branches[0][1].struct_eq(e0)

    e1 = ie.Piecewise(((t <= 3, t), (ie.ConstBool(True), ie.ConstInt(3))))
    e2 = ie.Piecewise(((t <= 5, e1), (ie.ConstBool(True), ie.ConstInt(2))))
    e3 = 3 + e2 // 2
    branches = e3.enumerate_all_cond_branches()

    assert len(branches) == 3
    assert e3.evaluate({i: 2, I: 5, t: 4, T: 10}) == 4


def test_eval_symbol_index_expr():
    dictionary = {}

    for i in range(1000):
        s = ie.Symbol(f"i{i}", idx=i * 2)
        dictionary[s] = i

    s = ie.Symbol("i1", idx=1000*2)
    assert (s + 3).evaluate(dictionary) == 4


if __name__ == "__main__":
    ie.IntIndexValue.__eq__ = ie.IntIndexValue.symb_eq  # type: ignore
    ie.IntIndexValue.__ne__ = ie.IntIndexValue.symb_ne  # type: ignore

    t = ie.Symbol("t", idx=0)
    T = ie.Symbol("T", is_bound=True, idx=1)

    e = (t + t * 2 == t - 3) & (T <= 3)

    e2 = ie.Piecewise(
        ((e, ie.Min((t, ie.ConstInt(3)))), (ie.ConstBool(True), ie.ConstInt(2)))
    )

    e3 = ie.IndexSequence((e2, t))

    print(e3)
    print(e3._codegen_str())
    ct = {t: 1, T: 3}
    start_time = time.perf_counter()
    for _ in range(1_000_000):
        e3.evaluate(ct)
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time} seconds")
    print(e3.evaluate(ct))
    print(e3.evaluate(ct, codegen=True))

    start_time = time.perf_counter()
    for _ in range(1_000_000):
        e3.evaluate(ct)
    end_time = time.perf_counter()
    print(f"Execution time (codegen): {end_time - start_time} seconds")

    print(e3.evaluate(ct))
    print(e3.evaluate(ct, codegen=True))
    ct[t] = 0
    ct[T] = 10
    print(e3.evaluate(ct, codegen=True))
    ct[t] = 5
    print(e3.evaluate(ct, codegen=True))
    ct[t] = 10
    print(e3.evaluate(ct, codegen=True))
