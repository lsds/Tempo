from __future__ import annotations

import builtins
import functools
import itertools
import math
import traceback
import typing
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Set,
    SupportsIndex,
    Tuple,
    Union,
    overload,
)

import sympy

from tempo.utils import logger

log = logger.get_logger(__name__)

""" BNF for the index expression language

<IndexExpr>          ::= <IndexAtom>
                       | <IndexSequence>

<IndexSequence>      ::= '[' <IndexAtom> (',' <IndexAtom>)* ']'

<IndexAtom>          ::= <IndexValue>
                       | <Slice>

<Slice>              ::= <IntIndexValue> ':' <IntIndexValue> [':' <IntIndexValue>]

<IndexValue>         ::= <IntIndexValue>
                       | <BoolIndexValue>

<IntIndexValue>      ::= <ConstInt>
                       | <Symbol>
                       | '-' <IntIndexValue>
                       | <IntBinaryExpr>
                       | <NAryExpr>
                       | <RandExpr>
                       | <Piecewise>
                       | '(' <IntIndexValue> ')'

<BoolIndexValue>     ::= ('True' | 'False')
                       | 'not' <BoolIndexValue>
                       | <BoolComparisonExpr>
                       | <BoolLogicExpr>
                       | '(' <BoolIndexValue> ')'

<ConstInt>           ::= integer_literal

<Symbol>             ::= identifier

<RandExpr>           ::= 'rand' '(' <IntIndexValue> ',' <IntIndexValue> ')'

<Piecewise>          ::= 'piecewise' '(' <ConditionExprPair> (',' <ConditionExprPair>)* ')'

<ConditionExprPair>  ::= '(' <BoolIndexValue> ',' <IntIndexValue> ')'

<IntBinaryExpr>      ::= <IntIndexValue> <IntBinaryOp> <IntIndexValue>

<IntBinaryOp>        ::= '+' | '-' | '*' | '//' | '/' | '%' | '**'

<BoolComparisonExpr> ::= <IntIndexValue> <ComparisonOp> <IntIndexValue>

<ComparisonOp>       ::= '<' | '<=' | '>' | '>=' | '==' | '!='

<BoolLogicExpr>      ::= <BoolIndexValue> ('and' | 'or') <BoolIndexValue>

<NAryExpr>           ::= 'min' '(' <IntIndexValue> (',' <IntIndexValue>)* ')'
                       | 'max' '(' <IntIndexValue> (',' <IntIndexValue>)* ')'
"""


def symbols(names: str) -> Tuple[Symbol, ...]:
    return tuple(Symbol(name) for name in names.split(" "))


def bound_symbols(names: str) -> Tuple[Symbol, ...]:
    return tuple(Symbol(name, is_bound=True) for name in names.split(" "))


class StructEqCheckCtxManager:
    def __init__(self) -> None:
        self.original_eq = IntIndexValue.__eq__
        self.original_ne = IntIndexValue.__ne__

    def __enter__(self) -> None:
        IntIndexValue.__eq__ = IntIndexValue.struct_eq  # type: ignore
        IntIndexValue.__ne__ = lambda x, y: not IntIndexValue.struct_eq(  # type: ignore
            x, y
        )

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        IntIndexValue.__eq__ = self.original_eq  # type: ignore
        IntIndexValue.__ne__ = self.original_ne  # type: ignore


def to_sympy_expr(ie: IndexExpr) -> Union[sympy.Expr, Tuple[sympy.Expr, ...]]:  # noqa: C901
    if isinstance(ie, IndexSequence):
        return tuple(_to_sympy_expr(member) for member in ie.members)
    else:
        return _to_sympy_expr(ie)


def _to_sympy_expr(ie: IndexExpr) -> sympy.Expr:  # noqa: C901
    """Convert an IndexExpr to a sympy expression.

    Args:
        ie: The IndexExpr to convert

    Returns:
        A sympy expression representing the IndexExpr
    """
    if isinstance(ie, ConstInt):
        return sympy.Integer(ie.const)
    elif isinstance(ie, Symbol):
        # NOTE: extended_nonnegative includes infinity
        return sympy.Symbol(ie.name, nonnegative=True, integer=True)
    elif isinstance(ie, Add):
        return _to_sympy_expr(ie.left_operand) + _to_sympy_expr(ie.right_operand)
    elif isinstance(ie, Sub):
        return _to_sympy_expr(ie.left_operand) - _to_sympy_expr(ie.right_operand)
    elif isinstance(ie, Mul):
        return _to_sympy_expr(ie.left_operand) * _to_sympy_expr(ie.right_operand)
    elif isinstance(ie, FloorDivision):
        return sympy.floor(_to_sympy_expr(ie.left_operand) / _to_sympy_expr(ie.right_operand))
    elif isinstance(ie, TrueDivision):
        return _to_sympy_expr(ie.left_operand) / _to_sympy_expr(ie.right_operand)
    elif isinstance(ie, Modulos):
        return _to_sympy_expr(ie.left_operand) % _to_sympy_expr(ie.right_operand)
    elif isinstance(ie, Pow):
        return _to_sympy_expr(ie.left_operand) ** _to_sympy_expr(ie.right_operand)
    elif isinstance(ie, Neg):
        return -_to_sympy_expr(ie.operand)
    elif isinstance(ie, Ceil):
        return sympy.ceiling(_to_sympy_expr(ie.operand))
    elif isinstance(ie, Floor):
        return sympy.floor(_to_sympy_expr(ie.operand))
    elif isinstance(ie, Min):
        return (
            sympy.Min(*[_to_sympy_expr(op) for op in ie.operands])
            .rewrite(sympy.Min, sympy.Piecewise)
            .rewrite(sympy.Max, sympy.Piecewise)
        )
    elif isinstance(ie, Max):
        return (
            sympy.Max(*[_to_sympy_expr(op) for op in ie.operands])
            .rewrite(sympy.Min, sympy.Piecewise)
            .rewrite(sympy.Max, sympy.Piecewise)
        )
    elif isinstance(ie, Piecewise):
        # Convert each condition and expression pair
        cond_expr_pairs = []
        for cond, expr in ie.conds_and_branches:
            sympy_cond = _to_sympy_expr(cond)
            sympy_expr = _to_sympy_expr(expr)
            cond_expr_pairs.append((sympy_expr, sympy_cond))
        return sympy.Piecewise(*cond_expr_pairs)
    elif isinstance(ie, Slice):
        # Convert slice to a range expression
        start = _to_sympy_expr(ie.start)
        stop = _to_sympy_expr(ie.stop)
        return sympy.Range(start, stop)
    elif isinstance(ie, ConstBool):
        return sympy.S(ie.const)
    elif isinstance(ie, Not):
        return ~_to_sympy_expr(ie.operand)
    elif isinstance(ie, And):
        return _to_sympy_expr(ie.left_operand) & _to_sympy_expr(ie.right_operand)
    elif isinstance(ie, Or):
        return _to_sympy_expr(ie.left_operand) | _to_sympy_expr(ie.right_operand)
    elif isinstance(ie, LessThan):
        return _to_sympy_expr(ie.left_operand) < _to_sympy_expr(ie.right_operand)
    elif isinstance(ie, LessThanOrEqual):
        return _to_sympy_expr(ie.left_operand) <= _to_sympy_expr(ie.right_operand)
    elif isinstance(ie, GreaterThan):
        return _to_sympy_expr(ie.left_operand) > _to_sympy_expr(ie.right_operand)
    elif isinstance(ie, GreaterThanOrEqual):
        return _to_sympy_expr(ie.left_operand) >= _to_sympy_expr(ie.right_operand)
    elif isinstance(ie, Equal):
        return _to_sympy_expr(ie.left_operand) == _to_sympy_expr(ie.right_operand)
    else:
        raise NotImplementedError(f"Conversion to sympy not implemented for {type(ie)}")


# def from_sympy_expr(expr: Union[sympy.Expr, Tuple[sympy.Expr, ...]]) -> IndexExpr:  # noqa: C901
#    """Convert a sympy expression to an IndexExpr.
#    Args:
#        expr: The sympy expression to convert
#    Returns:
#        An IndexExpr representing the sympy expression
#    """
#    if isinstance(expr, tuple):
#        return IndexSequence(tuple(from_sympy_expr(elem) for elem in expr))
#    elif isinstance(expr, sympy.Integer):
#        return ConstInt(int(expr))
#    elif isinstance(expr, sympy.Symbol):
#        return Symbol(str(expr.name))
#    elif isinstance(expr, sympy.Add):
#        return Add(from_sympy_expr(expr.args[0]), from_sympy_expr(expr.args[1]))
#    elif isinstance(expr, sympy.Mul):
#        return Mul(from_sympy_expr(expr.args[0]), from_sympy_expr(expr.args[1]))
#    elif isinstance(expr, sympy.Pow):
#        return Pow(from_sympy_expr(expr.args[0]), from_sympy_expr(expr.args[1]))
#    elif isinstance(expr, sympy.Mod):
#        return Modulos(from_sympy_expr(expr.args[0]), from_sympy_expr(expr.args[1]))
#    elif isinstance(expr, sympy.floor):
#        return Floor(from_sympy_expr(expr.args[0]))
#    elif isinstance(expr, sympy.ceiling):
#        return Ceil(from_sympy_expr(expr.args[0]))
#    elif isinstance(expr, sympy.Min):
#        return Min(*[from_sympy_expr(arg) for arg in expr.args])
#    elif isinstance(expr, sympy.Max):
#        return Max(*[from_sympy_expr(arg) for arg in expr.args])
#    elif isinstance(expr, sympy.Piecewise):
#        cond_expr_pairs = []
#        for expr_cond_pair in expr.args:
#            if len(expr_cond_pair) == 2:
#                expr_part, cond_part = expr_cond_pair
#                cond_expr_pairs.append((from_sympy_expr(cond_part), from_sympy_expr(expr_part)))
#        return Piecewise(cond_expr_pairs)
#    elif isinstance(expr, sympy.Range):
#        return Slice(from_sympy_expr(expr.start), from_sympy_expr(expr.stop))
#    elif isinstance(expr, sympy.Boolean):
#        return ConstBool(bool(expr))
#    elif isinstance(expr, sympy.Not):
#        return Not(from_sympy_expr(expr.args[0]))
#    elif isinstance(expr, sympy.And):
#        return And(from_sympy_expr(expr.args[0]), from_sympy_expr(expr.args[1]))
#    elif isinstance(expr, sympy.Or):
#        return Or(from_sympy_expr(expr.args[0]), from_sympy_expr(expr.args[1]))
#    elif isinstance(expr, sympy.LessThan):
#        return LessThan(from_sympy_expr(expr.args[0]), from_sympy_expr(expr.args[1]))
#    elif isinstance(expr, sympy.LessThanOrEqual):
#        return LessThanOrEqual(from_sympy_expr(expr.args[0]), from_sympy_expr(expr.args[1]))
#    elif isinstance(expr, sympy.GreaterThan):
#        return GreaterThan(from_sympy_expr(expr.args[0]), from_sympy_expr(expr.args[1]))
#    elif isinstance(expr, sympy.GreaterThanOrEqual):
#        return GreaterThanOrEqual(from_sympy_expr(expr.args[0]), from_sympy_expr(expr.args[1]))
#    elif isinstance(expr, sympy.Eq):
#        return Equal(from_sympy_expr(expr.args[0]), from_sympy_expr(expr.args[1]))
#    elif isinstance(expr, sympy.Rational):
#        # Convert rational to integer division
#        return FloorDivision(from_sympy_expr(expr.numerator), from_sympy_expr(expr.denominator))
#    elif isinstance(expr, sympy.NegativeOne):
#        return ConstInt(-1)
#    elif isinstance(expr, sympy.One):
#        return ConstInt(1)
#    elif isinstance(expr, sympy.Zero):
#        return ConstInt(0)
#    elif isinstance(expr, sympy.true):
#        return ConstBool(True)
#    elif isinstance(expr, sympy.false):
#        return ConstBool(False)
#    else:
#        raise NotImplementedError(f"Conversion from sympy not implemented for {type(expr)}")


# TODO: sympy does not seem powerful enough to do real simplification.
# def simplify(ie: IndexExpr, known_symbols: Optional[Mapping[Symbol, int]] = None) -> IndexExpr:
#    as_sympy = to_sympy_expr(ie)
#
#    if isinstance(as_sympy, tuple):
#        return IndexSequence(tuple(simplify(member, known_symbols) for member in as_sympy))
#
#    if known_symbols is not None:
#        for symbol, value in known_symbols.items():
#            as_sympy = as_sympy.subs(symbol, value)
#    vars = ie.vars_used()
#    for var in vars:
#        sympy_var = sympy.Symbol(var.name)
#        sympy_var.assumptions0
#
#    from tempo.core import global_objects as glob
#    static_bounds = glob.get_static_bounds_or_empty()
#    dyn_bounds = glob.get_dynamic_bounds_or_empty()
#
#    #TODO add bound constraints to sympy
#
#    simplified = sympy.simplify(as_sympy)
#
#    return from_sympy_expr(simplified)


def _sympy_equivalent(ie1: Any, ie2: Any) -> bool:
    """Check if two index expressions are equivalent by
    converting them to sympy expressions and comparing them."""

    ie1 = lift_to_ie(ie1)
    ie2 = lift_to_ie(ie2)

    # Fast paths
    if isinstance(ie1, Symbol) and isinstance(ie2, Symbol):
        return ie1.name == ie2.name
    if isinstance(ie1, ConstInt) and isinstance(ie2, ConstInt):
        return ie1.const == ie2.const
    if isinstance(ie1, ConstBool) and isinstance(ie2, ConstBool):
        return ie1.const == ie2.const

    as_sympy1 = to_sympy_expr(ie1)
    as_sympy2 = to_sympy_expr(ie2)
    # as_sympy1 = ie1._cached_sympy_expr
    # as_sympy2 = ie2._cached_sympy_expr

    if not isinstance(as_sympy1, tuple) == isinstance(as_sympy2, tuple):
        return False

    as_sympy1_tuple = as_sympy1 if isinstance(as_sympy1, tuple) else (as_sympy1,)
    as_sympy2_tuple = as_sympy2 if isinstance(as_sympy2, tuple) else (as_sympy2,)

    if len(as_sympy1_tuple) != len(as_sympy2_tuple):
        return False

    for a, b in zip(as_sympy1_tuple, as_sympy2_tuple, strict=False):
        if not isinstance(a, sympy.Range) == isinstance(b, sympy.Range):
            return False

        if isinstance(a, sympy.Range):
            if sympy.simplify(a.start - b.start) != sympy.Integer(0):
                return False
            if sympy.simplify(a.stop - b.stop) != sympy.Integer(0):
                return False
            if sympy.simplify(a.step - b.step) != sympy.Integer(0):
                return False
        else:
            try:
                if sympy.simplify(a - b) != sympy.Integer(0):
                    return False
            except Exception as e:
                print(f"Problem with {a} and {b}")
                print(a, b)
                print(sympy.simplify(a - b))
                print(sympy.simplify(a - b) != sympy.Integer(0))
                raise e

    return True


def struct_eq(*ies: Any) -> bool:
    if len(ies) < 2:
        return True

    lifted = [lift_to_ie(ie) for ie in ies]
    first = lifted[0]

    return all(first.struct_eq(ie) for ie in lifted[1:])


def logical_eq(*ies: Any) -> bool:
    if len(ies) < 2:
        return True

    lifted = [lift_to_ie(ie) for ie in ies]
    first = lifted[0]

    return all(_sympy_equivalent(first, ie) for ie in lifted[1:])


@dataclass(frozen=True, eq=False, slots=True)
class IndexExpr(SupportsIndex, ABC):
    children: List[IndexAtom] = field(
        default_factory=list, repr=False, compare=False, init=False, hash=False
    )
    eval_fast: Callable[[], Any] = field(default_factory=lambda: (lambda: 0), init=False)
    _creation_traceback: List[str] = field(
        init=False, default_factory=lambda: traceback.format_stack()[:-1]
    )
    _cached_sympy_expr: Optional[sympy.Expr] = field(default=None, init=False)

    @property
    def creation_traceback(self) -> str:
        return "\n".join([x.strip() for x in self._creation_traceback])

    def __index__(self) -> int:
        # TODO try to evaluate and return if static
        raise ValueError(
            "__index__ was called on IndexExpr. While IndexExprs can be passed \
                         as indexes to recurrent tensors, they cannot be used as indexes in \
                         other contexts (e.g. indexing a list)"
        )

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, tuple(self.children)))

    # def _cache_sympy_expr(self) -> None:
    #    #assert not hasattr(self, "_cached_sympy_expr")
    #    #self._cached_sympy_expr = to_sympy_expr(self)
    #    object.__setattr__(self, "_cached_sympy_expr", to_sympy_expr(self))

    # def __post_init__(self) -> None:
    #    self._cache_sympy_expr()
    #    self._cached_sympy_expr = to_sympy_expr(self)

    def cache_codegenerated_eval(self, index_values: Mapping[Symbol, int]) -> None:
        # codegen_str will be a simple string like "t + 1" or "min(t+3, T)"
        codegen_str = self._codegen_str(index_values)
        fn = compile(codegen_str, "<string>", "eval", optimize=2)

        locals_ = {**{k.name: k for k, v in index_values.items()}, "cd": index_values}
        globals_ = {"math": math}
        # eval_partial_ = functools.partial(eval, fn, globals_, locals_)
        # Use a lambda instead of a partial to avoid overhead???
        eval_partial_ = lambda: eval(fn, globals_, locals_)
        object.__setattr__(self, "eval_fast", eval_partial_)

    @abstractmethod
    def _codegen_str(self, index_values: Mapping[Symbol, int]) -> str:
        raise NotImplementedError()

    def logical_eq(self, other: Any) -> bool:
        return _sympy_equivalent(self, other)

    # def equivalent(self, other: Any) -> bool:
    #    from tempo.utils import isl as isl_utils

    #    if isinstance(self, Symbol) and isinstance(other, Symbol):
    #        return self.name == other.name
    #    if isinstance(self, ConstBool) and isinstance(other, ConstBool):
    #        return self.const == other.const
    #    if isinstance(self, ConstInt) and isinstance(other, ConstInt):
    #        return self.const == other.const

    #    return isl_utils.expr_eq(self, other)

    def struct_eq(self, other: Any) -> bool:
        """Compares two index expressions to check if they are equivalent.

        Because we overrode __eq__ in order to be used as an easy way to construct index
        expressions, we need to create a new method to check if two index expressions
          are equivalent.
        """
        if not isinstance(other, IndexExpr):
            try:
                other = lift_to_ie(other)
            except Exception:
                return False

        if type(self) is type(other) and len(self.children) == len(other.children):
            return all(
                map(
                    lambda x, y: x.struct_eq(y),
                    self.children,
                    other.children,  # TODO: these should really be sorted first
                )
            )
        return False

    def accesses_bound(self) -> bool:
        """Returns true if the index expression accesses the bound of any of the dimensions.

        For example, if the index expression is "t + 1" or "min(t+3, T)",
        then it does not access the bound
        of any of the dimensions. However, if the index expression is "T - 2" or "t+3:T" or ",
          then it
        does access the bound of the dimension t.
        """
        return any(x.accesses_bound() for x in self.children)

    # TODO remove
    def evaluate_shape(
        self, index_values: Mapping[Symbol, int]
    ) -> Tuple[Union[int, IntIndexValue], ...]:
        """Evaluates an index expr to compute the effect it has on the shape of a tensor."""
        raise NotImplementedError()

    def is_block_access(self) -> bool:
        return False

    def is_shrinking_slice(self) -> bool:
        return False

    def is_growing_slice(self) -> bool:
        return False

    def is_point(self) -> bool:
        raise NotImplementedError()

    def is_constant(self) -> bool:
        return all(x.is_constant() for x in self.children)

    def is_valid_initializer_expr(self) -> bool:
        """All children are either const or points and there is at least one constantpoint

        Returns:
            bool: True if the index expression is a valid initializer expression

        """
        if not self.children:
            return self.is_constant() or self.is_point()
        return all(x.is_constant() or x.is_point() for x in self.children)

    def is_boolean_expr(self) -> bool:
        return isinstance(self, BooleanIndexValue)

    def is_valid_set_item_expr(self) -> bool:
        return self.is_valid_initializer_expr() or self.is_boolean_expr()

    def enumerate_all_cond_branches(
        self,
    ) -> Sequence[Tuple[Optional[BooleanIndexValue], IndexExpr]]:
        children_enumerations = [child.enumerate_all_cond_branches() for child in self.children]

        res: List[Tuple[Optional[BooleanIndexValue], IndexExpr]] = []
        for prod in itertools.product(*children_enumerations):
            branches = tuple([p[1] for p in prod])
            branches = typing.cast(Tuple[IndexAtom, ...], branches)
            conds = [p[0] for p in prod]
            conds = [c for c in conds if c is not None]
            if len(conds) == 0:
                res.append((None, self.__class__(*branches)))
            elif len(conds) == 1:
                res.append((conds[0], self.__class__(*branches)))
            else:
                res.append((functools.reduce(And, conds), self.__class__(*branches)))
        return res  # type: ignore

    # TODO: pretty sure we only need these methods for Sequence and Slice, nothing else....
    # Every other expr can just return self
    def as_upper_bound_access(self) -> IndexExpr:
        """Returns a copy of the index expression with slice operations replaced with the
        upperbound (stop) access the slice.

        For example, if the index expression is "t+3:T", then the result is "T".
        """
        return self.__class__(*[x.as_upper_bound_access() for x in self.children])

    def as_lower_bound_access(self) -> IndexExpr:
        """Returns a copy of the index expression with slice operations replaced with the
        lowerbound (start) access the slice.

        For example, if the index expression is "t+3:T", then the result is "t+3".
        """
        return self.__class__(*[x.as_lower_bound_access() for x in self.children])

    def drop_modulos(self) -> IndexExpr:
        if len(self.children) == 0:
            return self
        raise NotImplementedError(f"No drop_modulos rule for {self.__class__}")

    def simplify_mins_and_maxes(self, aggressive: bool = False) -> IndexExpr:
        return self.__class__(*[x.simplify_mins_and_maxes(aggressive) for x in self.children])

    def bound_symbols_used(self) -> Sequence[Symbol]:
        bounds = set()
        for child in self.children:
            if isinstance(child, Symbol) and child.is_bound:
                bounds.add(child)
            else:
                bounds.update(child.bound_symbols_used())
        return list(bounds)

    def vars_used(self) -> Set[Symbol]:
        variables = set()
        for child in self.children:
            if isinstance(child, Symbol) and not child.is_bound:
                variables.add(child)
            else:
                assert isinstance(child, IndexAtom), f"Expected IndexValue, got {child}"
                variables.update(child.vars_used())
        return variables

    def __eq__(self, other: Any) -> bool:
        return self.struct_eq(other)

    def __ne__(self, other: Any) -> bool:
        return not self.struct_eq(other)


def lift_to_ie_atom(other: Any) -> IndexAtom:
    raised = lift_to_ie(other)
    assert isinstance(raised, IndexAtom), "Could not raise to index expression atom."
    return raised  # type: ignore


def lift_to_ie_slice(other: Any) -> Slice:
    raised = lift_to_ie(other)
    assert isinstance(raised, Slice), (
        f"Could not raise {other} ({type(other)}) to slice expr. Raise to {raised} ({type(raised)})"
    )
    return raised  # type: ignore


def lift_to_ie_val(other: Any) -> IndexValue:
    raised = lift_to_ie(other)
    assert isinstance(raised, IndexValue), (
        f"Could not raise {other} ({type(other)}) to value. Raised to {raised} ({type(raised)})"
    )
    return raised  # type: ignore


def lift_to_int_ie(other: Any) -> IntIndexValue:
    raised = lift_to_ie(other)
    assert isinstance(raised, IntIndexValue), (
        f"Could not raise {other} ({type(other)}) to int expr. Raised to {raised} ({type(raised)})"
    )
    return raised  # type: ignore


def lift_to_bool_ie(other: Any) -> BooleanIndexValue:
    raised = lift_to_ie(other)
    assert isinstance(raised, BooleanIndexValue), (
        f"Could not raise {other} ({type(other)}) to bool expr. Raised to {raised} ({type(raised)})"
    )
    return raised  # type: ignore


def lift_to_ie_seq(other: Any) -> IndexSequence:
    raised = lift_to_ie(other)
    assert isinstance(raised, IndexSequence), (
        f"Could not raise {other} ({type(other)}) to seq expr. Raised to {raised} ({type(raised)})"
    )
    return raised  # type: ignore


def evaluate(ie_: IndexExprLike, index_values: Mapping[Symbol, int]) -> Any:
    ie_ = lift_to_ie(ie_)
    return ie_.evaluate(index_values)  # type: ignore


def evaluate_int(ie_: IndexExprLike, index_values: Mapping[Symbol, int]) -> int:
    ie_ = lift_to_int_ie(ie_)
    return ie_.evaluate(index_values)


def lift_to_ie(x: Any) -> IndexExpr:
    if isinstance(x, IndexExpr):
        return x
    if isinstance(x, Sequence):
        inner_exprs = tuple(map(lift_to_ie_atom, x))
        return IndexSequence(inner_exprs)
    if type(x) is bool:
        return ConstBool(const=x)
    if type(x) is float:
        return ConstFloat(const=x)  # type: ignore
    if type(x) is int:
        return ConstInt(const=x)
    if type(x) is str:
        # return Symbol(x)
        # TODO we put this here so we avoid getting the active DG
        raise ValueError(f"Cannot raise string {x} to IndexExpr")
    if type(x) is builtins.slice:
        assert x.start is not None, "Slice start cannot be None for now"
        assert x.stop is not None, "Slice stop cannot be None for now"
        return slice_(
            lift_to_int_ie(x.start),
            lift_to_int_ie(x.stop),
            # ConstExpr(1)
            # if other.step is None
            # else raise_to_index_value_expr(other.step),
        )
    raise Exception(f"Cannot raise value {x} of class {x.__class__} to AST nodes.")


# def random(
#    lower_bound: Union[int, IndexValue], upper_bound: Union[int, IndexValue]
# ) -> RandExpr:
#    lb = raise_to_int_index_value_expr(lower_bound)
#    ub = raise_to_int_index_value_expr(upper_bound)
#    return RandExpr(lower_bound=lb, upper_bound=ub)


def min(  # noqa: A001, A002, A003
    arg1: Any,
    arg2: Optional[Any] = None,
    *args: Any,
) -> IntIndexValue:
    # TODO need to redefine the version which takes a iterable
    if arg2 is None:
        return lift_to_int_ie(arg1)
    lst = [arg1, arg2, *args]
    converted = tuple(map(lift_to_int_ie, lst))
    return Min(converted)


def max(  # noqa: A001, A002, A003
    arg1: Any,
    arg2: Optional[Any] = None,
    *args: Any,
) -> IntIndexValue:
    if arg2 is None:
        return lift_to_int_ie(arg1)
    lst = [arg1, arg2, *args]
    converted = tuple(map(lift_to_int_ie, lst))
    return Max(converted)


@dataclass(frozen=True, eq=False, slots=True)
class IndexAtom(IndexExpr, ABC):
    """An IndexExprAtom is any expression which can be used as an index, including slices.

    Args:
        IndexExpr (_type_): _description_
        ABC (_type_): _description_

    """

    def evaluate_shape(
        self, index_values: Mapping[Symbol, int]
    ) -> Tuple[Union[int, IntIndexValue], ...]:
        raise NotImplementedError()

    def remap(self, domain_map: Mapping[Symbol, IntIndexValue]) -> IndexAtom:
        raise NotImplementedError()

    def partial_eval(self, domain_map: Mapping[Symbol, int]) -> IndexAtom:
        raise NotImplementedError(f"Cannot partial eval {self} of type {type(self)}")


@dataclass(frozen=True, eq=False, slots=True)
class IndexSequence(IndexExpr):
    members: Sequence[IndexAtom]
    # seq_len: int = 1

    def __post_init__(self) -> None:
        self.children.extend(self.members)
        # self._cached_sympy_expr()
        # object.__setattr__(self, "seq_len", len(self.members))

    def __str__(self) -> str:
        return f"{','.join(map(str, self.members))}"

    def __len__(self) -> int:
        return len(self.members)

    def __iter__(self) -> typing.Iterator[IndexAtom]:
        return iter(self.members)

    @overload
    def __getitem__(self, key: int) -> IndexAtom:
        pass

    @overload
    def __getitem__(self, key: builtins.slice) -> IndexSequence:
        pass

    def __getitem__(self, key: Union[int, builtins.slice]) -> Union[IndexAtom, IndexSequence]:
        if isinstance(key, int):
            return self.members[key]
        return IndexSequence(self.members[key])

    __repr__ = __str__

    def replace_idx(self, idx: int, new_member: IndexAtomLike) -> IndexSequence:
        new_member = lift_to_ie_atom(new_member)
        return IndexSequence(
            (
                *self.members[:idx],
                new_member,
                *self.members[idx + 1 :],
            )
        )

    def skip_idx(self, idx: int) -> IndexSequence:
        return IndexSequence(
            (
                *self.members[:idx],
                *self.members[idx + 1 :],
            )
        )

    def num_slices(self) -> int:
        return sum(isinstance(x, Slice) for x in self.members)

    def prepend_member(self, new_member: IndexAtomLike) -> IndexSequence:
        new_member = lift_to_ie_atom(new_member)
        new_members = [new_member] + list(self.members)
        return IndexSequence(tuple(new_members))

    def append_member(self, new_member: IndexAtomLike) -> IndexSequence:
        new_member = lift_to_ie_atom(new_member)
        new_members = list(self.members) + [new_member]
        return IndexSequence(tuple(new_members))

    def remap(self, domain_map: Mapping[Symbol, IntIndexValue]) -> IndexSequence:
        return IndexSequence(
            tuple(x.remap(domain_map) for x in self.members)  # type: ignore
        )

    def _codegen_str(self, index_values: Mapping[Symbol, int]) -> str:
        str_ = f"({', '.join((x._codegen_str(index_values) for x in self.members))}"
        if len(self.members) == 1:
            str_ += ",)"
        else:
            str_ += ")"
        return str_

    def evaluate(self, index_values: Mapping[Symbol, int]) -> Tuple[Union[int, slice], ...]:
        return tuple(x.evaluate(index_values) for x in self.members)  # type: ignore

    def partial_eval(self, domain_map: Mapping[Symbol, builtins.int]) -> IndexSequence:
        inner = tuple(x.partial_eval(domain_map) for x in self.members)  # type: ignore
        return IndexSequence(inner)  # type: ignore

    def evaluate_shape(
        self, index_values: Mapping[Symbol, int]
    ) -> Tuple[Union[int, IntIndexValue], ...]:
        member_shapes = tuple(x.evaluate_shape(index_values) for x in self.members)
        flat = tuple(x for shape in member_shapes for x in shape)
        return flat

    def is_point(self) -> bool:
        return all(x.is_point() for x in self.members)

    def is_basis(self) -> bool:
        return all(isinstance(x, Symbol) and x.is_variable() for x in self.members)

    # TODO: remove these methods, prefer dg_utils ones
    def is_block_access(self) -> builtins.bool:
        return any(x.is_block_access() for x in self.members)

    def is_shrinking_slice(self) -> bool:
        return any(x.is_shrinking_slice() for x in self.members)

    def is_growing_slice(self) -> bool:
        return any(x.is_growing_slice() for x in self.members)

    def get_block_access_block_sizes(
        self,
    ) -> Tuple[Union[int, None], ...]:
        res = []
        for member in self.members:
            if member.is_block_access():
                res.append(member.get_block_access_block_size())  # type: ignore
            else:
                res.append(None)
        return tuple(res)

    def is_regressive(self) -> bool:
        return any(isinstance(x, Sub) for x in self.members)

    def enumerate_all_cond_branches(
        self,
    ) -> Sequence[Tuple[Optional[BooleanIndexValue], IndexExpr]]:
        children_enumerations = [child.enumerate_all_cond_branches() for child in self.children]

        res: List[Tuple[Optional[BooleanIndexValue], IndexExpr]] = []
        for prod in itertools.product(*children_enumerations):
            branches = tuple([p[1] for p in prod])
            branches = typing.cast(Tuple[IndexAtom, ...], branches)
            conds = [p[0] for p in prod]
            conds = [c for c in conds if c is not None]
            if len(conds) == 0:
                res.append((None, self.__class__(branches)))
            elif len(conds) == 1:
                res.append((conds[0], self.__class__(branches)))
            else:
                res.append((functools.reduce(And, conds), self.__class__(branches)))
        return res

    def drop_modulos(self) -> IndexExpr:
        return self.__class__(tuple(x.drop_modulos() for x in self.members))  # type: ignore

    def as_upper_bound_access(self) -> IndexSequence:
        return self.__class__(
            tuple(x.as_upper_bound_access() for x in self.members)  # type: ignore
        )

    # def increment_block_slices(
    #    self,
    # ) -> IndexSequence:
    #    return IndexSequence(
    #        tuple(
    #            Slice(x.as_lower_bound_access(), x.as_upper_bound_access() + 1)
    # if ((not x.is_constant()) and (not x.is_point()))
    # else x for x in self.members)  # type: ignore
    #    )

    # def decrement_block_slices(
    #    self,
    # ) -> IndexSequence:
    #    return IndexSequence(
    #        tuple(Slice(x.as_lower_bound_access(), x.as_upper_bound_access() - 1)
    # if ((not x.is_constant()) and (not x.is_point()))
    # else x for x in self.members)  # type: ignore
    #    )

    def as_lower_bound_access(self) -> IndexSequence:
        return self.__class__(
            tuple(x.as_lower_bound_access() for x in self.members)  # type: ignore
        )

    def simplify_mins_and_maxes(self, aggressive: bool = False) -> IndexSequence:
        return self.__class__(
            tuple(x.simplify_mins_and_maxes(aggressive) for x in self.members)  # type: ignore
        )

    def struct_eq(self, other: Any) -> bool:
        """Compares two index expressions to check if they are equivalent.

        Because we overrode __eq__ in order to be used as an easy way to construct index
        expressions, we need to create a new method to check if two index expressions
          are equivalent.
        """
        if not isinstance(other, IndexExpr):
            try:
                other = lift_to_ie_seq(other)
            except Exception:
                return False

        if type(self) is type(other) and len(self.members) == len(other.members):
            return all(
                map(
                    lambda x, y: x.struct_eq(y),
                    self.members,
                    other.members,
                )
            )
        return False


@dataclass(frozen=True, eq=False, slots=True)
class IndexValue(IndexAtom, ABC):
    """An IndexValueExpr is an expression which can be used as an index, but not a slice.

    Args:
        IndexExprAtom (_type_): _description_
        ABC (_type_): _description_

    """

    def evaluate(self, index_values: Mapping[Symbol, int]) -> Union[int, bool]:
        raise NotImplementedError()

    def is_point(self) -> bool:
        return True

    def evaluate_shape(
        self, index_values: Mapping[Symbol, int]
    ) -> Tuple[Union[int, IntIndexValue], ...]:
        # The shape of index values is a scalar, so they have shape ()
        return ()  # type: ignore

    def __truediv__(self, other: Any) -> IntIndexValue:
        raise NotImplementedError()

    def __rtruediv__(self, other: Any) -> IntIndexValue:
        raise NotImplementedError()

    def __pow__(self, other: Any) -> IntIndexValue:
        raise NotImplementedError()

    def __rpow__(self, other: Any) -> IntIndexValue:
        raise NotImplementedError()

    def __sub__(self, other: Any) -> IntIndexValue:
        raise NotImplementedError()

    def __rsub__(self, other: Any) -> IntIndexValue:
        raise NotImplementedError()

    def __add__(self, other: Any) -> IntIndexValue:
        raise NotImplementedError()

    def __radd__(self, other: Any) -> IntIndexValue:
        raise NotImplementedError()

    def __mul__(self, other: Any) -> IntIndexValue:
        raise NotImplementedError()

    def __rmul__(self, other: Any) -> IntIndexValue:
        raise NotImplementedError()


@dataclass(frozen=True, eq=False, slots=True)
class BooleanIndexValue(IndexValue, ABC):
    def evaluate(self, index_values: Mapping[Symbol, int]) -> bool:
        raise NotImplementedError()

    def remap(self, domain_map: Mapping[Symbol, IntIndexValue]) -> BooleanIndexValue:
        raise NotImplementedError()

    def eliminate_symbol(self, symbol: Symbol) -> BooleanIndexValue:
        raise NotImplementedError(f"Cannot eliminate symbol {symbol} from {self}")

    def partial_eval(self, domain_map: Mapping[Symbol, int]) -> BooleanIndexValue:
        raise NotImplementedError(f"Cannot partial eval {self} of type {type(self)}")

    def __invert__(self) -> BooleanIndexValue:
        # not not x -> x
        if isinstance(self, Not):
            return self.operand

        # not True -> False
        if self.struct_eq(ConstBool(True)):
            return ConstBool(False)
        # not False -> True
        if self.struct_eq(ConstBool(False)):
            return ConstBool(True)

        return Not(self)

    def __or__(self, other: Any) -> BooleanIndexValue:  # noqa: C901
        raised = lift_to_bool_ie(other)

        # False or x -> x
        if self.struct_eq(ConstBool(False)):
            return raised  # type: ignore

        # x or False -> x
        if raised.struct_eq(ConstBool(False)):
            return self  # type: ignore

        # True or x -> True
        if self.struct_eq(ConstBool(True)) or raised.struct_eq(ConstBool(True)):
            return ConstBool(True)

        # x or x -> x
        if self.struct_eq(raised):
            return self  # type: ignore

        # Absorption Law
        # x or (x and y) -> x
        if isinstance(raised, And) and (
            self.struct_eq(raised.left_operand) or self.struct_eq(raised.right_operand)
        ):
            return self

        # (x and y) or x -> x
        if isinstance(self, And) and (
            raised.struct_eq(self.left_operand) or raised.struct_eq(self.right_operand)
        ):
            return raised

        # Complement Law
        # x or not x -> True
        if (isinstance(raised, Not) and self.struct_eq(raised.operand)) or (
            isinstance(self, Not) and raised.struct_eq(self.operand)
        ):
            return ConstBool(True)

        return Or(self, raised)

    def __and__(self, other: Any) -> BooleanIndexValue:  # noqa: C901
        raised = lift_to_bool_ie(other)
        # True and x -> x
        if self.struct_eq(ConstBool(True)):
            return raised  # type: ignore
        # x and True -> x
        if raised.struct_eq(ConstBool(True)):
            return self  # type: ignore
        # False and x -> False
        if self.struct_eq(ConstBool(False)) or raised.struct_eq(ConstBool(False)):
            return ConstBool(False)
        # x and x -> x
        if self.struct_eq(raised):
            return self  # type: ignore

        # Absorption Law
        # x and (x or y) -> x
        if isinstance(raised, Or) and (
            self.struct_eq(raised.left_operand) or self.struct_eq(raised.right_operand)
        ):
            return self
        # (x or y) and x -> x
        if isinstance(self, Or) and (
            raised.struct_eq(self.left_operand) or raised.struct_eq(self.right_operand)
        ):
            return raised

        # Complement Law
        # x and not x -> False
        if (
            isinstance(raised, Not)
            and self.struct_eq(raised.operand)
            or isinstance(self, Not)
            and raised.struct_eq(self.operand)
        ):
            return ConstBool(False)

        return And(self, raised)


@dataclass(frozen=True, eq=False, slots=True)
class IntIndexValue(IndexValue, ABC):
    def evaluate(self, index_values: Mapping[Symbol, int]) -> int:
        raise NotImplementedError()

    # TODO this should return self if fails
    def try_eval(self, index_values: Mapping[Symbol, int]) -> Optional[int]:
        try:
            return self.evaluate(index_values)
        except Exception:
            return None

    def partial_eval(self, domain_map: Mapping[Symbol, int]) -> IntIndexValue:
        raise NotImplementedError(f"Cannot partial eval {self} of type {type(self)}")

    def remap(self, domain_map: Mapping[Symbol, IntIndexValue]) -> IntIndexValue:
        raise NotImplementedError()

    def __add__(self, other: Any) -> IntIndexValue:  # noqa: C901
        raised = lift_to_int_ie(other)
        # Handle identity element for addition: x + 0 = x
        if isinstance(raised, ConstInt) and raised.const == 0:
            return self
        if isinstance(self, ConstInt) and self.const == 0:
            return raised

        # Simplify addition when both operands are constants
        if isinstance(self, ConstInt) and isinstance(raised, ConstInt):
            return ConstInt(self.const + raised.const)

        # Simplification with additive inverses: x + (-x) -> 0 and (-x) + x -> 0
        if (
            (
                isinstance(self, ConstInt)
                and isinstance(raised, ConstInt)
                and self.const == -raised.const
            )
            or (isinstance(self, Neg) and self.operand.struct_eq(raised))
            or (isinstance(raised, Neg) and raised.operand.struct_eq(self))
        ):
            return ConstInt(0)

        # Simplification for expressions of the form x + (y - x) -> y and (y - x) + x -> y
        if isinstance(raised, Sub) and self.struct_eq(raised.right_operand):
            return raised.left_operand
        if isinstance(self, Sub) and raised.struct_eq(self.right_operand):
            return self.left_operand

        return Add(self, raised)

    def __radd__(self, other: Any) -> IntIndexValue:
        return self.__add__(other)

    def __sub__(self, other: Any) -> IntIndexValue:  # noqa: C901
        raised = lift_to_int_ie(other)

        # Handle subtraction of 0: x - 0 = x
        if isinstance(raised, ConstInt) and raised.const == 0:
            return self  # type: ignore

        # Simplify subtraction when both operands are constants
        if isinstance(self, ConstInt) and isinstance(raised, ConstInt):
            return ConstInt(self.const - raised.const)

        if isinstance(raised, Add) and self.struct_eq(raised.right_operand):
            return -raised.left_operand
        if isinstance(self, Add) and raised.struct_eq(self.right_operand):
            return self.left_operand

        # Simplification for x - x -> 0
        if self.struct_eq(raised):
            return ConstInt(0)

        return Sub(self, raised)

    def __rsub__(self, other: Any) -> IntIndexValue:
        raised = lift_to_int_ie(other)
        return raised.__sub__(self)

    def __mul__(self, other: Any) -> IntIndexValue:
        raised = lift_to_int_ie(other)

        # Multiplicative Identity
        if raised.struct_eq(ConstInt(1)):
            return self
        if self.struct_eq(ConstInt(1)):
            return raised

        # Multiplication by Zero
        if raised.struct_eq(ConstInt(0)) or self.struct_eq(ConstInt(0)):
            return ConstInt(0)

        ## Simplification when self or raised is a FloorDivision
        # if isinstance(self, FloorDivision):
        #    if self.right_operand.equivalent(raised):
        #        # (a // b) * b = a
        #        return self.left_operand
        #    elif isinstance(raised, FloorDivision) and self.right_operand.equivalent(
        #        raised.right_operand
        #    ):
        #        # (a // c) * (b // c) = (a * b) // c
        #        return Mul(self.left_operand, raised.left_operand) // self.right_operand

        # if isinstance(raised, FloorDivision):
        #    if raised.right_operand.equivalent(self):
        #        # b * (a // b) = a
        #        return raised.left_operand
        #    elif isinstance(self, FloorDivision) and raised.right_operand.equivalent(
        #        self.right_operand
        #    ):
        #        # (b // c) * (a // c) = (b * a) // c
        #        return (
        #            Mul(self.left_operand, raised.left_operand) // raised.right_operand
        #        )

        # Multiplication of Constants
        if isinstance(self, ConstInt) and isinstance(raised, ConstInt):
            return ConstInt(self.const * raised.const)

        ## Simplification with Negation
        # if isinstance(raised, Neg) or isinstance(self, Neg):
        #    negated_term: Neg = raised if isinstance(raised, Neg) else self  # type: ignore
        #    other_term = self if isinstance(raised, Neg) else raised
        #    return Neg(Mul(negated_term.operand, other_term))

        return Mul(self, raised)

    def __rmul__(self, other: Any) -> IntIndexValue:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> IntIndexValue:
        # raise ValueError(
        #    "Cannot use true division (/) as it may lead to float indexes. Use // instead."
        # )
        raised = lift_to_int_ie(other)
        return TrueDivision(self, raised)

    def __floordiv__(self, other: Any) -> IntIndexValue:  # noqa: C901
        raised = lift_to_int_ie(other)

        # Division by One
        if isinstance(raised, ConstInt) and raised.const == 1:
            return self

        # Zero Divided by Any Non-Zero Number
        if (
            isinstance(self, ConstInt)
            and self.const == 0
            and (
                not isinstance(raised, ConstInt)
                or (isinstance(raised, ConstInt) and raised.const != 0)
            )
        ):
            return ConstInt(0)

        # Division by Itself
        if self.struct_eq(raised):
            return ConstInt(1)

        # Handle Division by Zero as an error or special case
        if isinstance(raised, ConstInt) and raised.const == 0:
            raise ValueError("Division by zero is undefined.")

        # Simplification with Constants
        if isinstance(self, ConstInt) and isinstance(raised, ConstInt):
            # Ensure raised.const is not zero to avoid division by zero error
            if raised.const != 0:
                return ConstInt(self.const // raised.const)
        return FloorDivision(self, raised)

    def __rfloordiv__(self, other: Any) -> IntIndexValue:
        raised = lift_to_int_ie(other)
        return raised.__floordiv__(self)

    def __mod__(self, other: Any) -> IntIndexValue:
        #    return self - raised * (self // raised)
        raised = lift_to_int_ie(other)

        # modulo simplification: x % 1 -> 0, x % x -> 0, x % 0 -> error
        if isinstance(raised, ConstInt):
            if raised.const == 1:
                return ConstInt(0)
            elif raised.const == 0:
                raise ValueError("Cannot mod by 0")
        if self.struct_eq(raised):
            return ConstInt(0)
        return Modulos(self, raised)
        # return self - raised * (self // raised)

    def __pow__(self, other: Any) -> IntIndexValue:
        raised = lift_to_int_ie(other)
        return Pow(self, raised)

    def __rpow__(self, other: Any) -> IntIndexValue:
        raised = lift_to_int_ie(other)
        return raised.__pow__(self)

    def __neg__(self) -> IntIndexValue:
        raised = lift_to_int_ie(self)
        # Simplification for double negation: -(-x) -> x
        if isinstance(raised, Neg):
            return raised.operand

        # Simplification for negation of 0: -0 -> 0
        if isinstance(raised, ConstInt) and raised.const == 0:
            return raised
        return Neg(raised)

    def __lt__(self, other: Any) -> BooleanIndexValue:
        raised = lift_to_int_ie(other)
        return LessThan(self, raised)

    def __le__(self, other: Any) -> BooleanIndexValue:
        raised = lift_to_int_ie(other)
        return LessThanOrEqual(self, raised)

    def __gt__(self, other: Any) -> BooleanIndexValue:
        raised = lift_to_int_ie(other)
        return GreaterThan(self, raised)

    def __ge__(self, other: Any) -> BooleanIndexValue:
        raised = lift_to_int_ie(other)
        return GreaterThanOrEqual(self, raised)

    def symb_eq(self, other: Any) -> BooleanIndexValue:
        raised = lift_to_int_ie(other)
        return Equal(self, raised)

    def symb_ne(self, other: Any) -> BooleanIndexValue:
        raised = lift_to_int_ie(other)
        return Not(Equal(self, raised))

    def __eq__(self, other: Any) -> BooleanIndexValue:  # type: ignore
        return self.symb_eq(other)

    def __ne__(self, other: Any) -> BooleanIndexValue:  # type: ignore
        return self.symb_ne(other)


# @dataclass(frozen=True, eq=False, slots=True)
# class RandExpr(IntIndexValue):
#    lower_bound: IntIndexValue
#    upper_bound: IntIndexValue
#
#    def __post_init__(self) -> None:
#        self.children.extend([self.lower_bound, self.upper_bound])
#
#    def equivalent(self, other: Any) -> bool:
#        if isinstance(other, RandExpr):
#            return self.lower_bound.equivalent(
#                other.lower_bound
#            ) and self.upper_bound.equivalent(other.upper_bound)
#        return False
#
#    def accesses_bound(self) -> bool:
#        return False
#
#    def evaluate(self, index_values: Mapping[Symbol, int]) -> int:
#        lb = self.lower_bound.evaluate(index_values)
#        ub = self.upper_bound.evaluate(index_values)
#        assert isinstance(lb, int) and isinstance(ub, int)
#        return rand.randint(lb, ub)
#
#    def as_upper_bound_access(self) -> IndexExpr:
#        return self.upper_bound - 1
#
#    def as_lower_bound_access(self) -> IndexExpr:
#        return self.lower_bound


@dataclass(frozen=True, eq=False, slots=True)
class IntLeafExpr(IntIndexValue, ABC):
    def enumerate_all_cond_branches(
        self,
    ) -> Sequence[Tuple[Optional[BooleanIndexValue], IndexExpr]]:
        return [(None, self)]

    def as_upper_bound_access(self) -> IndexExpr:
        return self

    def as_lower_bound_access(self) -> IndexExpr:
        return self

    def simplify_mins_and_maxes(self, aggressive: bool = False) -> IntLeafExpr:
        return self


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class ConstInt(IntLeafExpr):
    const: int

    def __int__(self) -> int:
        return self.const

    def __bool__(self) -> bool:
        return bool(self.const)

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.const))

    def __post_init__(self) -> None:
        assert isinstance(self.const, int), "ConstInt must be an integer, got %s: %s" % (
            type(self.const),
            self.const,
        )
        # self._cached_sympy_expr()

    def __str__(self) -> str:
        return str(self.const)

    def remap(self, domain_map: Mapping[Symbol, IntIndexValue]) -> IntIndexValue:
        return ConstInt(self.const)

    def _codegen_str(self, index_values: Mapping[Symbol, int]) -> str:
        return str(self.const)

    def __repr__(self) -> str:
        return f"{self.const}"

    def struct_eq(self, other: Any) -> bool:
        if not isinstance(other, ConstInt):
            try:
                other = lift_to_int_ie(other)
            except Exception:
                return False
        if not isinstance(other, ConstInt):
            return False

        return self.const == other.const

    def accesses_bound(self) -> bool:
        return False

    def evaluate(self, index_values: Mapping[Symbol, int]) -> int:
        return self.const

    def partial_eval(self, domain_map: Mapping[Symbol, int]) -> IntIndexValue:
        return self

    def is_constant(self) -> bool:
        return True


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class ConstBool(BooleanIndexValue):
    const: bool

    def __int__(self) -> int:
        return 1 if self.const else 0

    def __bool__(self) -> bool:
        return self.const

    def enumerate_all_cond_branches(
        self,
    ) -> Sequence[Tuple[Optional[BooleanIndexValue], IndexExpr]]:
        return [(None, self)]

    def eliminate_symbol(self, symbol: Symbol) -> BooleanIndexValue:
        return self

    def remap(self, domain_map: Mapping[Symbol, IntIndexValue]) -> BooleanIndexValue:
        return ConstBool(self.const)

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.const))

    def __str__(self) -> str:
        if self.const:
            return "true"
        else:
            return "false"

    def _codegen_str(self, index_values: Mapping[Symbol, int]) -> str:
        return str(self.const).title()

    __repr__ = __str__

    def struct_eq(self, other: Any) -> bool:
        if not isinstance(other, ConstBool):
            try:
                other = lift_to_bool_ie(other)
            except Exception:
                return False
        if not isinstance(other, ConstBool):
            return False
        return self.const == other.const

    def accesses_bound(self) -> bool:
        return False

    def evaluate(self, index_values: Mapping[Symbol, int]) -> bool:
        return self.const

    def partial_eval(self, domain_map: Mapping[Symbol, int]) -> BooleanIndexValue:
        return self

    def is_constant(self) -> bool:
        return True


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class Symbol(IntLeafExpr):
    name: str
    is_bound: bool = False
    idx: int = -1

    def __hash__(self) -> int:
        # TODO return self.idx?
        return hash(self.name)

    def __str__(self) -> str:
        return self.name

    __repr__ = __str__

    def _codegen_str(self, index_values: Mapping[Symbol, int]) -> str:
        if self.is_bound and index_values[self] >= 0:
            return f"({index_values[self]})"
        return f"cd[{self.name}]"
        # return self.name

    # def __post_init__(self) -> None:
    #    if self.name.startswith("d"):
    #        if self.idx > 5:
    #            raise ValueError(f"Symbol {self} has index greater than 5")

    def remap(self, domain_map: Mapping[Symbol, IntIndexValue]) -> IntIndexValue:
        # mapping = domain_map[self]
        # if self.is_bound:
        #    return self
        # else:
        if self in domain_map:
            return domain_map[self]
        else:
            return self
        # return mapping

    def custom_repr(self) -> str:
        return f"Symbol({self.name}, bound={self.is_bound}, idx={self.idx})"

    def as_var(self) -> Symbol:
        return Symbol(
            self.name.lower(),
            is_bound=False,
            idx=(self.idx - 1 if self.is_bound else self.idx),
        )

    def as_bound(self) -> Symbol:
        return Symbol(
            self.name.upper(),
            is_bound=True,
            idx=(self.idx if self.is_bound else self.idx + 1),
        )

    def vars_used(self) -> Set[Symbol]:
        return {self} if not self.is_bound else set()

    def struct_eq(self, other: Any) -> bool:
        if not isinstance(other, Symbol):
            try:
                other = lift_to_int_ie(other)
            except Exception:
                return False
        if not isinstance(other, Symbol):
            return False

        # Either both are bound, or both are not bound
        if self.is_bound != other.is_bound:
            return False
        return self.name == other.name

    def accesses_bound(self) -> bool:
        return self.is_bound

    def evaluate(self, index_values: Mapping[Symbol, int]) -> int:
        # res = index_values[self]
        # if res is None:
        #    raise ValueError(f"Could not evaluate symbol {self} in {index_values}")
        # return res
        return index_values[self]

    def partial_eval(self, domain_map: Mapping[Symbol, int]) -> IntIndexValue:
        if self not in domain_map:
            return self
        val = domain_map[self]

        return ConstInt(val)

    def is_constant(self) -> bool:
        return self.is_bound

    def is_variable(self) -> bool:
        return not self.is_bound


def slice_(start: IntIndexValueLike, stop: IntIndexValueLike) -> IndexAtom:
    l = lift_to_int_ie(start)
    r = lift_to_int_ie(stop)

    if isinstance(l, ConstInt) and isinstance(r, ConstInt):
        assert l.const <= r.const, "Slice start must be less than or equal to stop"
    if r.struct_eq(l):
        raise ValueError(
            "Slice start and stop cannot be equal as this would result in an empty slice"
        )
    if r.struct_eq(l + 1):
        return l

    return Slice(l, r)


@dataclass(frozen=True, eq=False, slots=True)
class Slice(IndexAtom):
    start: IntIndexValue
    stop: IntIndexValue
    # step: IndexValueExpr

    def __post_init__(self) -> None:
        self.children.extend([self.start, self.stop])
        # self._cached_sympy_expr()

    def __str__(self) -> str:
        return f"(({self.start}):(({self.stop})-1))"

    def _codegen_str(self, index_values: Mapping[Symbol, int]) -> str:
        start = self.start._codegen_str(index_values)
        stop = self.stop._codegen_str(index_values)
        return f"slice(({start}), ({stop}))"

    def is_block_access(self) -> bool:
        left_is_block = (
            isinstance(self.start, Mul)
            and isinstance(self.start.left_operand, IntIndexValue)
            and isinstance(self.start.right_operand, ConstInt)
        )
        if left_is_block:
            return True

        # def is_block_stop(stop: IntIndexValue) -> bool:
        #    return (
        #        isinstance(stop, Mul)
        #        and isinstance(stop.left_operand, Add)
        #        and isinstance(stop.left_operand.left_operand, Symbol)
        #        and isinstance(stop.left_operand.right_operand, ConstInt)
        #        and isinstance(stop.right_operand, ConstInt)
        #    )

        ## e.g. (t+1)*16
        # right_is_block = is_block_stop(self.stop) and (
        #    self.start.left_operand.equivalent(self.stop.left_operand.left_operand)
        # )
        ## e.g. min((t+1)*16, t)
        ##TODO: need to generalize, since children are not guaranteed to be in this order
        # right_is_min_block = (
        #    isinstance(self.stop, Min)
        #    and is_block_stop(self.stop.children[0])
        #    and isinstance(self.stop.children[1], Symbol)
        #    and self.stop.children[1].equivalent(list(self.stop.vars_used())[0])
        # )

        # if left_is_block and (right_is_block or right_is_min_block):
        #    return True

        return False

    def is_shrinking_slice(self) -> bool:
        # e.g. t:T
        return (not self.start.is_constant()) and self.stop.is_constant()

    def is_growing_slice(self) -> bool:
        # e.g. 0:t
        return self.start.is_constant() and not self.stop.is_constant()

    def get_block_access_block_size(self) -> int:
        assert self.is_block_access(), "Slice must be a block access"
        return self.start.right_operand.const  # type: ignore

    __repr__ = __str__

    def remap(self, domain_map: Mapping[Symbol, IntIndexValue]) -> Slice:
        return Slice(
            self.start.remap(domain_map),  # type: ignore
            self.stop.remap(domain_map),  # type: ignore
            # self.step.remap(domain_map),
        )

    def struct_eq(self, other: Any) -> bool:
        if not isinstance(other, Slice):
            try:
                other = lift_to_ie_slice(other)
            except Exception:
                return False
        if not isinstance(other, Slice):
            return False
        start_eq = self.start.struct_eq(other.start)
        stop_eq = self.stop.struct_eq(other.stop)
        # step_eq = self.step.equivalent(other.step)
        return start_eq and stop_eq  # and step_eq

    def evaluate(self, index_values: Mapping[Symbol, int]) -> IndexAtom:
        return slice_(self.start.evaluate(index_values), self.stop.evaluate(index_values))

    def partial_eval(self, domain_map: Mapping[Symbol, int]) -> IntIndexValue:
        new_start = self.start.partial_eval(domain_map)  # type: ignore
        new_stop = self.stop.partial_eval(domain_map)  # type: ignore
        return slice_(new_start, new_stop)  # type: ignore

    def evaluate_shape(
        self, index_values: Optional[Mapping[Symbol, int]] = None
    ) -> Tuple[Union[int, IntIndexValue], ...]:
        if index_values is None:
            index_values = {}

        assert index_values is not None
        try:
            start_eval = self.start.evaluate(index_values)
            assert isinstance(start_eval, int), "Start must be an integer in shape eval"
            start: Union[int, IntIndexValue] = start_eval
        except KeyError:
            # Could not evaluate start, so we return the symbolic index
            start = self.start

        try:
            stop_eval = self.stop.evaluate(index_values)
            assert isinstance(stop_eval, int), "Stop must be an integer in shape eval"
            stop: Union[int, IntIndexValue] = stop_eval
        except KeyError:
            # Could not evaluate start, so we return the symbolic index
            stop = self.stop

        return (stop - start,)

    def is_point(self) -> bool:
        return False

    def as_upper_bound_access(self) -> IntIndexValue:
        return self.stop.as_upper_bound_access() - 1  # type: ignore

    def as_lower_bound_access(self) -> IntIndexValue:
        return self.start.as_lower_bound_access()  # type: ignore

    def drop_modulos(self) -> Slice:
        return self.__class__(self.start.drop_modulos(), self.stop.drop_modulos())  # type: ignore

    def simplify_mins_and_maxes(self, aggressive: bool = False) -> Slice:
        return self.__class__(
            self.start.simplify_mins_and_maxes(aggressive),  # type: ignore
            self.stop.simplify_mins_and_maxes(aggressive),  # type: ignore
        )


@dataclass(frozen=True, eq=False, slots=True)
class NAryExpr(IntIndexValue, ABC):
    operands: Tuple[IntIndexValue, ...] = field(default_factory=tuple)

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.operands))

    def __post_init__(self) -> None:
        self.children.extend(self.operands)
        # self._cached_sympy_expr()

    # TODO: What is  going on here? This needs docs.
    def enumerate_all_cond_branches(
        self,
    ) -> Sequence[Tuple[Optional[BooleanIndexValue], IndexExpr]]:
        children_enumerations = [child.enumerate_all_cond_branches() for child in self.children]

        res: List[Tuple[Optional[BooleanIndexValue], IndexExpr]] = []
        for prod in itertools.product(*children_enumerations):
            branches = [p[1] for p in prod]
            conds = [p[0] for p in prod]
            conds = [c for c in conds if c is not None]
            if len(conds) == 0:
                res.append((None, self.__class__(tuple(branches))))  # type: ignore
            elif len(conds) == 1:
                res.append((conds[0], self.__class__(tuple(branches))))  # type: ignore
            else:
                r = (functools.reduce(And, conds), self.__class__(tuple(branches)))  # type: ignore
                res.append(r)
        return res

    def as_upper_bound_access(self) -> IndexExpr:
        return self.__class__(
            tuple(x.as_upper_bound_access() for x in self.operands)  # type: ignore
        )

    def as_lower_bound_access(self) -> IndexExpr:
        return self.__class__(
            tuple(x.as_lower_bound_access() for x in self.operands)  # type: ignore
        )

    def drop_modulos(self) -> IndexExpr:
        return self.__class__(tuple(x.drop_modulos() for x in self.operands))  # type: ignore

    def remap(self, domain_map: Mapping[Symbol, IntIndexValue]) -> IntIndexValue:
        return self.__class__(
            tuple(x.remap(domain_map) for x in self.operands)  # type: ignore
        )

    def struct_eq(self, other: Any) -> bool:
        if not isinstance(other, NAryExpr):
            try:
                other = lift_to_ie(other)
            except Exception:
                return False
        if not isinstance(other, NAryExpr):
            return False

        if self.__class__ is not other.__class__:
            return False
        if len(self.operands) != len(other.operands):
            return False

        # Since order doesn't matter, we need to check if each operand in self
        # has a matching operand in other, and vice versa
        used_other_indices = set()
        for x in self.operands:
            found_match = False
            for i, y in enumerate(other.operands):
                if i not in used_other_indices and x.struct_eq(y):
                    used_other_indices.add(i)
                    found_match = True
                    break
            if not found_match:
                return False
        return True


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class Min(NAryExpr):
    def __str__(self) -> str:
        return f"min({','.join(map(str, self.children))})"

    __repr__ = __str__

    def _codegen_str(self, index_values: Mapping[Symbol, int]) -> str:
        return f"min({', '.join((x._codegen_str(index_values) for x in self.operands))})"

    def accesses_bound(self) -> bool:
        children_access_bound = [c.accesses_bound() for c in self.children]
        all_children_access_bound = all(children_access_bound)
        # So, if there is any child not accessing the bound,
        # that will by definition be smaller than the bound.
        return all_children_access_bound

    def evaluate(self, index_values: Mapping[Symbol, int]) -> int:
        return builtins.min([c.evaluate(index_values) for c in self.children])  # type: ignore

    def partial_eval(self, domain_map: Mapping[Symbol, int]) -> IntIndexValue:
        members = [c.partial_eval(domain_map) for c in self.children]  # type: ignore
        constints = [c for c in members if isinstance(c, ConstInt)]
        nonconstints = [c for c in members if not isinstance(c, ConstInt)]

        if len(constints) == 0:
            return self

        local_min = ConstInt(builtins.min([c.const for c in constints]))

        if len(nonconstints) == 0:
            return local_min

        return min(local_min, *nonconstints)

    def as_upper_bound_access(self) -> IntIndexValue:
        return Min(tuple(o.as_upper_bound_access() for o in self.operands))  # type: ignore

    def as_lower_bound_access(self) -> IntIndexValue:
        return Min(tuple(o.as_lower_bound_access() for o in self.operands))  # type: ignore

    # TODO: We need to remove this method. It's semantics is not clear.
    def simplify_mins_and_maxes(self, aggressive: bool = False) -> IntIndexValue:
        new_operands = [o.simplify_mins_and_maxes(aggressive) for o in self.operands]

        # Remove any bounds accesses
        new_operands = [o for o in new_operands if not (o.accesses_bound() or o.is_constant())]

        if len(new_operands) == 0:
            raise ValueError("Min with no non-bound operands")
        elif len(new_operands) == 1:
            return new_operands[0]  # type: ignore
        else:
            if aggressive:
                return new_operands[0]  # type: ignore
            return self.__class__(tuple(new_operands))  # type: ignore


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class Max(NAryExpr):
    def __str__(self) -> str:
        return f"max({','.join(map(str, self.children))})"

    __repr__ = __str__

    def _codegen_str(self, index_values: Mapping[Symbol, int]) -> str:
        # TODO builtins.min?
        return f"max({', '.join((x._codegen_str(index_values) for x in self.operands))})"

    def accesses_bound(self) -> bool:
        children_access_bound = [c.accesses_bound() for c in self.children]
        any_children_access_bound = any(children_access_bound)
        # So, if there is any child accessing the bound,
        # that will by definition be greater than any non-bound access
        return any_children_access_bound

    def evaluate(self, index_values: Mapping[Symbol, int]) -> int:
        return builtins.max([c.evaluate(index_values) for c in self.children])  # type: ignore

    def partial_eval(self, domain_map: Mapping[Symbol, int]) -> IntIndexValue:
        members = [c.partial_eval(domain_map) for c in self.children]  # type: ignore
        constints = [c for c in members if isinstance(c, ConstInt)]
        nonconstints = [c for c in members if not isinstance(c, ConstInt)]

        if len(constints) == 0:
            return self

        local_max = ConstInt(builtins.max([c.const for c in constints]))

        if len(nonconstints) == 0:
            return local_max

        return max(local_max, *nonconstints)

    def simplify_mins_and_maxes(self, aggressive: bool = False) -> IntIndexValue:
        new_operands = [o.simplify_mins_and_maxes(aggressive) for o in self.operands]

        # Remove any const accesses
        new_operands = [o for o in new_operands if not o.is_constant()]

        if len(new_operands) == 0:
            raise ValueError("Max with no non-const operands")
        elif len(new_operands) == 1:
            return new_operands[0]  # type: ignore
        else:
            if aggressive:
                return new_operands[0]  # type: ignore
            return self.__class__(tuple(new_operands))  # type: ignore


def piecewise(
    conds_and_branches: Sequence[Tuple[BooleanIndexValue, IntIndexValue]],
) -> IntIndexValue:
    # TODO catch and simplify patterns:
    # piecewise([(d1 >= 256 and ds0 == 0, 256), (not (d1 >= 256 and ds0 == 0), d1 + 1)]) - 1
    # into:
    # min(d1, 255) + (d1 - min(d1, 255)) * (ds0 != 0)

    # TODO: Try to convert to min max when possible

    return Piecewise(conds_and_branches)  # type: ignore


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class Piecewise(IntIndexValue):
    conds_and_branches: Sequence[Tuple[BooleanIndexValue, IntIndexValue]]

    def __post_init__(self) -> None:
        self.children.extend([x for objs in self.conds_and_branches for x in objs])
        # self._cached_sympy_expr()

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.conds_and_branches))

    def __str__(self) -> str:
        return (
            "piecewise("
            + ";".join(
                [
                    f" {i}: cond={str(cond)}, expr={str(expr)}"
                    for i, (cond, expr) in enumerate(self.conds_and_branches)
                ]
            )
            + ")"
        )

    __repr__ = __str__

    def _codegen_str(self, index_values: Mapping[Symbol, int]) -> str:
        res = ""
        for cond, expr in self.conds_and_branches:
            res += f"({expr._codegen_str(index_values)} if {cond._codegen_str(index_values)} else ("

        return res + "0" + ")" * (len(self.conds_and_branches) * 2)

    def evaluate(self, index_values: Mapping[Symbol, int]) -> int:
        for cond, expr in self.conds_and_branches:
            if cond.evaluate(index_values):
                return expr.evaluate(index_values)
        raise ValueError(
            f"No branch of piecewise {self} evaluated to true with index_values {index_values}"
        )

    def partial_eval(self, domain_map: Mapping[Symbol, int]) -> IntIndexValue:
        partially_evaled = [
            (cond.partial_eval(domain_map), expr.partial_eval(domain_map))  # type: ignore
            for cond, expr in self.conds_and_branches
        ]

        nonfalse = [(c, e) for c, e in partially_evaled if not c.struct_eq(ConstBool(False))]

        up_to_first_true = []
        for c, e in nonfalse:
            up_to_first_true.append((c, e))

            if c.struct_eq(ConstBool(True)):
                break

        if len(up_to_first_true) == 1:
            return up_to_first_true[0][1]

        return piecewise(up_to_first_true)  # type: ignore

    def enumerate_all_cond_branches(  # noqa: C901
        self,
    ) -> Sequence[Tuple[Optional[BooleanIndexValue], IndexExpr]]:
        children_enums: List[Tuple[Optional[BooleanIndexValue], IndexExpr]] = []
        for cond, branch in self.conds_and_branches:
            cond_enums = cond.enumerate_all_cond_branches()
            branch_enums = branch.enumerate_all_cond_branches()
            for cond_cond, cond_expr in cond_enums:
                for branch_cond, branch_expr in branch_enums:
                    assert cond_expr.is_boolean_expr()
                    final_cond: BooleanIndexValue = cond_expr  # type: ignore
                    if cond_cond:
                        final_cond = And(cond_cond, final_cond)
                    if branch_cond:
                        final_cond = And(branch_cond, final_cond)

                    children_enums.append((final_cond, branch_expr))
        return children_enums

    def as_upper_bound_access(self) -> IntIndexValue:
        return self.__class__(
            tuple(
                (cond, branch.as_upper_bound_access())  # type: ignore
                for cond, branch in self.conds_and_branches
            )  # type: ignore
        )

    def as_lower_bound_access(self) -> IntIndexValue:
        return self.__class__(
            tuple(
                (cond, branch.as_lower_bound_access())  # type: ignore
                for cond, branch in self.conds_and_branches
            )  # type: ignore
        )

    def drop_modulos(self) -> IntIndexValue:
        return self.__class__(
            tuple(
                (cond.drop_modulos(), branch.drop_modulos())  # type: ignore
                for cond, branch in self.conds_and_branches
            )  # type: ignore
        )

    def simplify_mins_and_maxes(self, aggressive: bool = False) -> IntIndexValue:
        if aggressive:
            # NOTE: The last branch tends to be the common case
            return self.conds_and_branches[1][-1]  # type: ignore
        return self.__class__(
            tuple(
                (cond, branch.simplify_mins_and_maxes(aggressive))  # type: ignore
                for cond, branch in self.conds_and_branches
            )  # type: ignore
        )

    def remap(self, domain_map: Mapping[Symbol, IntIndexValue]) -> IntIndexValue:
        return self.__class__(
            tuple(
                (cond.remap(domain_map), branch.remap(domain_map))  # type: ignore
                for cond, branch in self.conds_and_branches
            )  # type: ignore
        )

    def struct_eq(self, other: Any) -> bool:
        if not isinstance(other, Piecewise):
            try:
                other = lift_to_int_ie(other)
            except Exception:
                return False
        if not isinstance(other, Piecewise):
            return False

        if len(self.conds_and_branches) != len(other.conds_and_branches):
            return False

        return all(
            x[0].struct_eq(y[0]) and x[1].struct_eq(y[1])
            for x, y in zip(self.conds_and_branches, other.conds_and_branches, strict=True)
        )

    # def evaluate_shape(
    #    self, index_values: Mapping[Symbol, int]
    # ) -> Tuple[Union[int, IndexValue], ...]:
    #    pass #TODO if we end-up needing piecewise shapes


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class FloatBinaryExpr(IntIndexValue, ABC):
    left_operand: IntIndexValue
    right_operand: IntIndexValue
    symbol = ""
    operation: Callable[[int, int], float] = lambda x, y: x + y

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, (self.left_operand, self.right_operand)))

    def __post_init__(self) -> None:
        self.children.extend([self.left_operand, self.right_operand])
        # self._cached_sympy_expr()

    def __str__(self) -> str:
        return f"({self.left_operand}{self.symbol}{self.right_operand})"

    def _codegen_str(self, index_values: Mapping[Symbol, int]) -> str:
        l = self.left_operand._codegen_str(index_values)
        r = self.right_operand._codegen_str(index_values)

        return f"({l}{self.symbol}{r})"

    __repr__ = __str__

    def evaluate(self, index_values: Mapping[Symbol, int]) -> float:  # type: ignore
        left = self.left_operand.evaluate(index_values)
        right = self.right_operand.evaluate(index_values)
        # assert isinstance(left, int) and isinstance(right, int)
        return self.operation(left, right)

    def partial_eval(self, domain_map: Mapping[Symbol, int]) -> IntIndexValue:
        left = self.left_operand.partial_eval(domain_map)
        right = self.right_operand.partial_eval(domain_map)
        if isinstance(left, (ConstInt, ConstFloat)) and isinstance(right, (ConstInt, ConstFloat)):
            return lift_to_ie(self.operation(left.const, right.const))  # type: ignore

        # assert isinstance(left, int) and isinstance(right, int)
        return self.__class__(lift_to_ie(left), lift_to_ie(right))  # type: ignore

    def struct_eq(self, other: Any) -> bool:
        if not isinstance(other, FloatBinaryExpr):
            try:
                other = lift_to_ie(other)
            except Exception:
                return False
        if not isinstance(other, FloatBinaryExpr):
            return False
        if type(self) is type(other) and len(self.children) == len(other.children):
            return self.left_operand.struct_eq(other.left_operand) and self.right_operand.struct_eq(
                other.right_operand
            )
        return False

    def remap(self, domain_map: Mapping[Symbol, IntIndexValue]) -> IntIndexValue:
        return self.__class__(
            self.left_operand.remap(domain_map),  # type: ignore
            self.right_operand.remap(domain_map),  # type: ignore
        )

    def drop_modulos(self) -> IndexExpr:
        return self.__class__(
            self.left_operand.drop_modulos(),  # type: ignore
            self.right_operand.drop_modulos(),  # type: ignore
        )


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class IntBinaryExpr(IntIndexValue, ABC):
    left_operand: IntIndexValue
    right_operand: IntIndexValue
    symbol = ""
    operation: Callable[[int, int], int] = lambda x, y: x + y

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, (self.left_operand, self.right_operand)))

    def __post_init__(self) -> None:
        self.children.extend([self.left_operand, self.right_operand])
        # self._cached_sympy_expr()

    def __str__(self) -> str:
        return f"({self.left_operand}{self.symbol}{self.right_operand})"

    def _codegen_str(self, index_values: Mapping[Symbol, int]) -> str:
        l = self.left_operand._codegen_str(index_values)
        r = self.right_operand._codegen_str(index_values)
        return f"({l}{self.symbol}{r})"

    __repr__ = __str__

    def evaluate(self, index_values: Mapping[Symbol, int]) -> int:
        left = self.left_operand.evaluate(index_values)
        right = self.right_operand.evaluate(index_values)
        # assert isinstance(left, int) and isinstance(right, int)
        return self.operation(left, right)

    def partial_eval(self, domain_map: Mapping[Symbol, int]) -> IntIndexValue:
        left = self.left_operand.partial_eval(domain_map)
        right = self.right_operand.partial_eval(domain_map)
        if isinstance(left, (ConstInt, ConstFloat)) and isinstance(right, (ConstInt, ConstFloat)):
            return lift_to_ie(self.operation(left.const, right.const))  # type: ignore

        # assert isinstance(left, int) and isinstance(right, int)
        return self.__class__(lift_to_ie(left), lift_to_ie(right))  # type: ignore

    def drop_modulos(self) -> IndexExpr:
        return self.__class__(
            self.left_operand.drop_modulos(),  # type: ignore
            self.right_operand.drop_modulos(),  # type: ignore
        )

    def as_upper_bound_access(self) -> IntIndexValue:
        # TODO wouldnt this be wrong for for example, subtraction or division, where we want
        # to access the upper bound of the left operand, but the lower bound of the right operand?
        return self.__class__(
            self.left_operand.as_upper_bound_access(),  # type: ignore
            self.right_operand.as_upper_bound_access(),  # type: ignore
        )

    def as_lower_bound_access(self) -> IntIndexValue:
        return self.__class__(
            self.left_operand.as_lower_bound_access(),  # type: ignore
            self.right_operand.as_lower_bound_access(),  # type: ignore
        )

    def simplify_mins_and_maxes(self, aggressive: bool = False) -> IntIndexValue:
        return self.__class__(
            self.left_operand.simplify_mins_and_maxes(aggressive),  # type: ignore
            self.right_operand.simplify_mins_and_maxes(aggressive),  # type: ignore
        )

    def struct_eq(self, other: Any) -> bool:
        # NOTE: since most operations are reflexive, we need to compare both.
        # We override this in non-reflexive operations
        if not isinstance(other, IntBinaryExpr):
            try:
                other = lift_to_int_ie(other)
            except Exception:
                return False
        if not isinstance(other, IntBinaryExpr):
            return False
        if type(self) is type(other) and len(self.children) == len(other.children):
            return (
                self.left_operand.struct_eq(other.left_operand)
                and self.right_operand.struct_eq(other.right_operand)
            ) or (
                self.left_operand.struct_eq(other.right_operand)
                and self.right_operand.struct_eq(other.left_operand)
            )
        return False

    def remap(self, domain_map: Mapping[Symbol, IntIndexValue]) -> IntIndexValue:
        return self.__class__(
            self.left_operand.remap(domain_map),  # type: ignore
            self.right_operand.remap(domain_map),  # type: ignore
        )


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class BooleanLogicBinaryExpr(BooleanIndexValue, ABC):
    left_operand: BooleanIndexValue
    right_operand: BooleanIndexValue
    symbol = ""
    operation: Callable[[bool, bool], bool] = lambda x, y: x == y

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, (self.left_operand, self.right_operand)))

    def __post_init__(self) -> None:
        self.children.extend([self.left_operand, self.right_operand])
        # self._cached_sympy_expr()

    def __str__(self) -> str:
        return f"({self.left_operand}{self.symbol}{self.right_operand})"

    __repr__ = __str__

    def struct_eq(self, other: Any) -> bool:
        # NOTE: since most operations are reflexive, we need to compare both.
        if not isinstance(other, BooleanLogicBinaryExpr):
            try:
                other = lift_to_bool_ie(other)
            except Exception:
                return False
        if not isinstance(other, BooleanLogicBinaryExpr):
            return False

        if type(self) is type(other) and len(self.children) == len(other.children):
            # NOTE: And and Or are reflexive
            return (
                self.left_operand.struct_eq(other.left_operand)
                and self.right_operand.struct_eq(other.right_operand)
                or (
                    self.left_operand.struct_eq(other.right_operand)
                    and self.right_operand.struct_eq(other.left_operand)
                )
            )

        return False

    def evaluate(self, index_values: Mapping[Symbol, int]) -> bool:
        left = self.left_operand.evaluate(index_values)  # type: ignore
        right = self.right_operand.evaluate(index_values)  # type: ignore
        return self.operation(left, right)

    def partial_eval(self, domain_map: Mapping[Symbol, int]) -> BooleanIndexValue:
        left = self.left_operand.partial_eval(domain_map)  # type: ignore
        right = self.right_operand.partial_eval(domain_map)  # type: ignore
        if isinstance(left, (ConstBool,)) and isinstance(right, (ConstBool,)):
            return lift_to_ie(self.operation(left.const, right.const))  # type: ignore

        # assert isinstance(left, int) and isinstance(right, int)
        return self.__class__(lift_to_ie(left), lift_to_ie(right))  # type: ignore

    def _codegen_str(self, index_values: Mapping[Symbol, int]) -> str:
        l = self.left_operand._codegen_str(index_values)
        r = self.right_operand._codegen_str(index_values)
        return f"({l}{self.symbol}{r})"

    def remap(self, domain_map: Mapping[Symbol, IntIndexValue]) -> BooleanIndexValue:
        return self.__class__(
            self.left_operand.remap(domain_map),  # type: ignore
            self.right_operand.remap(domain_map),  # type: ignore
        )

    def drop_modulos(self) -> IndexExpr:
        return self.__class__(
            self.left_operand.drop_modulos(),  # type: ignore
            self.right_operand.drop_modulos(),  # type: ignore
        )


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class BooleanBinaryExpr(BooleanIndexValue, ABC):
    left_operand: IntIndexValue
    right_operand: IntIndexValue
    symbol = ""
    operation: Callable[[int, int], bool] = lambda x, y: x == y

    def struct_eq(self, other: Any) -> bool:
        # NOTE: since most operations are reflexive, we need to compare both.
        if not isinstance(other, BooleanBinaryExpr):
            try:
                other = lift_to_bool_ie(other)
            except Exception:
                return False
        if not isinstance(other, BooleanBinaryExpr):
            return False

        if type(self) is type(other) and len(self.children) == len(other.children):
            if isinstance(self, Equal):
                # NOTE: Equal is reflexive, so we need to compare both.
                return (
                    self.left_operand.struct_eq(other.left_operand)
                    and self.right_operand.struct_eq(other.right_operand)
                ) or (
                    self.left_operand.struct_eq(other.right_operand)
                    and self.right_operand.struct_eq(other.left_operand)
                )
            else:
                # NOTE: all others are not reflexive
                return self.left_operand.struct_eq(
                    other.left_operand
                ) and self.right_operand.struct_eq(other.right_operand)

        return False

    def eliminate_symbol(self, symbol: Symbol) -> BooleanIndexValue:
        assert not symbol.is_bound
        # Any condition involving the eliminated symbol becomes True
        variables = self.vars_used()
        for v in variables:
            if v.struct_eq(symbol):
                return ConstBool(True)
        return self

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, (self.left_operand, self.right_operand)))

    def __post_init__(self) -> None:
        self.children.extend([self.left_operand, self.right_operand])
        # self._cached_sympy_expr()

    def __str__(self) -> str:
        return f"({self.left_operand}{self.symbol}{self.right_operand})"

    def _codegen_str(self, index_values: Mapping[Symbol, int]) -> str:
        l = self.left_operand._codegen_str(index_values)
        r = self.right_operand._codegen_str(index_values)
        return f"({l}{self.symbol}{r})"

    __repr__ = __str__

    def evaluate(self, index_values: Mapping[Symbol, int]) -> bool:
        left = self.left_operand.evaluate(index_values)
        right = self.right_operand.evaluate(index_values)
        return self.operation(left, right)

    def partial_eval(self, domain_map: Mapping[Symbol, int]) -> BooleanIndexValue:
        left = self.left_operand.partial_eval(domain_map)
        right = self.right_operand.partial_eval(domain_map)
        if isinstance(left, (ConstInt, ConstFloat, ConstBool)) and isinstance(
            right, (ConstInt, ConstFloat, ConstBool)
        ):
            return lift_to_ie(self.operation(left.const, right.const))  # type: ignore

        # assert isinstance(left, int) and isinstance(right, int)
        return self.__class__(lift_to_ie(left), lift_to_ie(right))  # type: ignore

    def remap(self, domain_map: Mapping[Symbol, IntIndexValue]) -> BooleanIndexValue:
        return self.__class__(
            self.left_operand.remap(domain_map),  # type: ignore
            self.right_operand.remap(domain_map),  # type: ignore
        )

    def drop_modulos(self) -> IndexExpr:
        return self.__class__(
            self.left_operand.drop_modulos(),  # type: ignore
            self.right_operand.drop_modulos(),  # type: ignore
        )


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class Add(IntBinaryExpr):
    symbol = "+"
    operation: Callable[[int, int], int] = lambda x, y: x + y


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class Sub(IntBinaryExpr):
    symbol = "-"
    operation: Callable[[int, int], int] = lambda x, y: x - y

    def struct_eq(self, other: Any) -> bool:
        # NOTE: not-reflexive
        if not isinstance(other, Sub):
            try:
                other = lift_to_int_ie(other)
            except Exception:
                return False
        if not isinstance(other, Sub):
            return False
        if type(self) is type(other) and len(self.children) == len(other.children):
            return self.left_operand.struct_eq(other.left_operand) and self.right_operand.struct_eq(
                other.right_operand
            )
        return False


# @dataclass(repr=False, frozen=True, eq=False)
# class Division(IntBinaryExpr):
#    symbol = "/"
#    operation: Callable[[int, int], int] = lambda x, y: x / y


# TODO replace with composition of true div and floor
@dataclass(repr=False, frozen=True, eq=False, slots=True)
class FloorDivision(IntBinaryExpr):
    symbol = "//"
    operation: Callable[[int, int], int] = lambda x, y: x // y

    # def __post_init__(self) -> None:
    #    self.children.extend([self.left_operand, self.right_operand])
    #    if isinstance(self.left_operand, Sub):
    #        print(f"FloorDivision: {self}")
    #        print(self.creation_traceback)

    # NOTE: / and // seem to behave differently in isl
    # >>> import islpy as isl
    # >>> isl.UnionMap.read_from_str(isl.Context(), "[A] -> {S0[a] -> S1[a//2]}")
    # UnionMap("[A] -> { S0[a] -> S1[o0] : -1 + a <= 2o0 <= a }")
    # >>> isl.UnionMap.read_from_str(isl.Context(), "[A] -> {S0[a] -> S1[a/2]}")
    # UnionMap("[A] -> { S0[a] -> S1[o0] : 2o0 = a }")
    def struct_eq(self, other: Any) -> bool:
        # NOTE: not-reflexive
        if not isinstance(other, FloorDivision):
            try:
                other = lift_to_int_ie(other)
            except Exception:
                return False
        if not isinstance(other, FloorDivision):
            return False
        if type(self) is type(other) and len(self.children) == len(other.children):
            return self.left_operand.struct_eq(other.left_operand) and self.right_operand.struct_eq(
                other.right_operand
            )
        return False


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class TrueDivision(FloatBinaryExpr):
    symbol = "/"
    operation: Callable[[int, int], float] = lambda x, y: x / y

    def _codegen_str(self, index_values: Mapping[Symbol, int]) -> str:
        l = self.left_operand._codegen_str(index_values)
        r = self.right_operand._codegen_str(index_values)
        return f"int({l}/{r})"

    def struct_eq(self, other: Any) -> bool:
        # NOTE: not-reflexive
        if not isinstance(other, TrueDivision):
            try:
                other = lift_to_ie(other)
            except Exception:
                return False
        if not isinstance(other, TrueDivision):
            return False
        if type(self) is type(other) and len(self.children) == len(other.children):
            return self.left_operand.struct_eq(other.left_operand) and self.right_operand.struct_eq(
                other.right_operand
            )
        return False


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class Mul(IntBinaryExpr):
    symbol = "*"
    operation: Callable[[int, int], int] = lambda x, y: x * y


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class Modulos(IntBinaryExpr):
    symbol = "%"
    operation: Callable[[int, int], int] = lambda x, y: x % y

    def struct_eq(self, other: Any) -> bool:
        # NOTE: not-reflexive
        if not isinstance(other, Modulos):
            try:
                other = lift_to_int_ie(other)
            except Exception:
                return False
        if not isinstance(other, Modulos):
            return False
        if type(self) is type(other) and len(self.children) == len(other.children):
            return self.left_operand.struct_eq(other.left_operand) and self.right_operand.struct_eq(
                other.right_operand
            )
        return False

    def drop_modulos(self) -> IndexExpr:
        return self.right_operand


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class Pow(IntBinaryExpr):
    symbol = "**"
    operation: Callable[[int, int], int] = lambda x, y: x**y

    def struct_eq(self, other: Any) -> bool:
        # NOTE: not-reflexive
        if not isinstance(other, Pow):
            try:
                other = lift_to_int_ie(other)
            except Exception:
                return False
        if not isinstance(other, Pow):
            return False
        if type(self) is type(other) and len(self.children) == len(other.children):
            return self.left_operand.struct_eq(other.left_operand) and self.right_operand.struct_eq(
                other.right_operand
            )
        return False


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class And(BooleanLogicBinaryExpr):
    symbol = " and "
    operation: Callable[[bool, bool], bool] = lambda x, y: x and y

    def eliminate_symbol(self, symbol: Symbol) -> BooleanIndexValue:
        assert not symbol.is_bound
        left_operand = self.left_operand.eliminate_symbol(symbol)
        right_operand = self.right_operand.eliminate_symbol(symbol)
        return left_operand & right_operand


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class Or(BooleanLogicBinaryExpr):
    symbol = " or "
    operation: Callable[[bool, bool], bool] = lambda x, y: x or y

    def eliminate_symbol(self, symbol: Symbol) -> BooleanIndexValue:
        assert not symbol.is_bound
        left_operand = self.left_operand.eliminate_symbol(symbol)
        right_operand = self.right_operand.eliminate_symbol(symbol)
        return left_operand | right_operand


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class LessThan(BooleanBinaryExpr):
    symbol = "<"
    operation: Callable[[int, int], bool] = lambda x, y: x < y


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class LessThanOrEqual(BooleanBinaryExpr):
    symbol = "<="
    operation: Callable[[int, int], bool] = lambda x, y: x <= y


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class GreaterThan(BooleanBinaryExpr):
    symbol = ">"
    operation: Callable[[int, int], bool] = lambda x, y: x > y


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class GreaterThanOrEqual(BooleanBinaryExpr):
    symbol = ">="
    operation: Callable[[int, int], bool] = lambda x, y: x >= y


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class Equal(BooleanBinaryExpr):
    symbol = "="
    operation: Callable[[int, int], bool] = lambda x, y: x == y

    def _codegen_str(self, index_values: Mapping[Symbol, int]) -> str:
        l = self.left_operand._codegen_str(index_values)
        r = self.right_operand._codegen_str(index_values)
        return f"({l}=={r})"

    def struct_eq(self, other: Any) -> bool:
        # NOTE: since most operations are reflexive, we need to compare both.
        if not isinstance(other, Equal):
            try:
                other = lift_to_bool_ie(other)
            except Exception:
                return False
        if not isinstance(other, Equal):
            return False
        if type(self) is type(other) and len(self.children) == len(other.children):
            return (
                self.left_operand.struct_eq(other.left_operand)
                and self.right_operand.struct_eq(other.right_operand)
            ) or (
                self.left_operand.struct_eq(other.right_operand)
                and self.right_operand.struct_eq(other.left_operand)
            )
        return False


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class IntUnaryExpr(IntIndexValue, ABC):
    operand: IntIndexValue  # TODO because of ceil and floor, operand should be also float

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.operand))

    def __post_init__(self) -> None:
        self.children.append(self.operand)
        # self._cached_sympy_expr()

    def as_upper_bound_access(self) -> IntIndexValue:
        return self.__class__(self.operand.as_upper_bound_access())  # type: ignore

    def as_lower_bound_access(self) -> IntIndexValue:
        return self.__class__(self.operand.as_lower_bound_access())  # type: ignore

    def simplify_mins_and_maxes(self, aggressive: bool = False) -> IntIndexValue:
        return self.__class__(
            self.operand.simplify_mins_and_maxes(aggressive),  # type: ignore
        )

    def remap(self, domain_map: Mapping[Symbol, IntIndexValue]) -> IntIndexValue:
        return self.__class__(
            self.operand.remap(domain_map),  # type: ignore
        )

    def drop_modulos(self) -> IndexExpr:
        return self.__class__(self.operand.drop_modulos())  # type: ignore


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class Neg(IntUnaryExpr):
    def __str__(self) -> str:
        return f"(-{self.operand})"

    def _codegen_str(self, index_values: Mapping[Symbol, int]) -> str:
        return f"(-{self.operand._codegen_str(index_values)})"

    __repr__ = __str__

    def evaluate(self, index_values: Mapping[Symbol, int]) -> int:
        operand = self.operand.evaluate(index_values)
        assert isinstance(operand, int), "Operand must evaluate to an integer in negation"
        return -operand

    def partial_eval(self, domain_map: Mapping[Symbol, int]) -> IntIndexValue:
        o = self.operand.partial_eval(domain_map)
        if isinstance(o, (ConstInt, ConstFloat)):
            return lift_to_int_ie(-o.const)

        return self


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class Ceil(IntUnaryExpr):
    def __str__(self) -> str:
        return f"ceil({self.operand})"

    __repr__ = __str__

    def _codegen_str(self, index_values: Mapping[Symbol, int]) -> str:
        return f"math.ceil({self.operand._codegen_str(index_values)})"

    def evaluate(self, index_values: Mapping[Symbol, int]) -> int:
        operand = self.operand.evaluate(index_values)
        return math.ceil(operand)

    def partial_eval(self, domain_map: Mapping[Symbol, int]) -> IntIndexValue:
        o = self.operand.partial_eval(domain_map)
        if isinstance(o, (ConstInt, ConstFloat)):
            return lift_to_int_ie(math.ceil(o.const))

        return self


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class Floor(IntUnaryExpr):
    def __str__(self) -> str:
        return f"floor({self.operand})"

    __repr__ = __str__

    def _codegen_str(self, index_values: Mapping[Symbol, int]) -> str:
        return f"math.floor({self.operand._codegen_str(index_values)})"

    def evaluate(self, index_values: Mapping[Symbol, int]) -> int:
        operand = self.operand.evaluate(index_values)
        return math.floor(operand)

    def partial_eval(self, domain_map: Mapping[Symbol, int]) -> IntIndexValue:
        o = self.operand.partial_eval(domain_map)
        if isinstance(o, (ConstInt, ConstFloat)):
            return lift_to_int_ie(math.floor(o.const))

        return self


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class BooleanUnaryExpr(BooleanIndexValue, ABC):
    operand: BooleanIndexValue

    def struct_eq(self, other: Any) -> bool:
        if not isinstance(other, BooleanUnaryExpr):
            try:
                other = lift_to_bool_ie(other)
            except Exception:
                return False
        if not isinstance(other, BooleanUnaryExpr):
            return False
        if type(self) is type(other) and len(self.children) == len(other.children):
            return self.operand.struct_eq(other.operand)

        return False

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.operand))

    def __post_init__(self) -> None:
        self.children.append(self.operand)
        # self._cached_sympy_expr()

    def simplify_mins_and_maxes(self, aggressive: bool = False) -> IntIndexValue:
        return self.__class__(
            self.operand.simplify_mins_and_maxes(aggressive),  # type: ignore
        )

    def remap(self, domain_map: Mapping[Symbol, IntIndexValue]) -> BooleanIndexValue:
        return self.__class__(
            self.operand.remap(domain_map),  # type: ignore
        )

    def drop_modulos(self) -> IndexExpr:
        return self.__class__(self.operand.drop_modulos())  # type: ignore


@dataclass(repr=False, frozen=True, eq=False, slots=True)
class Not(BooleanUnaryExpr):
    def __str__(self) -> str:
        return f"(not ({self.operand}))"

    def _codegen_str(self, index_values: Mapping[Symbol, int]) -> str:
        return f"(not ({self.operand._codegen_str(index_values)}))"

    __repr__ = __str__

    def evaluate(self, index_values: Mapping[Symbol, int]) -> bool:
        operand = self.operand.evaluate(index_values)
        assert isinstance(operand, bool), "Operand must evaluate to a bool in Not"
        return not operand

    def partial_eval(self, domain_map: Mapping[Symbol, int]) -> BooleanIndexValue:
        o = self.operand.partial_eval(domain_map)  # type: ignore
        if isinstance(o, (ConstBool)):
            return lift_to_bool_ie(not o.const)

        return self

    def eliminate_symbol(self, symbol: Symbol) -> BooleanIndexValue:
        assert not symbol.is_bound
        # TODO this seems problematic...
        elim = self.operand.eliminate_symbol(symbol)

        # If when eliminating, everything is eliminated, we return True, as the whole condition
        # stopped existing. We don't want to return const(false) as that would disable everything
        if elim.struct_eq(ConstBool(True)):
            return ConstBool(True)
        else:
            return Not(elim)


BooleanIndexValueLike = Union[bool, BooleanIndexValue]
IntIndexValueLike = Union[int, IntIndexValue]
IndexAtomLike = Union[IndexAtom, int, bool, slice]
IndexExprLike = Union[IndexExpr, int, bool, slice, tuple[IndexAtomLike, ...]]
ConstFloat = ConstInt
