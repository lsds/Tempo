from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from tempo.core import index_expr as ie
from tempo.utils import logger

log = logger.get_logger(__name__)


@dataclass
class Domain:
    """A domain in Tempo essentially functions as a space bounded by the upper bounds."""

    variables: Sequence[ie.Symbol]
    _ubounds: Sequence[ie.Symbol]
    _ubound_overrides: Dict[ie.Symbol, ie.IntIndexValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.basis_expr = ie.IndexSequence(tuple(self.variables))
        self.full_range_expr = ie.IndexSequence(
            tuple(ie.slice_(ie.ConstInt(0), v.as_bound()) for v in self.variables)
        )

    def __iter__(self) -> Iterator[ie.Symbol]:
        return iter(self.variables)

    def __str__(self) -> str:
        return f"Domain({self.variables},{self.ubounds})"

    def __hash__(self) -> int:
        return hash((tuple(self.variables), tuple(self.ubounds)))

    __repr__ = __str__

    @property
    def ubounds(self) -> Sequence[ie.IntIndexValue]:
        return [self.get_ubound_override(b, b) for b in self._ubounds]

    @property
    def parameters(self) -> Sequence[ie.Symbol]:
        return self._ubounds

    def get_ubound_override(self, var: ie.Symbol, default: ie.Symbol) -> ie.IntIndexValue:
        with ie.StructEqCheckCtxManager():
            return self._ubound_overrides.get(var, default)

    def set_ubound_override(self, bound_symbol: ie.Symbol, val: ie.IntIndexValue) -> None:
        with ie.StructEqCheckCtxManager():
            self._ubound_overrides[bound_symbol] = val

    def del_ubound_override(self, bound_symbol: ie.Symbol) -> None:
        with ie.StructEqCheckCtxManager():
            if bound_symbol in self._ubound_overrides:
                del self._ubound_overrides[bound_symbol]

    def ubound_overrides_empty(self) -> bool:
        with ie.StructEqCheckCtxManager():
            return len(self._ubound_overrides) == 0

    def prepend_dim(self, var: ie.Symbol, ubound: ie.Symbol) -> Domain:
        return Domain(
            [var, *self.variables],
            [ubound, *self._ubounds],
            _ubound_overrides=self._ubound_overrides,
        )

    def append_dim(self, var: ie.Symbol, ubound: ie.Symbol) -> Domain:
        return Domain(
            [*self.variables, var],
            [*self._ubounds, ubound],
            _ubound_overrides=self._ubound_overrides,
        )

    def remove_dim(self, dim: ie.Symbol) -> Domain:
        with ie.StructEqCheckCtxManager():
            idx = self.find_variable_index(dim)
            new_ubound_overrides = {
                k: v for k, v in self._ubound_overrides.items() if not k.struct_eq(dim.as_bound())
            }
            return Domain(
                [v for i, v in enumerate(self.variables) if i != idx],
                [b for i, b in enumerate(self._ubounds) if i != idx],
                new_ubound_overrides,
            )

    def copy(self) -> Domain:
        return copy.deepcopy(self)

    def __getitem__(self, index: int) -> ie.Symbol:
        # Returns the variable at the index when ignoring disabled dimensions
        return self.variables[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Domain):
            return False

        return (
            len(self.variables) == len(other.variables)
            and all(a.struct_eq(b) for a, b in zip(self.variables, other.variables, strict=False))
            and all(a.struct_eq(b) for a, b in zip(self._ubounds, other._ubounds, strict=False))
            and all(a.struct_eq(b) for a, b in zip(self.ubounds, other.ubounds, strict=False))
            and self._ubound_overrides == other._ubound_overrides
        )

    def __ne__(self, value: object) -> bool:
        return not self.__eq__(value)

    @staticmethod
    def from_vars_and_bounds(
        variables: Sequence[ie.Symbol],
        bounds: Sequence[ie.Symbol],
        overrides: Optional[Dict[ie.Symbol, ie.IntIndexValue]] = None,
    ) -> Domain:
        return Domain([*variables], [*bounds], overrides or {})

    @staticmethod
    def union(*domains: DomainLike) -> Domain:
        domains_ = [Domain.from_(d) for d in domains]
        # assert all(d.condition.equivalent(ie.ConstBool(True)) for d in domains_)

        all_variables = {v for d in domains_ for v in d.variables}

        result = Domain.from_vars(all_variables)

        ubound_overrides = Domain._merge_ubound_overrides(domains_, result)
        result._ubound_overrides = ubound_overrides
        # log.debug("Unioning domains %s resulted in %s", domains_, result)
        return result

    @staticmethod
    def intersect(*domains: DomainLike) -> Domain:
        domains_ = [Domain.from_(d) for d in domains]
        # assert all(d.condition.equivalent(ie.ConstBool(True)) for d in domains_)

        all_variables = {v for d in domains_ for v in d.variables}

        variables_in_all = []
        for v in all_variables:
            shared_by_all = True
            for d in domains_:
                shared_by_all &= d.has_dim(v)
            if shared_by_all:
                variables_in_all.append(v)

        result = Domain.from_vars(variables_in_all)
        ubound_overrides = Domain._merge_ubound_overrides(domains_, result)
        result._ubound_overrides = ubound_overrides
        # log.debug("Intersecting domains %s resulted in %s", domains_, result)
        return result

    @staticmethod
    def _merge_ubound_overrides(  # noqa: C901
        domains_: Sequence[Domain], result: Domain
    ) -> Dict[ie.Symbol, ie.IntIndexValue]:
        with ie.StructEqCheckCtxManager():
            ubound_overrides = {}
            for d in domains_:
                for k, v in d._ubound_overrides.items():
                    if not result.has_dim(k.as_var()):
                        continue
                    if k not in ubound_overrides:
                        ubound_overrides[k] = v
                    else:
                        if v.struct_eq(ubound_overrides[k]):
                            continue
                        else:
                            raise ValueError(
                                f"Conflicting ubound overrides {v=} {ubound_overrides[k]}"
                            )

            # ubound_overrides = {}
            return ubound_overrides

    @staticmethod
    def difference(a: DomainLike, b: DomainLike) -> Domain:
        a = Domain.from_(a)
        b = Domain.from_(b)

        # assert a.condition.equivalent(ie.ConstBool(True))
        # assert b.condition.equivalent(ie.ConstBool(True))

        # if not (a.is_contained_in(b) or b.is_contained_in(a)):
        #    raise ValueError("Domains are not comparable")
        # assert b.is_contained_in(a)

        variables = [v for v in a.variables if not b.has_dim(v)]
        result = Domain.from_vars(variables)
        ubound_overrides = Domain._merge_ubound_overrides([a, b], result)
        result._ubound_overrides = ubound_overrides
        return result

    def has_dim(self, dim: ie.Symbol) -> bool:
        try:
            self.find_variable_index(dim)
            return True
        except ValueError:
            return False

    __contains__ = has_dim

    def indexed_real_domain(self, index: ie.IndexSequence) -> Domain:
        with ie.StructEqCheckCtxManager():
            assert len(index.members) == self.num_dimensions

            variables_involved = index.vars_used()

            new_dom = Domain.from_vars(
                variables_involved,
            )
            overrides = Domain._merge_ubound_overrides([self], new_dom)

            new_dom._ubound_overrides = overrides
        return new_dom

    def is_contained_in(self, other: Domain) -> bool:
        for v in self.variables:
            if not other.has_dim(v):
                return False

        ## TODO: needs improvement
        # if len(self._ubound_overrides) > len(other._ubound_overrides):
        #    return False
        return True

    def get_corresponding_lbound(self, var: ie.Symbol) -> ie.IntIndexValue:
        return ie.ConstInt(0)

    def get_corresponding_ubound(self, var: ie.Symbol) -> ie.IntIndexValue:
        idx = self.find_variable_index(var)
        ubound = self._ubounds[idx]
        overriden = self.get_ubound_override(ubound, ubound)
        return overriden

    def find_variable_index(self, var: ie.Symbol) -> int:
        idx = -1
        for i, v in enumerate(self.variables):
            if v.struct_eq(var):
                idx = i
                break
        if idx == -1:
            raise ValueError(f"Variable not in domain, {var=} {self.variables=}")
        return idx

    def __len__(self) -> int:
        return self.num_dimensions

    @property
    def num_dimensions(self) -> int:
        return len(self.variables)

    @property
    def paired_vars_and_upper_bounds(self) -> Sequence[Tuple[ie.Symbol, ie.Symbol]]:
        return list(zip(self.variables, self._ubounds, strict=False))

    @property
    def lbounds(self) -> Sequence[int]:
        return [0 for _ in self.variables]

    @property
    def linearized_count_expr(self) -> ie.IntIndexValue:
        state: ie.IntIndexValue = ie.ConstInt(1)
        count: ie.IntIndexValue = ie.ConstInt(0)
        for v, b in reversed(list(zip(self.variables, self.ubounds, strict=False))):
            count += v * state
            state = state * b
        return count

    @property
    def full_index_expr(self) -> ie.IndexSequence:
        return ie.lift_to_ie_seq(
            tuple(slice(lb, ub) for lb, ub in zip(self.lbounds, self.ubounds, strict=False))
        )

    @property
    def lb_expr(self) -> ie.IndexSequence:
        return ie.lift_to_ie_seq(tuple(self.lbounds))

    @property
    def lex_prev_expr(self) -> ie.IndexSequence:
        vars_ = self.variables

        if len(vars_) == 1:
            return ie.IndexSequence((vars_[0] - 1,))

        prev_indices: List[ie.IntIndexValue] = [*self.variables]
        bounds = self.ubounds

        still_needs_dec: ie.BooleanIndexValue = ie.ConstBool(True)

        for i in range(len(prev_indices) - 1, -1, -1):
            v = vars_[i]
            V = bounds[i]

            if i != 0:
                prev_indices[i] = ie.piecewise(
                    (
                        ((v > 0) & still_needs_dec, v - 1),  # type: ignore
                        ((v == 0) & still_needs_dec, V - 1),  # type: ignore
                        (
                            ~(  # type: ignore
                                ((v > 0) & still_needs_dec)  # type: ignore
                                | ((v == 0) & still_needs_dec)  # type: ignore
                            ),
                            v,
                        ),
                    )
                )

                still_needs_dec = still_needs_dec & (v == 0)  # type: ignore
            else:
                prev_indices[i] = ie.piecewise(
                    (
                        ((v > 0) & still_needs_dec, v - 1),  # type: ignore
                        (~still_needs_dec, v),  # type: ignore
                        # (~((v > 0) & still_needs_dec), v),
                    )
                )

        new_one = ie.IndexSequence(tuple(prev_indices))
        return new_one

    @property
    def lex_next_expr(self) -> ie.IndexSequence:
        vars_ = self.variables
        next_indices: List[ie.IntIndexValue] = [*self.variables]
        bounds = self.ubounds

        still_needs_inc: ie.BooleanIndexValue = ie.ConstBool(True)

        for i in range(len(next_indices) - 1, -1, -1):
            v = vars_[i]
            V = bounds[i]

            lex_next_val = ie.piecewise(
                (
                    (
                        (v < V - 1) & still_needs_inc,  # type: ignore
                        v + 1,  # type: ignore
                    ),  # Increment if below upper bound
                    (
                        (v == V - 1) & still_needs_inc,  # type: ignore
                        ie.ConstInt(0),
                    ),  # Reset to 0 if at upper bound
                    (ie.ConstBool(True), v),  # Otherwise, keep the same value
                )
            )
            next_indices[i] = lex_next_val

            still_needs_inc = still_needs_inc & (v == V - 1)  # type: ignore

        return ie.IndexSequence(tuple(next_indices))

    def is_empty(self) -> bool:
        return self.num_dimensions == 0

    def select_vars(self, *variables: ie.Symbol) -> Domain:
        with ie.StructEqCheckCtxManager():
            # assert self.condition.equivalent(ie.ConstBool(True))

            idxs = list({self.find_variable_index(v) for v in variables})

            pairs = [(self.variables[i], i) for i in idxs]
            sorted_pairs = sorted(pairs, key=lambda p: p[1])
            vars_sorted = [p[0] for p in sorted_pairs]

            universe = Domain.universe()
            bounds = [universe._ubounds[universe.find_variable_index(v)] for v in vars_sorted]
            overrides = {
                v: self._ubound_overrides[v] for v in bounds if v in self._ubound_overrides
            }
            return Domain(vars_sorted, bounds, overrides)

    @staticmethod
    def empty() -> Domain:
        return Domain([], [])

    # TODO remove probably
    @staticmethod
    def universe() -> Domain:
        from tempo.core.global_objects import get_active_dg

        return get_active_dg().universe  # type: ignore

    @staticmethod
    def from_vars(
        variables: Iterable[ie.Symbol],
        overrides: Optional[Dict[ie.Symbol, ie.IntIndexValue]] = None,
    ) -> Domain:
        universe = Domain.universe()
        dom = universe.select_vars(*variables)
        dom._ubound_overrides = overrides or {}

        return dom

    @staticmethod
    def from_(domain_like: DomainLike, none_is_empty: bool = False) -> Domain:
        """Convert a domain-like object to a Domain. None is treated as the active domain,
        unless none_is_empty is True, in which case it is treated as the empty domain.

        Args:
            domain_like (DomainLike): The domain-like object to convert.
            none_is_empty (bool, optional): If True, None is treated as the empty domain.
            Defaults to False, in which case None is treated as the active domain.

        Raises:
            ValueError: If the domain-like object cannot be converted to a Domain.

        Returns:
            Domain: The converted Domain.
        """
        if domain_like is None:
            from tempo.core.global_objects import get_active_domain_or_empty

            return Domain.empty() if none_is_empty else get_active_domain_or_empty()

        if isinstance(domain_like, ie.Symbol):
            domain_like = [domain_like]

        if isinstance(domain_like, Domain):
            return domain_like
        elif isinstance(domain_like, Iterable):
            return Domain.from_vars(domain_like)

        raise ValueError(f"Cannot convert {domain_like} to a Domain")


DomainLike = Union[None, Domain, Iterable[ie.Symbol]]
