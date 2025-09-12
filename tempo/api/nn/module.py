from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Optional, Protocol, Union

from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.core.domain import Domain, DomainLike
from tempo.core.dtype import DataType
from tempo.core.shape import ShapeLike


# type declaration of function initializer
class InitFn(Protocol):
    def __call__(
        self, *, shape: ShapeLike, dtype: DataType, domain: DomainLike
    ) -> RecurrentTensor: ...


# InitFn = Callable[[ShapeLike, DataType, DomainLike], RecurrentTensor]
MaybeInitFn = Optional[InitFn]
MaybeInitFnList = Optional[list[MaybeInitFn]]
MaybeInitFnOrList = Union[MaybeInitFn, MaybeInitFnList]


class Module(ABC):
    def __init__(
        self,
        domain: DomainLike = None,  # TODO: eventually, rename to update_domain
        independent_domain: DomainLike = None,
    ) -> None:
        self._reg_parameters: list[RecurrentTensor] = []
        self._parameters: list[RecurrentTensor] = []
        self._buffers: list[RecurrentTensor] = []

        if independent_domain is None:
            independent_domain = Domain.empty()

        self.indep_dom = Domain.from_(independent_domain, none_is_empty=True)
        self.update_dom = Domain.from_(domain, none_is_empty=True)
        intersection = Domain.intersect(self.indep_dom, self.update_dom)
        assert intersection == Domain.empty(), (
            f"{self.indep_dom=} {self.update_dom=} cannot intersect"
        )
        self.full_dom = Domain.union(self.update_dom, self.indep_dom)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @property
    def reg_parameters(self) -> list[RecurrentTensor]:
        params = [*self._reg_parameters]
        for module in self.__dict__.values():
            if isinstance(module, Module):
                params.extend(module.reg_parameters)
        return params

    @property
    def parameters(self) -> list[RecurrentTensor]:
        params = [*self._parameters]
        for module in self.__dict__.values():
            if isinstance(module, Module):
                params.extend(module.parameters)
        return params

    @property
    def buffers(self) -> list[RecurrentTensor]:
        buffers = [*self._buffers]
        for module in self.__dict__.values():
            if isinstance(module, Module):
                buffers.extend(module.buffers)
        return buffers

    def param_from_init(
        self, initial_value: RecurrentTensor, regularize: bool = True
    ) -> RecurrentTensor:
        shape, dtype = initial_value.shape, initial_value.dtype
        p = RecurrentTensor.placeholder(
            shape,
            dtype,
            requires_grad=True,
            domain=self.full_dom,
        )
        p[(*self.update_dom.lbounds, *self.indep_dom.variables)] = initial_value
        self._parameters.append(p)
        if regularize:
            self._reg_parameters.append(p)
        return p

    def bias_from_init(self, initial_value: RecurrentTensor) -> RecurrentTensor:
        return self.param_from_init(initial_value, regularize=False)

    def buffer_from_init(self, initial_value: RecurrentTensor) -> RecurrentTensor:
        shape, dtype = initial_value.shape, initial_value.dtype
        b = RecurrentTensor.placeholder(
            shape,
            dtype,
            requires_grad=True,
            domain=self.full_dom,
        )
        b[(*self.update_dom.lbounds, *self.indep_dom.variables)] = initial_value
        self._buffers.append(b)
        return b

    def fixed(self) -> None:
        for p in self.parameters:
            p[True] = p.previous
        for b in self.buffers:
            b[True] = b.previous
            # p[self.domain.variables] = p[
            #    (*self.accum_dom.lex_prev_expr.members, *self.indep_dom.variables)
            # ]
            # p[self.domain.variables] = p[
            #    (*self.accum_dom.lex_prev_expr.members, *self.indep_dom.variables)
            # ]

    def state_dict(self) -> Mapping[str, Any]:
        # clz_name = self.__class__.__name__

        ret: dict[str, Any] = {}
        for i, param in enumerate(self._parameters):
            full_name = f"param_{i}"
            ret[full_name] = param
        for i, buffer in enumerate(self._buffers):
            full_name = f"buffer_{i}"
            ret[full_name] = buffer

        # Iterate over members of the class
        for name, member in self.__dict__.items():
            if isinstance(member, Module):
                full_name = f"{name}"
                if full_name in ret:
                    raise ValueError(f"Duplicate state_dict key: {full_name}")
                ret[full_name] = member.state_dict()
        return ret

    def flat_state_dict(self) -> Mapping[str, Any]:
        import optree

        paths, leaves, _ = optree.tree_flatten_with_path(self.state_dict())
        flat_dict = {
            ".".join(str(k) for k in path): leaf for path, leaf in zip(paths, leaves, strict=True)
        }
        return flat_dict

    # TODO this won't work exactly. What we probably want is to override the parameters and buffers
    # conditions with the ones in the state_dict
    # def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:

    #    clz_name = self.__class__.__name__

    #    param_keys = [k for k in state_dict.keys() if f"{clz_name}.param_" in k]
    #    buffer_keys = [k for k in state_dict.keys() if f"{clz_name}.buffer_" in k]

    #    for i, key in enumerate(param_keys):

    #    for name, member in self.__dict__.items():
    #        member_name = "".join(name.split(".")[1:])
    #        if isinstance(member, RecurrentTensor):
    #            self.__setattr__(member_name, member)
    #        elif isinstance(member, Mapping):
    #            self.__getattribute__(member_name).load_state_dict(state_dict[name])
    #        else:
    #            raise ValueError(f"Unknown member type {type(member)}")

    def __getitem__(self, key: Any) -> Module:
        # return a copy of the module with parameters and buffers indexed by key
        cpy = copy.deepcopy(self)

        for p in cpy.parameters:
            p._underlying = p.copy_with_no_symbolic_index()[key]._underlying
        for b in cpy.buffers:
            b._underlying = b.copy_with_no_symbolic_index()[key]._underlying

        return cpy

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


class Sequential(Module):
    def __init__(
        self,
        *modules: Module,
        domain: DomainLike = None,
        independent_domain: DomainLike = None,
    ) -> None:
        super().__init__(domain, independent_domain)

        self.modules = list(modules)

    def forward(self, x: RecurrentTensor) -> Any:
        for module in self.modules:
            x = module.forward(x)
        return x

    @property
    def parameters(self) -> list[RecurrentTensor]:
        params = []
        for module in self.modules:
            params.extend(module.parameters)
        return params

    @property
    def reg_parameters(self) -> list[RecurrentTensor]:
        params = []
        for module in self.modules:
            params.extend(module.reg_parameters)
        return params

    @property
    def buffers(self) -> list[RecurrentTensor]:
        buffers = []
        for module in self.modules:
            buffers.extend(module.buffers)
        return buffers

    # def state_dict(self) -> Mapping[str, Any]:
    #    clz_name = self.__class__.__name__

    #    ret = {}
    #    for i, module in enumerate(self.modules):
    #        full_name = f"{clz_name}.module_{i}"
    #        ret[full_name] = module.state_dict()
    #    return ret
