from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

from tempo.api.nn import utils
from tempo.api.recurrent_tensor import MaybeRecurrentTensor, RecurrentTensor
from tempo.core.domain import Domain, DomainLike


class Optimizer:
    def __init__(
        self,
        params: list[RecurrentTensor],
        buffers: list[RecurrentTensor],
        lr: MaybeRecurrentTensor,
        max_norm: MaybeRecurrentTensor | None = None,
        norm_type: MaybeRecurrentTensor = 2.0,
        independent_domain: DomainLike = None,
    ):
        self.max_norm, self.norm_type = max_norm, norm_type
        assert len(params) != 0, "optimizer must have at least one param"
        self.domain = params[0].domain

        # TODO Do they have to all have the same domain though?
        for x in params:
            if x.domain != self.domain:
                raise ValueError("All params must have the same domain")
            if x.requires_grad is None:
                x.requires_grad = True

        if independent_domain is None:
            independent_domain = Domain.empty()

        self.indep_dom = Domain.from_(independent_domain)
        self.accum_dom = Domain.difference(self.domain, self.indep_dom)

        self._init_expr = (*self.accum_dom.lb_expr.members, *self.indep_dom.variables)
        self._prev_expr = (
            *self.accum_dom.lex_prev_expr.members,
            *self.indep_dom.variables,
        )

        # TODO shouldn't this filter out params that don't require grad?
        self.params: list[RecurrentTensor] = list(set(params))
        self.buffers: list[RecurrentTensor] = list(set(buffers))
        self.lr = RecurrentTensor.lift(lr)

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError

    # @abstractmethod
    # def step_when(self, when: ie.BooleanIndexValueLike = True) -> None:
    #    raise NotImplementedError


# TODO implement SGD in terms of LARS
class SGD(Optimizer):
    def __init__(
        self,
        params: list[RecurrentTensor],
        buffers: list[RecurrentTensor],
        lr: MaybeRecurrentTensor = 0.001,
        momentum: MaybeRecurrentTensor = 0.0,
        weight_decay: MaybeRecurrentTensor = 0.0,
        nesterov: bool = False,
        max_norm: MaybeRecurrentTensor | None = None,
        norm_type: MaybeRecurrentTensor = 2.0,
        independent_domain: DomainLike = None,
    ):
        super().__init__(
            params, buffers, lr, max_norm, norm_type, independent_domain=independent_domain
        )
        self.momentum, self.wd, self.nesterov = momentum, weight_decay, nesterov

    # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    def step(self) -> None:
        grads: Sequence[RecurrentTensor] = [p.grad for p in self.params]  # type: ignore

        if self.max_norm:
            grads = utils.clip_norm(grads, self.max_norm, self.norm_type)[0]

        for p, g in zip(self.params, grads, strict=False):
            if self.wd != 0.0:
                g = g + self.wd * p[self._prev_expr]
                # g = g + self.wd * p
            if self.momentum != 0.0:
                p_m = RecurrentTensor.placeholder_like(p)
                p_m[self._init_expr] = RecurrentTensor.zeros(p.shape, p.dtype)
                p_m[True] = self.momentum * p_m[self._prev_expr] + g[self._prev_expr]
                g = (g + self.momentum * p_m) if self.nesterov else p_m
            p[True] = p[self._prev_expr].detach() - g[self._prev_expr].detach() * self.lr

    # def step_when(self, cond: ie.BooleanIndexValueLike = True) -> None:
    #    # TODO we will have to write an accumulator for the gradients
    #    # TODO this accumulator will need to be liftable by the lifter
    #    # TODO
    #    grads: Sequence[RecurrentTensor] = [p.grad for p in self.params]  # type: ignore
    #    grads = []
    #    for p in self.params:
    #        g = p.grad
    #        assert g is not None
    #        accum_g = RecurrentTensor.placeholder_like(g)
    #        accum_g[self._init_expr] = g
    #        # TODO could simplify with above by not putting this cond below in (and lift ofc)
    #        accum_g[cond] = g
    #        accum_g[True] = accum_g[self._prev_expr] + g[self._prev_expr]

    #        grads.append(accum_g)

    #    if self.max_norm:
    #        grads = utils.clip_norm(grads, self.max_norm, self.norm_type)[0]

    #    for p, g in zip(self.params, grads):
    #        if self.wd != 0.0:
    #            g = g + self.wd * p
    #        if self.momentum != 0.0:
    #            p_m = RecurrentTensor.placeholder_like(p)
    #            p_m[self._init_expr] = RecurrentTensor.zeros(p.shape, p.dtype)
    #            p_m[cond] = self.momentum * p_m[self._prev_expr] + g[self._prev_expr]
    #            p_m[True] = p_m[self._prev_expr]
    #            g = (g + self.momentum * p_m) if self.nesterov else p_m
    #        p[cond] = p[self._prev_expr].detach() - g[self._prev_expr].detach() * self.lr
    #        # TODO if a "True" condition is already present, do not register a new one
    #        p[True] = p[self._prev_expr]


# LAMB is essentially just the trust ratio part of LARS applied to Adam/W so if we just set
# the trust ratio to 1.0 its just Adam/W.
def AdamW(  # noqa: N802
    params: list[RecurrentTensor],
    buffers: list[RecurrentTensor],
    lr: MaybeRecurrentTensor = 0.001,
    b1: MaybeRecurrentTensor = 0.9,
    b2: MaybeRecurrentTensor = 0.999,
    eps: MaybeRecurrentTensor = 1e-8,
    wd: MaybeRecurrentTensor = 0.01,
    max_norm: MaybeRecurrentTensor | None = None,
    norm_type: MaybeRecurrentTensor = 2.0,
    independent_domain: DomainLike = None,
) -> LAMB:
    return LAMB(
        params,
        buffers,
        lr,
        b1,
        b2,
        eps,
        wd,
        adam=True,
        max_norm=max_norm,
        norm_type=norm_type,
        independent_domain=independent_domain,
    )


def Adam(  # noqa: N802
    params: list[RecurrentTensor],
    buffers: list[RecurrentTensor],
    lr: MaybeRecurrentTensor = 0.001,
    b1: MaybeRecurrentTensor = 0.9,
    b2: MaybeRecurrentTensor = 0.999,
    eps: MaybeRecurrentTensor = 1e-8,
    max_norm: MaybeRecurrentTensor | None = None,
    norm_type: MaybeRecurrentTensor = 2.0,
    independent_domain: DomainLike = None,
) -> LAMB:
    return LAMB(
        params,
        buffers,
        lr,
        b1,
        b2,
        eps,
        0.0,
        adam=True,
        max_norm=max_norm,
        norm_type=norm_type,
        independent_domain=independent_domain,
    )


class LAMB(Optimizer):
    def __init__(
        self,
        params: list[RecurrentTensor],
        buffers: list[RecurrentTensor],
        lr: MaybeRecurrentTensor = 0.001,
        b1: MaybeRecurrentTensor = 0.9,
        b2: MaybeRecurrentTensor = 0.999,
        eps: MaybeRecurrentTensor = 1e-6,
        wd: MaybeRecurrentTensor = 0.0,
        adam: bool = False,
        max_norm: MaybeRecurrentTensor | None = None,
        norm_type: MaybeRecurrentTensor = 2.0,
        independent_domain: DomainLike = None,
    ):
        super().__init__(
            params, buffers, lr, max_norm, norm_type, independent_domain=independent_domain
        )
        self.b1, self.b2, self.eps, self.wd = b1, b2, eps, wd

        self.adam = adam

    def step(self) -> None:
        steps = RecurrentTensor.lift(self.accum_dom.linearized_count_expr) + 1

        grads: Sequence[RecurrentTensor] = [p.grad for p in self.params]  # type: ignore

        if self.max_norm:
            grads, total_norm = utils.clip_norm(grads, self.max_norm, self.norm_type)

        for p, g in zip(self.params, grads, strict=False):
            if self.wd != 0.0:
                g = g + self.wd * p.previous.detach()

            # How to update m
            m = RecurrentTensor.placeholder_like(p)
            m[self._init_expr] = (1.0 - self.b1) * g
            m[True] = self.b1 * m[self._prev_expr] + (1.0 - self.b1) * g

            # How to update v
            v = RecurrentTensor.placeholder_like(p)
            v[self._init_expr] = (1.0 - self.b2) * (g**2)
            v[True] = self.b2 * v[self._prev_expr] + (1.0 - self.b2) * (g**2)

            m_hat = m / (1.0 - self.b1**steps)
            v_hat = v / (1.0 - self.b2**steps)

            up = m_hat / (v_hat.sqrt() + self.eps)
            assert isinstance(up, RecurrentTensor)

            if not self.adam:
                r1 = p.detach().squared().sum().sqrt()
                r2 = up.squared().sum().sqrt()
                r = RecurrentTensor.where(r1 > 0, RecurrentTensor.where(r2 > 0, r1 / r2, 1.0), 1.0)
            else:
                r = RecurrentTensor.lift(1.0)
            p[True] = p[self._prev_expr].detach() - up[self._prev_expr] * r * self.lr
