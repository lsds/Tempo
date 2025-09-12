from __future__ import annotations

from abc import ABC, abstractmethod

import optree

from tempo.api.nn.module import Module
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.api.rl.datatypes import (
    ActionsRecurrentTensor,
    ObservationsRecurrentTensor,
    ValueRecurrentTensor,
)
from tempo.core import index_expr as ie


class Network(Module, ABC):
    def __call__(
        self, obs: ObservationsRecurrentTensor | RecurrentTensor
    ) -> tuple[ActionsRecurrentTensor, ValueRecurrentTensor | None]:
        return self.forward(obs)  # type: ignore

    @abstractmethod
    def forward(
        self, obs: ObservationsRecurrentTensor | RecurrentTensor
    ) -> tuple[ActionsRecurrentTensor | None, ValueRecurrentTensor | None]:
        raise NotImplementedError

    @abstractmethod
    def qvalues(self) -> RecurrentTensor:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, actions: ActionsRecurrentTensor | RecurrentTensor) -> RecurrentTensor:
        raise NotImplementedError

    @abstractmethod
    def entropy(self) -> RecurrentTensor:
        raise NotImplementedError

    def polyak_update(  # noqa: C901
        self,
        target: Network,
        tau: float,
        update_var: ie.Symbol | None = None,
        update_freq: int = 1,
    ) -> None:
        if tau > 1 or tau < 0:
            raise ValueError("For polyak, tau must be in [0, 1]")

        self_state = self.state_dict()
        target_state = target.state_dict()

        assert optree.tree_structure(self_state) == optree.tree_structure(  # type: ignore
            target_state  # type: ignore
        )
        dom = self.full_dom

        # TODO eventually, let the optimizer find this. Maybe it already does, IDK.
        if tau == 1:

            def update(self_state: RecurrentTensor, target_state: RecurrentTensor) -> None:
                self_state.clear_placeholder_branches()
                self_state.init = target_state.init
                if update_var is not None:
                    self_state[update_var % update_freq == 0] = target_state
                else:
                    self_state[dom.variables] = target_state
                self_state[True] = self_state.previous

        else:

            def update(self_state: RecurrentTensor, target_state: RecurrentTensor) -> None:
                self_state.clear_placeholder_branches()
                self_state.init = target_state.init
                if update_var is not None:
                    self_state[update_var % update_freq == 0] = (
                        self_state.previous * (1 - tau) + target_state
                    )
                else:
                    self_state[dom.variables] = self_state.previous * (1 - tau) + target_state
                self_state[True] = self_state.previous

        optree.tree_map(
            update,
            self_state,  # type: ignore
            target_state,  # type: ignore
        )

    def copy_every(self, num_steps: int = 1, var: ie.Symbol | None = None) -> None:
        self.polyak_update(self, 1.0, update_var=var, update_freq=num_steps)

    # @staticmethod
    # def from_env(env:Env, domain: DomainLike = None, independent_domain: DomainLike = None):
    #    raise NotImplementedError
