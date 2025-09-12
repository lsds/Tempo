from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym

from tempo.api.recurrent_tensor import MaybeRecurrentTensor, RecurrentTensor, lift
from tempo.api.rl.datatypes import (
    ObservationsRecurrentTensor,
    RewardsRecurrentTensor,
    TerminationsRecurrentTensor,
    TruncationsRecurrentTensor,
)
from tempo.api.rl.env import udfs
from tempo.api.rl.env.env_desc import EnvDesc
from tempo.api.rl.env.env_registry.env_registry import EnvBuildCtx, EnvRegistry
from tempo.api.rl.env.wrappers import TempoEnvWrapper
from tempo.core.domain import DomainLike
from tempo.core.dtype import DataType, dtypes
from tempo.core.global_objects import get_active_exec_cfg
from tempo.core.shape import StaticShape
from tempo.core.symbolic_tensor import SymbolicTensor


@dataclass(frozen=True)
class Env:
    _env_desc: EnvDesc

    @staticmethod
    def make_env(
        name: str,
        num_envs: int | None = None,
        max_episode_steps: int | None = None,
        wrappers: Sequence[type[TempoEnvWrapper]] | None = None,
        **kwargs: Any,
    ) -> Env:
        """Creates and register a supported environment in the context env registry.
        The name passed in must be of the form ``<env_set>.<env_name>``. Where env_set
        is the identifier for the set of environments the environment belongs to (e.g. gym).

        Args:
            name (str): The .-separated name of the environment to create. E.g. gym.CartPole-v0
            wrappers (Optional[Sequence[Type[TempoEnvWrapper]]], optional):
                A list of wrapper classes to apply to the environment. Defaults to None.
            num_envs (Optional[int], optional): Number of envs to create. Defaults to None.

        Returns:
            Env:  A symbolic object that may be used as an env with step and reset functions.

        """

        exec_cfg = get_active_exec_cfg()

        # TODO we should not be importing backend here in the API...
        from tempo.core.dl_backend import DLBackend

        backend = DLBackend.get_backend(exec_cfg.backend)

        env_set, env_name = name.split(".", maxsplit=1)

        env_build_ctx = EnvBuildCtx(
            name=env_name,
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            kwargs=kwargs,
            exec_cfg=exec_cfg,
            backend=backend(),
        )

        desc_ = EnvRegistry.get_env_description(name, env_build_ctx, wrappers=wrappers)
        return Env(desc_)

    @property
    def observation_space(self) -> gym.Space:
        return self._env_desc.single_observation_space  # type:ignore

    @property
    def action_space(self) -> gym.Space:
        return self._env_desc.single_action_space  # type:ignore

    @property
    def observation_shape(self) -> StaticShape:
        return self._env_desc.observation_shape  # type:ignore

    @property
    def observation_dtype(self) -> DataType:
        return self._env_desc.observation_dtype  # type:ignore

    @property
    def action_shape(self) -> StaticShape:
        return self._env_desc.action_shape  # type:ignore

    @property
    def action_dtype(self) -> DataType:
        return self._env_desc.action_dtype  # type:ignore

    def reset(
        self,
        seed: MaybeRecurrentTensor | None = None,
        domain: DomainLike = None,
        # ) -> Tuple[ObservationsRecurrentTensor, Dict[str, InfoRecurrentTensor]]:
    ) -> ObservationsRecurrentTensor:
        if seed is not None:
            seed = lift(seed)
            assert dtypes.is_integer(seed.dtype), (
                f"Expect seed to have int dtype, found {seed.dtype}"
            )
        desc = udfs.get_reset_udf_desc(self._env_desc, seed is not None)
        return ObservationsRecurrentTensor(
            RecurrentTensor(
                SymbolicTensor.udf(
                    desc, [seed._underlying] if seed is not None else [], domain=domain
                )[0]
            )
        )

    def step(
        self,
        actions: MaybeRecurrentTensor,
        domain: DomainLike = None,
    ) -> tuple[
        ObservationsRecurrentTensor,
        RewardsRecurrentTensor,
        TerminationsRecurrentTensor,
        TruncationsRecurrentTensor,
    ]:
        actions = lift(actions)
        assert actions._underlying is not None, "Actions must be initialized tensor"
        assert actions.shape == self.action_shape, (
            f"Actions shape must match expected shape. \
          {actions.shape=}, {self.action_shape=}"
        )
        assert actions.dtype == self.action_dtype, (
            f"Actions must match expected dtype. \
            {actions.dtype=}, {self.action_dtype=}"
        )

        desc = udfs.get_step_udf_desc(self._env_desc)

        if domain is None:
            domain = actions.domain

        obs, rew, term, trunc = SymbolicTensor.udf(desc, [actions._underlying], domain=domain)

        assert obs.shape == self.observation_shape, "Observations must have correct shape"
        assert obs.dtype == self.observation_dtype, "Observations must have correct dtype"

        return (
            ObservationsRecurrentTensor(RecurrentTensor(obs)),
            RewardsRecurrentTensor(RecurrentTensor(rew)),
            TerminationsRecurrentTensor(RecurrentTensor(term)),
            TruncationsRecurrentTensor(RecurrentTensor(trunc)),
        )
