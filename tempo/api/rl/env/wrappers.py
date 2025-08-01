from __future__ import annotations

import math
from typing import Any, Callable, Dict, Generic, Tuple, Union

import gymnasium as gym

from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import BackendTensorT, BackendTensorTPyTree
from tempo.core.dtype import dtypes

# mypy: ignore-errors


# class TempoEnvWrapper(gym.vector.VectorWrapper, Generic[BackendTensorT]):
class TempoEnvWrapper(gym.Wrapper, Generic[BackendTensorT]):
    def __init__(self, env: RuntimeEnv, exec_cfg: ExecutionConfig):
        super().__init__(env)
        self.env = env
        self.exec_cfg = exec_cfg

        self._closed = False  # Default to False in the wrapper

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.env, name)

    @property
    def closed(self) -> bool:
        if hasattr(self.env, "closed"):
            return self.env.closed  # type: ignore
        return self._closed

    def close(self) -> None:
        # Call the environment's close method if available, then set the wrapper's _closed state
        if hasattr(self.env, "close") and callable(self.env.close):
            self.env.close()
        self._closed = True

    def reset(self, **kwargs: Any) -> Tuple[BackendTensorT, BackendTensorTPyTree]:
        raise NotImplementedError

    def step(
        self, action: BackendTensorT
    ) -> Tuple[
        BackendTensorT,
        BackendTensorT,
        BackendTensorT,
        BackendTensorT,
        BackendTensorTPyTree,
    ]:
        raise NotImplementedError


RuntimeEnv = Union[gym.Env, TempoEnvWrapper]


class BoxToDiscreteWrapper(TempoEnvWrapper):
    def __init__(self, env: RuntimeEnv, exec_cfg: ExecutionConfig, dof: int, low: int, high: int):
        super().__init__(env, exec_cfg)

        self.low = low
        self.high = high
        self.dof = dof

        from tempo.runtime.backends.backend import DLBackend

        self.backend = DLBackend.get_backend(exec_cfg.backend)

        assert isinstance(env.action_space, gym.spaces.Box)
        if isinstance(env, gym.vector.VectorEnv) or hasattr(env, "num_envs"):
            bs = env.num_envs  # type: ignore

            self._single_box_shape: Tuple[int, ...] = env.single_action_space.shape  # type: ignore
            self._full_box_shape: Tuple[int, ...] = (bs,) + self._single_box_shape
            self._single_box_shape_prod = math.prod(self._single_box_shape)
            self.single_action_space = gym.spaces.Discrete(
                dof ** (self._single_box_shape_prod // bs)
            )
            self.stack_axis = 1
        else:
            self._single_box_shape: Tuple[int, ...] = env.action_space.shape  # type: ignore
            self._full_box_shape: Tuple[int, ...] = self._single_box_shape
            self._single_box_shape_prod = math.prod(self._single_box_shape)
            self.single_action_space = self.action_space
            self.stack_axis = 0

        self.action_space = gym.spaces.Discrete(dof**self._single_box_shape_prod)
        self.backend_float32 = self.backend.to_backend_datatype(dtypes.float32)

    def rescale(self, action: BackendTensorT) -> BackendTensorT:
        return (action / (self.dof - 1) * (self.high - self.low)) + self.low  # type: ignore

    def reset(self, **kwargs: Dict[str, Any]) -> Tuple[BackendTensorT, BackendTensorTPyTree]:
        return self.env.reset(**kwargs)  # type: ignore

    def step(
        self, action: BackendTensorT
    ) -> Tuple[
        BackendTensorT,
        BackendTensorT,
        BackendTensorT,
        BackendTensorT,
        BackendTensorTPyTree,
    ]:
        div = self.backend.cast_backend_dtype(action, self.backend_float32)  # type: ignore
        real_action = []
        for _ in range(self._single_box_shape_prod):
            mod = div % self.dof  # type: ignore
            div = div // self.dof  # type: ignore
            real_action.append(self.rescale(mod))  # type: ignore

        # TODO broken as axis is not given
        real_action = self.backend.reshape(
            self.backend.stack(real_action, axis=self.stack_axis), self._full_box_shape
        )

        return self.env.step(real_action)  # type: ignore


class ToBackendTensorTWrapper(TempoEnvWrapper):
    def __init__(
        self,
        env: RuntimeEnv,
        exec_cfg: ExecutionConfig,
        to_backend_tensor: Callable[[Any], BackendTensorT],
        from_backend_tensor: Callable[[BackendTensorT], Any],
    ) -> None:
        super().__init__(env, exec_cfg)
        self.to_backend_tensor = to_backend_tensor
        self.from_backend_tensor = from_backend_tensor

    def reset(self, **kwargs: Dict[str, Any]) -> Tuple[BackendTensorT, BackendTensorTPyTree]:
        o, info = self.env.reset(**kwargs)
        o = self.to_backend_tensor(o)
        # info = optree.tree_map(self.to_backend_tensor, info)  # type: ignore
        info = {}
        return o, info

    def step(
        self, action: BackendTensorT
    ) -> Tuple[
        BackendTensorT,
        BackendTensorT,
        BackendTensorT,
        BackendTensorT,
        BackendTensorTPyTree,
    ]:
        action = self.from_backend_tensor(action)
        o, r, term, trunc, info = self.env.step(action)
        o = self.to_backend_tensor(o)
        r = self.to_backend_tensor(r)
        term = self.to_backend_tensor(term)
        trunc = self.to_backend_tensor(trunc)
        # info = optree.tree_map(self.to_backend_tensor, info)  # type: ignore
        info = {}
        return o, r, term, trunc, info


class DoneToTermTruncAPIConverterWrapper(TempoEnvWrapper):
    def __init__(
        self,
        env: RuntimeEnv,
        exec_cfg: ExecutionConfig,
    ) -> None:
        super().__init__(env, exec_cfg)

    def reset(self, **kwargs: Dict[str, Any]) -> Tuple[BackendTensorT, BackendTensorTPyTree]:
        o = self.env.reset(**kwargs)
        return o, {}

    def step(
        self, action: BackendTensorT
    ) -> Tuple[
        BackendTensorT,
        BackendTensorT,
        BackendTensorT,
        BackendTensorT,
        BackendTensorTPyTree,
    ]:
        o, r, done, info = self.env.step(action)  # type: ignore

        # trunc = info.get("TimeLimit.truncated", False * done)  # type: ignore

        return o, r, done, done, info  # type: ignore


class SinglePrecisionRewardWrapper(TempoEnvWrapper):
    def __init__(self, env: RuntimeEnv, exec_cfg: ExecutionConfig):
        super().__init__(env, exec_cfg)
        from tempo.runtime.backends.backend import DLBackend

        self.backend = DLBackend.get_backend(exec_cfg.backend)
        self.backend_dtype = self.backend.to_backend_datatype(dtypes.float32)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        o, r, term, trunc, info = self.env.step(action)
        r_ = self.backend.cast_backend_dtype(r, self.backend_dtype)
        return o, r_, term, trunc, info  # type: ignore


# class MergeTruncationsAndTerminations(gym.Wrapper):
#    def __init__(self, env: gym.vector.VectorEnv, dev):
#        super().__init__(env)
#        self.env = env
#        self.num_envs = env.num_envs
#        self.dev = dev
#
#    def reset(self, **kwargs):
#        obs, data = self.env.reset(**kwargs)
#        return obs, data
#
#    def step(self, action):
#        outs = self.env.step(action)
#        o, r, term, trunc, info = outs
#
#        d = torch.logical_or(torch.tensor(term), torch.tensor(trunc))
#
#        return o, r, d, info


#
class NoAutoResetWrapper(TempoEnvWrapper):
    def __init__(self, env: RuntimeEnv, exec_cfg: ExecutionConfig):
        super().__init__(env, exec_cfg)
        from tempo.runtime.backends.backend import DLBackend

        self.backend = DLBackend.get_backend(exec_cfg.backend)

        self.dev = self.backend.to_backend_device_obj(exec_cfg.dev)

        self.bool_dtype = self.backend.to_backend_datatype(dtypes.bool_)
        self.num_envs = env.num_envs if hasattr(env, "num_envs") else 1  # type: ignore

        self.done_envs_mask = self.backend.zeros_tensor(
            (self.num_envs,), dtype=self.bool_dtype, dev=self.dev
        )

    def reset(self):
        # self.done_envs_mask.fill_(0.0)
        self.done_envs_mask = self.backend.zeros_tensor(
            (self.num_envs,), dtype=self.bool_dtype, dev=self.dev
        )
        return self.env.reset()

    def step(self, action):
        o, r, term, trunc, info = self.env.step(action)
        masked_reward = (1 - self.done_envs_mask) * r

        self.done_envs_mask = self.done_envs_mask | (term | trunc)

        return o, masked_reward, self.done_envs_mask, info


class PermuteObservationChannelAxis(TempoEnvWrapper):
    def __init__(self, env: RuntimeEnv, exec_cfg: ExecutionConfig):
        super().__init__(env, exec_cfg)

        from tempo.runtime.backends.backend import DLBackend

        self.backend = DLBackend.get_backend(exec_cfg.backend)

        self.permutation = (0, 3, 2, 1)

    def reset(self):
        obs, info = self.env.reset()
        return self.backend.permute(obs, self.permutation), info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        return self.backend.permute(obs, self.permutation), rew, term, trunc, info


#
#
# class RemoveBatchDimWrapper(gym.Wrapper):
#    def __init__(self, env):
#        super().__init__(env)
#
#        self.observation_space = remove_batch_dim(self.env.observation_space)
#        self.action_space = remove_batch_dim(self.env.action_space)
#
#
# class PermuteObservationChannelAxis(gym.Wrapper):
#    def __init__(self, env):
#        super().__init__(env)
#
#    def reset(self):
#        obs = self.env.reset()
#        return obs.permute(0, 3, 2, 1)
#
#    def step(self, action):
#        obs, rew, done, info = self.env.step(action)
#        return obs.permute(0, 3, 2, 1), rew, done, info
