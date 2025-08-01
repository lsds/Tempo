from __future__ import annotations

from functools import partial
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, Union

import optree
import torch

from tempo.api.nn.module import MaybeInitFn
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import BackendTensorT
from tempo.core.device import device
from tempo.core.dtype import DataType, DataTypeLike, dtypes
from tempo.core.index_expr import Symbol
from tempo.core.shape import Shape
from tempo.runtime.backends.backend import DLBackend
from tempo.utils.logger import get_logger

logger = get_logger(__name__)

StateDictT = Dict[str, BackendTensorT]


def flatten_state_dict(state_dict: StateDictT) -> StateDictT:
    """Flatten a nested state dict into a flat dictionary with dot-separated keys."""
    paths, leaves, _ = optree.tree_flatten_with_path(state_dict)
    flat_dict: StateDictT = {
        ".".join(str(k) for k in path): leaf for path, leaf in zip(paths, leaves, strict=True)
    }
    return flat_dict


def place_on_device(state_dict: StateDictT, exec_cfg: ExecutionConfig) -> StateDictT:
    """Place the state dict on a device."""

    dev = device.from_(exec_cfg.dev)
    bend = DLBackend.get_backend(exec_cfg.backend)
    dev = bend.to_backend_device_obj(dev)

    result: StateDictT = optree.tree_map(lambda x: bend.to_device(x, dev), state_dict)
    return result


def cast_to(state_dict: StateDictT, exec_cfg: ExecutionConfig, dtype: DataType) -> StateDictT:
    """Cast all tensors to a given dtype."""

    bend = DLBackend.get_backend(exec_cfg.backend)
    bend_dtype = bend.to_backend_datatype(dtype)

    result: StateDictT = optree.tree_map(
        lambda x: bend.cast_backend_dtype(x, bend_dtype), state_dict
    )
    return result


def as_backend_tensor_type(state_dict: StateDictT, exec_cfg: ExecutionConfig) -> StateDictT:
    """Convert the state dict to backend tensors."""

    bend = DLBackend.get_backend(exec_cfg.backend)

    result: StateDictT = optree.tree_map(lambda x: bend.from_dlpack(x), state_dict)
    return result


class StateDictLoader:
    """State loader with string builder pattern."""

    def __init__(
        self,
        state_dict: Optional[StateDictT],
        current_path: Tuple[str, ...] = (),
        used_symbols: Optional[Set[Symbol]] = None,
    ):
        self._current_path: Tuple[str, ...] = tuple(current_path)
        self._used_symbols: Set[Symbol] = set(used_symbols) if used_symbols is not None else set()
        self.state_dict: Optional[StateDictT] = state_dict

    def _replace_placeholders(self, pattern: str, mapping: Dict[Symbol, int]) -> str:
        """Replace placeholders in a pattern with actual symbol values."""
        result = pattern
        assert self._used_symbols.issubset(set(mapping.keys())), (
            f"Used symbols {self._used_symbols} not a subset of {mapping=} for key {pattern}"
        )
        for key, value in mapping.items():
            result = result.replace(self._symbol_key(key), str(value))
        return result

    def _symbol_key(self, symbol: Symbol) -> str:
        return "<" + str(symbol) + ">"

    @staticmethod
    def empty() -> StateDictLoader:
        return StateDictLoader(None)

    def get_structure(self) -> Dict[str, Tuple[Shape, DataType]]:
        """Get the structure of the state dict."""
        if self.state_dict is None:
            return {}

        def get_tensor_structure(tensor: BackendTensorT) -> Tuple[Shape, DataType]:
            shape = Shape.from_(tensor.shape)
            dtype = dtypes.implied(tensor)

            # TODO: only good if guaranteed torch...
            # mean = bend.mean(tensor)
            # std = bend.std(tensor)

            return shape, dtype  # , mean, std

        return optree.tree_map(get_tensor_structure, self.state_dict)  # type: ignore

    @staticmethod
    def from_torch_checkpoint(
        statedict_path: PathLike,
        exec_cfg: ExecutionConfig,
        cast_to_dtype: DataTypeLike = None,
        **kwargs: Any,
    ) -> StateDictLoader:
        """Create a state loader from a checkpoint path."""
        statedict_path = Path(statedict_path)

        # Load the state dict
        state_dict = torch.load(statedict_path, mmap=True, map_location="cpu")

        flat_state_dict = flatten_state_dict(state_dict)

        # TODO we may come to need finer grained control over this. For now it's fine.
        desired_dtype = dtypes.from_(cast_to_dtype, none_dtype=dtypes.default_float)
        backend_state_dict = as_backend_tensor_type(flat_state_dict, exec_cfg)
        cast_state_dict = cast_to(backend_state_dict, exec_cfg, desired_dtype)
        on_device_state_dict = place_on_device(cast_state_dict, exec_cfg)

        return StateDictLoader(on_device_state_dict, **kwargs)

    def _assert_not_initialized(self) -> None:
        assert len(self._current_path) == 0, "StateDictLoader is already initialized"
        assert self._used_symbols == set(), "StateDictLoader is already initialized"

    def load_tensor(self, suffix: str = "") -> MaybeInitFn:
        """Load a tensor using the current path + suffix.

        Args:
            suffix: Optional suffix to append to the current path
        Returns:
            A partial function for init_from_statedict, or None if no state dict
        """
        if self.state_dict is None:
            return None

        key = ".".join(self._current_path + (suffix,))

        logger.info("Registering state dict initializer for tensor %s", key)

        return partial(
            RecurrentTensor.init_from_statedict,
            flat_state_dict=self.state_dict,
            key=lambda ts: self._replace_placeholders(key, ts),
            skip_cast_dev_and_bend=True,
        )

    def append(self, component: Union[str, Symbol]) -> StateDictLoader:
        """Append a component to the current path (string builder pattern)."""

        if isinstance(component, Symbol):
            self._used_symbols.add(component)
            component = self._symbol_key(component)

        return StateDictLoader(
            self.state_dict, self._current_path + (component,), set(self._used_symbols)
        )
