from collections.abc import Callable
from typing import Any


def make_sink(name: str, sink_data: dict[str, Any]) -> Callable[[Any], None]:
    """
    Creates a sink function that stores the passed array_value into sink_data under the given name.

    name: The key under which values will be stored.
    sink_data: Dictionary that will hold the data.
    return: A function that takes a value and stores it in sink_data[name].
    """

    def _store(array_value: Any) -> None:
        # import torch
        # assert not torch.isnan(array_value).any(), f"NaN value in {name}"
        sink_data[name] = array_value

    return _store


def make_step_by_step_sink(name: str, sink_data: dict[str, Any]) -> Callable[[Any], None]:
    """
    Creates a sink function that stores the passed array_value into sink_data under the given name.

    name: The key under which values will be stored.
    sink_data: Dictionary that will hold the data.
    return: A function that takes a value and stores it in sink_data[name].
    """
    sink_data[name] = []

    def _store(array_value: Any) -> None:
        # import torch
        # assert not torch.isnan(array_value).any(), f"NaN value in {name}"
        sink_data[name].append(array_value)

    return _store
