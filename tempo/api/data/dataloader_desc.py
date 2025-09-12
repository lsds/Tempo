import numbers
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import optree
import torch
from torch.utils.data import Dataset as TorchDataset

from tempo.core.dtype import DataType, dtypes
from tempo.core.shape import Shape


@dataclass
class DatasetInfo:
    """Information about the structure and types of dataset samples."""

    treespec: optree.PyTreeSpec
    sample_shapes: Sequence[Shape]
    sample_dtypes: Sequence[DataType]
    size: int

    @property
    def num_outputs(self) -> int:
        return len(self.sample_shapes)


def infer_sample_info(dataset: TorchDataset) -> DatasetInfo:
    """Infer the structure, shapes, and dtypes from a dataset's first sample."""
    size = len(dataset)
    sample = dataset[0]

    # Extract all tensor leaves and their structure
    # TODO there should be a runtime global "registered_backend_tensor_types" to use here
    def is_leaf(x: Any) -> bool:
        return isinstance(x, (torch.Tensor, np.ndarray, numbers.Number))

    leaves, treespec = optree.tree_flatten(sample, is_leaf=is_leaf)

    # Get shapes and dtypes from tensor leaves
    shapes = []
    dtypes_ = []
    for leaf in leaves:
        # if not isinstance(leaf, torch.Tensor):
        #    raise ValueError(f"Expected all leaves to be tensors, got {type(leaf)}")
        leaf_torch_tensor = torch.tensor(leaf)

        shapes.append(Shape.from_(tuple(leaf_torch_tensor.shape)))
        dtypes_.append(dtypes.from_np(leaf_torch_tensor.numpy().dtype))

    ds_info = DatasetInfo(treespec, shapes, dtypes_, size)
    return ds_info


@dataclass(frozen=True)
class DataLoaderDesc:
    """Description of a dataloader that can be instantiated at runtime."""

    _id_counter: ClassVar[int] = 0

    dataset_factory: Callable[[], TorchDataset]
    batch_size: int | None
    shuffle: bool
    num_workers: int
    dataset_info: DatasetInfo
    kwargs: dict[str, Any]
    unique_id: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "unique_id", DataLoaderDesc._id_counter)
        DataLoaderDesc._id_counter += 1

    @property
    def state_id(self) -> str:
        return f"dataloader_{self.unique_id}"

    @property
    def batch_shapes(self) -> Sequence[Shape]:
        if self.batch_size is None:
            return self.dataset_info.sample_shapes

        return [
            Shape((self.batch_size,) + shape._shape) for shape in self.dataset_info.sample_shapes
        ]

    @property
    def batch_dtypes(self) -> Sequence[DataType]:
        return self.dataset_info.sample_dtypes

    def unflatten_batch(self, flat_tensors: Sequence[Any]) -> Any:
        """Reconstruct the original batch structure from flat tensors."""
        return optree.tree_unflatten(self.dataset_info.treespec, flat_tensors)
