from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from torch.utils.data import Dataset as TorchDataset

from tempo.api.data import udfs
from tempo.api.data.dataloader_desc import DataLoaderDesc, DatasetInfo, infer_sample_info
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.core.domain import DomainLike
from tempo.core.symbolic_tensor import SymbolicTensor


@dataclass(frozen=True)
class DataLoader:
    """Symbolic dataloader that can be instantiated at runtime."""

    _dataloader_desc: DataLoaderDesc

    @staticmethod
    def from_torch_dataset(
        dataset_factory: Callable[[], TorchDataset],
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> DataLoader:
        """Create a DataLoader from a PyTorch dataset factory.

        Args:
            dataset_factory: Function that creates a new dataset instance
            batch_size: Number of samples per batch (None for no batching, default)
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            **kwargs: Additional arguments passed to torch.utils.data.DataLoader
        """
        # Create temporary dataset to infer shapes/dtypes
        temp_dataset = dataset_factory()
        sample_info = infer_sample_info(temp_dataset)
        del temp_dataset

        desc = DataLoaderDesc(
            dataset_factory=dataset_factory,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            dataset_info=sample_info,
            kwargs=kwargs,
        )
        return DataLoader(desc)

    def sample(
        self,
        domain: DomainLike = None,
    ) -> Sequence[RecurrentTensor]:
        """Get the next batch from the dataloader."""
        desc = udfs.get_next_batch_udf_desc(self._dataloader_desc)
        symbolic_items = SymbolicTensor.udf(desc, [], domain=domain)
        return [RecurrentTensor(rt) for rt in symbolic_items]

    @property
    def sample_info(self) -> DatasetInfo:
        return self._dataloader_desc.dataset_info
