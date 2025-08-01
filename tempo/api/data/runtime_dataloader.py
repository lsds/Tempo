from abc import ABC, abstractmethod
from typing import Generic, Sequence

import optree
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from tempo.api.data.dataloader_desc import DataLoaderDesc
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import BackendTensorT
from tempo.core.device import DeviceLike
from tempo.runtime.backends.backend import DLBackend, DLBackendName


class RuntimeDataLoader(ABC, Generic[BackendTensorT]):
    # @abstractmethod
    # def reset(self) -> None:
    #    pass

    @abstractmethod
    def next_batch(self) -> Sequence[BackendTensorT]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class TorchRuntimeDataLoader(RuntimeDataLoader[torch.Tensor]):
    """Runtime instance of a dataloader."""

    def __init__(self, desc: DataLoaderDesc, exec_cfg: ExecutionConfig):
        self.desc = desc

        self.exec_cfg = exec_cfg

        # Create dataset and dataloader
        # TODO allow dataset factory to be given storage path, so we can store it in run folder?
        self.dataset = desc.dataset_factory()
        self.dataloader = TorchDataLoader(
            self.dataset,
            batch_size=desc.batch_size,
            shuffle=desc.shuffle,
            num_workers=desc.num_workers,
            generator=torch.Generator().manual_seed(exec_cfg.seed),
            drop_last=True,  # Drop last batch if it's not full size
            **desc.kwargs,
        )
        self.pytree_structure = optree.tree_structure(next(iter(self.dataloader)))
        self._iterator = iter(self.dataloader)

    def next_batch(self) -> Sequence[BackendTensorT]:
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            batch = next(self._iterator)

        # TODO we may want to, instead of flattening, return to users a similar structure of
        # recurrent tensors.
        # Flatten the batch structure
        flat_batch = self.pytree_structure.flatten_up_to(batch)

        # NOTE: if bs=1, it will have a batch dimension, but if None, it will not
        # This aligns with the behavior of tempo, so we are good.
        # if self.desc.batch_size is None:
        #    flat_batch = [item[0] for item in flat_batch]

        return tuple(flat_batch)

    def close(self) -> None:
        self.dataloader = None


# A wrapper that uses to_device to move the batch to the correct device.
class ToDeviceDataLoaderWrapper(RuntimeDataLoader[BackendTensorT]):
    def __init__(
        self,
        dataloader: RuntimeDataLoader,
        backend: DLBackend,
        desired_device: DeviceLike,
    ):
        self.dataloader = dataloader
        self.desired_device = desired_device

        # TODO: should not create backend here, but be given one
        # Update emission context to include backend
        self.backend = backend
        self.backend_dev = self.backend.to_backend_device_obj(self.desired_device)

    def next_batch(self) -> Sequence[BackendTensorT]:
        b = self.dataloader.next_batch()

        converted_batch = [
            self.backend.to_device(self.backend.from_dlpack(item), self.backend_dev) for item in b
        ]

        return converted_batch

    def close(self) -> None:
        self.dataloader.close()


def get_runtime_dataloader(desc: DataLoaderDesc, exec_cfg: ExecutionConfig) -> RuntimeDataLoader:
    loader: RuntimeDataLoader = TorchRuntimeDataLoader(desc, exec_cfg)

    backend = DLBackend.get_backend(exec_cfg.backend)
    backend.configure(exec_cfg)

    if DLBackendName.str_to_enum(exec_cfg.backend) != DLBackendName.TORCH:
        loader = ToDeviceDataLoaderWrapper(loader, backend, exec_cfg.dev)

    return loader
