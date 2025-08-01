# from typing import Any, Dict, List, Tuple, Union
#
# import torch
#
# from tempo.core.configs import ExecutionConfig
# from tempo.core.datatypes import NestedList, TensorId
# from tempo.core.domain import Domain
# from tempo.core.dtype import DataType
# from tempo.core.shape import Shape
# from tempo.runtime.tensor_store import RuntimeTensor, TensorStore
# from tempo.utils import logger
#
# log = logger.get_logger(__name__)
#
#
# def _concat_tensors(  # noqa: C901
#    nested_list: NestedList, item: Tuple[Union[int, slice], ...], depth: int
# ) -> torch.Tensor:
#    current_index = item[depth]
#    is_last_dim = depth == len(item) - 1
#
#    if type(current_index) is slice:
#        slice_range = list(
#            range(
#                int(current_index.start),
#                int(current_index.stop),
#                current_index.step if current_index.step is not None else 1,
#            )
#        )
#        tensors = []
#
#        for idx in slice_range:
#            if is_last_dim:
#                tensors.append(nested_list[idx])
#            else:
#                tensors.append(_concat_tensors(nested_list[idx], item, depth + 1))
#
#        return torch.stack(tensors, dim=0)
#    else:
#        if is_last_dim:
#            return nested_list[current_index]
#        else:
#            try:
#                return _concat_tensors(nested_list[current_index], item, depth + 1)
#            except Exception as e:
#                # print(
#                #    f"Tried to access {self.tensor_id} with domain {self.domain}, \
#                #          isl {self.domain.isl_domain} at {item} but failed"
#                # )
#                raise e
#
#
# def ensure_capacity(
#    nested_list: NestedList, item: Tuple[Union[int, slice], ...], depth: int
# ) -> None:
#    current_index = item[depth]
#    max_index = (
#        current_index.stop - 1 if isinstance(current_index, slice) else current_index
#    )
#
#    if depth == len(item) - 1:
#        while len(nested_list) <= max_index + 500:
#            nested_list.append(
#                None
#            )  # TODO instead of appending None, append empty tensor
#        return
#    else:
#        diff = (max_index - len(nested_list) + 1) + 500
#        # new_lists = [[] * diff] #NOTE: every list points to the same ref
#        new_lists: List[List[Any]] = [[] for _ in range(diff)]
#        nested_list.extend(new_lists)
#        for sublist in nested_list:
#            ensure_capacity(sublist, item, depth + 1)
#
#
# def split_and_set(  # noqa: C901
#    tensor: torch.Tensor,
#    nested_list: NestedList,
#    item: Tuple[Union[int, slice], ...],
#    depth: int,
# ) -> None:
#    current_index = item[depth]
#    is_last_dim = depth == len(item) - 1
#
#    if type(current_index) is slice:
#        if current_index.stop >= len(nested_list):
#            ensure_capacity(nested_list, item, depth)
#        slice_range = list(range(*current_index.indices(tensor.shape[depth])))
#        tensors = tensor.unbind(dim=0)
#        if is_last_dim:
#            for slice_idx, t in zip(slice_range, tensors):
#                nested_list[slice_idx] = (
#                    t  # TODO instead of assigning tensor, index_put t b_index
#                )
#            return
#        else:
#            for slice_idx, t in zip(slice_range, tensors):
#                split_and_set(
#                    t,
#                    nested_list[slice_idx],
#                    item,
#                    depth + 1,
#                )
#    else:
#        if current_index >= len(nested_list):
#            print(f"Ensuring capacity for {current_index} at depth {depth}")
#            ensure_capacity(nested_list, item, depth)
#        if is_last_dim:
#            # TODO instead of assigning tensor, index_put at b_index
#
#            nested_list[current_index] = tensor
#            return
#        else:
#            split_and_set(
#                tensor,
#                nested_list[current_index],
#                item,
#                depth + 1,
#            )
#
#
# class PyTorchRuntimeTensor(RuntimeTensor[torch.Tensor]):
#    def __init__(
#        self,
#        exec_cfg: ExecutionConfig,
#        tensor_id: TensorId,
#        shape: Shape,
#        dtype: DataType,
#        domain: Domain,
#    ) -> None:
#        super().__init__()
#        self.exec_cfg = exec_cfg
#        self.tensor_id = tensor_id
#        self.shape = shape
#        self.dtype = dtype
#
#        self.domain = domain
#        self.domain_size = len(domain)
#
#        if self.domain_size != 0:
#            self._tensor: NestedList = []
#            # Initially allocate space for 1000 elements in each dim
#            ensure_capacity(self._tensor, (1000,) * self.domain_size, 0)
#
#    def __str__(self) -> str:
#        return f"Tensor(tensor_id={self.tensor_id}, shape={self.shape},\
#              dtype={self.dtype}, domain={self.domain})"
#
#    def __getitem__(  # noqa: C901
#        self, item: Tuple[Union[int, slice], ...]
#    ) -> torch.Tensor:
#
#        # log.debug(
#        #    f"{self.tensor_id=}, {self.shape=}, {self.dtype=}, {self.domain=}, {item=}"
#        # )
#        # assert len(item) == self.domain_size
#
#        if self.domain_size == 0:
#            return self._tensor
#        else:
#            return _concat_tensors(self._tensor, item, 0)
#
#    def __setitem__(  # noqa: C901
#        self, item: Tuple[Union[int, slice], ...], value: torch.Tensor
#    ) -> None:
#        # assert len(item) == self.domain_size
#
#        # b_index = item[0]
#        # item = item[1:]  # remove batch dimension
#
#        # Ensure sufficient capacity in self._tensor for the new indices
#        # ensure_capacity(self._tensor, 0)
#        # Now, perform the splitting and setting
#        if self.domain_size == 0:
#            self._tensor = value
#        else:
#            split_and_set(value, self._tensor, item, 0)
#
#    def flush(self) -> None:
#        """This method clears the tensor of any remaining data."""
#        self._tensor = []
#
#    def deallocate(self, item: Tuple[Union[int, slice], ...]) -> None:
#        assert len(item) == self.domain_size
#        if len(item) == 0:
#            self._tensor = None
#        else:
#            tensor = self._tensor
#            for i in item[:-1]:
#                tensor = tensor[i]
#            tensor[item[-1]] = None
#
#    def offload(self, item: Tuple[Union[int, slice], ...]) -> None:
#        assert len(item) == self.domain_size
#        if len(item) == 0:
#            self._tensor = self._tensor.to("cpu")
#        else:
#            tensor = self._tensor
#            for i in item[:-1]:
#                tensor = tensor[i]
#            tensor[item[-1]] = tensor[item[-1]].to("cpu")
#
#    def fetch(self, item: Tuple[Union[int, slice], ...]) -> None:
#        assert len(item) == self.domain_size
#        if len(item) == 0:
#            self._tensor = self._tensor.to(self.exec_cfg.dev)
#        else:
#            tensor = self._tensor
#            for i in item[:-1]:
#                tensor = tensor[i]
#            tensor[item[-1]] = tensor[item[-1]].to(self.exec_cfg.dev)
#
#
# class PyTorchTensorStore(TensorStore[torch.Tensor]):
#    def __init__(
#        self,
#        exec_cfg: ExecutionConfig,
#        tensor_descs: List[Tuple[TensorId, Shape, DataType, Domain]],
#    ):
#        super().__init__(exec_cfg, tensor_descs)
#        self.tensors: Dict[TensorId, PyTorchRuntimeTensor] = {}
#        for tensor_id, shape, dtype, domain in tensor_descs:
#            self.tensors[tensor_id] = PyTorchRuntimeTensor(
#                exec_cfg, tensor_id, shape, dtype, domain
#            )
#
#    def __getitem__(self, item: TensorId) -> PyTorchRuntimeTensor:
#        return self.tensors[item]
#
#    def flush(self) -> None:
#        """This method clears the tensor store of any remaining data, preparing
#        it for the next execution."""
#        for tensor in self.tensors.values():
#            tensor.flush()
#
