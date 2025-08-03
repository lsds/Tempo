from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Set, Tuple

from tempo.core import device
from tempo.core import isl_types as islt
from tempo.core import tensor_op as top
from tempo.core.datatypes import OpId, TensorId
from tempo.core.device import DeviceGroup
from tempo.core.schedule.execution_schedule import ExecutionSchedule
from tempo.core.storage_methods import StorageMethod
from tempo.core.tensor_op import TensorOp


@dataclass
class AnalysisCtx:
    _isl_ctx: islt.Context  # type: ignore
    _isl_domains: Dict[OpId, islt.UnionSet] = field(default_factory=dict)
    _additional_val_constraints: Optional[islt.UnionMap] = None
    _execution_schedule: Optional[ExecutionSchedule] = None
    _isl_execution_schedule: Optional[islt.Schedule] = None
    _donatable_args: Optional[Dict[OpId, Tuple[int, ...]]] = None
    _all_donatable_args: Optional[Dict[OpId, Tuple[int, ...]]] = None
    _needed_merge_copies: Optional[Dict[OpId, Sequence[bool]]] = None
    _tensor_is_donated: Optional[Dict[TensorId, bool]] = None
    _memory_requirement_bytes: Optional[Dict[OpId, int]] = None
    _tensor_storage_classes: Optional[Dict[TensorId, StorageMethod]] = None
    _tensor_prealloc_value: Optional[Dict[TensorId, Any]] = None
    _groups: Optional[Dict[int, Set[TensorOp]]] = None
    _device_assignment: Optional[Dict[OpId, DeviceGroup]] = None
    _compilation_time_breakdown: Optional[Dict[str, Any]] = None
    _buffer_stored_output_tensor_positions: Optional[Dict[OpId, Tuple[int, ...]]] = None
    _broadcast_elim_has_run: bool = False
    _is_incremental_algo: bool = False

    def __deepcopy__(self, memo: Dict[int, Any]) -> AnalysisCtx:
        return AnalysisCtx(
            self._isl_ctx,
            {k: v.copy() for k, v in self._isl_domains.items()},
            (
                self._additional_val_constraints.copy()
                if self._additional_val_constraints is not None
                else None
            ),
            copy.deepcopy(self._execution_schedule, memo),
            self._isl_execution_schedule.copy()
            if self._isl_execution_schedule is not None
            else None,
            copy.deepcopy(self._donatable_args, memo),
            copy.deepcopy(self._all_donatable_args, memo),
            copy.deepcopy(self._needed_merge_copies, memo),
            copy.deepcopy(self._tensor_is_donated, memo),
            copy.deepcopy(self._memory_requirement_bytes, memo),
            copy.deepcopy(self._tensor_storage_classes, memo),
            copy.deepcopy(self._tensor_prealloc_value, memo),
            copy.deepcopy(self._groups, memo),
            copy.deepcopy(self._device_assignment, memo),
            copy.deepcopy(self._compilation_time_breakdown, memo),
            copy.deepcopy(self._buffer_stored_output_tensor_positions, memo),
            self._broadcast_elim_has_run,
            self._is_incremental_algo,
        )

    @property
    def isl_ctx(self) -> islt.Context:
        if self._isl_ctx is None:
            raise ValueError("ISL context not set")
        return self._isl_ctx

    @property
    def isl_domains(self) -> Dict[OpId, islt.UnionSet]:
        if self._isl_domains is None:
            raise ValueError("ISL domains not set")
        return self._isl_domains

    @property
    def isl_execution_schedule(self) -> islt.Schedule:
        if self._isl_execution_schedule is None:
            raise ValueError("ISL execution schedule not set")
        return self._isl_execution_schedule

    @property
    def execution_schedule(self) -> ExecutionSchedule:
        if self._execution_schedule is None:
            raise ValueError("Execution schedule not set")
        return self._execution_schedule

    @property
    def donatable_args(self) -> Dict[OpId, Tuple[int, ...]]:
        if self._donatable_args is None:
            raise ValueError("Donatable args not set")
        return self._donatable_args

    @property
    def tensor_is_donated(self) -> Dict[TensorId, bool]:
        if self._tensor_is_donated is None:
            raise ValueError("Tensor donation not set")
        return self._tensor_is_donated

    @property
    def memory_requirement_bytes(self) -> Dict[OpId, int]:
        if self._memory_requirement_bytes is None:
            raise ValueError("Memory requirement not set")
        return self._memory_requirement_bytes

    @property
    def tensor_storage_classes(self) -> Dict[TensorId, StorageMethod]:
        if self._tensor_storage_classes is None:
            raise ValueError("Tensor storage classes not set")
        return self._tensor_storage_classes

    @property
    def tensor_prealloc_value(self) -> Dict[TensorId, Any]:
        if self._tensor_prealloc_value is None:
            raise ValueError("Tensor prealloc value not set")
        return self._tensor_prealloc_value

    @property
    def buffer_stored_output_tensor_positions(self) -> Dict[OpId, Tuple[int, ...]]:
        if self._buffer_stored_output_tensor_positions is None:
            raise ValueError("Buffer stored output tensor positions not set")
        return self._buffer_stored_output_tensor_positions

    def get_or_make_domain(self, op: TensorOp) -> islt.UnionSet:
        from tempo.utils.isl import isl_domain_from_op

        if op.op_id in self.isl_domains:
            isl_dom = self.isl_domains[op.op_id]
        else:
            isl_dom = isl_domain_from_op(op, self.isl_ctx)
            self.isl_domains[op.op_id] = isl_dom
        return isl_dom

    @property
    def compilation_time_breakdown(self) -> Dict[str, Any]:
        if self._compilation_time_breakdown is None:
            raise ValueError("Compilation time breakdown not set")
        return self._compilation_time_breakdown

    def get_op_device(self, op: top.TensorOp) -> device.DeviceGroup:
        dev_assignment = self._device_assignment
        if dev_assignment is not None:
            return dev_assignment[op.op_id]
        else:
            return device.device.cpu
