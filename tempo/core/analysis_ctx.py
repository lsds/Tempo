from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import islpy as isl

from tempo.core import device
from tempo.core import tensor_op as top
from tempo.core.datatypes import OpId, TensorId
from tempo.core.device import DeviceGroup
from tempo.core.schedule.execution_schedule import ExecutionSchedule
from tempo.core.storage_methods import StorageMethod
from tempo.core.tensor_op import TensorOp
from tempo.utils.common import Timer


@dataclass
class ScheduleConstraints:
    # TODO: eventually move domains in here.
    domain: isl.UnionSet
    validity: isl.UnionMap
    proximity: isl.UnionMap
    coincidence: isl.UnionMap

    def copy(self) -> ScheduleConstraints:
        return ScheduleConstraints(
            self.domain.copy(),
            self.validity.copy(),
            self.proximity.copy(),
            self.coincidence.copy(),
        )


@dataclass
class AnalysisCtx:
    _isl_ctx: isl.Context  # type: ignore
    _isl_domains: dict[OpId, isl.UnionSet] = field(default_factory=dict)
    _cached_schedule_constraints: ScheduleConstraints | None = None
    _execution_schedule: ExecutionSchedule | None = None
    _isl_execution_schedule: isl.Schedule | None = None
    _donatable_args: dict[OpId, tuple[int, ...]] | None = None
    _all_donatable_args: dict[OpId, tuple[int, ...]] | None = None
    _needed_merge_copies: dict[OpId, Sequence[bool]] | None = None
    _tensor_is_donated: dict[TensorId, bool] | None = None
    _memory_requirement_bytes: dict[OpId, int] | None = None
    _tensor_storage_classes: dict[TensorId, StorageMethod] | None = None
    _tensor_prealloc_value: dict[TensorId, Any] | None = None
    _groups: dict[int, set[TensorOp]] | None = None
    _device_assignment: dict[OpId, DeviceGroup] | None = None
    _compilation_profile_ms: dict[str, Any] | None = None
    _buffer_stored_output_tensor_positions: dict[OpId, tuple[int, ...]] | None = None
    _broadcast_elim_has_run: bool = False
    _is_incremental_algo: bool = False
    _loading_time_timer: Timer | None = None

    # def __post_init__(self):
    #    self._cached_schedule_constraints = ScheduleConstraints(
    #        domain=isl.UnionSet("[] -> { }", context=self._isl_ctx),
    #        validity=isl.UnionMap("[] -> { }", context=self._isl_ctx),
    #        proximity=isl.UnionMap("[] -> { }", context=self._isl_ctx),
    #        coincidence=isl.UnionMap("[] -> { }", context=self._isl_ctx),
    #    )

    def __deepcopy__(self, memo: dict[int, Any]) -> AnalysisCtx:
        return AnalysisCtx(
            self._isl_ctx,
            {k: v.copy() for k, v in self._isl_domains.items()},
            (
                self._cached_schedule_constraints.copy()
                if self._cached_schedule_constraints is not None
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
            copy.deepcopy(self._compilation_profile_ms, memo),
            copy.deepcopy(self._buffer_stored_output_tensor_positions, memo),
            self._broadcast_elim_has_run,
            self._is_incremental_algo,
            copy.deepcopy(self._loading_time_timer, memo),
        )

    def _copy_if_not_none(self, obj: Any) -> Any:
        """Helper method to copy an object if it's not None, otherwise return None."""
        return (
            (obj.copy() if hasattr(obj, "copy") else copy.deepcopy(obj))
            if obj is not None
            else None
        )

    def shallow_copy(self) -> AnalysisCtx:
        """Create a shallow copy of this AnalysisCtx instance."""
        return AnalysisCtx(
            self._isl_ctx,
            self._copy_if_not_none(self._isl_domains),
            self._copy_if_not_none(self._cached_schedule_constraints),
            self._copy_if_not_none(self._execution_schedule),
            self._copy_if_not_none(self._isl_execution_schedule),
            self._copy_if_not_none(self._donatable_args),
            self._copy_if_not_none(self._all_donatable_args),
            self._copy_if_not_none(self._needed_merge_copies),
            self._copy_if_not_none(self._tensor_is_donated),
            self._copy_if_not_none(self._memory_requirement_bytes),
            self._copy_if_not_none(self._tensor_storage_classes),
            self._copy_if_not_none(self._tensor_prealloc_value),
            self._copy_if_not_none(self._groups),
            self._copy_if_not_none(self._device_assignment),
            self._copy_if_not_none(self._compilation_profile_ms),
            self._copy_if_not_none(self._buffer_stored_output_tensor_positions),
            self._broadcast_elim_has_run,
            self._is_incremental_algo,
            self._copy_if_not_none(self._loading_time_timer),
        )

    @property
    def isl_ctx(self) -> isl.Context:
        if self._isl_ctx is None:
            raise ValueError("ISL context not set")
        return self._isl_ctx

    @property
    def isl_domains(self) -> dict[OpId, isl.UnionSet]:
        if self._isl_domains is None:
            raise ValueError("ISL domains not set")
        return self._isl_domains

    @property
    def isl_execution_schedule(self) -> isl.Schedule:
        if self._isl_execution_schedule is None:
            raise ValueError("ISL execution schedule not set")
        return self._isl_execution_schedule

    @property
    def execution_schedule(self) -> ExecutionSchedule:
        if self._execution_schedule is None:
            raise ValueError("Execution schedule not set")
        return self._execution_schedule

    @property
    def donatable_args(self) -> dict[OpId, tuple[int, ...]]:
        if self._donatable_args is None:
            raise ValueError("Donatable args not set")
        return self._donatable_args

    @property
    def tensor_is_donated(self) -> dict[TensorId, bool]:
        if self._tensor_is_donated is None:
            raise ValueError("Tensor donation not set")
        return self._tensor_is_donated

    @property
    def memory_requirement_bytes(self) -> dict[OpId, int]:
        if self._memory_requirement_bytes is None:
            raise ValueError("Memory requirement not set")
        return self._memory_requirement_bytes

    @property
    def tensor_storage_classes(self) -> dict[TensorId, StorageMethod]:
        if self._tensor_storage_classes is None:
            raise ValueError("Tensor storage classes not set")
        return self._tensor_storage_classes

    @property
    def tensor_prealloc_value(self) -> dict[TensorId, Any]:
        if self._tensor_prealloc_value is None:
            raise ValueError("Tensor prealloc value not set")
        return self._tensor_prealloc_value

    @property
    def buffer_stored_output_tensor_positions(self) -> dict[OpId, tuple[int, ...]]:
        if self._buffer_stored_output_tensor_positions is None:
            raise ValueError("Buffer stored output tensor positions not set")
        return self._buffer_stored_output_tensor_positions

    @property
    def cached_schedule_constraints(self) -> ScheduleConstraints:
        if self._cached_schedule_constraints is None:
            raise ValueError("Cached schedule constraints not set")
        return self._cached_schedule_constraints

    def get_or_make_domain(self, op: TensorOp) -> isl.UnionSet:
        from tempo.utils.isl import isl_domain_from_op

        if op.op_id in self.isl_domains:
            isl_dom = self.isl_domains[op.op_id]
        else:
            isl_dom = isl_domain_from_op(op, self.isl_ctx)
            self.isl_domains[op.op_id] = isl_dom
        return isl_dom

    @property
    def compilation_profile_ms(self) -> dict[str, Any]:
        if self._compilation_profile_ms is None:
            raise ValueError("Compilation profile not set")
        return self._compilation_profile_ms

    def get_op_device(self, op: top.TensorOp) -> device.DeviceGroup:
        dev_assignment = self._device_assignment
        if dev_assignment is not None:
            return dev_assignment[op.op_id]
        else:
            return device.device.cpu

    @property
    def load_timer(self) -> Timer:
        timer = self._loading_time_timer or Timer()
        return timer
