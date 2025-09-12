from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
    Generic,
)

from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.analysis_ctx import AnalysisCtx
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import BackendTensorT, OpInId, OpOutId, TensorId
from tempo.core.dependence_graph import PDG
from tempo.core.device import DeviceGroup, device
from tempo.core.dl_backend import DLBackend
from tempo.core.dl_backends import DLBackendName
from tempo.core.dtype import DataType, dtypes
from tempo.core.external_state_store import ExternalStateStore
from tempo.core.schedule.execution_schedule import ExecInstruction
from tempo.core.shape import Shape
from tempo.core.symbol_dict import SymbolDict
from tempo.core.tensor_op import TensorOp
from tempo.core.thunk import Thunk, ThunkExecutionCtx
from tempo.core.thunk_emitter import ThunkEmissionCtx, ThunkEmitter
from tempo.runtime.inplace_buffer_thunk_wrapper import has_buffer_stored_outputs
from tempo.runtime.tensor_store.block_runtime_tensor import BlockRuntimeTensor
from tempo.runtime.tensor_store.tensor_store import (
    PreallocRuntimeTensor,
    RuntimeTensor,
    TensorStore,
)
from tempo.utils import logger
from tempo.utils.dg_utils import is_initialization_merge

log = logger.get_logger(__name__)


def dump_debug_info(
    launcher: BaseThunkLauncher,
    e: Exception,
    merge_branch: int | None = None,
    cond: bool = False,
) -> None:
    if merge_branch is not None:
        if cond:
            log.error("Error during Merge %s branch condition", merge_branch)
        else:
            log.error("Error during Merge %s branch", merge_branch)

    log.error(
        "Error executing op %s with symbols: %s. Error: %s",
        launcher.op,
        launcher.thunk_exec_ctx.symbol_values,
        e,
    )
    print(f"Loop Counters: {dict(launcher.loop_counters_and_bounds.items())}")
    idxs = {k: k.idx for k, v in launcher.loop_counters_and_bounds.items()}
    print(f"Loop counter idx: {idxs}")
    print(f"Domain Map: {dict(launcher.domain_map.items())}")
    print(f"Domain map key idx: { {k: k.idx for k in launcher.domain_map.keys()} }")
    print(f"Indexes: {launcher.input_index_exprs}")
    print(f"Remapped indexes: {[e.remap(launcher.domain_map) for e in launcher.input_index_exprs]}")
    a = [
        e.remap(launcher.domain_map).evaluate(launcher.loop_counters_and_bounds)
        for e in launcher.input_index_exprs
    ]
    print(f"Evaluated remapped indexes: {a}")
    print("Op Creation Traceback:")
    print(f"{launcher.op.creation_traceback}")
    # print(f"Input tensors: {arg_tensors}")


@dataclass(frozen=True, slots=True)
class ThunkLauncherFactoryCtx(Generic[BackendTensorT]):
    thunk_emitter: ThunkEmitter[BackendTensorT]
    external_state_store: ExternalStateStore | None
    tensor_store: TensorStore[BackendTensorT]
    backend: DLBackend
    loop_counters_and_bounds: Mapping[ie.Symbol, int]
    compilation_ctx: CompilationCtx

    @property
    def dg(self) -> PDG:
        return self.compilation_ctx.dg

    @property
    def exec_cfg(self) -> ExecutionConfig:
        return self.compilation_ctx.exec_cfg

    @property
    def analysis_ctx(self) -> AnalysisCtx:
        return self.compilation_ctx.analysis_ctx


@dataclass(frozen=True, slots=True)
class ThunkLauncher(Generic[BackendTensorT], ABC):
    @abstractmethod
    def launch(self) -> None: ...


@dataclass(frozen=True, slots=True)
class BaseThunkLauncher(ThunkLauncher[BackendTensorT]):
    op: TensorOp
    thunk: Thunk[BackendTensorT]
    expected_input_shapes: list[Shape]
    expected_input_dtypes: list[DataType]
    expected_input_devices: list[DeviceGroup]
    input_tensors: list[tuple[RuntimeTensor[BackendTensorT], Callable[[], tuple[int | slice, ...]]]]
    input_index_exprs: list[ie.IndexSequence]
    input_conds: list[ie.BooleanIndexValue | None]
    requires_copy: list[bool]
    out_expr_eval: Callable[[], tuple[int, ...]]
    expected_output_shapes: list[Shape]
    expected_output_dtypes: list[DataType]
    output_tensors: list[RuntimeTensor[BackendTensorT]]
    buffer_out_positions: tuple[int, ...]
    device: DeviceGroup
    backend: DLBackend[BackendTensorT]
    thunk_exec_ctx: ThunkExecutionCtx
    domain_map: Mapping[ie.Symbol, ie.IntIndexValue]
    loop_counters_and_bounds: Mapping[ie.Symbol, int]
    arg_fns: list[Any] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        # ordinary input tensors – use all-int fast path
        arg_fns = []
        for i, (t, fn) in enumerate(self.input_tensors):
            if self.op.domain.basis_expr.struct_eq(self.input_index_exprs[i]):
                arg_fns.append(lambda out_ts, rt=t, ev=fn: rt.all_int_fast_path(out_ts))
            elif self.input_index_exprs[i].is_point():
                arg_fns.append(lambda out_ts, rt=t, ev=fn: rt.all_int_fast_path(ev()))
            else:
                arg_fns.append(lambda out_ts, rt=t, ev=fn: rt[ev()])

        object.__setattr__(self, "arg_fns", arg_fns)

    def _debug_checks(
        self,
        tensors: tuple[BackendTensorT, ...],
        input_: bool = True,
    ) -> None:
        # if input_:
        #     for i, tensor in enumerate(tensors):
        #         if tensor.is_deleted():
        #             print(f"Index {i} is deleted.")
        #             print([t[0].tensor_id for t in self.input_tensors])

        # TODO check dtypes
        # import jax.numpy as jnp

        # import torch
        # for i, tensor in enumerate(tensors):
        # if not input_ and isinstance(self.op, top.PadOp):
        #    continue

        # if input_ and isinstance(self.op, top.ValToValOp):
        #    continue

        # if torch.isnan(tensor).any():
        #    raise ValueError(f"({input_=}) NaNs in {'input' if input_ else 'output'}
        #  tensor {i} for op {self.op}.")
        # if torch.isinf(tensor).any():
        #    raise ValueError(f"({input_=}) Inf in {'input' if input_ else 'output'}
        #  tensor {i} for op {self.op}.")

        # for i, tensor in enumerate(tensors):
        #    if jnp.isnan(tensor).any():
        #        raise ValueError(
        #            f"({input_=}) NaNs in {'input' if input_ else
        # 'output'} tensor {i} for op {self.op}."
        #        )
        #    if jnp.isinf(tensor).any():
        #        raise ValueError(
        #            f"({input_=}) Inf in {'input' if input_ else
        #  'output'} tensor {i} for op {self.op}."
        #        )
        runtime_shapes = [t.shape for t in tensors]
        expected_shapes = [
            s.evaluate(self.thunk_exec_ctx.symbol_values)
            for s in (self.expected_input_shapes if input_ else self.expected_output_shapes)
        ]
        if not all(a == b for a, b in zip(runtime_shapes, expected_shapes, strict=False)):
            print(f"Creation traceback of op with error:\n{self.op.creation_traceback}")
            raise ValueError(
                f"({input_=}) Shapes do not match expected shapes for op {self.op}.\
                    Got {runtime_shapes} but expected {expected_shapes}"
            )

        runtime_devices = [self.backend.device(t) for t in tensors]
        input_expect_devs = [
            self.backend.to_backend_device_obj(d) for d in self.expected_input_devices
        ]
        out_dev = self.backend.to_backend_device_obj(self.device)

        if not all(
            a == b
            for a, b in zip(
                runtime_devices,
                (input_expect_devs if input_ else [out_dev] * len(tensors)),
                strict=False,
            )
        ):
            expected = input_expect_devs if input_ else [out_dev] * len(tensors)
            raise ValueError(
                f"({input_=}) devices do not match expected device for op {self.op}. \
                  Got {runtime_devices} but expected {expected}"
            )

    def launch(self) -> None:
        # runtime values of domains
        out_ts = self.out_expr_eval()

        arg_tensors = tuple(fn(out_ts) for fn in self.arg_fns)

        outs = self.thunk(arg_tensors, self.thunk_exec_ctx)  # type: ignore

        for i, out_tt in enumerate(self.output_tensors):
            # stored output.
            out_tt.all_int_fast_path_set(out_ts, outs[i])  # type: ignore


@dataclass(frozen=True, slots=True)
class DebugThunkLauncher(BaseThunkLauncher[BackendTensorT]):
    def launch(self) -> None:
        try:
            arg_tensors = tuple(t[eval_fn()] for t, eval_fn in self.input_tensors)

            self._debug_checks(arg_tensors, True)
            outs = self.thunk(arg_tensors, self.thunk_exec_ctx)  # type: ignore
            self._debug_checks(outs, False)
        except Exception as e:
            dump_debug_info(self, e)
            raise e
        out_ts = self.out_expr_eval()

        for i, out_tt in enumerate(self.output_tensors):
            out_tt.all_int_fast_path_set(out_ts, outs[i])  # type: ignore


@dataclass(frozen=True, slots=True)
class AllIntFastPathThunkLauncher(BaseThunkLauncher[BackendTensorT]):
    def launch(self) -> None:
        # Uses the all_int_fast_path to get data from the tensor store.
        args = tuple(t.all_int_fast_path(eval_fn()) for t, eval_fn in self.input_tensors)

        outs = self.thunk(args, self.thunk_exec_ctx)  # type: ignore

        out_ts = self.out_expr_eval()
        for i, out_tt in enumerate(self.output_tensors):
            out_tt.all_int_fast_path_set(out_ts, outs[i])  # type: ignore


@dataclass(frozen=True, slots=True)
class SingleInputAllIntFastPathThunkLauncher(BaseThunkLauncher[BackendTensorT]):
    eval_fn: Callable[[], tuple[int | slice]] = lambda: (0,)
    input_tensor: RuntimeTensor[BackendTensorT] = None  # type: ignore

    def __post_init__(self) -> None:
        assert len(self.input_tensors) == 1
        object.__setattr__(self, "eval_fn", self.input_tensors[0][1])
        object.__setattr__(self, "input_tensor", self.input_tensors[0][0])

    def launch(self) -> None:
        # Uses the all_int_fast_path to get data from the tensor store.
        args = (self.input_tensor.all_int_fast_path(self.eval_fn()),)

        outs = self.thunk(args, self.thunk_exec_ctx)  # type: ignore

        out_ts = self.out_expr_eval()
        for i, out_tt in enumerate(self.output_tensors):
            out_tt.all_int_fast_path_set(out_ts, outs[i])  # type: ignore


@dataclass(frozen=True, slots=True)
class TwoInputAllIntFastPathThunkLauncher(BaseThunkLauncher[BackendTensorT]):
    eval_fn0: Callable[[], tuple[int | slice]] = lambda: (0,)
    input_tensor0: RuntimeTensor[BackendTensorT] = None  # type: ignore
    eval_fn1: Callable[[], tuple[int | slice]] = lambda: (0,)
    input_tensor1: RuntimeTensor[BackendTensorT] = None  # type: ignore

    def __post_init__(self) -> None:
        assert len(self.input_tensors) == 2
        object.__setattr__(self, "eval_fn0", self.input_tensors[0][1])
        object.__setattr__(self, "input_tensor0", self.input_tensors[0][0])
        object.__setattr__(self, "eval_fn1", self.input_tensors[1][1])
        object.__setattr__(self, "input_tensor1", self.input_tensors[1][0])

    def launch(self) -> None:
        # Uses the all_int_fast_path to get data from the tensor store.
        args = (
            self.input_tensor0.all_int_fast_path(self.eval_fn0()),
            self.input_tensor1.all_int_fast_path(self.eval_fn1()),
        )

        outs = self.thunk(args, self.thunk_exec_ctx)  # type: ignore

        out_ts = self.out_expr_eval()
        for i, out_tt in enumerate(self.output_tensors):
            out_tt.all_int_fast_path_set(out_ts, outs[i])  # type: ignore


@dataclass(frozen=True, slots=True)
class ThreeInputAllIntFastPathThunkLauncher(BaseThunkLauncher[BackendTensorT]):
    eval_fn0: Callable[[], tuple[int | slice]] = lambda: (0,)
    input_tensor0: RuntimeTensor[BackendTensorT] = None  # type: ignore
    eval_fn1: Callable[[], tuple[int | slice]] = lambda: (0,)
    input_tensor1: RuntimeTensor[BackendTensorT] = None  # type: ignore
    eval_fn2: Callable[[], tuple[int | slice]] = lambda: (0,)
    input_tensor2: RuntimeTensor[BackendTensorT] = None  # type: ignore

    def __post_init__(self) -> None:
        assert len(self.input_tensors) == 3
        object.__setattr__(self, "eval_fn0", self.input_tensors[0][1])
        object.__setattr__(self, "input_tensor0", self.input_tensors[0][0])
        object.__setattr__(self, "eval_fn1", self.input_tensors[1][1])
        object.__setattr__(self, "input_tensor1", self.input_tensors[1][0])
        object.__setattr__(self, "eval_fn2", self.input_tensors[2][1])
        object.__setattr__(self, "input_tensor2", self.input_tensors[2][0])

    def launch(self) -> None:
        args = (
            self.input_tensor0.all_int_fast_path(self.eval_fn0()),
            self.input_tensor1.all_int_fast_path(self.eval_fn1()),
            self.input_tensor2.all_int_fast_path(self.eval_fn2()),
        )
        outs = self.thunk(args, self.thunk_exec_ctx)  # type: ignore
        out_ts = self.out_expr_eval()
        for i, out_tt in enumerate(self.output_tensors):
            out_tt.all_int_fast_path_set(out_ts, outs[i])  # type: ignore


@dataclass(frozen=True, slots=True)
class FourInputAllIntFastPathThunkLauncher(BaseThunkLauncher[BackendTensorT]):
    eval_fn0: Callable[[], tuple[int | slice]] = lambda: (0,)
    input_tensor0: RuntimeTensor[BackendTensorT] = None  # type: ignore
    eval_fn1: Callable[[], tuple[int | slice]] = lambda: (0,)
    input_tensor1: RuntimeTensor[BackendTensorT] = None  # type: ignore
    eval_fn2: Callable[[], tuple[int | slice]] = lambda: (0,)
    input_tensor2: RuntimeTensor[BackendTensorT] = None  # type: ignore
    eval_fn3: Callable[[], tuple[int | slice]] = lambda: (0,)
    input_tensor3: RuntimeTensor[BackendTensorT] = None  # type: ignore

    def __post_init__(self) -> None:
        assert len(self.input_tensors) == 4
        object.__setattr__(self, "eval_fn0", self.input_tensors[0][1])
        object.__setattr__(self, "input_tensor0", self.input_tensors[0][0])
        object.__setattr__(self, "eval_fn1", self.input_tensors[1][1])
        object.__setattr__(self, "input_tensor1", self.input_tensors[1][0])
        object.__setattr__(self, "eval_fn2", self.input_tensors[2][1])
        object.__setattr__(self, "input_tensor2", self.input_tensors[2][0])
        object.__setattr__(self, "eval_fn3", self.input_tensors[3][1])
        object.__setattr__(self, "input_tensor3", self.input_tensors[3][0])

    def launch(self) -> None:
        args = (
            self.input_tensor0.all_int_fast_path(self.eval_fn0()),
            self.input_tensor1.all_int_fast_path(self.eval_fn1()),
            self.input_tensor2.all_int_fast_path(self.eval_fn2()),
            self.input_tensor3.all_int_fast_path(self.eval_fn3()),
        )
        outs = self.thunk(args, self.thunk_exec_ctx)  # type: ignore
        out_ts = self.out_expr_eval()
        for i, out_tt in enumerate(self.output_tensors):
            out_tt.all_int_fast_path_set(out_ts, outs[i])  # type: ignore


class NoInputsThunkLauncher(BaseThunkLauncher[BackendTensorT]):
    def launch(self) -> None:
        outs = self.thunk((), self.thunk_exec_ctx)  # type: ignore

        out_ts = self.out_expr_eval()
        for i, out_tt in enumerate(self.output_tensors):
            out_tt.all_int_fast_path_set(out_ts, outs[i])  # type: ignore


@dataclass(frozen=True, slots=True)
class EvalSymbolThunkLauncher(BaseThunkLauncher[BackendTensorT]):
    out_tensor: RuntimeTensor[BackendTensorT] = None  # type: ignore
    lift: Callable[[Any], BackendTensorT] = lambda x: x
    # remapped: ie.IndexSequence = None  # type: ignore

    def __post_init__(self) -> None:
        assert isinstance(self.op, top.EvalSymbolOp)

        dev = self.backend.to_backend_device_obj(self.device)
        dtype = self.backend.to_backend_datatype(self.op.dtype)

        idx = self.op.domain.find_variable_index(self.op.symbol)
        lift = lambda x: self.backend.fast_int_lift(x[idx], device=dev, dtype=dtype)

        object.__setattr__(self, "out_tensor", self.output_tensors[0])
        object.__setattr__(self, "lift", lift)

    def launch(self) -> None:
        val = self.out_expr_eval()
        # NOTE: val is going to be a tuple of length 1 containing the value of the symbol
        # We need to extract the value from the tuple and pass it to the tensor store
        # as the value to actually set.
        self.out_tensor.all_int_fast_path_set(val, self.lift(val))  # type: ignore


@dataclass(frozen=True, slots=True)
class NoOutputsThunkLauncher(BaseThunkLauncher[BackendTensorT]):
    def launch(self) -> None:
        arg_tensors = tuple(t[eval_fn()] for t, eval_fn in self.input_tensors)
        self.thunk(arg_tensors, self.thunk_exec_ctx)


@dataclass(frozen=True, slots=True)
class EvalOnceAndFastPathSingleInputNoOutputsThunkLauncher(BaseThunkLauncher[BackendTensorT]):
    eval_fn: Callable[[], tuple[int | slice]] = lambda: (0,)
    input_tensor: RuntimeTensor[BackendTensorT] = None  # type: ignore

    def __post_init__(self) -> None:
        assert isinstance(self.op, top.TensorOp)
        object.__setattr__(self, "eval_fn", self.input_tensors[0][1])
        object.__setattr__(self, "input_tensor", self.input_tensors[0][0])

    def launch(self) -> None:
        try:
            self.thunk(
                (self.input_tensor.all_int_fast_path(self.eval_fn()),),
                self.thunk_exec_ctx,
            )
        except Exception as e:
            log.error(
                "Error executing op %s with domain map: %s. Error: %s",
                self.op,
                dict(self.domain_map.items()),
                e,
            )
            raise e


@dataclass(frozen=True, slots=True)
class EvalOnceAndFastPathThunkLauncher(BaseThunkLauncher[BackendTensorT]):
    eval_fn: Callable[[], tuple[int | slice]] = lambda: (0,)

    def __post_init__(self) -> None:
        assert isinstance(self.op, top.TensorOp)
        object.__setattr__(self, "eval_fn", self.input_tensors[0][1])

    def launch(self) -> None:
        # Uses the all_int_fast_path to get data from the tensor store. Eval only once.
        point = self.eval_fn()
        args = tuple(t.all_int_fast_path(point) for t, _ in self.input_tensors)

        outs = self.thunk(args, self.thunk_exec_ctx)  # type: ignore

        out_ts = self.out_expr_eval()

        for i, out_tt in enumerate(self.output_tensors):
            out_tt.all_int_fast_path_set(out_ts, outs[i])  # type: ignore


@dataclass(frozen=True, slots=True)
class SingleInputEvalOnceAndFastPathThunkLauncher(BaseThunkLauncher[BackendTensorT]):
    eval_fn: Callable[[], tuple[int | slice]] = lambda: (0,)
    input_tensor: RuntimeTensor[BackendTensorT] = None  # type: ignore

    def __post_init__(self) -> None:
        assert isinstance(self.op, top.TensorOp)
        assert len(self.input_tensors) == 1
        object.__setattr__(self, "eval_fn", self.input_tensors[0][1])
        object.__setattr__(self, "input_tensor", self.input_tensors[0][0])

    def launch(self) -> None:
        # Uses the all_int_fast_path to get data from the tensor store. Eval only once.
        point = self.eval_fn()
        args = (self.input_tensor.all_int_fast_path(point),)

        outs = self.thunk(args, self.thunk_exec_ctx)  # type: ignore

        out_ts = self.out_expr_eval()

        for i, out_tt in enumerate(self.output_tensors):
            out_tt.all_int_fast_path_set(out_ts, outs[i])  # type: ignore


@dataclass(frozen=True, slots=True)
class TwoInputEvalOnceAndFastPathThunkLauncher(BaseThunkLauncher[BackendTensorT]):
    eval_fn0: Callable[[], tuple[int | slice]] = lambda: (0,)
    input_tensor0: RuntimeTensor[BackendTensorT] = None  # type: ignore
    input_tensor1: RuntimeTensor[BackendTensorT] = None  # type: ignore

    def __post_init__(self) -> None:
        assert isinstance(self.op, top.TensorOp)
        assert len(self.input_tensors) == 2
        object.__setattr__(self, "eval_fn0", self.input_tensors[0][1])
        object.__setattr__(self, "input_tensor0", self.input_tensors[0][0])
        object.__setattr__(self, "input_tensor1", self.input_tensors[1][0])

    def launch(self) -> None:
        # Uses the all_int_fast_path to get data from the tensor store. Eval only once.
        point = self.eval_fn0()
        args = (
            self.input_tensor0.all_int_fast_path(point),
            self.input_tensor1.all_int_fast_path(point),
        )

        outs = self.thunk(args, self.thunk_exec_ctx)  # type: ignore

        out_ts = self.out_expr_eval()

        for i, out_tt in enumerate(self.output_tensors):
            out_tt.all_int_fast_path_set(out_ts, outs[i])  # type: ignore


@dataclass(frozen=True, slots=True)
class ThreeInputEvalOnceAndFastPathThunkLauncher(BaseThunkLauncher[BackendTensorT]):
    eval_fn0: Callable[[], tuple[int | slice]] = lambda: (0,)
    input_tensor0: RuntimeTensor[BackendTensorT] = None  # type: ignore
    input_tensor1: RuntimeTensor[BackendTensorT] = None  # type: ignore
    input_tensor2: RuntimeTensor[BackendTensorT] = None  # type: ignore

    def __post_init__(self) -> None:
        assert isinstance(self.op, top.TensorOp)
        assert len(self.input_tensors) == 3
        object.__setattr__(self, "eval_fn0", self.input_tensors[0][1])
        object.__setattr__(self, "input_tensor0", self.input_tensors[0][0])
        object.__setattr__(self, "input_tensor1", self.input_tensors[1][0])
        object.__setattr__(self, "input_tensor2", self.input_tensors[2][0])

    def launch(self) -> None:
        # Uses the all_int_fast_path to get data from the tensor store. Eval only once.
        point = self.eval_fn0()
        args = (
            self.input_tensor0.all_int_fast_path(point),
            self.input_tensor1.all_int_fast_path(point),
            self.input_tensor2.all_int_fast_path(point),
        )

        outs = self.thunk(args, self.thunk_exec_ctx)  # type: ignore

        out_ts = self.out_expr_eval()

        for i, out_tt in enumerate(self.output_tensors):
            out_tt.all_int_fast_path_set(out_ts, outs[i])  # type: ignore


@dataclass(frozen=True, slots=True)
class FourInputEvalOnceAndFastPathThunkLauncher(BaseThunkLauncher[BackendTensorT]):
    eval_fn0: Callable[[], tuple[int | slice]] = lambda: (0,)
    input_tensor0: RuntimeTensor[BackendTensorT] = None  # type: ignore
    input_tensor1: RuntimeTensor[BackendTensorT] = None  # type: ignore
    input_tensor2: RuntimeTensor[BackendTensorT] = None  # type: ignore
    input_tensor3: RuntimeTensor[BackendTensorT] = None  # type: ignore

    def __post_init__(self) -> None:
        assert isinstance(self.op, top.TensorOp)
        assert len(self.input_tensors) == 4
        object.__setattr__(self, "eval_fn0", self.input_tensors[0][1])
        object.__setattr__(self, "input_tensor0", self.input_tensors[0][0])
        object.__setattr__(self, "input_tensor1", self.input_tensors[1][0])
        object.__setattr__(self, "input_tensor2", self.input_tensors[2][0])
        object.__setattr__(self, "input_tensor3", self.input_tensors[3][0])

    def launch(self) -> None:
        # Uses the all_int_fast_path to get data from the tensor store. Eval only once.
        point = self.eval_fn0()
        args = (
            self.input_tensor0.all_int_fast_path(point),
            self.input_tensor1.all_int_fast_path(point),
            self.input_tensor2.all_int_fast_path(point),
            self.input_tensor3.all_int_fast_path(point),
        )

        outs = self.thunk(args, self.thunk_exec_ctx)  # type: ignore

        out_ts = self.out_expr_eval()

        for i, out_tt in enumerate(self.output_tensors):
            out_tt.all_int_fast_path_set(out_ts, outs[i])  # type: ignore


# NOTE: Trick: we can avoid calling the thunk. All we need is to set the output tensor to the
# chosen input tensor. The thunk is just an identity function.
@dataclass(frozen=True, slots=True)
class MergeThunkLauncher(BaseThunkLauncher[BackendTensorT]):
    loop_counters_and_bounds: Mapping[ie.Symbol, int]

    zipped_conds_and_exprs: Sequence[
        tuple[
            Callable[[], bool],
            Callable[[], tuple[int | slice, ...]],
        ]
    ] = ()
    out_tt: RuntimeTensor[BackendTensorT] = None  # type: ignore

    def __post_init__(self) -> None:
        assert isinstance(self.op, top.MergeOp)

        zipped = []

        for branch_cond, (t, eval_fn), copy_ in list(
            zip(self.input_conds, self.input_tensors, self.requires_copy, strict=False)
        ):
            assert not copy_, "Copies were disabled. Reenable."
            assert branch_cond is not None
            remapped_cond = branch_cond.remap(self.domain_map)
            remapped_cond.cache_codegenerated_eval(self.loop_counters_and_bounds)
            # if copy_:
            #    fn_ = lambda t=t, eval_fn=eval_fn: self.backend.copy(t[eval_fn()])
            # else:
            fn_ = lambda t=t, eval_fn=eval_fn: t[eval_fn()]
            zipped.append((remapped_cond.eval_fast, fn_))

        object.__setattr__(self, "zipped_conds_and_exprs", zipped)
        object.__setattr__(self, "out_tt", self.output_tensors[0])

    def launch(self) -> None:
        out_ts = self.out_expr_eval()
        for branch_cond_fn, expr_fn in self.zipped_conds_and_exprs:  # type: ignore
            if branch_cond_fn():
                self.out_tt.all_int_fast_path_set(out_ts, expr_fn())  # type: ignore
                return
        raise ValueError("No branch condition was satisfied")


@dataclass(frozen=True, slots=True)
class DebugMergeThunkLauncher(MergeThunkLauncher[BackendTensorT]):
    def launch(self) -> None:
        out_ts = self.out_expr_eval()
        for i, (branch_cond_fn, expr_fn) in enumerate(self.zipped_conds_and_exprs):  # type: ignore
            try:
                branch_res = branch_cond_fn()
            except Exception as e:
                dump_debug_info(self, e, merge_branch=i, cond=True)
                raise e
            if branch_res:
                try:
                    args_ = expr_fn()
                    # self._debug_checks(args_, input_=True)
                    self.out_tt.all_int_fast_path_set(out_ts, args_)  # type: ignore

                except Exception as e:
                    dump_debug_info(self, e, merge_branch=i)
                    raise e

                return
        raise ValueError(f"No branch condition was satisfied for op {self.op}")


@dataclass(frozen=True, slots=True)
class FastInitializationMergeThunkLauncher(BaseThunkLauncher[BackendTensorT]):
    """For the first execution of the thunk, we take the slow and safe path. After that, we
    we switch to a fast path that only executes the second branch without checking the condition.
    """

    _launch: Callable = lambda: None

    def __post_init__(self) -> None:
        assert isinstance(self.op, top.MergeOp)

        t0, eval_fn0 = self.input_tensors[0]
        needs_copy0 = self.requires_copy[0]
        assert not needs_copy0, "Copies were disabled. Reenable."
        assert self.input_conds[0] is not None
        cond0 = self.input_conds[0].remap(self.domain_map)
        cond0.cache_codegenerated_eval(self.loop_counters_and_bounds)
        # if needs_copy0:
        #    fn0 = lambda t0=t0, eval_fn0=eval_fn0: self.backend.copy(
        #        t0.all_int_fast_path(eval_fn0())
        #    )
        # else:
        fn0 = lambda t0=t0, eval_fn0=eval_fn0: t0.all_int_fast_path(eval_fn0())

        t1, eval_fn1 = self.input_tensors[1]
        needs_copy1 = self.requires_copy[1]
        assert not needs_copy1, "Copies were disabled. Reenable."
        assert self.input_conds[1] is not None
        out_t = self.output_tensors[0]
        # if needs_copy1:
        #    fn1 = lambda t1=t1, eval_fn1=eval_fn1: self.backend.copy(
        #        t1.all_int_fast_path(eval_fn1())
        #    )
        # else:
        fn1 = lambda t1=t1, eval_fn1=eval_fn1: t1.all_int_fast_path(eval_fn1())

        def _second_branch_fast_path() -> None:
            out_t.all_int_fast_path_set(self.out_expr_eval(), fn1())  # type: ignore

        def _slow_path() -> None:
            if cond0.eval_fast():
                out_t.all_int_fast_path_set(self.out_expr_eval(), fn0())  # type: ignore
            else:
                _second_branch_fast_path()
            object.__setattr__(self, "_launch", _second_branch_fast_path)

        object.__setattr__(self, "_launch", _slow_path)

    def launch(self) -> None:
        self._launch()


@dataclass(frozen=True, slots=True)
class SetSymbolValuesThunkLauncherWrapper(ThunkLauncher[BackendTensorT]):
    symbol_values: SymbolDict
    loop_counters_and_bounds: Mapping[ie.Symbol, int]
    domain_map: Mapping[ie.Symbol, ie.IndexValue]
    inner: ThunkLauncher[BackendTensorT]
    _domain_map_keys: list[ie.Symbol] = field(default_factory=lambda: [], init=False)

    def __post_init__(self) -> None:
        for v in self.domain_map.values():
            v.cache_codegenerated_eval(self.loop_counters_and_bounds)
        object.__setattr__(self, "_domain_map_keys", list(self.domain_map.keys()))

    def launch(self) -> None:
        # TODO eventually, if we have dynamic bounds, we will need to update those as well
        for k in self._domain_map_keys:
            self.symbol_values[k] = self.domain_map[k].eval_fast()

        self.inner.launch()


@dataclass(frozen=True, slots=True)
class BufferStoredTensorThunkLauncher(BaseThunkLauncher[BackendTensorT]):
    update_fns: list[Any] = field(default_factory=list, init=False)  # type: ignore
    example_brt: BlockRuntimeTensor = None  # type: ignore

    def __post_init__(self) -> None:
        dev_ = self.backend.to_backend_device_obj(self.device)
        bend_int = self.backend.to_backend_datatype(dtypes.default_int)

        arg_fns = []
        for i, ((t, fn), _) in enumerate(
            zip(self.input_tensors, self.input_index_exprs, strict=False)
        ):
            if self.op.domain.basis_expr.struct_eq(self.input_index_exprs[i]):
                arg_fns.append(lambda out_ts, rt=t, ev=fn: rt.all_int_fast_path(out_ts))
            elif self.input_index_exprs[i].is_point():
                # def _fn(out_ts, rt=t, ev=fn, idx_expr=idx_expr, i=i):
                #    print(f"self.op: {self.op}, input tensor: {rt.tensor_id}, i: {i}")
                #    print(f"out_ts: {out_ts}, idx_expr: {idx_expr}")
                #    print(f"eval fn: {ev()}")
                #    return rt.all_int_fast_path(ev())

                # arg_fns.append(_fn)
                arg_fns.append(lambda out_ts, rt=t, ev=fn: rt.all_int_fast_path(ev()))
            else:
                arg_fns.append(lambda out_ts, rt=t, ev=fn: rt[ev()])

        # used to store functions to get blocks
        buf_arg_fns = []
        # used to store functions to get indices
        idx_arg_fns = []
        buf_inputs = []
        inx_inputs = []
        # new_inputs = self.input_tensors  # will be extended

        # per output choose the correct update strategy
        upd = []  # type: ignore
        for pos, rt in enumerate(self.output_tensors):
            # block-stored outputs
            if pos in self.buffer_out_positions:
                assert isinstance(rt, PreallocRuntimeTensor)

                # (a) extra kernel inputs – backing buffer & slice index
                def _buf_fn(
                    out_ts: tuple[int, ...],
                    rt: PreallocRuntimeTensor = rt,
                ) -> BackendTensorT:
                    key, _ = rt.extract_write_key_and_indexes(out_ts)
                    return rt.get_backing_buffer(key)  # type: ignore

                # NOTE: JAX supports integers as traced inputs
                lift_idx = lambda x, bend_int=bend_int, dev=dev_: self.backend.fast_int_lift(
                    x,
                    dtype=bend_int,
                    device=dev,
                )
                if self.backend.get_backend_name() == DLBackendName.TORCH:

                    def _idx_fn(
                        out_ts: tuple[int, ...],
                        lift: Callable[[int], BackendTensorT] = lift_idx,
                        rt: PreallocRuntimeTensor = rt,
                    ) -> tuple[tuple[BackendTensorT, ...], ...]:
                        _, idxs = rt.extract_write_key_and_indexes(out_ts)

                        return tuple((lift(idx[0]),) for idx in idxs)
                else:

                    def _idx_fn(
                        out_ts: tuple[int, ...],
                        lift: Callable[[int], BackendTensorT] = lift_idx,  # Not used
                        rt: PreallocRuntimeTensor = rt,
                    ) -> tuple[tuple[BackendTensorT, ...], ...]:
                        _, idxs = rt.extract_write_key_and_indexes(out_ts)
                        return tuple((idx[0],) for idx in idxs)  # type: ignore

                buf_inputs.append((rt, _buf_fn))
                inx_inputs.append((rt, _idx_fn))
                buf_arg_fns.append(_buf_fn)
                idx_arg_fns.append(_idx_fn)

                def _upd(
                    out_ts: tuple[int, ...],
                    val: BackendTensorT,
                    rt: BlockRuntimeTensor = rt,
                ) -> None:
                    key, _ = rt.extract_write_key_and_indexes(out_ts)
                    rt.replace_backing_buffer(key, val)

                upd.append(_upd)
            # ---- point-stored (normal) outputs ----------------------------
            else:

                def _upd(
                    out_ts: tuple[int, ...],
                    val: BackendTensorT,
                    rt: PreallocRuntimeTensor = rt,
                ) -> None:
                    rt.all_int_fast_path_set(out_ts, val)

                upd.append(_upd)

        # arg_fns should be [*orginal_inputs_fns, *buf_fns, *idx_fns]
        arg_fns.extend(buf_arg_fns)  # type: ignore
        arg_fns.extend(idx_arg_fns)  # type: ignore
        # new_inputs.extend(buf_inputs)
        # new_inputs.extend(inx_inputs)

        # finalise frozen dataclass fields
        # object.__setattr__(self, "input_tensors", new_inputs)
        object.__setattr__(self, "arg_fns", arg_fns)
        object.__setattr__(self, "update_fns", upd)

    def launch(self) -> None:
        out_ts = self.out_expr_eval()

        args = tuple(fn(out_ts) for fn in self.arg_fns)  # no reflection
        outs = self.thunk(args, self.thunk_exec_ctx)  # type: ignore
        for val, write_fn in zip(outs, self.update_fns, strict=True):
            write_fn(out_ts, val)


class ThunkLauncherFactory(Generic[BackendTensorT]):
    def __init__(self, prep_ctx: ThunkLauncherFactoryCtx[BackendTensorT]) -> None:
        self.prep_ctx = prep_ctx

        self.counter_per_launcher_class: Counter[type[ThunkLauncher[BackendTensorT]]] = Counter()

    def get_launcher_percentages(self) -> dict[str, str]:
        total = sum(self.counter_per_launcher_class.values())
        return {
            k.__name__: f"{round(v / total * 100, 1)}%"
            for k, v in self.counter_per_launcher_class.items()
        }

    def emit_thunk_launcher(
        self, exec_sched_item: ExecInstruction
    ) -> ThunkLauncher[BackendTensorT]:
        prep_ctx = self.prep_ctx
        dg = prep_ctx.dg

        op_id = exec_sched_item.op_id
        op = dg.ops_by_id[op_id].op

        raw_thunk = prep_ctx.thunk_emitter.emit_thunk_for_op(
            op,
            ThunkEmissionCtx(
                exec_sched_item.domain_map,
                prep_ctx.loop_counters_and_bounds,
                prep_ctx.analysis_ctx.get_op_device(op),
                dg.static_bounds,
                prep_ctx.external_state_store,
                prep_ctx.backend,
                prep_ctx.compilation_ctx,
            ),
        )

        # Prepare input tensors
        input_tensors = self._prepare_input_tensors(op, exec_sched_item.domain_map)
        index_exprs = [dep.expr for _, dep in dg.get_flat_direct_dependencies(op)]
        conds = [dep.cond for _, dep in dg.get_flat_direct_dependencies(op)]
        # Prepare output tensors
        output_tensors = self._prepare_output_tensors(op)

        basis_expr = op.domain.basis_expr.remap(exec_sched_item.domain_map)
        basis_expr.cache_codegenerated_eval(prep_ctx.loop_counters_and_bounds)

        # Get expected shapes and dtypes
        expected_input_shapes, expected_input_dtypes, expected_input_devices = (
            self._get_expected_shapes_dtypes(op, True)
        )
        expected_output_shapes, expected_output_dtypes, _ = self._get_expected_shapes_dtypes(
            op, False
        )

        merge_copy_analysis = prep_ctx.analysis_ctx._needed_merge_copies

        # TODO this sort of is coupled to the merge copy analysis.
        # We shouldn't need to know about this format here.
        requires_copy: list[bool] = [False] * len(input_tensors)
        if merge_copy_analysis is not None and op_id in merge_copy_analysis:
            requires_copy = list(merge_copy_analysis[op_id])

        assert all(not rc for rc in requires_copy), "Copies should not be needed."

        # Determine the appropriate thunk class
        thunk_launcher_cls = self._determine_thunk_launcher_class(op, exec_sched_item, raw_thunk)

        self.counter_per_launcher_class[thunk_launcher_cls] += 1

        # Create symbol values
        symbol_values = self._create_symbol_values(dg)

        # Create thunk execution context
        thunk_exec_ctx = ThunkExecutionCtx(
            symbol_values=symbol_values,  # type: ignore
            exec_cfg=prep_ctx.exec_cfg,
            universe_basis_expr=dg.universe.basis_expr,
            external_state_store=(prep_ctx.external_state_store if op.is_stateful else None),
        )

        if prep_ctx.analysis_ctx._buffer_stored_output_tensor_positions is not None:
            buffer_out_positions = prep_ctx.analysis_ctx._buffer_stored_output_tensor_positions.get(
                op_id, ()
            )
        else:
            buffer_out_positions = ()

        # Instantiate and return the prepared thunk
        wrapped_thunk = thunk_launcher_cls(
            op,
            raw_thunk,
            expected_input_shapes,
            expected_input_dtypes,
            expected_input_devices,
            input_tensors,
            index_exprs,
            conds,
            requires_copy,
            basis_expr.eval_fast,
            expected_output_shapes,
            expected_output_dtypes,
            output_tensors,
            buffer_out_positions,
            device.from_(prep_ctx.analysis_ctx.get_op_device(op)),
            prep_ctx.backend,
            thunk_exec_ctx,
            exec_sched_item.domain_map,
            prep_ctx.loop_counters_and_bounds,
        )

        if prep_ctx.exec_cfg.executor_debug_mode or self._requires_symbol_setter(op):
            return SetSymbolValuesThunkLauncherWrapper(
                symbol_values,
                prep_ctx.loop_counters_and_bounds,
                exec_sched_item.domain_map,
                wrapped_thunk,
            )

        return wrapped_thunk

    def _requires_symbol_setter(self, op: top.TensorOp) -> bool:
        if isinstance(op, (top.RandOp, top.ReshapeOp, top.ExpandOp, top.PadOp, top.IndexSliceOp)):
            return op.is_dynamic()

        if isinstance(op, top.UDFOp):
            return op.desc.needs_symbol_setter

        if (
            isinstance(op, top.EvalSymbolOp)
            and not self.prep_ctx.exec_cfg.enable_eval_symbol_launcher
        ):
            return True

        if isinstance(op, top.ExecDataflowOp):
            return op.is_dynamic()

        return False

    def _prepare_input_tensors(
        self, op: top.TensorOp, domain_map: Mapping[ie.Symbol, ie.IntIndexValue]
    ) -> list[tuple[RuntimeTensor[BackendTensorT], Callable[[], tuple[int | slice, ...]]]]:
        all_dependencies = self.prep_ctx.dg.get_flat_direct_dependencies(op)
        log.debug("Preparing thunk for op %s with dependencies %s", op, all_dependencies)

        input_tensors_dict: dict[
            OpInId,
            tuple[
                RuntimeTensor[BackendTensorT],
                Callable[[], tuple[int | slice, ...]],
            ],
        ] = {}
        for dep_op, dep in all_dependencies:
            e = dep.expr.remap(domain_map)
            e.cache_codegenerated_eval(self.prep_ctx.loop_counters_and_bounds)

            input_tensors_dict[dep.sink_in_idx] = (
                self.prep_ctx.tensor_store[TensorId(dep_op.op_id, dep.src_out_idx)],
                e.eval_fast,
            )
        log.debug("Input tensors for op %s: %s", op, input_tensors_dict)
        return [input_tensors_dict[OpInId(i)] for i in range(len(input_tensors_dict.keys()))]

    def _prepare_output_tensors(self, op: top.TensorOp) -> list[RuntimeTensor[BackendTensorT]]:
        return [
            self.prep_ctx.tensor_store[TensorId(op.op_id, OpOutId(output))]
            for output in range(self.prep_ctx.dg.ops_by_id[op.op_id].num_outputs)
        ]

    def _get_expected_shapes_dtypes(
        self, op: top.TensorOp, is_input: bool
    ) -> tuple[list[Shape], list[DataType], list[DeviceGroup]]:
        shapes_dict: dict[Any, Shape]
        dtypes_dict: dict[Any, DataType]
        id_cls: type

        if is_input:
            shapes_dict = self.prep_ctx.dg.get_input_shapes(op)
            dtypes_dict = self.prep_ctx.dg.get_input_dtypes(op)
            id_cls = OpInId
        else:
            shapes_dict = self.prep_ctx.dg.get_output_shapes(op)
            dtypes_dict = self.prep_ctx.dg.get_output_dtypes(op)
            id_cls = OpOutId

        shapes = [shapes_dict[id_cls(i)] for i in range(len(shapes_dict))]  # type: ignore
        dtypes = [dtypes_dict[id_cls(i)] for i in range(len(shapes_dict))]  # type: ignore

        comp_ctx = CompilationCtx(
            self.prep_ctx.dg, self.prep_ctx.analysis_ctx, self.prep_ctx.exec_cfg
        )
        devices = comp_ctx.get_input_devices_list(op) if is_input else []

        return shapes, dtypes, devices

    def _determine_thunk_launcher_class(  # noqa: C901
        self, op: top.TensorOp, sched_item: ExecInstruction, thunk: Thunk[BackendTensorT]
    ) -> type[BaseThunkLauncher]:
        allow_custom_launchers = self.prep_ctx.exec_cfg.enable_custom_thunk_launchers

        all_dependencies = self.prep_ctx.dg.get_flat_direct_dependencies(op)
        num_inputs = len(all_dependencies)
        num_outputs = self.prep_ctx.dg.ops_by_id[op.op_id].num_outputs

        # all_inputs_basis = all(dep[1].expr.is_basis() for dep in all_dependencies)

        if self.prep_ctx.exec_cfg.executor_debug_mode:
            if isinstance(op, top.MergeOp):
                return DebugMergeThunkLauncher
            return DebugThunkLauncher

        if isinstance(op, top.MergeOp):
            if allow_custom_launchers and is_initialization_merge(self.prep_ctx.dg, op):
                # NOTE: We can only leverage the fast path if the domain map does not have consts.
                # In that case, that schedule item will likely represent the initialization
                # of the merge op.

                no_const_in_domain_map = all(
                    not isinstance(v, ie.ConstInt) for v in sched_item.domain_map.values()
                )
                all_inputs_point = all(dep[1].expr.is_point() for dep in all_dependencies)
                if (
                    all_inputs_point
                    and no_const_in_domain_map
                    and len(dict(sched_item.domain_map.items())) == 1
                ):
                    return FastInitializationMergeThunkLauncher
            return MergeThunkLauncher
        elif self.prep_ctx.exec_cfg.enable_inplace_write and has_buffer_stored_outputs(
            self.prep_ctx.dg, op, self.prep_ctx.analysis_ctx
        ):
            return BufferStoredTensorThunkLauncher
        elif (
            isinstance(op, top.EvalSymbolOp)
            and (not op.symbol.is_bound)
            and self.prep_ctx.exec_cfg.enable_eval_symbol_launcher
        ):
            return EvalSymbolThunkLauncher

        if not allow_custom_launchers:
            return BaseThunkLauncher

        all_exprs_equal = all(
            dep[1].expr.struct_eq(all_dependencies[0][1].expr) for dep in all_dependencies
        )
        all_inputs_point = all(dep[1].expr.is_point() for dep in all_dependencies)
        if num_inputs == 0:
            return NoInputsThunkLauncher
        elif num_outputs == 0:
            if num_inputs == 1 and all_inputs_point:
                return EvalOnceAndFastPathSingleInputNoOutputsThunkLauncher
            return NoOutputsThunkLauncher
        elif all_exprs_equal and all_inputs_point:
            if num_inputs == 1:
                return SingleInputEvalOnceAndFastPathThunkLauncher
            elif num_inputs == 2:
                return TwoInputEvalOnceAndFastPathThunkLauncher
            elif num_inputs == 3:
                return ThreeInputEvalOnceAndFastPathThunkLauncher
            elif num_inputs == 4:
                return FourInputEvalOnceAndFastPathThunkLauncher
            else:
                return EvalOnceAndFastPathThunkLauncher
        elif all_inputs_point:
            if num_inputs == 1:
                return SingleInputAllIntFastPathThunkLauncher
            elif num_inputs == 2:
                return TwoInputAllIntFastPathThunkLauncher
            elif num_inputs == 3:
                return ThreeInputAllIntFastPathThunkLauncher
            elif num_inputs == 4:
                return FourInputAllIntFastPathThunkLauncher
            else:
                return AllIntFastPathThunkLauncher
        else:
            return BaseThunkLauncher

    def _create_symbol_values(self, dg: PDG) -> SymbolDict:
        symbol_values = SymbolDict(len(dg.universe) * 2)

        keys = list(dg.universe.variables) + list(dg.universe.parameters)
        symbol_values.load_keys(keys)

        for k in dg.universe.variables:
            symbol_values[k] = -1
        for k in dg.universe.parameters:
            symbol_values[k] = -1
        for k, v in dg.static_bounds.items():
            symbol_values[k] = v

        return symbol_values
