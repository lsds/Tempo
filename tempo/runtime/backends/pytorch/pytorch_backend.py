from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import numpy as np
import optree

from tempo.core.analysis_ctx import AnalysisCtx
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import OpId
from tempo.core.dependence_graph import PDG
from tempo.core.device import DeviceGroup, DeviceLike, device
from tempo.core.dl_backend import DLBackend
from tempo.core.dl_backends import DLBackendName
from tempo.core.dtype import (
    INVERSE_TORCH_TO_TEMPO_DTYPES_DICT,
    TORCH_TO_TEMPO_DTYPES_DICT,
    DataType,
    dtypes,
)
from tempo.core.fast_object_pool import ObjectPool
from tempo.core.shape import StaticShape, StaticShapeLike
from tempo.core.thunk import Thunk
from tempo.core.thunk_emitter import ThunkEmitter
from tempo.core.utils import bytes_to_human_readable
from tempo.utils import logger
from tempo.utils.common import Timer

log = logger.get_logger(__name__)


SCALAR_SHAPE = ()
try:
    import torch
    import torch._dynamo.config
    import torch.utils
    import torch.utils.dlpack

    # def checks(i_: int, t: torch.Tensor):
    #    torch._check_is_size(i_)
    #    torch._constrain_as_size(i_, min=0, max=t.shape[0])

    # torch.fx.wrap("checks")

    # from torch.fx import symbolic_trace

    op_src = torch.ops.aten

    pinned_tensor_cache: dict[tuple[tuple[int, ...], torch.dtype], ObjectPool[torch.Tensor]] = {}

    class PyTorchBackend(DLBackend[torch.Tensor]):
        backend_cpu = torch.device("cpu")  # Set the class-level CPU device
        pinned_memory_enabled = False

        # Use the imported dtype dicts
        TORCH_TO_TEMPO_DTYPES_DICT = TORCH_TO_TEMPO_DTYPES_DICT
        INVERSE_TORCH_TO_TEMPO_DTYPES_DICT = INVERSE_TORCH_TO_TEMPO_DTYPES_DICT

        index_tensor: torch.Tensor = None  # type: ignore

        @staticmethod
        def configure(exec_cfg: ExecutionConfig) -> None:
            torch._dynamo.config.cache_size_limit = 512
            torch._dynamo.config.accumulated_cache_size_limit = 512

            # Disable all guards
            torch._dynamo.config.skip_fsdp_guards = True
            torch._dynamo.config.skip_nnmodule_hook_guards = True
            torch._dynamo.config.guard_nn_modules = False
            torch._dynamo.config.do_not_emit_runtime_asserts = True
            torch._dynamo.config.guard_nn_modules_using_dict_tags = False

            # No dynamic shapes
            torch._dynamo.config.dynamic_shapes = False
            torch._dynamo.config.assume_static_by_default = True

            # No backward
            torch._dynamo.config.capture_autograd_function = False

            ##NOTE: Appears to be needed to support narrow...
            torch._dynamo.config.capture_scalar_outputs = True

            # torch._dynamo.config.verbose = True
            # NOTE: useful for debugging
            # torch._dynamo.config.error_on_recompile = True

            # torch._dynamo.config.capture_func_transforms = False

            np.random.seed(exec_cfg.seed)
            torch.manual_seed(exec_cfg.seed)

            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            if exec_cfg.deterministic:
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            # torch.no_grad().__enter__()

            torch.set_grad_enabled(False)
            torch.inference_mode(True).__enter__()
            torch.no_grad().__enter__()

            dev = device.from_(exec_cfg.dev)

            dev_torch = PyTorchBackend.to_backend_device_obj(dev)

            ## NOTE: Perform a dummy allocation to ensure that the memory allocator is initialized
            buffer = torch.ones((1024,), dtype=torch.float32, device=dev_torch)
            del buffer
            # if (
            #    dev == device.gpu
            #    and os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", "0") == "0"
            # ):
            #    torch_dev = PyTorchBackend.to_backend_device_obj(exec_cfg.dev)
            #    total_mem = torch.cuda.get_device_properties(torch_dev).total_memory
            #    target_bytes = int(total_mem * 0.9)
            #    buffer = torch.ones(target_bytes // 4, dtype=torch.float32, device=torch_dev)
            #    del buffer

            PyTorchBackend.index_tensor = torch.arange(
                start=0, end=exec_cfg.M, dtype=torch.int64, device=dev_torch
            )

            PyTorchBackend.pinned_memory_enabled = exec_cfg.torch_pinned_memory_enabled

            # Do a pinned memory allocation so torch caches it
            if (
                exec_cfg.enable_swap
                and exec_cfg.torch_pinned_prealloc_size_bytes > 0
                and exec_cfg.dev != "cpu"
            ):
                import psutil

                # NOTE: Allocate at most 50% of the CPU memory
                ps_util_cpu_mem_max_bytes = psutil.virtual_memory().total * 0.80
                amount = min(
                    exec_cfg.torch_pinned_prealloc_size_bytes,
                    ps_util_cpu_mem_max_bytes,
                )
                log.info(
                    "Preallocating %s bytes of pinned=%s CPU memory. This should take ~%s seconds.",
                    bytes_to_human_readable(amount),
                    exec_cfg.torch_pinned_memory_enabled,
                    amount / (2**30) if exec_cfg.torch_pinned_memory_enabled else 0.1,
                )
                with Timer() as timer:
                    buffer = torch.empty(
                        int(amount),
                        dtype=torch.uint8,
                        device=torch.device("cpu"),
                        pin_memory=exec_cfg.torch_pinned_memory_enabled,
                    )
                log.info(
                    "Preallocated %s bytes of CPU memory. This took %s seconds.",
                    bytes_to_human_readable(amount),
                    timer.elapsed_s,
                )
                del buffer

        @staticmethod
        def sync() -> None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        @staticmethod
        def get_backend_name() -> DLBackendName:
            """Get the name of the backend."""
            return DLBackendName.TORCH

        @staticmethod
        def get_thunk_emitter_cls() -> type[ThunkEmitter]:
            from tempo.runtime.backends.pytorch.pytorch_thunk_emitter import PytorchThunkEmitter

            return PytorchThunkEmitter  # type:ignore

        @staticmethod
        def to_backend_device_obj(dev: DeviceLike) -> Any:
            dev = device.from_(dev)

            torch_dev_str = "cpu"
            if dev == device.cpu:
                torch_dev_str = "cpu"
            elif dev == device.gpu:
                torch_dev_str = "cuda:0"
            elif dev == device.fake_gpu:
                torch_dev_str = "cpu"
            else:
                raise ValueError(f"Unknown device {dev}")

            return torch.device(torch_dev_str)

        @staticmethod
        def device(tensor: torch.Tensor) -> Any:
            return tensor.device

        @staticmethod
        def to_device(tensor: torch.Tensor, dev: Any, **kwargs: Any) -> torch.Tensor:
            if dev.type == tensor.device.type:
                return tensor
            if dev.type == PyTorchBackend.backend_cpu.type:
                pool = PyTorchBackend._get_or_create_pool(tensor)
                buffer = pool.borrow()
                buffer = buffer.copy_(tensor, non_blocking=False)
                return buffer
            else:
                temp = tensor.to(dev, non_blocking=True)

                pool = PyTorchBackend._get_or_create_pool(tensor)
                pool.recycle(tensor)

                return temp

        @staticmethod
        def _get_or_create_pool(tensor: torch.Tensor) -> ObjectPool[torch.Tensor]:
            key = (tensor.shape, tensor.dtype)
            if key in pinned_tensor_cache:
                return pinned_tensor_cache[key]

            pool = ObjectPool(
                lambda: torch.empty(
                    tensor.shape,
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    device=PyTorchBackend.backend_cpu,
                    pin_memory=PyTorchBackend.pinned_memory_enabled,
                )
            )
            pinned_tensor_cache[key] = pool
            return pool

        @staticmethod
        def to_backend_datatype(dtype: DataType) -> Any:
            return PyTorchBackend.INVERSE_TORCH_TO_TEMPO_DTYPES_DICT[dtype]

        @staticmethod
        def to_tpo_dtype(backend_dtype: Any) -> DataType:
            """Convert a PyTorch dtype to a Tempo dtype."""
            return PyTorchBackend.TORCH_TO_TEMPO_DTYPES_DICT[backend_dtype]

        @staticmethod
        def cast_backend_dtype(tensor: torch.Tensor, dtype: Any) -> torch.Tensor:
            return tensor.to(dtype)

        @staticmethod
        def to_backend_shape(shape: StaticShapeLike) -> Any:
            shape = StaticShape.from_(shape)
            return shape._shape

        # @staticmethod
        # def to_dlpack(tensor: BackendTensorT) -> DLPackObject:
        #    return torch.utils.dlpack.
        @staticmethod
        def from_dlpack(ext_tensor: Any) -> torch.Tensor:
            return torch.from_dlpack(ext_tensor)

        @staticmethod
        def zeros_tensor(shape: tuple[int, ...], dtype: Any, dev: Any) -> torch.Tensor:
            return torch.zeros(size=shape, dtype=dtype, device=dev)

        @staticmethod
        def ones_tensor(shape: tuple[int, ...], dtype: Any, dev: Any) -> torch.Tensor:
            return torch.ones(size=shape, dtype=dtype, device=dev)

        @staticmethod
        def full_tensor(
            fill_value: Any,
            shape: StaticShapeLike | None = None,
            dtype: Any | None = None,
            device: Any | None = None,
        ) -> torch.Tensor:
            return torch.full(size=shape, fill_value=fill_value, dtype=dtype, device=device)

        @staticmethod
        def fast_int_lift(
            fill_value: int,
            dtype: Any | None = None,
            device: Any | None = None,
        ) -> torch.Tensor:
            # return torch.full(size=(), fill_value=fill_value, dtype=dtype, device=device)
            # TODO: preallocate an arrange tensor of size M, return a view at index fill_value?
            # tensor = torch.empty((), dtype=dtype, device=device)
            # tensor.fill_(fill_value)
            # return tensor
            return PyTorchBackend.index_tensor[fill_value]

        @staticmethod
        def lift_tensor(
            data: Any,
            shape: StaticShapeLike | None = None,
            dtype: Any | None = None,
            device: Any | None = None,
        ) -> torch.Tensor:
            # If already a torch tensor, just move/cast as needed
            # if isinstance(data, torch.Tensor):
            #    x = data
            #    if dtype is not None and x.dtype != dtype:
            #        x = x.to(dtype)
            #    if device is not None and x.device != device:
            #        x = x.to(device)
            #    if shape is not None and tuple(x.shape) != tuple(shape):
            #        x = x.expand(shape)
            #    return x
            assert device is not None, "Device is required"

            # Convert to numpy array for uniform handling
            np_data = np.array(data)
            target_shape = shape if shape is not None else np_data.shape
            from tempo.runtime.backends.pytorch.pytorch_backend import PyTorchBackend

            torch_dtype = (
                dtype
                if dtype is not None
                else PyTorchBackend.to_backend_datatype(dtypes.implied(np_data))
            )
            tensor = torch.empty(target_shape, dtype=torch_dtype, device=device)

            if np_data.shape == () or np_data.size == 1:
                tensor.fill_(np_data.item())
            else:
                # TODO: I think this will be a blocking copy. To make it non-blocking, we need to
                # use a pinned tensor. If we try that, the torch-numpy data will need to be copied
                # to a pinned CPU tensor. Sigh...
                # Broadcast/copy data
                tensor.copy_(torch.from_numpy(np_data).expand(target_shape))
            return tensor

        @staticmethod
        def unbind(tensor: torch.Tensor, axis: int) -> Sequence[torch.Tensor]:
            return op_src.unbind(tensor, axis)  # type:ignore

        @staticmethod
        def stack(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
            return op_src.stack(tensors, 0)  # type:ignore

        @staticmethod
        def get_stack_fn(
            tensors: Sequence[torch.Tensor],
        ) -> Callable[[Sequence[torch.Tensor]], torch.Tensor]:
            return partial(op_src.stack, dim=0)  # type:ignore

        @staticmethod
        def reshape(tensor: torch.Tensor, shape: StaticShapeLike) -> torch.Tensor:
            return op_src.reshape(tensor, shape)  # type:ignore

        @staticmethod
        def permute(tensor: torch.Tensor, axes: Sequence[int]) -> torch.Tensor:
            return op_src.permute(tensor, axes)  # type:ignore

        @staticmethod
        def to_numpy(tensor: torch.Tensor) -> np.ndarray:
            return tensor.detach().cpu().numpy()

        @staticmethod
        def get_inplace_set_fn(
            tensor: torch.Tensor,
            item: Sequence[int | slice],
            value: torch.Tensor,
            traceable: bool = False,
        ) -> Callable[[torch.Tensor, Sequence[int | slice], torch.Tensor], torch.Tensor]:
            if traceable:

                def fn_(t: torch.Tensor, i: Sequence[int | slice], v: torch.Tensor) -> torch.Tensor:
                    return torch.index_put_(t, i, v)
            else:

                def fn_(t: torch.Tensor, i: Sequence[int | slice], v: torch.Tensor) -> torch.Tensor:
                    t[i] = v
                    return t

            return fn_

        @staticmethod
        def copy(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.clone()

        @staticmethod
        def _symbolically_trace_tempo_thunk(
            execution_func: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]],
            inputs: Sequence[torch.Tensor],
        ) -> Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]:
            from torch.fx import symbolic_trace
            from torch.fx._symbolic_trace import PH

            any_nested_args = any(isinstance(x, tuple) for x in inputs)
            if any_nested_args:
                tracers = (optree.tree_map(lambda x: PH, inputs),)
                traced_func_ = symbolic_trace(execution_func, concrete_args=tracers)
            else:
                traced_func_ = symbolic_trace(execution_func)

            # print(f"traced_func_.code={traced_func_.code}")
            # for attr in dir(traced_func_):
            #    if attr.startswith("_tensor_constant"):
            #        attr_val = getattr(traced_func_, attr)
            #        if isinstance(attr_val, int) or (
            #            isinstance(attr_val, torch.Tensor) and attr_val.numel() == 1
            #        ):
            #            print(f"attr={attr} = {attr_val}")
            traced_func = traced_func_.forward  # type: ignore

            return traced_func  # type: ignore

        @staticmethod
        def trace_codegen_thunk_jit_trace(
            execution_func: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]],
            op_id: OpId,
            exec_cfg: ExecutionConfig,
            example_inputs: Sequence[torch.Tensor],
            donatable_args: Sequence[int],
        ) -> Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]:
            # execution_func = lambda ins: execution_func((i.contiguous() for i in ins))
            # execution_func = lambda ins: tuple(o.contiguous() for o in execution_func_(ins))

            if len(example_inputs) == 0:
                # If it has no inputs, we cannot trace it using tracer objects.
                # Symbolic tracing instead
                traced_func = execution_func
            else:
                execution_func = PyTorchBackend._symbolically_trace_tempo_thunk(
                    execution_func, example_inputs
                )

                traced_func = torch.jit.trace(
                    execution_func,
                    (example_inputs,),
                    strict=True,
                    check_trace=False,
                    optimize=True,
                    _store_inputs=False,
                    _force_outplace=True,  # default is False
                )
                # try:
                #    traced_func = torch.jit.optimize_for_inference(traced_func)
                # except Exception as e:
                #    log.error("Failed to optimize for inference, continuing...")
                #    # print the exception
                #    print(e)

                # print("------------------------------TRACED STRT -----------------------------")
                # print(traced_func.code)
                # print("-----------------------------TRACED END ------------------------------")

            return traced_func  # type: ignore

        @staticmethod
        def trace_codegen_thunk_symbolic_trace_only(
            execution_func: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]],
            op_id: OpId,
            exec_cfg: ExecutionConfig,
            inputs: Sequence[torch.Tensor],
            donatable_args: Sequence[int],
        ) -> Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]:
            # NOTE: Unlike torch.jit.trace, torch.fx.symbolic_trace performs actual tracing
            # of the function, and does not analyse any other code. This means that as a result,
            # we get only the operations that are executed in the function, and not any other
            # stuff.
            return PyTorchBackend._symbolically_trace_tempo_thunk(execution_func, inputs)

        @staticmethod
        def trace_codegen_thunk_torch_compile(
            execution_func: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]],
            op_id: OpId,
            exec_cfg: ExecutionConfig,
            inputs: Sequence[torch.Tensor],
            donatable_args: Sequence[int],
        ) -> Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]:
            # NOTE: Can only set mode or options. Modes set these options:
            # DEFAULTS:
            # {
            # "default": {},
            # enable cudagraphs
            # "reduce-overhead": {
            #    "triton.cudagraphs": True,
            # },
            # enable max-autotune
            # "max-autotune-no-cudagraphs": {
            #    "max_autotune": True,
            #    "coordinate_descent_tuning": True,
            # },
            # enable max-autotune
            # enable cudagraphs
            # "max-autotune": {
            #    "max_autotune": True,
            #    "triton.cudagraphs": True,
            #    "coordinate_descent_tuning": True,
            # },
            # }

            # NOTE: Torch.compile has trouble with no inputs
            if len(inputs) == 0:
                return execution_func

            traced_func = PyTorchBackend._symbolically_trace_tempo_thunk(execution_func, inputs)

            # if "narrow" in traced_func_.code:
            #    log.info("Using JIT trace due to narrow in code for op %s", op_id)
            #    return PyTorchBackend.trace_codegen_thunk_jit_trace(
            #        execution_func, op_id, exec_cfg, inputs, donatable_args
            #    )

            params = {
                "model": traced_func,
                "fullgraph": True,
                "dynamic": False,
                "backend": "inductor",  # tvm
                # "mode": "max-autotune-no-cudagraphs",
                # "mode": "max-autotune",
                # "mode": "reduce-overhead",
                # "mode": "default",
                # "backend": "openxla", #openxla_eval
            }

            opts = {
                "max_autotune": False,
                # "decompose_mem_bound_mm": True,  # decompose some memory bound matmul/bmm to mul
                "coordinate_descent_tuning": True,
                "triton.cudagraphs": False,  # NOTE: When on, incorrect results and worse perf..
                # Disable cudagraph trees
                "triton.cudagraph_trees": False,
                # "triton.tile_reductions": True,
                # "triton.prefer_nd_tiling": True,
                #'max_fusion_size': 32
                # "shape_padding": False,
                # "epilogue_fusion": True,
                # "permute_fusion": True,
                # assume_aligned_inputs means that we assume that inputs will be aligned;
                # we generate code using this assumption, and clone tensors before use if they
                # aren't aligned. In the common case, most inputs will be aligned.
                # "assume_aligned_inputs": True,
                # "padding_stride_threshold": 320,
                ## We need to lower memory usage
                # "memory_planning": False,
            }

            params["options"] = opts

            exec_func = torch.compile(**params)

            # if "mode" in params and (
            #    params["mode"] == "reduce-overhead" or params["mode"] == "max-autotune"
            # ):

            #    def thunk_and_copy(inputs_: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
            #        #inputs__ = tuple(x.contiguous() for x in inputs_)
            #        #inputs__ = tuple(x.clone() for x in inputs_ if not type(x) == int)
            #        #torch.compiler.cudagraph_mark_step_begin()
            #        inputs__ = inputs_
            #        outs = exec_func(inputs__)
            #        return tuple(x.clone() for x in outs)
            #        #return outs

            #    return thunk_and_copy  # type: ignore

            return exec_func  # type: ignore

        @staticmethod
        def trace_codegen_thunk_torch_export(
            execution_func: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]],
            op_id: OpId,
            exec_cfg: ExecutionConfig,
            inputs: Sequence[torch.Tensor],
            donatable_args: Sequence[int],
        ) -> Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]:
            # NOTE: Unlike torch.jit.trace, torch.fx.symbolic_trace performs actual tracing
            # of the function, and does not analyse any other code. This means that as a result,
            # we get only the operations that are executed in the function, and not any other
            # stuff.

            num_args = len(inputs)
            if num_args == 0:
                return execution_func
            from torch.fx import symbolic_trace

            # wrapped = PyTorchBackend.get_wrapped_func(num_args, execution_func)
            traced_func = symbolic_trace(execution_func)

            # print("------------------------------TRACED STRT -----------------------------")
            # traced_func.print_readable()
            # print("-----------------------------TRACED END ------------------------------")

            # export_program = export(traced_func, inputs)
            import torch._inductor.config as config
            from torch._export import aot_compile

            config.cpp_wrapper = False
            config.b2b_gemm_pass = True
            config.aggressive_fusion = False

            config.triton.cudagraphs = True
            config.triton.cudagraph_trees = False

            # NOTE: passing cpp_wrapper=False seems to not work. We still get a warning saying
            # that cudagraphs are skipped due to cpp_wrapper=True.
            path = aot_compile(
                traced_func,
                (inputs,),
                remove_runtime_assertions=True,
                options={"cpp_wrapper": False},
            )
            # print(f"AOT compiled to {path}")
            # dev = PyTorchBackend.to_backend_device_obj(device.from_(exec_cfg.dev))
            # compiled_program = aot_load(path, "cuda")

            if exec_cfg.dev == "cpu":
                runner = torch._C._aoti.AOTIModelContainerRunnerCpu(path, 1)
            elif exec_cfg.dev == "gpu" or exec_cfg.dev.startswith("cuda"):
                runner = torch._C._aoti.AOTIModelContainerRunnerCuda(path, 1, "cuda")

            return runner.run  # type: ignore

        @staticmethod
        def trace_codegen_thunk(
            execution_func: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]],
            op_id: OpId,
            dev: DeviceGroup,
            exec_cfg: ExecutionConfig,
            inputs: Sequence[torch.Tensor],
            donatable_args: Sequence[int],
            analysis_ctx: AnalysisCtx,
            parent_graph: PDG,
        ) -> Thunk[torch.Tensor]:
            # torch_dev = PyTorchBackend.to_backend_device_obj(dev)

            # def wrapped_fun(inputs_: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
            #    outs = execution_func(inputs_)
            #    # Guarantee that the outputs we get are contiguous in memory
            #    # return tuple(o.contiguous().to(torch_dev) for o in outs)
            #    #return tuple(o.to(torch_dev) for o in outs)
            #    return tuple(o.contiguous() for o in outs)

            wrapped_fun = execution_func

            with torch.no_grad():
                # NOTE: we have to do this because torch offers no way to set the device
                # for compilation.
                # options: jit, compile, export, symbolic_trace
                if exec_cfg.torch_compilation_backend == "jit":
                    fun = PyTorchBackend.trace_codegen_thunk_jit_trace(
                        wrapped_fun, op_id, exec_cfg, inputs, donatable_args
                    )
                elif exec_cfg.torch_compilation_backend == "compile":
                    fun = PyTorchBackend.trace_codegen_thunk_torch_compile(
                        wrapped_fun, op_id, exec_cfg, inputs, donatable_args
                    )
                elif exec_cfg.torch_compilation_backend == "export":
                    fun = PyTorchBackend.trace_codegen_thunk_torch_export(
                        wrapped_fun, op_id, exec_cfg, inputs, donatable_args
                    )
                elif exec_cfg.torch_compilation_backend == "symbolic_trace":
                    fun = PyTorchBackend.trace_codegen_thunk_symbolic_trace_only(
                        wrapped_fun, op_id, exec_cfg, inputs, donatable_args
                    )
                elif exec_cfg.torch_compilation_backend == "no_compile":
                    fun = wrapped_fun
                else:
                    raise ValueError(
                        f"Unknown subbackend codegen: {exec_cfg.torch_compilation_backend}"
                    )
            # log.info(
            #   "Warming up the kernel for %s with input types %s", op_id, [type(i) for i in inputs]
            # )
            # fun(inputs)
            # log.info("Kernel warmed up for %s", op_id)
            fn = lambda ins, ctx: fun(ins)  # type: ignore

            return fn

    DLBackend.register_backend(DLBackendName.TORCH, PyTorchBackend)


except ImportError as e:
    raise ImportError(
        "PyTorch is not installed. Please install PyTorch to use the PyTorch backend."
    ) from e
except Exception as e:
    raise e
    raise e
