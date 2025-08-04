import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch

from tempo.core.analysis_ctx import AnalysisCtx
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import OpId, PDGId
from tempo.core.dependence_graph import PDG
from tempo.core.device import DeviceGroup, DeviceLike, device
from tempo.core.dtype import INVERSE_JAX_TO_TEMPO_DTYPES_DICT, JAX_TO_TEMPO_DTYPES_DICT, DataType
from tempo.core.shape import StaticShape
from tempo.core.thunk import Thunk
from tempo.runtime.backends.backend import DLBackend, DLBackendName
from tempo.runtime.thunk_emitter import ThunkEmitter
from tempo.utils import logger


def strip_hlo_constants(hlo_text: str, summarize: bool = True) -> str:
    """
    Remove or summarize large hex constant blocks from HLO text.

    Args:
        hlo_text: The full HLO string from `as_text()`.
        summarize: If True, replace hex with a short summary. If False, remove completely.

    Returns:
        Cleaned HLO string.
    """
    # Regex matches constants like:
    #   constant.123 = f32[...] constant({... huge hex ...})
    pattern = re.compile(
        r"(constant\.[^\s=]+ *= *[^ ]+ *\[[^\]]*\] *constant\()([0-9A-F]+)(\))", re.MULTILINE
    )

    if summarize:

        def repl(match: re.Match) -> str:
            hex_data = match.group(2)
            return f"{match.group(1)}<hex_blob[{len(hex_data)} chars]>{match.group(3)}"
    else:

        def repl(match: re.Match) -> str:
            return ""

    return pattern.sub(repl, hlo_text)


log = logger.get_logger(__name__)

SCALAR_SHAPE = ()
try:
    ## Suppress specific deprecation warnings from JAX and related libraries

    import warnings

    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"jax.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"jax.*",
    )

    warnings.filterwarnings("ignore")
    os.environ["JAX_LOG_LEVEL"] = "0"

    import jax

    def print_jax_compiled(
        op_id: OpId,
        thunk_: jax.stages.Compiled,
        with_cost_analysis: bool = False,
        with_memory_analysis: bool = False,
    ) -> None:
        print(f"======= Thunk ({op_id}) IR =======")
        print(strip_hlo_constants(thunk_.as_text()))
        if with_cost_analysis:
            print(thunk_.cost_analysis())
        if with_memory_analysis:
            print(thunk_.memory_analysis())
        print("================================")

    # from jax import Array
    from jax import numpy as jnp
    from jax._src.dlpack import from_dlpack as jax_from_dlpack
    from jax.dtypes import canonicalize_dtype

    stack_cache: Dict[
        Tuple[Any, Tuple[int, ...], int], Callable[[List[jnp.ndarray]], jnp.ndarray]
    ] = {}
    if jax.__version__ < "0.4.18":
        from jax.lib import xla_bridge, xla_client

        def create_stack_computation(
            dtype: jnp.dtype, shape: tuple[int, ...], num_tensors: int, backend: str
        ) -> Callable[[List[jnp.ndarray]], jnp.ndarray]:
            """Build the XLA computation for stacking tensors."""
            # Get the XLA builder
            builder = xla_client.XlaBuilder("stack_tensors")

            # Convert JAX dtype to the canonical dtype (e.g., np.dtype to dtype understood by XLA)
            canonical_dtype = canonicalize_dtype(dtype)

            # Create placeholders for the inputs
            inputs = [
                xla_client.ops.Parameter(
                    builder, i, xla_client.Shape.array_shape(canonical_dtype, shape)
                )
                for i in range(num_tensors)
            ]

            ## Unsqueezing tensors (adding a new axis at position 0)
            unsqueezed = [
                xla_client.ops.BroadcastInDim(
                    i,
                    shape=(1, *shape),
                    broadcast_dimensions=tuple(range(1, len(shape) + 1)),
                )
                for i in inputs
            ]

            # Concatenating along the new axis
            _ = xla_client.ops.ConcatInDim(builder, unsqueezed, dimension=0)

            # Build the computation
            built = xla_client._xla.mlir.xla_computation_to_mlir_module(builder.build())

            comp_backend = xla_bridge.get_backend(backend)
            compiled_computation = comp_backend.compile(built)
            return lambda xs: compiled_computation.execute(xs)[0]  # type: ignore
    else:
        from jax._src.lib import xla_client
        from jax.lib import xla_bridge  # , xla_client

        def stack_fn(args: List[jnp.ndarray]) -> jnp.ndarray:
            unsqueezed = [jnp.expand_dims(x, axis=0) for x in args]
            return jnp.concatenate(unsqueezed, axis=0)

        def create_stack_computation(
            dtype: jnp.dtype, shape: tuple[int, ...], num_tensors: int, backend: str
        ) -> Callable[[List[jnp.ndarray]], jnp.ndarray]:
            # example_inputs = [jnp.zeros(shape, dtype) for _ in range(num_tensors)]
            jitted = jax.jit(stack_fn, backend=backend)

            return jitted  # type: ignore

            # xla_comp = jitted.lower(example_inputs)
            ## OLD == #TODO: active one
            # compiled_call = xla_comp.compile()
            # fn = compiled_call  # type: ignore
            ## NEW ==
            ##compiled_call = xla_comp.compile()._executable.call
            ##fn = lambda xs: compiled_call(*xs)[0]  # type: ignore

            # return fn

    def get_or_create_compiled_stack(
        tensors: List[jnp.ndarray],
    ) -> Callable[[List[jnp.ndarray]], jnp.ndarray]:
        t = tensors[0]
        shape = t.shape
        dtype = t.dtype

        num_tensors = len(tensors)

        key = (dtype, shape, num_tensors)
        compiled_computation = stack_cache.get(key, None)
        if compiled_computation is None:
            for dev in t.devices():
                backend = dev.platform
                break  # type: ignore
            compiled_computation = create_stack_computation(dtype, shape, num_tensors, backend)
            stack_cache[key] = compiled_computation

            # Warm up
            compiled_computation(tensors)

        return compiled_computation  # type: ignore

    unbind_impl = jax.jit(
        lambda x, axis: tuple(jnp.moveaxis(x, axis, 0)),
        static_argnums=(1,),
        donate_argnums=(0,),
        inline=True,
    )
    jit_kernel_cache: Dict[Tuple[PDGId, OpId], Thunk[jnp.ndarray]] = {}

    reshape_impl = jax.jit(jnp.reshape, static_argnums=(1,), inline=True)

    def update_in_place_int(
        x: jnp.ndarray,
        idx: Union[int, slice, Sequence[Union[int, slice]]],
        value: Union[float, jnp.ndarray],
    ) -> jnp.ndarray:
        """Updates the tensor `x` at index `idx` with the new `value` in a JAX-compatible way.

        Args:
            x (jnp.ndarray): The input tensor to modify.
            idx (Union[int, slice, Sequence[Union[int, slice]]]): The index or indices
                (int, slice, or sequence of int/slice) to update.
            value (Union[float, jnp.ndarray]): The value to set at the given index or indices.

        Returns:
            jnp.ndarray: The updated tensor.

        """
        # print(f"Presumably tracing update_in_place with {x.shape=}, {idx=}, {value.shape=}")
        return x.at[idx].set(value, indices_are_sorted=True, unique_indices=True)

    def update_in_place_slice(
        x: jnp.ndarray,
        idx: Union[int, slice, Sequence[Union[int, slice]]],
        value: Union[float, jnp.ndarray],
    ) -> jnp.ndarray:
        return x.at[idx[0] : idx[1]].set(  # type: ignore
            value, indices_are_sorted=True, unique_indices=True
        )

    permute_impl = jax.jit(jnp.transpose, static_argnums=(1,), donate_argnums=(0,), inline=True)

    TORCH_TO_JAX_DEVICE_DICT = {
        torch.device("cpu"): jax.devices("cpu")[0],
    }
    try:
        TORCH_TO_JAX_DEVICE_DICT[torch.device("cuda:0")] = jax.devices("gpu")[0]
        TORCH_TO_JAX_DEVICE_DICT[torch.device("cuda")] = jax.devices("gpu")[0]
    except Exception:
        pass
    JAX_TO_TORCH_DEVICE_DICT = {v: k for k, v in TORCH_TO_JAX_DEVICE_DICT.items()}
    torch_cpu = torch.device("cpu")

    from tempo.runtime.backends.pytorch.pytorch_backend import PyTorchBackend

    pinned_memory_enabled = False

    class JaxBackend(DLBackend[jnp.ndarray]):
        backend_cpu = jax.devices("cpu")[0]  # Set the class-level CPU device

        # Use the imported dtype dicts
        JAX_TO_TEMPO_DTYPES_DICT = JAX_TO_TEMPO_DTYPES_DICT
        INVERSE_JAX_TO_TEMPO_DTYPES_DICT = INVERSE_JAX_TO_TEMPO_DTYPES_DICT

        @staticmethod
        def configure(exec_cfg: ExecutionConfig) -> None:
            JaxBackend.pinned_memory_enabled = exec_cfg.torch_pinned_memory_enabled

            # torch._dynamo.config.cache_size_limit = 64
            np.random.seed(exec_cfg.seed)
            try:
                jax.config.update("jax_compiler_enable_remat_pass", False)
            except Exception:
                pass
            try:
                jax.config.update("enable_remat_opt_pass", False)
            except Exception:
                pass
            if exec_cfg.enable_x64:
                jax.config.update("jax_enable_x64", True)
            # The following flag removes extra copies introduced by DUS (dynamic update slice) when
            # used in conjunction with custom NVIDIA kernels (like cuBLAS for GEMMs).
            # --xla_gpu_enable_custom_fusions=true
            # --xla_gpu_enable_address_computation_fusion=true

            # jax.config.update("xla_gpu_enable_custom_fusions", True)
            # jax.config.update("xla_gpu_enable_address_computation_fusion", True)
            # jax.config.update("xla_gpu_enable_triton_gemm", True)
            # jax.config.update("xla_gpu_enable_triton_gemm_any", True)
            # jax.config.update("xla_gpu_enable_triton_softmax_fusion", True)

            # import absl.logging
            # absl.logging.set_verbosity(absl.logging.FATAL)

            # import warnings
            # warnings.filterwarnings("once", category=FutureWarning)
            # warnings.filterwarnings("once", category=DeprecationWarning)
            # warnings.filterwarnings("once", category=UserWarning)
            # warnings.filterwarnings("ignore")

            # NOTE: disables UserWarnings about x64 ints. Unfortunately, does not work for tree_map
            import logging

            logger = logging.getLogger("jax._src.xla_bridge")
            logger.setLevel(logging.ERROR)

            import warnings

            # Suppress specific deprecation warnings from JAX and related libraries
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                module=r"brax.*",  # Regex pattern to match modules starting with 'brax'
            )
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            if exec_cfg.torch_pinned_memory_enabled:
                PyTorchBackend.configure(exec_cfg)

        @staticmethod
        def get_backend_name() -> DLBackendName:
            """Get the name of the backend."""
            return DLBackendName.JAX

        @staticmethod
        def sync() -> None:
            # See: https://github.com/jax-ml/jax/issues/4335
            (jax.device_put(0.0) + 0).block_until_ready()

        @staticmethod
        def unbind(tensor: jnp.ndarray, axis: int) -> Sequence[jnp.ndarray]:
            return unbind_impl(tensor, axis)  # type: ignore

        @staticmethod
        def stack(tensors: Sequence[jnp.ndarray]) -> jnp.ndarray:
            return get_or_create_compiled_stack(tensors)(tensors)  # type: ignore

        @staticmethod
        def get_stack_fn(
            tensors: Sequence[jnp.ndarray],
        ) -> Callable[[Sequence[jnp.ndarray]], jnp.ndarray]:
            return get_or_create_compiled_stack(tensors)  # type: ignore
            # return stack_impl  # type: ignore

        @staticmethod
        def reshape(tensor: jnp.ndarray, shape: Tuple[int, ...]) -> jnp.ndarray:
            return reshape_impl(tensor, shape)

        # @staticmethod
        # def inplace_get(tensor: jnp.ndarray, item: Sequence[Union[int, slice]]) -> jnp.ndarray:

        #    #import torch
        #    #import torch.utils.dlpack as tdl
        #    #torch_from_dlpack = tdl.from_dlpack
        #    #jax_from_dlpack = jax.dlpack.from_dlpack
        #    #inplace_view = lambda x, idx: jax_from_dlpack(torch_from_dlpack(x)[idx])
        #    return tensor[item]

        @staticmethod
        def get_inplace_set_fn(
            tensor: jnp.ndarray,
            item: Sequence[Union[int, slice]],
            value: jnp.ndarray,
            traceable: bool = False,
        ) -> Callable[[jnp.ndarray, Sequence[Union[int, slice]], jnp.ndarray], jnp.ndarray]:
            for dev in tensor.devices():
                backend = dev.platform
                break  # type: ignore

            inplace_set_impl = jax.jit(
                update_in_place_int,
                donate_argnums=(0,),
                static_argnums=(),
                backend=backend,
                inline=True,
            )

            if item is not None and value is not None:
                # warmup
                inplace_set_impl(tensor, item, value)

            return inplace_set_impl  # type: ignore

        @staticmethod
        def permute(tensor: jnp.ndarray, axes: Tuple[int, ...]) -> jnp.ndarray:
            return permute_impl(tensor, axes)  # type:ignore

        @staticmethod
        def get_thunk_emitter_cls() -> Type[ThunkEmitter]:
            from tempo.runtime.backends.jax.jax_thunk_emitter import JaxThunkEmitter

            return JaxThunkEmitter  # type: ignore

        @staticmethod
        def device(tensor: jnp.ndarray) -> Any:
            for t_dev in tensor.devices():  # noqa: B007
                break
            return t_dev

        @staticmethod
        def to_device(tensor: jnp.ndarray, dev: Any, compiling: bool = False) -> jnp.ndarray:
            if JaxBackend.pinned_memory_enabled and not compiling:
                if dev == JaxBackend.backend_cpu:
                    for t_dev in tensor.devices():  # noqa: B007
                        break
                    if t_dev == dev:
                        return tensor
                    return PyTorchBackend.to_device(PyTorchBackend.from_dlpack(tensor), torch_cpu)  # type: ignore
                else:
                    if type(tensor) is torch.Tensor:
                        torch_dev = JAX_TO_TORCH_DEVICE_DICT[dev]
                        # NOTE: Host to CPU here may block... it's complicated and has to do with
                        # dlpack's internal signaling.
                        on_dev = PyTorchBackend.to_device(
                            tensor, torch_dev
                        )  # is already a torch tensor
                        return JaxBackend.from_dlpack(on_dev)
                    else:
                        # NOTE: Used during initial param loading to GPU
                        for t_dev in tensor.devices():  # noqa: B007
                            break
                        if t_dev == dev:
                            return tensor
                        return jax.device_put(tensor, dev)
            else:
                ret = jax.device_put(tensor, dev)
            return ret

        # @staticmethod
        # def to_device(tensor: jnp.ndarray, dev: Any) -> jnp.ndarray:
        #    for t_dev in tensor.devices():
        #        break
        #    if t_dev == dev:
        #        return tensor
        #    return jax.device_put(tensor, dev)

        # @staticmethod
        # def to_cpu(tensor: jnp.ndarray) -> jnp.ndarray:
        #    for t_dev in tensor.devices():
        #        break
        #    if t_dev == cpu:
        #        return tensor

        ##
        #    return jax.device_put(tensor, cpu)#.block_until_ready()

        @staticmethod
        def to_backend_datatype(dtype: DataType) -> Any:
            return JaxBackend.INVERSE_JAX_TO_TEMPO_DTYPES_DICT[dtype]

        @staticmethod
        def to_tpo_dtype(backend_dtype: Any) -> DataType:
            """Convert a JAX dtype to a Tempo dtype."""
            return JaxBackend.JAX_TO_TEMPO_DTYPES_DICT[backend_dtype]

        @staticmethod
        def cast_backend_dtype(tensor: jnp.ndarray, dtype: Any) -> jnp.ndarray:
            return tensor.astype(dtype)

        @staticmethod
        def to_backend_device_obj(dev: DeviceLike) -> Any:
            dev = device.from_(dev)
            jax_dev_str = "cpu"
            if dev == device.cpu:
                jax_dev_str = "cpu"
            elif dev == device.gpu:
                jax_dev_str = "gpu"
            elif dev == device.fake_gpu:
                jax_dev_str = "cpu"
            return jax.devices(jax_dev_str)[0]

        @staticmethod
        def to_backend_shape(shape: int | Tuple[int] | StaticShape) -> Any:
            shape = StaticShape.from_(shape)
            return shape._shape

        @staticmethod
        def from_dlpack(ext_tensor: Any) -> jnp.ndarray:
            return jax_from_dlpack(ext_tensor, copy=False)

        @staticmethod
        def zeros_tensor(shape: Tuple[int, ...], dtype: Any, dev: Any) -> jnp.ndarray:
            return jnp.zeros(shape=shape, dtype=dtype, device=dev)

        @staticmethod
        def ones_tensor(shape: Tuple[int, ...], dtype: Any, dev: Any) -> jnp.ndarray:
            return jnp.ones(shape=shape, dtype=dtype, device=dev)

        @staticmethod
        def fast_int_lift(
            fill_value: int,
            dtype: Optional[Any] = None,
            device: Optional[Any] = None,
        ) -> jnp.ndarray:
            # TODO: what if we just returned an actual int?
            # mainly, we just want to avoid waiting for the tensor to be created,
            # which often involves device syncs.
            return jnp.full(shape=(), fill_value=fill_value, dtype=dtype, device=device)

        @staticmethod
        def full_tensor(
            fill_value: Any,
            shape: Tuple[int, ...] = SCALAR_SHAPE,
            dtype: Optional[Any] = None,
            device: Optional[Any] = None,
        ) -> jnp.ndarray:
            return jnp.full(shape=shape, fill_value=fill_value, dtype=dtype, device=device)

        @staticmethod
        def lift_tensor(
            data: Any,
            shape: Optional[Tuple[int, ...]] = None,
            dtype: Optional[Any] = None,
            device: Optional[Any] = None,
        ) -> jnp.ndarray:
            x = jnp.asarray(data, dtype=dtype, device=device)
            if shape is not None:
                x = jnp.broadcast_to(x, shape)
            return x

        @staticmethod
        def copy(tensor: jnp.ndarray) -> jnp.ndarray:
            return tensor.copy()

        @staticmethod
        def clear_jit_cache() -> None:
            """Clear the jit cache."""
            jit_kernel_cache.clear()

        @staticmethod
        def trace_codegen_thunk(
            execution_func: Callable[[Tuple[jnp.ndarray, ...]], Tuple[jnp.ndarray, ...]],
            op_id: OpId,
            dev: DeviceGroup,
            exec_cfg: ExecutionConfig,
            example_inputs: Tuple[jnp.ndarray, ...],
            donatable_args: Sequence[int],
            analysis_ctx: AnalysisCtx,
            parent_graph: PDG,
        ) -> Thunk[jnp.ndarray]:  # type: ignore
            if (parent_graph.pdg_id, op_id) in jit_kernel_cache:
                log.info("Using cached JIT kernel for %s", op_id)
                return jit_kernel_cache[(parent_graph.pdg_id, op_id)]

            assert isinstance(example_inputs, tuple)
            # if len(example_inputs) > 0:
            #    assert isinstance(example_inputs[0], jnp.ndarray)

            clone_ins = lambda ins: tuple(
                jnp.array(x) if isinstance(x, jnp.ndarray) else x for x in ins
            )
            with jax.disable_jit(disable=False):
                log.info("Jitting %s with %s donations", op_id, donatable_args)

                if donatable_args:

                    def wrapped_fun(*args: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
                        return execution_func(args)

                    thunk_ = jax.jit(
                        wrapped_fun,
                        device=JaxBackend.to_backend_device_obj(dev),
                        donate_argnums=tuple(donatable_args),
                    )

                    ### thunk_lower = thunk_.lower(*clone_ins(example_inputs))  # type: ignore
                    ### # OLD ===
                    ### thunk_ = thunk_lower.compile()  # type: ignore
                    ### # print_jax_compiled(op_id, thunk_)
                    ### # Argument unpacker
                    thunk = lambda ins, ctx: thunk_(*ins)
                    # NEW ===
                    # thunk_ = thunk_lower.compile()._executable.call  # type: ignore
                    # thunk = lambda ins, ctx: thunk_(*ins)

                else:
                    thunk_ = jax.jit(execution_func, device=JaxBackend.to_backend_device_obj(dev))
                    ### thunk_lower = thunk_.lower(clone_ins(example_inputs))  # type: ignore

                    ### # OLD ==
                    ### thunk_ = thunk_lower.compile()  # type: ignore
                    thunk = lambda ins, ctx: thunk_(ins)
                    ## print_jax_compiled(op_id, thunk_)

                    # NEW ===
                    # thunk_ = thunk_lower.compile()._executable.call  # type: ignore
                    # thunk = lambda ins, ctx: thunk_(*ins)

                # TODO: cloning affects outputs here.
                if exec_cfg.profile_codegen_kernels:
                    import time

                    start_time = time.perf_counter_ns()
                    for _ in range(100):
                        thunk(clone_ins(example_inputs), None)  # type: ignore
                    JaxBackend.sync()
                    end_time = time.perf_counter_ns()
                    elapsed_sec = (end_time - start_time) / 1e9 / 100
                    print(f"Time taken - op_id: {op_id} - {elapsed_sec} seconds")
                # warmup
                thunk(clone_ins(example_inputs), None)  # type: ignore

            jit_kernel_cache[(parent_graph.pdg_id, op_id)] = thunk
            return thunk

    DLBackend.register_backend(DLBackendName.JAX, JaxBackend)


except ImportError as e:
    raise ImportError("JAX is not installed. Please install JAX to use the JAX backend.") from e
except Exception as e:
    print(e)
    raise e
