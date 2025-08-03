from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

from tempo.core import index_expr as ie


def get_default_path() -> str:
    return f"./runs/{time.strftime('%Y%m%d-%H%M%S')}/"


@dataclass
class ExecutionConfig:
    """
    Core configuration for system execution.

    Attributes:
        path: Directory path to store (debug) execution artifacts.
        dev: Device to run on (e.g., "cpu", "cuda:0").
        backend: Deep Learning Backend to use (e.g., "torch", "jax").
        torch_compilation_backend: Backend to use for torch compilation (e.g., "jit", "compile").
        enable_dead_code_elim: Remove unused ops.
        enable_broadcast_elim: Remove unnecessary broadcasts.
        enable_constant_folding: Fold constants at compile time.
        enable_duplicate_code_elim: Eliminate duplicated code blocks.
        enable_algebraic_optimizer: Apply algebraic rewrites.
        enable_vectorization: Vectorize takes temporal dims and makes them spatial, speeding up
          execution at the cost of increased memory use.
        enable_incrementalization: Incrementalize breaks large spatial dims into smaller
          spatial dims by adding a temporal dimension. E.g. 1x 1024 --> 8x 128. This lowers
          memory use at the cost of speed.
        incrementalize_mem_threshold: Op memory threshold (bytes) to apply incrementalization.
          Incrementalization is applied to all ops with memory use above this threshold.
        incrementalize_fixed_block_size: (Optional) Force inc block size globally. Tempo
          picks a block size automatically otherwise.
        enable_statifying_incrementalization: Static inc is like incrementalization, but
          used to break dynamic shapes into static shapes, rather than lowering memory use.
          It does this by using a fixed block size, and adding padding and masking where needed.
        inc_statify_block_size: Block size for statified incrementalization. Not automatic yet.
        enable_fold_pads_n_masks_into_storage: Optimize away masks/padding into runtime storage.
        enable_dataflow_grouping: Enable automatic grouping of Ops into DataflowOps.
        enable_conservative_grouping: Favor grouping safety (valid schedules) over performance.
        min_dataflow_group_size: Minimum size for dataflow groups to be grouped.
        max_dataflow_group_size: Maximum size for dataflow groups to be grouped.
        enable_group_fusions: Try fusing dataflow groups into larger groups.
        enable_group_fissions: Allow splitting large dataflow groups (to lower kernel mem. use).
        enable_codegen_dataflows: Enable code generation for dataflows.
        enable_device_assignment: Enable device assignment analysis.
        enable_donation_analysis: Enable JAX-like buffer donation logic.
        enable_ast_promo: Promote some schedule instructions into batch instructions. E.g.:
          Loop(c0 = 0, c0 <100) --> Deallocate(x, c0) becomes Deallocate(x, 0:100)
        enable_isolate_loop_conditions: Removes internal Ifs from schedule by
          partially unrolling loops.
        runtime_tensor_prefetch_amount_point: Number of tensors to prefetch at runtime when
          encountering a fetch instruction for point stored tensors.
        runtime_tensor_prefetch_amount_block: Number of blocks to prefetch at runtime when
          encountering a fetch instruction for block stored tensors.
        enable_hybrid_tensorstore: Enable customizing Tensor storage types based on dynamic
          access patterns. Otherwise, all tensors are point stored.
        enable_gc: Enable runtime garbage collection.
        gc_bytes_min: Minimum memory (in bytes) to trigger GC for a tensor.
        enable_swap: Enable runtime swapping to cpu memory.
        swap_bytes_min: Minimum memory (in bytes) to trigger swapping for a tensor.
        enable_symbol_prealloc_store: Enable pre-allocating on-device symbol iotas.
        enable_point_store_fallback: Enable falling back to point store
        min_block_store_tensor_point_size: int = 4 * (2**10) # 4KB
        deterministic: Enable deterministic execution (lowers performance, for reproducibility).
        seed: Random seed for determinism.
        visualize_pipeline_stages: Render graph after each compilation pipeline stage.
        visualize_debug_stages: Render graphs after key stages.
        validate_pipeline_stages: Run validation checks after each compilation pipeline stage.
        render_schedule: Render the execution schedule as python and graph.
        profile_codegen_kernels: Profile how long each compiled kernel takes.
        executor_debug_mode: Enable runtime debug checks.
        enable_index_ops: Enable indexing operations: IndexAddOp and IndexSelectOp.
        enable_matmul_ops: Enable optimized matmul operations: MatMulOp.
        default_dim_upper_bound_size: When we need a dimension size, but don't have one,
            use this size.
        bound_size_estimates: Similar to above, but can be provided per-symbol.
        enable_x64: Enable 64-bit floats and ints (slower but more accurate).
        num_executor_pool_workers: Number of parallel executors (None = auto).
        enable_parallel_block_detection: Enable schedule parallelism analysis.
        algebraic_optims_to_disable: List of algebraic optimization names (as strings)
            to disable in the optimizer registry builder.
    """

    # GENERAL SYSTEM =================================================
    path: str = get_default_path()
    dev: str = "cpu"
    backend: str = "torch"
    torch_compilation_backend: str = "compile"  # Can also try "jit"
    # COMPILER TRANSFORMS & Analysis ============================================
    # OPTIMIZER ----------------------------------------------------
    enable_dead_code_elim: bool = True
    enable_broadcast_elim: bool = True
    enable_constant_folding: bool = True
    enable_duplicate_code_elim: bool = True
    enable_algebraic_optimizer: bool = True
    algebraic_optims_to_disable: Iterable[str] = field(default_factory=tuple)
    enable_domain_reduction: bool = True
    enable_inplace_write: bool = True
    enable_lazy_slice: bool = True
    # VECTORIZATION ----------------------------------------------
    enable_vectorization: bool = True
    reject_vec_groups_smaller_than: int = 5
    # If group has X nodes, and X*ratio external dependents, reject.
    reject_vec_groups_with_external_deps_ratio_greater_than: float = 0.2
    # INCREMENTALIZATION ---------------------------------
    enable_incrementalization: bool = True
    incrementalization_percentile: int = 20
    # Memory threshold at which we trigger incrementalization for an Op.
    incrementalize_mem_threshold: int = 16 * (2**30)  # 16GB
    # STATIC INCREMENTALIZATION ---------------------------------
    enable_statifying_incrementalization: bool = True
    inc_statify_block_size: int = 256
    enable_fold_pads_into_storage: bool = True
    enable_pad_mask_removal: bool = True
    # DATAFLOW GROUPING --------------------------------------------
    enable_dataflow_grouping: bool = True
    enable_conservative_grouping: bool = False
    min_dataflow_group_size: int = 1
    max_dataflow_group_size: int = 200_000
    enable_group_fusions: bool = True
    enable_group_fissions: bool = False
    # DATAFLOW CODEGEN --------------------------------------------
    enable_codegen_dataflows: bool = True
    # ANALYSIS ----------------------------------------------------
    enable_device_assignment: bool = True
    enable_donation_analysis: bool = True
    # SCHEDULE BUILD -------------------------------------------
    enable_ast_promo: bool = True
    enable_isolate_loop_conditions: bool = True
    runtime_tensor_prefetch_amount_point: int = 10
    runtime_tensor_prefetch_amount_block: int = 5
    # RUNTIME MEMORY MANAGEMENT ======================================
    enable_hybrid_tensorstore: bool = True
    enable_gc: bool = True
    gc_bytes_min: int = 0
    enable_swap: bool = False
    swap_bytes_min: int = 32 * (2**20)  # 32MB #NOTE: We don't want to swap much.
    enable_symbol_prealloc_store: bool = True
    enable_point_store_fallback: bool = True
    min_block_store_tensor_point_size: int = int(0.8 * (2**20))  # 0.8MB
    # DEBUG =========================================================
    deterministic: bool = False
    seed: int = 0
    visualize_pipeline_stages: bool = False
    visualize_debug_stages: bool = True
    validate_pipeline_stages: bool = False
    render_schedule: bool = False
    profile_codegen_kernels: bool = False
    executor_debug_mode: bool = False
    # PROFILING =====================================================
    enable_exec_op_profiling: bool = False
    exec_op_profiling_sync_after_each: bool = False
    # OPTIONAL PRIMITIVE OPERATIONS ================================
    enable_index_ops: bool = True
    enable_matmul_ops: bool = True
    # DIM ESTIMATES =================================================
    bound_size_estimates: Mapping[ie.Symbol, int] = field(default_factory=dict)
    default_dim_upper_bound_size: int = 500
    # DEPRECATED ====================================================
    enable_x64: bool = True
    num_executor_pool_workers: Optional[int] = None
    enable_parallel_block_detection: bool = False
    # NOTE: Not currently in use
    torch_pinned_prealloc_size_bytes: int = 250 * (2**30)  # 250GB
    torch_pinned_memory_enabled: bool = True
    # OTHER =========================================================
    M: int = 20_000  # A number assumed larger than every temporal dimension.

    def __post_init__(self) -> None:
        if (
            self.enable_statifying_incrementalization and self.enable_fold_pads_into_storage
        ) and not self.enable_hybrid_tensorstore:
            raise ValueError("StatifyInc with pad/mask folding requires hybrid tensorstore.")

    @staticmethod
    def default() -> ExecutionConfig:
        return ExecutionConfig()

    @staticmethod
    def test_cfg() -> ExecutionConfig:
        return ExecutionConfig(
            path="./test_run/",
            dev="cpu",
            backend="torch",
            deterministic=True,
            seed=0,
            executor_debug_mode=True,
            gc_bytes_min=0,
            enable_dataflow_grouping=False,
            enable_constant_folding=False,
            enable_dead_code_elim=False,
            enable_domain_reduction=False,
            render_schedule=False,
            visualize_pipeline_stages=False,
            visualize_debug_stages=False,
            validate_pipeline_stages=True,
            profile_codegen_kernels=False,
            enable_duplicate_code_elim=False,
            enable_algebraic_optimizer=False,
            enable_broadcast_elim=False,
            enable_incrementalization=False,
            enable_vectorization=False,
            enable_gc=False,
            enable_swap=False,
            enable_parallel_block_detection=False,
            enable_donation_analysis=False,
            enable_hybrid_tensorstore=False,
            enable_x64=True,
            enable_device_assignment=False,
            enable_codegen_dataflows=False,
            enable_ast_promo=False,
            enable_isolate_loop_conditions=False,
            enable_matmul_ops=True,
            enable_index_ops=True,
            enable_statifying_incrementalization=False,
            enable_fold_pads_into_storage=False,
            enable_pad_mask_removal=False,
            enable_group_fusions=False,
            enable_group_fissions=False,
            enable_conservative_grouping=False,
            enable_inplace_write=False,
            enable_lazy_slice=False,
            enable_exec_op_profiling=False,
            exec_op_profiling_sync_after_each=False,
            enable_symbol_prealloc_store=False,
            torch_pinned_prealloc_size_bytes=0,
            torch_compilation_backend="jit",
            algebraic_optims_to_disable=(),
            reject_vec_groups_smaller_than=5,
            reject_vec_groups_with_external_deps_ratio_greater_than=0.2,
            incrementalization_percentile=50,
            incrementalize_mem_threshold=16 * (2**30),
            inc_statify_block_size=256,
            min_dataflow_group_size=1,
            max_dataflow_group_size=200_000,
            runtime_tensor_prefetch_amount_point=10,
            runtime_tensor_prefetch_amount_block=5,
            swap_bytes_min=32 * (2**20),
            enable_point_store_fallback=False,
            min_block_store_tensor_point_size=int(0.8 * (2**20)),
            bound_size_estimates={},
            default_dim_upper_bound_size=500,
            num_executor_pool_workers=None,
            torch_pinned_memory_enabled=False,
            M=20_000,
        )

    @staticmethod
    def debug_cfg() -> ExecutionConfig:
        return ExecutionConfig(
            path="./debug_run/",
            dev="cpu",
            backend="torch",
            deterministic=True,
            seed=0,
            executor_debug_mode=True,
            gc_bytes_min=0,  # 1MiB
            enable_dataflow_grouping=True,
            enable_constant_folding=True,
            enable_dead_code_elim=True,
            enable_domain_reduction=True,
            render_schedule=True,
            visualize_pipeline_stages=True,
            visualize_debug_stages=True,
            validate_pipeline_stages=True,
            profile_codegen_kernels=False,
            enable_duplicate_code_elim=True,
            enable_algebraic_optimizer=True,
            enable_broadcast_elim=True,
            enable_incrementalization=False,
            enable_vectorization=False,
            enable_gc=True,
            enable_swap=False,
            enable_parallel_block_detection=False,
            enable_donation_analysis=False,
            enable_hybrid_tensorstore=True,
            enable_x64=True,
            enable_device_assignment=False,
            enable_codegen_dataflows=False,
            enable_ast_promo=False,
            enable_isolate_loop_conditions=False,
            enable_matmul_ops=True,
            enable_index_ops=True,
            enable_statifying_incrementalization=True,
            inc_statify_block_size=10,
            enable_fold_pads_into_storage=True,
            enable_pad_mask_removal=True,
            enable_group_fusions=False,
            enable_group_fissions=False,
            enable_conservative_grouping=False,
            enable_inplace_write=False,
            enable_lazy_slice=False,
            enable_exec_op_profiling=False,
            exec_op_profiling_sync_after_each=False,
            enable_symbol_prealloc_store=False,
            torch_pinned_prealloc_size_bytes=0,
        )
