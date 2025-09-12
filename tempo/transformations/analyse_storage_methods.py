import csv
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import TensorId
from tempo.core.dependence_graph import PDG, DependencyData
from tempo.core.domain import Domain
from tempo.core.dtype import DataType, dtypes
from tempo.core.storage_methods import (
    DEFAULT_PREALLOC_VALUE,
    BlockStore,
    CircularBufferStore,
    DontStore,
    EvalSymbolStore,
    PointStore,
    StorageMethod,
)
from tempo.core.utils import bytes_to_human_readable
from tempo.transformations.compilation_pass import Transformation
from tempo.utils import isl as isl_utils
from tempo.utils import logger
from tempo.utils.dg_utils import (
    is_block_access,
    is_const_block_access,
    is_proto_block_access,
    is_range_access,
    is_window_access,
)
from tempo.utils.memory_estimator import MemoryEstimator

log = logger.get_logger(__name__)


def get_default_prealloc_value_for_dtype(dtype: DataType) -> Any:
    if dtypes.is_integer(dtype):
        return 0
    else:
        return DEFAULT_PREALLOC_VALUE


def _get_max_val_for_shape(
    snk_dom: Domain, static_bounds: Mapping[ie.Symbol, int], shape_evaled: ie.IntIndexValueLike
) -> int | None:
    max_val: int | None = None
    if isinstance(shape_evaled, ie.IndexExpr):
        max_val_expr = isl_utils.int_index_val_max(
            shape_evaled,
            # snk_dom,
            condition=None,
            known_symbols=static_bounds,
        )
        if max_val_expr is not None:
            if isinstance(max_val_expr, ie.ConstInt):
                max_val = max_val_expr.const
            else:
                raise NotImplementedError(f"TODO: support {type(max_val_expr)}")
    else:
        max_val = shape_evaled
    return max_val


def extract_block_size_from_expr(
    expr: ie.IndexSequence,
    snk_dom: Domain,
    src_domain: Domain,
    static_bounds: Mapping[ie.Symbol, int],
) -> Sequence[tuple[ie.Symbol, int | None]]:
    slices_and_dims = [
        (e, d) for e, d in zip(expr.members, src_domain.variables, strict=False) if not e.is_point()
    ]
    dims_and_block_sizes: list[tuple[ie.Symbol, int | None]] = []
    for slice_, dim in slices_and_dims:
        shape_evaled = slice_.evaluate_shape(static_bounds)[0]
        max_val = _get_max_val_for_shape(snk_dom, static_bounds, shape_evaled)

        dims_and_block_sizes.append((dim, max_val))

    return dims_and_block_sizes


def _get_block_sizes_per_dim(
    new_dg: PDG,
    exec_cfg: ExecutionConfig,
    op: top.TensorOp,
    dependents: Sequence[tuple[top.TensorOp, DependencyData]],
) -> Sequence[tuple[ie.Symbol, int]]:
    non_point_accesses = [
        (dep_op, dep_data.expr) for dep_op, dep_data in dependents if not dep_data.expr.is_point()
    ]

    block_sizes_per_dim: dict[ie.Symbol, int] = {}
    for dep_op, expr in non_point_accesses:
        dims_and_block_sizes_ = extract_block_size_from_expr(
            expr, dep_op.domain, op.domain, new_dg.static_bounds
        )
        for dim, block_size in dims_and_block_sizes_:
            if dim not in block_sizes_per_dim:
                bs = block_size
            else:
                bs = max_block_size(block_sizes_per_dim[dim], block_size)
            if bs is None:
                try:
                    bs = new_dg.static_bounds[dim.as_bound()]
                except KeyError:
                    # NOTE: use big M for unknown bounds
                    # TODO: improve on this
                    log.warning(
                        "Unknown bound for %s, using big M %d for block alloc",
                        dim,
                        exec_cfg.M,
                    )
                    bs = exec_cfg.M
            block_sizes_per_dim[dim] = bs

        # dims_and_block_sizes = list(block_sizes_per_dim.items())
    dims_and_block_sizes = list(block_sizes_per_dim.items())
    return dims_and_block_sizes


def max_block_size(a: int | None, b: int | None) -> int | None:
    if a is None or b is None:
        return None
    else:
        return max(a, b)


def _get_window_sizes_per_dim(
    new_dg: PDG, op: top.TensorOp, dependents: Sequence[tuple[top.TensorOp, DependencyData]]
) -> Sequence[tuple[ie.Symbol, int]]:
    dims_and_window_sizes = []
    for dep_op, dep_data in dependents:
        expr = dep_data.expr
        num_slices = 0
        for i, member in enumerate(expr.members):
            if is_window_access(member):
                var = op.domain.variables[i]
                assert num_slices == 0, "For now, should be 0"
                shape_ = member.evaluate_shape(new_dg.static_bounds)[num_slices]
                max_val = _get_max_val_for_shape(dep_op.domain, new_dg.static_bounds, shape_)
                if max_val is None:
                    raise ValueError(f"Max val is None for {member} in {dep_op}")
                dims_and_window_sizes = [(var, max_val)]
            if not member.is_point():
                num_slices += 1

    dims_and_window_sizes = list(set(dims_and_window_sizes))
    assert len(dims_and_window_sizes) == 1, "For now, should be 1"

    return dims_and_window_sizes


def fallback_to_point_if_need(
    ctx: CompilationCtx,
    tensor_id: TensorId,
    mem_est: MemoryEstimator,
    storage_method: StorageMethod,
) -> StorageMethod:
    if isinstance(storage_method, PointStore):
        return storage_method

    dependents = ctx.dg.get_tensor_flat_direct_dependents(tensor_id)
    tensor_point_size = mem_est.estimate_tensor_point_size_bytes(
        tensor_id.op_id, tensor_id.output_id
    )

    would_be_swapped = ctx.exec_cfg.enable_swap and tensor_point_size >= ctx.exec_cfg.swap_bytes_min

    is_folded_pad = any(not ie.struct_eq(d.expr, d.isl_expr) for _, d in dependents)

    # TODO: this should not check the expression in general but just the temporal dimension
    # which is relevant for the storage method.
    any_moving_point_accesses = any(
        d.expr.is_point() and not d.expr.is_constant() for _, d in dependents
    )

    reason = None

    # NOTE: For small tensors, or tensors which are also point accessed, we fall back to point
    # store for performance reasons.
    if ctx.exec_cfg.enable_point_store_fallback and not would_be_swapped and not is_folded_pad:
        if any_moving_point_accesses:
            reason = "Falling back from {} to PointStore() for {} due to point accesses.".format(
                storage_method,
                tensor_id,
            )

        if tensor_point_size < ctx.exec_cfg.min_block_store_tensor_point_size:
            reason = "Falling back from {} to PointStore() for {} due to small tensor ({}).".format(
                storage_method,
                tensor_id,
                bytes_to_human_readable(tensor_point_size),
            )

    if reason is None:
        return storage_method

    log.info(reason)
    return PointStore()


class AnalyseStorageMethods(Transformation):
    def _run(self) -> tuple[PDG, bool]:
        new_dg = self.ctx.dg

        self.mem_est = MemoryEstimator(self.ctx)

        if self.ctx.analysis_ctx._tensor_prealloc_value is None:
            self.ctx.analysis_ctx._tensor_prealloc_value = {}

        storage_method_map: dict[TensorId, StorageMethod] = {}

        self.ctx.analysis_ctx._tensor_storage_classes = storage_method_map

        tensor_descs = new_dg.get_all_tensor_descriptions()

        for tensor_id, _, _, _ in tensor_descs:
            chosen_storage = self._determine_storage_method(
                new_dg, tensor_id, self.ctx.exec_cfg.enable_hybrid_tensorstore
            )
            if not isinstance(chosen_storage, (PointStore, DontStore)):
                log.info("Storing %s in %s.", tensor_id, chosen_storage)
            storage_method_map[tensor_id] = chosen_storage

        percentage_map = self._get_percentage_map(storage_method_map)

        # Save this as csv (avoid error from file not existing)
        os.makedirs(self.ctx.exec_cfg.path, exist_ok=True)
        with open(Path(self.ctx.exec_cfg.path) / "storage_methods.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=percentage_map.keys())
            writer.writeheader()
            writer.writerow(percentage_map)

        log.info("Breakdown of storage methods: %s", percentage_map)
        self.ctx.analysis_ctx._tensor_storage_classes = storage_method_map
        return new_dg, True

    def _determine_storage_method(
        self, new_dg: PDG, tensor_id: TensorId, hybrid_store_enabled: bool
    ) -> StorageMethod:
        op = new_dg.ops_by_id[tensor_id.op_id].op
        dependents = new_dg.get_tensor_flat_direct_dependents(tensor_id)

        num_dependents = len(dependents)

        if num_dependents == 0:
            # NOTE: There can be tensors that are not accessed at all, like a UDF output
            return DontStore()

        if isinstance(op, top.EvalSymbolOp) and self.ctx.exec_cfg.enable_symbol_prealloc_store:
            return EvalSymbolStore(symbol=op.symbol)

        if not hybrid_store_enabled:
            return PointStore()

        # NOTE: Don't make storage decisions based on depedents which execute once.
        # NOTE: Idea is that we generally prefer paying a one-time concat cost over
        # potentially paying a cost for each write.
        dependents = [(o, d) for o, d in dependents if not o.domain.is_empty()]

        any_window_accesses = any(
            is_window_access(m) for _, d in dependents for m in d.expr.members
        )
        any_block_accesses = any(is_block_access(m) for _, d in dependents for m in d.expr.members)
        any_proto_block_accesses = any(
            is_proto_block_access(m) for _, d in dependents for m in d.expr.members
        )
        any_common_const_block_accesses = any(
            # ((not o.domain.is_empty()) and is_const_block_access(m))
            is_const_block_access(m)
            for o, d in dependents
            for m in d.expr.members
        )

        any_type_block_accesses = (
            any_common_const_block_accesses or any_block_accesses or any_proto_block_accesses
        )

        any_range_accesses = any(is_range_access(m) for _, d in dependents for m in d.expr.members)

        default_prealloc_value_for_dtype = get_default_prealloc_value_for_dtype(
            new_dg.get_tensor_dtype(tensor_id)
        )
        prealloc_value = self.ctx.analysis_ctx.tensor_prealloc_value.get(
            tensor_id, default_prealloc_value_for_dtype
        )

        store: StorageMethod = PointStore()
        if any_window_accesses and not any_type_block_accesses:
            dims_and_window_sizes = _get_window_sizes_per_dim(new_dg, op, dependents)
            store = CircularBufferStore(
                prealloc_value=prealloc_value,
                dims_and_base_buffer_sizes=tuple(dims_and_window_sizes),
            )
            # return PreallocCircularBufferStore(tuple(dims_and_window_sizes), 4)

        # NOTE: Any block or fixed size slice access creates a block store.
        elif any_range_accesses:
            dims_and_block_sizes = _get_block_sizes_per_dim(
                new_dg, self.ctx.exec_cfg, op, dependents
            )

            store = BlockStore(
                prealloc_value=prealloc_value,
                dims_and_base_buffer_sizes=tuple(dims_and_block_sizes),
            )

        store = fallback_to_point_if_need(self.ctx, tensor_id, self.mem_est, store)

        return store

    def _get_percentage_map(
        self, storage_method_map: dict[TensorId, StorageMethod]
    ) -> dict[str, float]:
        m_counts = dict.fromkeys({s.__class__ for s in storage_method_map.values()}, 0)
        for tensor_id in storage_method_map:
            method = storage_method_map[tensor_id]
            m_counts[method.__class__] += 1

        percentage_map = {
            m.__name__: round(c / len(storage_method_map) * 100, 2) for m, c in m_counts.items()
        }

        return percentage_map
