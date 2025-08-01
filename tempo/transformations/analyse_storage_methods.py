import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
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
from tempo.utils.dg_utils import is_block_access, is_const_block_access, is_window_access
from tempo.utils.memory_estimator import MemoryEstimator

log = logger.get_logger(__name__)


def get_default_prealloc_value_for_dtype(dtype: DataType) -> Any:
    if dtypes.is_integer(dtype):
        return 0
    else:
        return DEFAULT_PREALLOC_VALUE


# TODO make sure tensors are not huge if prealloced
# bytes_est = dg.estimate_tensor_size_bytes(
#    tensor_id.op_id,
#    tensor_id.output_id,
#    bound_size_estimate=exec_cfg.bound_size_estimate,
# )
# if bytes_est < MAX_BYTES and all(
#    b in dg.static_bounds for b in domain.ubounds
# ):
# Statically bounded checks.


def _get_max_val_for_shape(
    snk_dom: Domain, static_bounds: Mapping[ie.Symbol, int], shape_evaled: ie.IntIndexValueLike
) -> Optional[int]:
    max_val: Optional[int] = None
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
) -> Sequence[Tuple[ie.Symbol, Union[int, None]]]:
    slices_and_dims = [
        (e, d) for e, d in zip(expr.members, src_domain.variables, strict=False) if not e.is_point()
    ]
    dims_and_block_sizes: List[Tuple[ie.Symbol, Union[int, None]]] = []
    for slice_, dim in slices_and_dims:
        shape_evaled = slice_.evaluate_shape(static_bounds)[0]
        max_val = _get_max_val_for_shape(snk_dom, static_bounds, shape_evaled)

        dims_and_block_sizes.append((dim, max_val))

    return dims_and_block_sizes


def _get_block_sizes_per_dim(
    new_dg: PDG,
    exec_cfg: ExecutionConfig,
    op: top.TensorOp,
    dependents: Sequence[Tuple[top.TensorOp, DependencyData]],
) -> Sequence[Tuple[ie.Symbol, int]]:
    non_point_accesses = [
        (dep_op, dep_data.expr) for dep_op, dep_data in dependents if not dep_data.expr.is_point()
    ]

    block_sizes_per_dim: Dict[ie.Symbol, int] = {}
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


def max_block_size(a: Union[int, None], b: Union[int, None]) -> Union[int, None]:
    if a is None or b is None:
        return None
    else:
        return max(a, b)


def _get_window_sizes_per_dim(
    new_dg: PDG, op: top.TensorOp, dependents: Sequence[Tuple[top.TensorOp, DependencyData]]
) -> Sequence[Tuple[ie.Symbol, int]]:
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


class AnalyseStorageMethods(Transformation):
    def _run(self) -> Tuple[PDG, bool]:
        new_dg = self.ctx.dg

        self.mem_est = MemoryEstimator(self.ctx)

        if self.ctx.analysis_ctx._tensor_prealloc_value is None:
            self.ctx.analysis_ctx._tensor_prealloc_value = {}

        storage_method_map: Dict[TensorId, StorageMethod] = {}

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

        tensor_point_size = self.mem_est.estimate_tensor_point_size_bytes(
            tensor_id.op_id, tensor_id.output_id
        )

        any_window_accesses = any(
            is_window_access(m) for _, d in dependents for m in d.expr.members
        )
        any_block_accesses = any(is_block_access(m) for _, d in dependents for m in d.expr.members)
        any_common_const_block_accesses = any(
            # ((not o.domain.is_empty()) and is_const_block_access(m))
            is_const_block_access(m)
            for o, d in dependents
            for m in d.expr.members
        )

        any_type_block_accesses = any_common_const_block_accesses or any_block_accesses

        # assert (
        #    sum([any_common_const_block_accesses, any_window_accesses, any_block_accesses]) <= 1
        # ), f"Found conflicting access patters {[d.expr for _, d in dependents]} for op {op}"
        any_range_accesses = any_type_block_accesses or any_window_accesses

        default_prealloc_value_for_dtype = get_default_prealloc_value_for_dtype(
            new_dg.get_tensor_dtype(tensor_id)
        )
        prealloc_value = self.ctx.analysis_ctx.tensor_prealloc_value.get(
            tensor_id, default_prealloc_value_for_dtype
        )

        if any_window_accesses and not any_type_block_accesses:
            dims_and_window_sizes = _get_window_sizes_per_dim(new_dg, op, dependents)
            print(f"Op {op} with dependents {dependents} has window sizes {dims_and_window_sizes}")
            return CircularBufferStore(
                prealloc_value=prealloc_value,
                dims_and_base_buffer_sizes=tuple(dims_and_window_sizes),
            )
            # return PreallocCircularBufferStore(tuple(dims_and_window_sizes), 4)

        # NOTE: Any block or fixed size slice access creates a block store.
        if any_range_accesses:
            dims_and_block_sizes = _get_block_sizes_per_dim(
                new_dg, self.ctx.exec_cfg, op, dependents
            )
            would_be_swapped = (
                self.ctx.exec_cfg.enable_swap
                and tensor_point_size >= self.ctx.exec_cfg.swap_bytes_min
            )

            any_point_accesses = any(d.expr.is_point() for _, d in dependents)
            # NOTE: In the RL use-case, we have found that a single point access on a block-stored
            # tensor leads to a significant performance drop. Thus, we might as well just use
            # point store.
            if (
                self.ctx.exec_cfg.enable_point_store_fallback
                and any_point_accesses
                and not would_be_swapped
            ):
                log.info("Falling back to point store for %s due to point accesses.", tensor_id)
                return PointStore()

            # NOTE: For small tensors, we can just pay the price of concatenating, to avoid the
            # cost of in-place writes.
            # NOTE: For large tensors, that price becomes too high.
            if (
                self.ctx.exec_cfg.enable_point_store_fallback
                and tensor_point_size < self.ctx.exec_cfg.min_block_store_tensor_point_size
            ):
                log.info(
                    "Falling back to point store for %s due to small tensor (%s).",
                    tensor_id,
                    bytes_to_human_readable(tensor_point_size),
                )
                return PointStore()

            ret_val = BlockStore(
                prealloc_value=prealloc_value,
                dims_and_base_buffer_sizes=tuple(dims_and_block_sizes),
            )
            return ret_val

        return PointStore()

    def _get_percentage_map(
        self, storage_method_map: Dict[TensorId, StorageMethod]
    ) -> Dict[str, float]:
        m_counts = dict.fromkeys({s.__class__ for s in storage_method_map.values()}, 0)
        for tensor_id in storage_method_map:
            method = storage_method_map[tensor_id]
            m_counts[method.__class__] += 1

        percentage_map = {
            m.__name__: round(c / len(storage_method_map) * 100, 2) for m, c in m_counts.items()
        }

        return percentage_map
