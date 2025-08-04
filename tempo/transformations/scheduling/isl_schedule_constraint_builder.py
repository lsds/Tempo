from typing import Callable, Dict, Tuple

import islpy as isl

from tempo.core import index_expr as ie
from tempo.core import isl_types as islt
from tempo.core import tensor_ops as top
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.datatypes import TensorId
from tempo.core.dependence_graph import PDG, DependencyData
from tempo.core.device import device
from tempo.core.domain import Domain
from tempo.core.op_tags import BACKWARD_REGION_TAG, REGION_TAG
from tempo.core.storage_methods import DontStore, EvalSymbolStore
from tempo.core.utils import bytes_to_human_readable
from tempo.utils import logger
from tempo.utils.dg_utils import get_block_access_var, is_block_access
from tempo.utils.isl import (
    dependence_to_isl_map,
    op_id_to_exec_name,
    rename_union_set_tuples,
    tensor_id_to_fetch_stmt,
    tensor_id_to_gc_stmt,
    tensor_id_to_offload_stmt,
)
from tempo.utils.memory_estimator import MemoryEstimator

log = logger.get_logger(__name__)


def get_affine_expr(
    snk_domain: Domain,
    src_domain: Domain,
    edge_data: DependencyData,
    dg: PDG,
    isl_ctx: islt.Context,
) -> ie.IndexSequence:
    expr = edge_data.isl_expr
    new_expr = expr.drop_modulos()

    # num_cond_branches = len(new_expr.enumerate_all_cond_branches())
    # if num_cond_branches > 1:
    #    #TODO: Unfortunately,
    #    # this is not able to simplify piecewise exprs to min/max for some reason.
    #    new_expr = simplify_dependence_expr(
    #        expr, snk_domain, src_domain, edge_data.cond, dg.static_bounds, isl_ctx
    #    )

    # TODO: piecewise exprs are giving not struct_eq, but they are equivalent.??
    if not new_expr.struct_eq(expr) and str(new_expr) != str(expr):
        log.info("Mapped non-affine %s to affine %s", expr, new_expr)
    assert isinstance(new_expr, ie.IndexSequence)
    return new_expr


class IslScheduleConstraintsBuilder:
    """To compute the schedule, we need to provide validity/proximity/coincidence constraints.

    Tempo defines its computations as:
    1. Sink[basis] = f(Source[IndexExprSequence])

    But ISL requires constraints of the form:
    2. Source[basis] -> Sink[IndexExprSequence]

    Where = is assignment and -> is "happens-before".

    So instead we build the mapping thinking of the arrows as "depends-on" relationships.
    Later, we reverse the map, turning the arrows into "happens-before" relationships.
    """

    def __init__(self, ctx: CompilationCtx) -> None:
        self.ctx = ctx
        self._dg = dg = ctx.dg
        self._cfg = ctx.exec_cfg
        self._isl_ctx = ctx.analysis_ctx.isl_ctx
        self.analysis_ctx = ctx.analysis_ctx

        self._all_edges = dg.get_all_edges(include_control=True)
        self._data_edges = []
        self._control_edges = []

        self._tids_to_swap: isl.Set[TensorId] = set()

        self.any_window_access_in_dg = self.analysis_ctx._is_incremental_algo

        for snk, src, d in self._all_edges:
            if d.is_control_edge:
                self._control_edges.append((snk, src, d))
            else:
                self._data_edges.append((snk, src, d))

        self._universe_bounds_str = ",".join(
            [
                f"{b}={dg.static_bounds[b]}"
                for b in self._dg.universe.parameters
                if b in dg.static_bounds
            ]
        )

        self._mem_man_ops: Dict[str, islt.UnionSet] = {}

        self.additional_proximity_constraints: islt.UnionMap = self._make_map(" ")
        self.additional_coincidence_constraints: islt.UnionMap = self._make_map(" ")

        self.mem_est = MemoryEstimator(ctx)

    def _requires_swap(  # noqa: C901
        self,
        dep_graph: PDG,
        tensor_id: TensorId,
    ) -> bool:
        size = self.mem_est.estimate_tensor_point_size_bytes(tensor_id.op_id, tensor_id.output_id)
        # NOTE: Small tensors are not worth swapping
        if size <= self._cfg.swap_bytes_min:
            log.debug("Tensor %s is too small to swap", tensor_id)
            return False

        # NOTE: Heuristics: If there are no block accesses, then we do not need to swap
        # If there are window accesses -> window algorithm, then we do not need to swap
        # if not self.any_block_access_in_dg:
        #    log.info("No block accesses in DG, skipping swap globally")
        #    return False
        if self.any_window_access_in_dg:
            log.debug("Any window accesses in DG, skipping swap globally")
            return False

        prod_op = dep_graph.ops_by_id[tensor_id.op_id].op

        # NOTE: Do not swap tensors with only iteration dimensions.
        if len(prod_op.domain) <= 1:
            log.debug("Tensor %s has only iteration dimensions, skipping swap", tensor_id)
            return False

        # NOTE: If a tensor has incrementalization dims in its domain, it is not worth swapping.
        # NOTE: This is because the tile will be immediately consumed
        if any(v.name.startswith("di") for v in prod_op.domain.variables):
            log.debug("Tensor %s has incrementalization dims, skipping swap", tensor_id)
            return False

        # NOTE: No point in swapping tensors in these storage classes
        storage_classes = self.analysis_ctx._tensor_storage_classes
        if storage_classes is not None:
            stor = storage_classes[tensor_id]
            if isinstance(stor, DontStore):
                log.debug("Tensor %s is DontStore, skipping swap", tensor_id)
                return False
            if isinstance(stor, EvalSymbolStore):
                log.debug("Tensor %s is EvalSymbolStore, skipping swap", tensor_id)
                return False

        # NOTE: No point in swapping tensors on CPU
        dev = self.ctx.analysis_ctx.get_op_device(dep_graph.ops_by_id[tensor_id.op_id].op)
        if dev == device.cpu:
            log.debug("Tensor %s is on CPU, skipping swap", tensor_id)
            return False

        # NOTE: We know that offloading env steps is a bad idea in general
        # NOTE: In particular because they are usually immediately copied into a merge.
        if isinstance(prod_op, top.UDFOp) and (
            prod_op.desc.thunk_name == "EnvStep" or prod_op.desc.thunk_name == "EnvReset"
        ):
            log.debug("Tensor %s is an env step or reset, skipping swap", tensor_id)
            return False

        deps = self._dg.get_tensor_flat_direct_dependents(tensor_id)
        nonbasis_consumers = [
            (op_, edge)
            for op_, edge in deps
            if not edge.isl_expr.struct_eq(prod_op.domain.basis_expr)
        ]
        if len(nonbasis_consumers) > 1:
            log.debug(
                "Found more than one non-basis consumer for tensor %s. Skipping swap",
                tensor_id,
            )
            return False

        for op, _ in deps:
            if BACKWARD_REGION_TAG in op.flat_tags.get(REGION_TAG, ()):
                if tensor_id not in self._tids_to_swap:
                    log.info(
                        "Will swap tensor %s with point size: %s",
                        tensor_id,
                        bytes_to_human_readable(size),
                    )
                    self._tids_to_swap.add(tensor_id)
                return True

        # NOTE: If all dependents do basis point accesses, then they can be scheduled together.
        # Thus, it is not worth offloading the tensor.
        if all(edge.expr.struct_eq(prod_op.domain.basis_expr) for _, edge in deps):
            log.debug("Tensor %s is a basis point access, skipping swap", tensor_id)
            return False

        if tensor_id not in self._tids_to_swap:
            log.info(
                "Will swap tensor %s with point size: %s",
                tensor_id,
                bytes_to_human_readable(size),
            )
            self._tids_to_swap.add(tensor_id)

        return True

    def _requires_gc(  # noqa: C901
        self,
        dep_graph: PDG,
        tensor_id: TensorId,
    ) -> bool:
        # NOTE: If gc_bytes_min is 0 all tensors are eligible for GC.
        if self._cfg.gc_bytes_min > 0:
            size = dep_graph.estimate_tensor_size_bytes(
                tensor_id.op_id,
                tensor_id.output_id,
                bound_size_estimate=self._cfg.default_dim_upper_bound_size,
            )

            if size < self._cfg.gc_bytes_min:
                return False

        storage_classes = self.analysis_ctx._tensor_storage_classes
        if storage_classes is not None:
            stor = storage_classes[tensor_id]
            if isinstance(stor, DontStore):
                return False
            if isinstance(stor, EvalSymbolStore):
                return False

        return True

    def _make_map(self, constraints: str) -> islt.UnionMap:
        map_with_bounds = f"[{self._universe_bounds_str}] -> {{ {constraints} }}"
        try:
            map_ = isl.UnionMap.read_from_str(self.analysis_ctx.isl_ctx, map_with_bounds)
        except Exception as e:
            log.error("Error when reading constraint: %s", map_with_bounds)
            raise e
        return map_

    def _build_gc_constraints(  # noqa: C901
        self, simplify_exprs: bool = False, validity: bool = False
    ) -> islt.UnionMap:
        union_map = self._make_map(" ")
        if not self._cfg.enable_gc:
            return union_map

        # NOTE: The idea here, is that for each tensor, we create a GC statement that happens after
        # each of the dependents. Each of the dependents reads the tensor using some expr e.
        # We can only GC

        for t_id, _, _, domain in self._dg.get_all_tensor_descriptions():
            if self._requires_gc(self._dg, t_id):
                prod_op = self._dg.ops_by_id[t_id.op_id].op
                prod_exec_name = op_id_to_exec_name(t_id.op_id)
                gc_name = tensor_id_to_gc_stmt(t_id)

                consumers = self._dg.get_tensor_flat_direct_dependents(t_id)
                prod_isl_dom = self.analysis_ctx.get_or_make_domain(prod_op)
                self._mem_man_ops[gc_name] = prod_isl_dom

                # NOTE: at least, GC must happen after the op is computed
                cg_after_prod = dependence_to_isl_map(
                    domain.basis_expr,
                    domain,
                    domain,
                    gc_name,
                    prod_exec_name,
                    # snk_dom_isl=prod_isl_dom,
                    # src_dom_isl=prod_isl_dom,
                    ctx=self._isl_ctx,
                )
                union_map = union_map.union(cg_after_prod)

                ## NOTE: This is a good idea but needs work. It will often pick the iteration
                ## dimension, because tensors often have no other dimension.
                # if len(domain) > 1 and self.mem_est.estimate_tensor_point_size_bytes(
                #   t_id.op_id, t_id.output_id
                # ) < 1 * (2**20):  # 1MB
                #   # NOTE: For small tensors, we want to motivate GC to happen in batches
                #   # for efficiency
                #   # TODO: This selection of domain -1 is a bit biased to having iterations in
                #   # middle
                #   # NOTE: Hack: Find t by picking the largest dimension by size
                #   #curr_size = -1
                #   #curr_idx = 0
                #   #for idx, var in enumerate(domain.variables):
                #   #    size = self._dg.static_bounds.get(var.as_bound(), -1)
                #   #    if size > curr_size:
                #   #        curr_size = size
                #   #        curr_idx = idx
                #   last_idx = len(domain) - 1
                #   last_var = domain.variables[last_idx]

                #   if True:
                #       #last_var = domain.variables[curr_idx]
                #       print(f"Picked last var: {last_var}")
                #       print(f"Picked last var: {last_var}")
                #       print(f"Picked last var: {last_var}")
                #       print(f"Picked last var: {last_var}")
                #       print(f"Picked last var: {last_var}")

                #       if not validity:
                #           # NOTE: proximal to previous and next
                #           for prox_expr in [
                #               domain.basis_expr.remap({last_var: last_var - 1}),
                #               domain.basis_expr.remap({last_var: last_var + 1}),
                #           ]:
                #               # Motivate proximity to previous gc
                #               cg_close_to_prev_gc = dependence_to_isl_map(
                #                   prox_expr,
                #                   domain,
                #                   domain,
                #                   gc_name,
                #                   gc_name,
                #                   ctx=self._isl_ctx,
                #               )
                #               union_map = union_map.union(cg_close_to_prev_gc)

                # NOTE: motivate/force deletion right after last consumer
                for consumer_op, dep_data in consumers:
                    cons_name = op_id_to_exec_name(consumer_op.op_id)
                    expr = dep_data.isl_expr
                    if not validity:
                        expr = get_affine_expr(
                            consumer_op.domain, domain, dep_data, self._dg, self._isl_ctx
                        )

                    if simplify_exprs:
                        expr = expr.as_lower_bound_access().simplify_mins_and_maxes(aggressive=True)

                    # NOTE: We make a constraint that uses the GC stmt as the src stmt so that
                    # after reversing, the constraint is of the form
                    # "GC --(depends on)--> Op[all the values that depend on src[basis]]"
                    # Which encodes that the GC of src at basis must happen after all the
                    # reads of src at that point
                    gc_after_consumer = dependence_to_isl_map(
                        expr,
                        consumer_op.domain,
                        domain,
                        cons_name,
                        gc_name,
                        dep_data.cond,
                        # snk_dom_isl=cons_isl_dom,
                        # src_dom_isl=prod_isl_dom,
                        ctx=self._isl_ctx,
                    ).reverse()

                    union_map = union_map.union(gc_after_consumer)

                    # NOTE: these should only be added for the last accessor only
                    # NOTE: One way to do that is to check if there is a single accessor.
                    # TODO: Think about how to generalize this more.
                    any_statify_dims = any(
                        "ds" in v.name for v in Domain.union(consumer_op.domain, domain)
                    )
                    any_inc_dims = any(
                        "di" in v.name for v in Domain.union(consumer_op.domain, domain)
                    )
                    if (
                        validity
                        and any(is_block_access(e) for e in expr.members)
                        and not any_statify_dims
                        and any_inc_dims
                    ):
                        for e in expr.members:
                            symbol = get_block_access_var(e)
                            if symbol is not None:
                                break
                        if symbol is None:
                            raise ValueError("No block access symbol found")
                        new_expr = expr.remap({symbol: symbol - 1})  # type: ignore

                        next_cons_only_after_prev_gc = dependence_to_isl_map(
                            new_expr,
                            consumer_op.domain,
                            domain,
                            cons_name,
                            gc_name,
                            dep_data.cond,
                            ctx=self._isl_ctx,
                        )
                        union_map = union_map.union(next_cons_only_after_prev_gc)

        return union_map.reverse()

    def _is_donated_to_consumer(
        self, tensor_id: TensorId, cons: top.TensorOp, dep_data: DependencyData
    ) -> bool:
        if self.analysis_ctx._tensor_is_donated is not None:
            donated = self.analysis_ctx._tensor_is_donated[tensor_id]
            if donated:
                # TODO: all donatable args, not just the used ones
                cons_donatable_args = self.analysis_ctx.donatable_args[cons.op_id]
                if dep_data.sink_in_idx in cons_donatable_args:
                    return True

        return False

    def _build_swap_constraints_single_exec(  # noqa: CCR001
        self, simplify: bool = False, proximity: bool = False, coincidence: bool = False
    ) -> islt.UnionMap:
        validity = (not proximity) and (not coincidence)
        union_map = self._make_map(" ")

        if not self._cfg.enable_swap:
            return union_map

        for t_id, _, _, prod_dom in self._dg.get_all_tensor_descriptions():
            op_with_id = self._dg.ops_by_id[t_id.op_id].op
            if self._requires_swap(self._dg, t_id):
                union_map = self._build_tid_swap_constraints(
                    simplify, validity, union_map, t_id, prod_dom, op_with_id
                )

        return union_map.reverse()

    def _get_fetch_offload_exprs(
        self, isl_expr: ie.IndexSequence, simplify: bool
    ) -> Tuple[ie.IndexSequence, ie.IndexSequence]:
        fetch_expr = isl_expr
        offload_expr = isl_expr
        if simplify:
            fetch_expr = fetch_expr.as_upper_bound_access().simplify_mins_and_maxes(aggressive=True)
            offload_expr = offload_expr.as_lower_bound_access().simplify_mins_and_maxes(
                aggressive=True
            )
        return fetch_expr, offload_expr

    def _build_tid_swap_constraints(
        self,
        simplify: bool,
        validity: bool,
        union_map: islt.UnionMap,
        t_id: TensorId,
        prod_dom: islt.UnionSet,
        op_with_id: top.TensorOp,
    ) -> islt.UnionMap:
        # TODO: Remember the domain condition we commented out.

        prod_isl_dom = self.analysis_ctx.get_or_make_domain(op_with_id)

        # NOTE: Initial swap out must happen after the computation of the tensor
        # NOTE: we do not reverse this one on purpose. We want the swap out to happen
        # after the computation of the tensor
        prod_exec_name = op_id_to_exec_name(t_id.op_id)

        gc_name = tensor_id_to_gc_stmt(t_id)

        # Make one constraint that swaps out the tensor immediately after it is computed
        init_offload_name = tensor_id_to_offload_stmt(t_id, 0)
        fetch_name = tensor_id_to_fetch_stmt(t_id, 1)
        offload_name = tensor_id_to_offload_stmt(t_id, 1)

        self._mem_man_ops[init_offload_name] = prod_isl_dom
        self._mem_man_ops[fetch_name] = prod_isl_dom
        self._mem_man_ops[offload_name] = prod_isl_dom

        init_offload_after_exec = dependence_to_isl_map(
            prod_dom.basis_expr,
            prod_dom,
            prod_dom,
            init_offload_name,
            prod_exec_name,
            ctx=self._isl_ctx,
        )
        union_map = union_map.union(init_offload_after_exec)

        deps = self._dg.get_tensor_flat_direct_dependents(t_id)
        basis_consumers = [
            (op_, edge) for op_, edge in deps if edge.isl_expr.struct_eq(prod_dom.basis_expr)
        ]
        nonbasis_consumers = [
            (op_, edge) for op_, edge in deps if not edge.isl_expr.struct_eq(prod_dom.basis_expr)
        ]

        # NOTE: We do not want to force the GC to be close to the initial swap out or fetch.
        if self._cfg.enable_gc:
            # NOTE: Swap operations must happen before GC
            for swap_op_name in [
                init_offload_name,
                fetch_name,
                offload_name,
            ]:
                if validity or swap_op_name == offload_name:
                    gc_after_swaps = dependence_to_isl_map(
                        prod_dom.basis_expr,
                        prod_dom,
                        prod_dom,
                        gc_name,
                        swap_op_name,
                        ctx=self._isl_ctx,
                    )
                    union_map = union_map.union(gc_after_swaps)

        # NOTE: We want to schedule the initial offload after the basis consumers
        # so that the basis consumers can use the tensor before it is swapped out.
        # NOTE: We also want it to be close, the consumers, so no validity check.
        for cons_op, cons_dep_data in basis_consumers:
            cons_name = op_id_to_exec_name(cons_op.op_id)
            fetch_expr, offload_expr = self._get_fetch_offload_exprs(
                cons_dep_data.isl_expr, simplify
            )
            # initial offload should be after basis deps
            init_offload_after_basis_consumer = dependence_to_isl_map(
                offload_expr,
                cons_op.domain,
                prod_dom,
                cons_name,
                init_offload_name,
                cons_dep_data.cond,
                ctx=self._isl_ctx,
            ).reverse()
            union_map = union_map.union(init_offload_after_basis_consumer)

        if validity:
            # NOTE: The fetch must happen after the initial offload,
            # but not necessarily close to it
            fetch_after_init_offload = dependence_to_isl_map(
                prod_dom.basis_expr,
                prod_dom,
                prod_dom,
                fetch_name,
                init_offload_name,
                ctx=self._isl_ctx,
            )
            union_map = union_map.union(fetch_after_init_offload)

            # TODO: DO we want proximity between the fetch and the offload
            # Just tab to change
            offload_after_fetch = dependence_to_isl_map(
                prod_dom.basis_expr,
                prod_dom,
                prod_dom,
                offload_name,
                fetch_name,
                ctx=self._isl_ctx,
            )
            union_map = union_map.union(offload_after_fetch)

        # NOW, for the non-basis consumers, which we hope is just one,
        # we want to schedule the fetch before the consumer and the offload after the consumer.
        for cons_op, cons_dep_data in nonbasis_consumers:
            fetch_expr, offload_expr = self._get_fetch_offload_exprs(
                cons_dep_data.isl_expr, simplify
            )
            consumer_name = op_id_to_exec_name(cons_op.op_id)
            # NOTE: Swap in must happen before the computation that uses the value
            # Goal: Fetch[p] -> Cons[phi-1(p)]
            # Start: Cons[p] = f(Fetch[phi(p)])
            consumer_after_fetch = dependence_to_isl_map(
                fetch_expr,
                cons_op.domain,
                prod_dom,
                consumer_name,
                fetch_name,
                cons_dep_data.cond,
                ctx=self._isl_ctx,
            )
            union_map = union_map.union(consumer_after_fetch)

            # NOTE: Swap out must happen after the computation that uses the value
            # Goal: Off[p] -> Cons[phi-1(p)].
            # (Cons[p] -> Off[phi(p)]).reverse()
            offload_after_consumer = dependence_to_isl_map(
                offload_expr,
                cons_op.domain,
                prod_dom,
                consumer_name,
                offload_name,
                cons_dep_data.cond,
                ctx=self._isl_ctx,
            ).reverse()
            # NOTE: Need to revert cause it's phi-1
            union_map = union_map.union(offload_after_consumer)
        return union_map

    def _build_edge_constraints(  # noqa: C901
        self,
        filter_fun: Callable[[top.TensorOp, top.TensorOp, ie.IndexSequence], bool],
        # simplify_exprs: bool = False,
        is_validity: bool,
    ) -> islt.UnionMap:
        union_map = self._make_map(" ")
        simplify_exprs = not is_validity

        edges_to_use = self._data_edges + (self._control_edges if is_validity else [])

        for sink, src, dep_data in edges_to_use:
            sink_name = op_id_to_exec_name(sink.op_id)
            src_name = op_id_to_exec_name(src.op_id)
            read_expr = dep_data.isl_expr

            # read_expr = get_affine_expr(read_expr)

            # snk_isl_dom = self._dg.analysis_ctx.get_or_make_domain(sink)
            # src_isl_dom = self._dg.analysis_ctx.get_or_make_domain(src)

            if filter_fun(sink, src, read_expr):
                if simplify_exprs:
                    read_expr = get_affine_expr(
                        sink.domain, src.domain, dep_data, self._dg, self._isl_ctx
                    )
                    read_expr = read_expr.as_upper_bound_access().simplify_mins_and_maxes(
                        aggressive=True
                    )
                constraint_map = dependence_to_isl_map(
                    read_expr,
                    sink.domain,
                    src.domain,
                    sink_name,
                    src_name,
                    dep_data.cond,
                    # snk_dom_isl=snk_isl_dom,
                    # src_dom_isl=src_isl_dom,
                    ctx=self._isl_ctx,
                )

                # snk_isl_dom = self._dg.analysis_ctx.get_or_make_domain(sink)
                # src_isl_dom = self._dg.analysis_ctx.get_or_make_domain(src)
                # constraint_map = constraint_map.intersect_domain(
                #    snk_isl_dom
                # ).intersect_range(src_isl_dom)

                union_map = union_map.union(constraint_map)
        result = union_map.reverse()
        return result

    def _get_validity_constraints(self) -> islt.UnionMap:
        edge_constraints = self._build_edge_constraints(
            lambda sink, src, expr: True, is_validity=True
        )

        constraints = edge_constraints
        gc_constraints = self._build_gc_constraints(validity=True)
        constraints = edge_constraints.union(gc_constraints)
        if self.analysis_ctx._isl_execution_schedule is not None:
            swap_constraints = self._build_swap_constraints_single_exec()
            constraints = constraints.union(swap_constraints)

        if self.analysis_ctx._additional_val_constraints is not None:
            # print(f"Additional constraints: {self._dg.analysis_ctx._additional_val_constraints}")
            constraints = constraints.union(
                self.analysis_ctx._additional_val_constraints,
            )

        log.debug("Validity constraints: %s", str(constraints))
        return constraints

    def _get_proximity_constraints(self) -> islt.UnionMap:
        # NOTE: When we detect a windowed algorithm, do not generate proximity constraints
        # to upper bounds.
        # if self.any_window_access_in_dg:
        #    edge_constraints = self._build_edge_constraints(
        #        # lambda sink, src, expr: not any(is_block_access(e) for e in expr.members),
        #        lambda sink, src, expr: not expr.accesses_bound(),
        #        is_validity=False,
        #    )
        # else:
        # NOTE: if incremental, we want proximity, else, we do not want proximity between backward and forward.
        # NOTE: But, if

        edge_constraints = self._build_edge_constraints(
            lambda sink, src, expr: True,
            is_validity=False,
        )

        # edge_constraints = self._build_edge_constraints(
        #    lambda sink, src, expr: True, simplify_exprs=True
        # )

        constraints = edge_constraints
        gc_constraints = self._build_gc_constraints(True)
        constraints = edge_constraints.union(gc_constraints)
        if self.analysis_ctx._isl_execution_schedule is not None:
            # NOTE: was False on simplify...
            swap_constraints = self._build_swap_constraints_single_exec(True, proximity=True)
            constraints = constraints.union(swap_constraints).union(
                self.additional_proximity_constraints
            )
        log.debug("Proximity constraints: %s", str(constraints))
        return constraints

    def _get_coincidence_constraints(self) -> islt.UnionMap:
        """We generate coincidence constraints only for guaranteeing that sources co-execute
        with their users. This will be a problem if a source is shared. So we shouldn't share
        sources or otherwise we ought to remove this.

        Returns:
            isl.UnionMap: representing the coincidence constraints.

        """


        edge_constraints = self._build_edge_constraints(
            lambda sink, src, expr: True,
            is_validity=False,
        )

        constraints = edge_constraints
        gc_constraints = self._build_gc_constraints(True)
        constraints = gc_constraints  # edge_constraints.union(gc_constraints)
        if self.analysis_ctx._isl_execution_schedule is not None:
            swap_constraints = self._build_swap_constraints_single_exec(True, coincidence=True)
            constraints = constraints.union(swap_constraints).union(
                self.additional_coincidence_constraints
            )
        log.debug("Coincidence constraints: %s", str(constraints))
        return constraints

    def _get_domain(self) -> islt.UnionSet:  # noqa: C901
        """For each variable return the domain from 0 to N where N is a parameter

        Returns:
            isl.UnionSet: a union set that represents the domain of each statement

        """
        # domains = isl.UnionSet("[" + self._universe_bounds_str + "] -> { }")

        domains = isl.UnionSet("[] -> { }", context=self._isl_ctx)
        for node in self._dg.nodes:
            # exec_stmt_name = op_id_to_exec_name(node.op_id)
            dom = self.analysis_ctx.get_or_make_domain(node)
            if not dom.is_empty():
                # isl_domain = dom.as_set().set_tuple_name(
                #   exec_stmt_name
                # )
                domains = domains.union(dom)
            else:
                log.warning("Node %s has empty domain", node.op_id)

        for name, isl_domain in self._mem_man_ops.items():
            if not isl_domain.is_empty():
                isl_domain = rename_union_set_tuples(isl_domain, name)
                domains = domains.union(isl_domain)

        return domains

    def _try_simplify(
        self, constraints: islt.UnionMap, domain: islt.UnionSet, params: islt.Set
    ) -> islt.UnionMap:
        try:
            constraints = constraints.gist_domain(domain)
            constraints = constraints.gist_range(domain)
            constraints = constraints.gist_params(params)
            constraints = constraints.coalesce().coalesce()
        except Exception as e:
            log.error("Error when simplifying constraints: %s", e)
        return constraints

    def build_schedule_constraints(self) -> islt.ScheduleConstraints:
        log.info(
            "Building schedule constraints with gc: %s and swap: %s",
            self._cfg.enable_gc,
            self._cfg.enable_swap,
        )

        # TODO use isl utils method
        params = isl.Set(f"[{self._universe_bounds_str}] -> {{ : }}", context=self._isl_ctx)
        val = isl.UnionMap.read_from_str(self._isl_ctx, str(self._get_validity_constraints()))

        prox = isl.UnionMap.read_from_str(self._isl_ctx, str(self._get_proximity_constraints()))
        coin = isl.UnionMap.read_from_str(self._isl_ctx, str(self._get_coincidence_constraints()))

        # NOTE: domain has to be last so we generate the mem_man ops first
        domain = (
            isl.UnionSet.read_from_str(self._isl_ctx, str(self._get_domain()))
            .gist_params(params)
            .coalesce()
            .coalesce()
        )
        # print(f"Domain: {domain}")
        # print(f"Gisted domain: {domain.gist_params(params)}")
        # print(f"Intersected domain: {domain.intersect_params(params)}")
        # print(f"Gisted then intersected domain:
        # {domain.gist_params(params).intersect_params(params)}")
        # print(f"Intersected then gisted domain:
        # {domain.intersect_params(params).gist_params(params)}")

        val = val.intersect_domain(domain).intersect_range(domain).coalesce().coalesce()
        prox = prox.intersect_domain(domain).intersect_range(domain).coalesce().coalesce()
        coin = coin.intersect_domain(domain).intersect_range(domain).coalesce().coalesce()

        ## NOTE: this is important to ensure that lex_prevs do not go out of bounds

        val = self._try_simplify(val, domain, params)
        prox = self._try_simplify(prox, domain, params)
        coin = self._try_simplify(coin, domain, params)

        # print("Domains:")
        # print(str(domain).replace("; ", ";\n"))
        # print()
        # print("Validity:")
        # print(str(val).replace("; ", ";\n"))
        # print()
        # print(f"Proximity:")
        # print(str(prox).replace("; ", ";\n"))
        # print()
        # print(f"Coincidence:")
        # print(str(coin).replace("; ", ";\n"))

        # log.info("Domain: %s", str(domain))
        # log.info("Validity: %s", str(val))
        # log.info("Prox: %s", str(prox))

        sc = isl.ScheduleConstraints.on_domain(domain)
        assert sc.get_ctx() == self._isl_ctx
        # sc_str = str(sc)
        # sc = isl.ScheduleConstraints(sc_str, ctx=self._isl_ctx)
        # sc = sc.set_domain(domain)
        sc = sc.set_context(params)
        sc = sc.set_validity(val)
        sc = sc.set_proximity(prox)
        # sc = sc.set_coincidence(coin)
        return sc
