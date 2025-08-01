from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

import islpy as isl

from tempo.core import isl_types as islt
from tempo.core import tensor_ops as top
from tempo.core.analysis_ctx import AnalysisCtx
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import OpId, OpInId, OpOutId, TensorId
from tempo.core.dependence_graph import PDG, DependencyData
from tempo.core.dtype import DataType
from tempo.core.shape import Shape
from tempo.core.storage_methods import BlockStore
from tempo.transformations.compilation_pass import Transformation
from tempo.utils import logger
from tempo.utils.isl import (
    dependence_to_isl_map,
    get_parameters_and_var_bounds_strs,
    op_id_to_exec_name,
    rename_union_set_tuples,
)

log = logger.get_logger(__name__)


def _find_donatable_args_conservative(  # noqa: C901
    new_dg: PDG, analysis_ctx: AnalysisCtx, op: top.TensorOp
) -> Set[int]:
    donatable_args: Set[int] = set()
    for depy_op, dep_data in new_dg.get_flat_direct_dependencies(op):
        # NOTE: If it is not a point, it requires concatenation, and thus, can be donated.
        # That is, unless it is stored in a block tensor.
        if not dep_data.expr.is_point():
            storages = analysis_ctx._tensor_storage_classes

            if storages is not None:
                if isinstance(storages[TensorId(depy_op.op_id, dep_data.src_out_idx)], BlockStore):
                    # If the source is block stored, we cannot donate
                    continue

            donatable_args.add(int(dep_data.sink_in_idx))
            continue
        if len(
            new_dg.get_tensor_flat_direct_dependents(TensorId(depy_op.op_id, dep_data.src_out_idx))
        ) == 1 and op.domain.is_contained_in(depy_op.domain):
            is_valid_donation = True
            # NOTE: this is essentially intersecting the domains and checking if the result
            # is a non-constant point
            # dep_data.expr.intersect_domain(op.domain).is_basis
            for j, v in enumerate(depy_op.domain.variables):
                # If op domain does not have dim it does not matter what it is.
                if op.domain.has_dim(v):
                    if dep_data.expr.members[j].is_constant() or (
                        not dep_data.expr.members[j].is_point()
                    ):
                        # if not dep_data.expr.members[j].equivalent(v):
                        is_valid_donation = False
                        break
            if is_valid_donation:
                donatable_args.add(int(dep_data.sink_in_idx))
    return donatable_args


# def points_where_a_executes_before_b(
#        dg: DependenceGraph,
#        a_op: top.TensorOp,
#        b_op: top.TensorOp,
#        isl_ctx: isl.Context,
# ) -> isl.UnionSet:
#    schedule = dg.analysis_ctx.isl_execution_schedule
#    sched_map: isl.UnionMap = schedule.get_map()
#
#    universe_bounds_str = ",".join(
#        [
#            (f"{b}={dg.static_bounds[b]}" if b in dg.static_bounds else f"{b}")
#            for b in dg.universe.parameters
#        ]
#    )
#    params_set = isl.Set(f"[{universe_bounds_str}] -> {{ : }}", context=isl_ctx)
#
#    a_name = op_id_to_exec_name(a_op.op_id)
#    b_name = op_id_to_exec_name(b_op.op_id)
#
#    a_dom = dg.analysis_ctx.get_or_make_domain(a_op)
#    b_dom = dg.analysis_ctx.get_or_make_domain(b_op)
#
#    a_sched = sched_map.intersect_domain(a_dom)  # op[p] -> theta(op[p])
#    b_sched = sched_map.intersect_domain(b_dom)  # op[p] -> theta(op[p])
#
#    can_donate = True
#    our_composed_map = our_rev_dep_map.apply_range(
#        our_sched
#    )  # dy[p'] -> theta(op[e-1(p')])
#
#    our_points_scheduled = dy_dom.apply(our_composed_map)
#
#    additional_val_constraints = None
#    for competitor_op, competitor_data in dg.get_flat_direct_dependents(dy_op):
#        if competitor_op.op_id == op.op_id:
#            continue
#        competitor_name = op_id_to_exec_name(competitor_op.op_id)
#        competitor_dep_map: isl.UnionMap = dependence_to_isl_map(
#            competitor_data.expr,
#            competitor_op.domain,
#            dy_op.domain,
#            competitor_name,
#            dy_name,
#            competitor_data.cond,
#            ctx=isl_ctx,
#        )
#
#        competitor_rev_map = (
#            competitor_dep_map.reverse()
#        )  # dy[p'] -> competitor[e-1(p')]
#
#        competitor_dom = competitor_dep_map.domain()
#        competitor_dom = competitor_dom.intersect_params(params_set)
#        competitor_sched = sched_map.intersect_domain(competitor_dom)
#
#        competitor_composed_map = competitor_rev_map.apply_range(
#            competitor_sched
#        )  # dy[p'] -> theta(competitor[e-1(p')])
#
#        # TODO could we just take the range of the composed map?
#        competitor_points_scheduled = dy_dom.apply(
#            competitor_composed_map
#        )  # theta(competitor[e-1(p')])
#
#        points_where_competitor_is_after_us = our_points_scheduled.lex_le_union_set(
#            competitor_points_scheduled
#        )
#        points_where_competitor_is_after_us = (
#            points_where_competitor_is_after_us.intersect(
#                dy_dom_identity.apply_domain(our_composed_map).apply_range(
#                    competitor_composed_map
#                )
#            )
#        )


def _find_donatable_args_isl_vectorized(  # noqa: C901
    ctx: CompilationCtx,
    analysis_ctx: AnalysisCtx,
    op: top.TensorOp,
    known_donations: Set[int],
) -> Set[int]:
    dg = ctx.dg
    analysis_ctx = ctx.analysis_ctx
    schedule = analysis_ctx.isl_execution_schedule
    isl_ctx = ctx.analysis_ctx.isl_ctx
    sched_map: islt.UnionMap = schedule.get_map()

    # universe_bounds_str = ",".join(
    #    [
    #        (f"{b}={dg.static_bounds[b]}" if b in dg.static_bounds else f"{b}")
    #        for b in dg.universe.parameters
    #    ]
    # )
    universe_params_str, _ = get_parameters_and_var_bounds_strs(dg.universe)
    params_set = isl.Set(f"[{universe_params_str}] -> {{ : }}", context=isl_ctx)

    donatable_args: Set[int] = set()

    our_name = op_id_to_exec_name(op.op_id)

    # Iterate the dependencies we are interested in possibly donating
    for dy_op, dy_data in dg.get_flat_direct_dependencies(op):
        # If the dependence is a constant or a range, we are not interested
        # (This is already handled by the conservative check)
        if (not dy_data.expr.is_point()) or dy_data.expr.is_constant():
            continue

        # if len(dy_data.expr.enumerate_all_cond_branches()) > 1:
        #    continue

        # NOTE: if it is a min/max situation, it might depend on same point several times
        if not dy_data.expr.simplify_mins_and_maxes().struct_eq(dy_data.expr):
            continue

        # if len(dy_data.expr.enumerate_all_cond_branches()) > 1:
        #    continue

        # NOTE: If the op domain is bigger, it will point to the same point several times
        if len(op.domain) > len(dy_op.domain):
            continue

        # TODO if any competitor for donation is not a point access, we cannot donate
        # because our analysis is not powerful enough to handle this case???

        if dy_data.sink_in_idx in known_donations:
            continue

        # Having passed other checks, we now need to check if op is always the last to access
        # any given point of dy_op's domain. If so, we can donate.

        # We know the dependence is a moving point.
        # Now we need to check that every other dependent accesses it before us.
        dy_name = op_id_to_exec_name(dy_op.op_id)
        dy_isl_dom = analysis_ctx.get_or_make_domain(dy_op)
        # TODO do we need to rename? doubt it.
        dy_dom = rename_union_set_tuples(dy_isl_dom, dy_name).intersect_params(params_set)
        dy_dom_identity = dy_dom.identity()  # Maps each domain point to itself

        dep_map: islt.UnionMap = dependence_to_isl_map(  # op[p] -> dy[e(p)]
            dy_data.expr,
            op.domain,
            dy_op.domain,
            our_name,
            dy_name,
            dy_data.cond,
            ctx=isl_ctx,
        )
        our_rev_dep_map = dep_map.reverse()  # dy[p'] -> op[e-1(p')]

        # our_dom = dep_map.domain()
        our_dom = analysis_ctx.get_or_make_domain(op)
        our_sched = sched_map.intersect_domain(our_dom)  # op[p] -> theta(op[p])

        can_donate = True
        our_composed_map = our_rev_dep_map.apply_range(our_sched)  # dy[p'] -> theta(op[e-1(p')])

        our_points_scheduled = dy_dom.apply(our_composed_map)

        additional_val_constraints = None
        for competitor_op, competitor_data in dg.get_tensor_flat_direct_dependents(
            TensorId(dy_op.op_id, dy_data.src_out_idx)
        ):
            if competitor_op.op_id == op.op_id:
                continue
            competitor_name = op_id_to_exec_name(competitor_op.op_id)
            competitor_dep_map: islt.UnionMap = dependence_to_isl_map(
                competitor_data.expr,
                competitor_op.domain,
                dy_op.domain,
                competitor_name,
                dy_name,
                competitor_data.cond,
                ctx=isl_ctx,
            )

            competitor_rev_map = competitor_dep_map.reverse()  # dy[p'] -> competitor[e-1(p')]

            # competitor_dom = competitor_dep_map.domain()
            competitor_dom = analysis_ctx.get_or_make_domain(competitor_op)
            competitor_dom = competitor_dom.intersect_params(params_set)
            competitor_sched = sched_map.intersect_domain(competitor_dom)

            competitor_composed_map = competitor_rev_map.apply_range(
                competitor_sched
            )  # dy[p'] -> theta(competitor[e-1(p')])

            # TODO could we just take the range of the composed map?
            competitor_points_scheduled = dy_dom.apply(
                competitor_composed_map
            )  # theta(competitor[e-1(p')])

            points_where_competitor_is_after_us = our_points_scheduled.lex_le_union_set(
                competitor_points_scheduled
            )
            # print(f"dy_op: {dy_op}")
            # print(f"us: {op}")
            # print(f"competitor_op: {competitor_op}")
            # print(f"dy_dom: {dy_dom}")
            # print(f"our dom: {our_dom}")
            # print(f"competitor dom: {competitor_dom}")
            # print(f"our points scheduled: {our_points_scheduled}")
            # print(f"points_where_competitor_is_after_us: {points_where_competitor_is_after_us}")
            # print(f"competitor points scheduled: {competitor_points_scheduled}")

            # TODO: it looks like this is at fault. It's yielding an empty map
            # print(f"dy_dom_identity: {dy_dom_identity}")
            # print(f"our_composed_map: {our_composed_map}")
            # print(f"competitor_composed_map: {competitor_composed_map}")
            dy_dom_mapped_to_depy_doms = dy_dom_identity.apply_domain(our_composed_map).apply_range(
                competitor_composed_map
            )
            # print(f"dy_dom_mapped_to_depy_doms: {dy_dom_mapped_to_depy_doms}")
            points_where_competitor_is_after_us = points_where_competitor_is_after_us.intersect(
                dy_dom_mapped_to_depy_doms
            )
            # print()
            # print()
            # TODO: there might be a way to start with this map, then compare only these points.
            # point_pairs = dy_dom_identity
            #   .apply_domain(our_composed_map).apply_range(competitor_composed_map)
            # points_where_competitor_is_after_us =
            #   point_pairs.intersect(isl.Map(f"[i] -> [j] : i < j"))

            # Check if any dependent points are scheuled later than our dependent points
            ## To compare union sets, we need to use

            if not points_where_competitor_is_after_us.is_empty():
                # If any dependent point is scheduled after our point, we cannot donate
                can_donate = False
                break
            else:
                # NOTE: it is empty, is the reverse empty?
                # TODO: this is a fix needed because for some cases it breaks...
                rev_dy_dom_mapped_to_depy_doms = dy_dom_identity.apply_domain(
                    competitor_composed_map
                ).apply_range(our_composed_map)
                if (
                    dy_dom_mapped_to_depy_doms.is_empty()
                    and rev_dy_dom_mapped_to_depy_doms.is_empty()
                ):
                    can_donate = False
                    break

                additional_val_constraint = (
                    dy_dom_identity.apply_range(our_rev_dep_map)
                    .apply_domain(competitor_rev_map)
                    .intersect_range(our_dom)
                    .intersect_domain(competitor_dom)
                )
                if additional_val_constraints is None:
                    additional_val_constraints = additional_val_constraint
                else:
                    additional_val_constraints = additional_val_constraints.union(
                        additional_val_constraint
                    )
        if can_donate:
            donatable_args.add(dy_data.sink_in_idx)
            if additional_val_constraints is not None:
                if analysis_ctx._additional_val_constraints is None:
                    analysis_ctx._additional_val_constraints = additional_val_constraints
                else:
                    analysis_ctx._additional_val_constraints = (
                        analysis_ctx._additional_val_constraints.union(additional_val_constraints)
                    )

    return donatable_args


def _filter_undonatable_args(
    exec_cfg: ExecutionConfig, dg: PDG, op: top.TensorOp, args_to_donate: Tuple[int, ...]
) -> Tuple[int, ...]:
    for i, (depy_op, depy_data) in enumerate(dg.get_flat_direct_dependencies(op)):
        if (
            i in args_to_donate
            and exec_cfg.enable_symbol_prealloc_store
            and isinstance(depy_op, top.EvalSymbolOp)
            and depy_data.is_unconditional_basis()
        ):
            args_to_donate = tuple(x for x in args_to_donate if x != i)
    return args_to_donate


def _donate_only_needed_args(
    dg: PDG, op: top.TensorOp, args_to_donate: Tuple[int, ...]
) -> Tuple[int, ...]:
    # If it is a UDF, we cannot donate any arguments to it.
    if isinstance(op, top.UDFOp):
        return ()

    # Basically, we want to donate only the arguments that are needed. Which is two conditions:
    # 1. There is an output of the op that has the same shape and dtype as the donated input
    # 2. The output is not already covered by another donation

    # We first find the outputs that are covered by a donation
    shapes = dg.get_output_shapes(op)
    dtypes = dg.get_output_dtypes(op)
    oid_to_spec: Dict[OpOutId, Tuple[Shape, DataType]] = {
        oid: (
            shapes[oid].simplify().try_resolve(dg.static_bounds).simplify(),
            dtypes[oid],
        )
        for oid in shapes
    }
    spec_to_oid: Dict[Tuple[Shape, DataType], Set[OpOutId]] = {
        spec: set() for _, spec in oid_to_spec.items()
    }
    for oid, spec in oid_to_spec.items():
        spec_to_oid[spec].add(oid)

    outputs_covered_by_donation: Set[OpOutId] = set()
    donations_to_keep: Set[int] = set()
    if isinstance(op, top.MergeOp):
        assert len(args_to_donate) > 0
        outputs_covered_by_donation.add(OpOutId(0))
        # NOTE: Max because the last condition is usually the least restrictive
        donations_to_keep.add(max(args_to_donate))
    else:
        for i in args_to_donate:
            in_id = OpInId(i)
            shape = (
                dg.get_input_shape(op, in_id).simplify().try_resolve(dg.static_bounds).simplify()
            )
            dtype = dg.get_input_dtype(op, in_id)

            if (shape, dtype) in spec_to_oid:
                oids = spec_to_oid[(shape, dtype)]

                for oid in oids:
                    if oid in outputs_covered_by_donation:
                        continue

                    outputs_covered_by_donation.add(oid)
                    donations_to_keep.add(i)

    return tuple(sorted(donations_to_keep))


class AnalyseDonations(Transformation):
    def _run(self) -> Tuple[PDG, bool]:
        new_dg = self.ctx.dg

        tensor_is_donated: Dict[TensorId, bool] = defaultdict(lambda: False)
        donation_map: Dict[OpId, Tuple[int, ...]] = {}
        all_donation_map: Dict[OpId, Tuple[int, ...]] = {}
        donated_to: Dict[TensorId, Tuple[top.TensorOp, DependencyData]] = {}

        total_args = 0
        donated_args = 0
        for op in new_dg.nodes:
            if isinstance(op, top.UDFOp) or not self.ctx.exec_cfg.enable_donation_analysis:
                donatable_args = set()
            else:
                donatable_args_cons = _find_donatable_args_conservative(
                    new_dg, self.ctx.analysis_ctx, op
                )
                donatable_args_isl = _find_donatable_args_isl_vectorized(
                    self.ctx, self.ctx.analysis_ctx, op, donatable_args_cons
                )
                donatable_args = donatable_args_cons.union(donatable_args_isl)
            args_to_donate = tuple(donatable_args)
            all_args_to_donate = tuple(args_to_donate)
            args_to_donate = (
                _donate_only_needed_args(new_dg, op, args_to_donate) if args_to_donate else ()
            )
            args_to_donate = (
                _filter_undonatable_args(self.ctx.exec_cfg, new_dg, op, args_to_donate)
                if args_to_donate
                else ()
            )
            donation_map[op.op_id] = args_to_donate
            all_donation_map[op.op_id] = all_args_to_donate

            for src_op, src_data in new_dg.get_flat_direct_dependencies(op):
                tid = TensorId(src_op.op_id, src_data.src_out_idx)
                if int(src_data.sink_in_idx) in args_to_donate and (
                    src_data.expr.is_point() and not src_data.expr.is_constant()
                ):
                    donated_to[tid] = (op, src_data)
                    tensor_is_donated[tid] = True
                else:
                    tensor_is_donated[tid] = False

            total_args += len(new_dg.get_flat_direct_dependencies(op))
            donated_args += len(args_to_donate)

        percentage_donated = round(donated_args / total_args * 100, 2) if total_args > 0 else 0

        path = Path(self.ctx.exec_cfg.path) / "donations.txt"
        with open(path, "w") as f:
            f.write(str(percentage_donated))

        log.info(
            "Found %d donatable arguments out of %d total arguments (%.2f%%)",
            donated_args,
            total_args,
            percentage_donated,
        )
        self.ctx.analysis_ctx._donatable_args = donation_map
        self.ctx.analysis_ctx._all_donatable_args = donation_map
        self.ctx.analysis_ctx._tensor_is_donated = tensor_is_donated
        return new_dg, True
