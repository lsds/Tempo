from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.datatypes import OpInId, OpOutId
from tempo.core.dependence_graph import PDG, DependencyData
from tempo.transformations.compilation_pass import Transformation
from tempo.transformations.incrementalization.apply_masks import apply_mask_to_edge, op_needs_mask
from tempo.transformations.incrementalization.inc_core import (
    IncKind,
    IncRoundCtx,
    PadInfo,
)
from tempo.transformations.incrementalization.incrementalization_mechanism import (
    create_inc_symbol_and_block_idxs,
    perform_incrementalization,
)
from tempo.utils import isl as isl_utils
from tempo.utils import logger
from tempo.utils.dg_utils import (
    is_expanded_dim,
    recursively_follow_dim_through_dependents_until_elimination,
    recursively_follow_op_in_dim_through_dependencies,
)

log = logger.get_logger(__name__)


@dataclass(frozen=True)
class MaskInfo:
    op_to_mask: top.TensorOp
    op_to_mask_src_op: top.TensorOp
    in_id: OpInId
    in_dim: int
    # mask_val: float #TODO: store mask value here.
    # TODO mask function should not receive that much information about padding.


@dataclass(frozen=True)
class DynDimEliminationInfo:
    elim_op: top.TensorOp
    in_id: OpInId
    elim_op_in_dim: int
    dyn_dim_size: ie.IntIndexValue
    dyn_src_op: top.TensorOp
    dyn_src_op_dom_var: ie.Symbol
    masks_needed: Sequence[MaskInfo]


def get_dyn_dim_idx(expr: ie.IndexSequence, static_bounds: Mapping[ie.Symbol, int]) -> int | None:
    for i, m in enumerate(expr.members):
        val_tuple = m.evaluate_shape(static_bounds)
        if len(val_tuple) == 0:
            continue
        val = val_tuple[0]
        if isinstance(val, int):
            continue
        elif isinstance(val, ie.IntIndexValue):
            val = val.partial_eval(static_bounds)
            if isinstance(val, ie.ConstInt):
                continue
        return i

    return None


def find_dyn_dim_in_sink_input_side(
    sink: top.TensorOp, expr: ie.IndexSequence, expr_dyn_dim: int
) -> int:
    num_slices_before_expr_dim = sum(
        1 for m in expr.members[:expr_dyn_dim] if isinstance(m, ie.Slice)
    )
    return num_slices_before_expr_dim


def find_dim_elimination_points(
    dg: PDG,
    snk: top.TensorOp,
    dep_data: DependencyData,
    src_op: top.TensorOp,
    dim: int,
) -> tuple[
    list[tuple[top.TensorOp, int, OpInId]],
    list[MaskInfo],
]:
    # Tracks the ops that eliminate dim
    eliminating_ops: list[tuple[top.TensorOp, int, OpInId]] = []
    masks_to_apply: list[MaskInfo] = []

    def should_recurr(
        dg: PDG,
        op: top.TensorOp,
        dep: DependencyData,
        depy_op: top.TensorOp,
        out_dim: int,
        in_dim: int,
    ) -> bool:
        # Keep going until dim is eliminated.
        return True

    def effect(
        dg: PDG,
        op: top.TensorOp,
        edge_data: DependencyData,
        src_op: top.TensorOp,
        op_out_dim: dict[OpOutId, int],
        op_in_dim: int,
        state: Any,
    ) -> tuple[PDG, Any]:
        needs_mask = op_needs_mask(dg, op, edge_data, op_in_dim)

        if needs_mask:
            masks_to_apply.append(
                MaskInfo(
                    op_to_mask=op,
                    op_to_mask_src_op=src_op,
                    in_id=edge_data.sink_in_idx,
                    in_dim=op_in_dim,
                )
            )

        if len(op_out_dim) == 0:  # Eliminated
            eliminating_ops.append((op, op_in_dim, edge_data.sink_in_idx))

        return dg, state

    recursively_follow_dim_through_dependents_until_elimination(
        dg,
        snk,
        dep_data,
        src_op,
        dim,
        should_recurr,
        effect,
    )
    return list(set(eliminating_ops)), list(set(masks_to_apply))


def find_all_relevant_pad_infos(dg: PDG, inc_ctx: IncRoundCtx, m: MaskInfo) -> list[PadInfo]:
    pad_infos: dict[top.PadOp, PadInfo] = {}

    if m.op_to_mask not in inc_ctx.op_mapping:
        return []

    def should_recurr(
        dg: PDG,
        op: top.TensorOp,
        depy_data: DependencyData,
        depy_op: top.TensorOp,
        op_out_dim: int,
        depy_out_dim: int,
        ctx: Any,
    ) -> bool:
        if isinstance(op, top.PadOp) and op.dim == op_out_dim:  # and op.is_mask_pad(): ???
            assert op in inc_ctx.padding_applied
            return False
        if is_expanded_dim(dg, op, op_out_dim):
            return False
        return True

    # PDG, TensorOp, int, Dict[OpInId, int]
    def effect(
        dg: PDG,
        op: top.TensorOp,
        op_out_dim: int,
        op_in_dims: dict[OpInId, int],
        ctx: Any,
    ) -> tuple[PDG, Any]:
        if isinstance(op, top.PadOp) and op.dim == op_out_dim:
            assert op in inc_ctx.padding_applied
            pad_infos[op] = inc_ctx.padding_applied[op]

        return dg, ctx

    recursively_follow_op_in_dim_through_dependencies(
        dg,
        inc_ctx.op_mapping[m.op_to_mask],
        m.in_id,
        m.in_dim,
        should_recurr,
        effect,
    )

    return list(pad_infos.values())


# TODO: maybe also capture the paths from elim to srcs?
def find_all_dependency_dynamic_ops(
    dg: PDG, elim_op: top.TensorOp, in_id: OpInId, in_dim: int
) -> set[top.TensorOp]:
    dynamic_ops: set[top.TensorOp] = set()

    dynamic_ops.add(elim_op)

    def should_recurr(
        dg: PDG,
        op: top.TensorOp,
        depy_data: DependencyData,
        depy_op: top.TensorOp,
        op_out_dim: int,
        depy_out_dim: int,
        ctx: Any,
    ) -> bool:
        ##NOTE: This should mean that they are included in dynamic_ops
        # if isinstance(op, top.IndexSliceOp):
        #    # NOTE: dim is sliced. It is effectively no longer the same dim.
        #    if op_out_dim == op.dim:
        #        return False
        # elif (
        #    #NOTE: if is expanded into a dynamic size
        #    is_expanded_dim(dg, op, op_out_dim)
        #    #and len(list(ie.lift_to_int_ie(op.sizes.at(op_out_dim)).vars_used())) != 0
        # ):
        #    # NOTE: Dim is expanded into existance here
        #    return False
        return True

    def effect(
        dg: PDG,
        op: top.TensorOp,
        op_out_dim: int,
        op_in_dims: dict[OpInId, int],
        ctx: Any,
    ) -> tuple[PDG, Any]:
        dynamic_ops.add(op)
        return dg, ctx

    recursively_follow_op_in_dim_through_dependencies(
        dg,
        elim_op,
        in_id,
        in_dim,
        should_recurr,
        effect,
    )

    return dynamic_ops


def setup_round_ctx(
    ctx: CompilationCtx,
    inc_dims_and_input_ids_grouped_by_start_op: dict[top.TensorOp, list[tuple[OpInId, int]]],
    dim_size: ie.IntIndexValue,
    round_num: int,
    block_size: int,
    all_dyn_ops: set[top.TensorOp],
) -> IncRoundCtx:
    # NOTE: eg dim_size = t+1-0 or T-t.
    num_blocks = ie.Ceil((dim_size) / block_size)
    num_blocks = isl_utils.simplify_int_index_value(num_blocks, known_symbols=ctx.dg.static_bounds)

    inc_var, block_idx = create_inc_symbol_and_block_idxs(
        ctx.dg, round_num, block_size, num_blocks, inc_var_name="ds", allow_reuse_symbol=False
    )

    inc_start_ops = set(inc_dims_and_input_ids_grouped_by_start_op.keys())

    ctx = IncRoundCtx(
        kind=IncKind.STATIFYING,
        inc_start_ops=inc_start_ops,  # type: ignore
        start_op_inputs_and_dims=inc_dims_and_input_ids_grouped_by_start_op,  # type: ignore
        inc_var=inc_var,
        dim_size=dim_size,
        block_size=block_size,
        block_idx=block_idx,
        num_blocks=num_blocks,
        comp_ctx=ctx,
        # add_dim=add_dim,
        needs_incrementalization=lambda op, comp_ctx, initial: True,
        # NOTE: Has to be incremental, because bound is dynamic.
        finalize_incremental=True,
        all_dynamic_ops=all_dyn_ops,
    )

    return ctx


class StatifyingIncrementalize(Transformation):
    """This transformation helps minimize dynamic regions of the graph by incrementalizing the
    execution into fixed size blocks.
    """

    def __init__(self, ctx: CompilationCtx):
        self.ctx = ctx

    def _run(self) -> tuple[PDG, bool]:
        # Keeps track of how many incrementalisations have been carried out

        op_inc_count = 0
        new_dg = self.ctx.dg
        tot_masks_applied = 0
        tot_pads_applied = 0

        round_num = 0

        # NOTE: Keep going until we can't find any more dynamic dim eliminations.
        while True:
            # if round_num > 0:
            #   break
            # NOTE: You are not allowed to be dynamic on two dimensions at once (especially
            # not using the same variable)
            # Thus, we can group dim_infos by size to get all elimination points for a given size.

            all_elimination_infos = self._get_all_dyn_dim_eliminations(new_dg)
            if len(all_elimination_infos) == 0:
                break

            # NOTE: Now, organize them by dynamic dim size.
            dim_infos_by_size: dict[ie.IntIndexValue, list[DynDimEliminationInfo]] = {}
            for info in all_elimination_infos:
                dim_infos_by_size.setdefault(info.dyn_dim_size, []).append(info)

            # NOTE: Finally, pick one dynamic dim size and do the incrementalization for that.
            dyn_dim_size = list(dim_infos_by_size.keys())[0]
            chosen_infos = dim_infos_by_size[dyn_dim_size]

            all_dyn_ops = {
                op
                for info in all_elimination_infos
                for op in find_all_dependency_dynamic_ops(
                    new_dg, info.elim_op, info.in_id, info.elim_op_in_dim
                )
            }

            max_val_expr = isl_utils.int_index_val_max(
                dyn_dim_size,
                known_symbols=self.ctx.dg.static_bounds,
            )
            max_val = -float("inf")
            if max_val_expr is not None:
                assert isinstance(max_val_expr, ie.ConstInt), "TODO: support D0 as max val?"
                max_val = max_val_expr.const

            block_size = int(min(self.ctx.exec_cfg.inc_statify_block_size, max_val))

            inc_dims_and_input_ids_grouped_by_start_op: dict[
                top.TensorOp, set[tuple[OpInId, int]]
            ] = {}
            for info in chosen_infos:
                inc_dims_and_input_ids_grouped_by_start_op.setdefault(info.elim_op, set()).add(
                    (info.in_id, info.elim_op_in_dim)
                )
            inc_dims_and_input_ids_grouped_by_start_op = {
                k: list(v) for k, v in inc_dims_and_input_ids_grouped_by_start_op.items()
            }

            inc_ctx = setup_round_ctx(
                self.ctx,
                inc_dims_and_input_ids_grouped_by_start_op,
                dyn_dim_size,
                round_num,
                block_size,
                all_dyn_ops,
            )
            new_dg = perform_incrementalization(new_dg, inc_ctx)

            # TODO: we need to add masks to dynamic index_slices too?
            masks_needed = {mi for info in chosen_infos for mi in info.masks_needed}
            # print(f"masks_needed: {masks_needed}")
            tot_pads_applied += len(inc_ctx.padding_applied)
            # DGRenderer(
            #    CompilationCtx(new_dg, self.ctx.analysis_ctx, self.ctx.exec_cfg),
            #    self.ctx.exec_cfg.path + f"statifying_incrementalize_{round_num}",
            # ).render()

            for m in masks_needed:
                relevant_pad_infos = find_all_relevant_pad_infos(new_dg, inc_ctx, m)
                if len(relevant_pad_infos) == 0:
                    continue
                mapped_snk_op = inc_ctx.op_mapping[m.op_to_mask]

                mapped_src_op, edge_data = new_dg.get_flat_direct_dependencies(mapped_snk_op)[
                    m.in_id
                ]

                apply_mask_to_edge(
                    self.ctx,
                    mapped_snk_op,
                    edge_data,
                    mapped_src_op,
                    info.dyn_src_op_dom_var,
                    m.in_dim,
                    relevant_pad_infos,
                )
                tot_masks_applied += 1
            # DGRenderer(
            #    CompilationCtx(new_dg, self.ctx.analysis_ctx, self.ctx.exec_cfg),
            #    self.ctx.exec_cfg.path + f"statifying_incrementalize_{round_num}",
            # ).render()
            # new_ctx_, _, _ = DeadCodeElimination(
            #    CompilationCtx(new_dg, self.ctx.analysis_ctx, self.ctx.exec_cfg)
            # ).run()
            # new_dg = new_ctx_.dg
            from tempo.core import global_objects as glob

            glob.set_active_dg(new_dg)

            op_inc_count += len(inc_ctx.op_mapping)
            round_num += 1

        log.info(
            "Performed %d incrementalizations in %d rounds (%d masks, %d pads)",
            op_inc_count,
            round_num,
            tot_masks_applied,
            tot_pads_applied,
        )

        return new_dg, op_inc_count > 0 or tot_masks_applied > 0

    def _get_var_and_snk_side_spatial_dim(
        self,
        new_dg: PDG,
        dyn_dim_snk: top.TensorOp,
        dyn_dim_src: top.TensorOp,
        dyn_dim_dep: DependencyData,
    ) -> tuple[ie.Symbol | None, int | None]:
        dyn_temporal_dim_idx = get_dyn_dim_idx(dyn_dim_dep.expr, new_dg.static_bounds)

        # NOTE: First check for dynamism in the edge
        if dyn_temporal_dim_idx is not None:
            snk_side_spatial_dim = find_dyn_dim_in_sink_input_side(
                dyn_dim_snk, dyn_dim_dep.expr, dyn_temporal_dim_idx
            )
            dyn_dim_src_var = dyn_dim_src.domain.variables[dyn_temporal_dim_idx]
            return dyn_dim_src_var, snk_side_spatial_dim

        # NOTE: dynamism may also come from expands and or index_slice and or index_select...
        if dyn_dim_src.is_dynamic():
            if isinstance(dyn_dim_src, top.ExpandOp):
                szs = dyn_dim_src.sizes
                vars_ = list(szs.vars_used())
                if len(vars_) == 1:
                    dyn_dim_src_var = vars_[0]
                    idx_var_used = tuple(
                        i
                        for i, e in enumerate(szs._shape)
                        if isinstance(e, ie.IntIndexValue) and dyn_dim_src_var in e.vars_used()
                    )
                    if len(idx_var_used) == 1:
                        snk_side_spatial_dim = idx_var_used[0] + dyn_dim_dep.expr.num_slices()
                return dyn_dim_src_var, snk_side_spatial_dim
            elif isinstance(dyn_dim_src, top.IndexSliceOp):
                vars_used = list(ie.lift_to_int_ie(dyn_dim_src.length).vars_used())
                if len(vars_used) == 1:
                    dyn_dim_src_var = vars_used[0]
                    dim = dyn_dim_src.dim
                    snk_side_spatial_dim = dim + dyn_dim_dep.expr.num_slices()
                    return dyn_dim_src_var, snk_side_spatial_dim
            elif isinstance(dyn_dim_src, top.IndexSelectOp):
                ...  # TODO

        return None, None

    def _get_all_dyn_dim_eliminations(self, new_dg: PDG) -> list[DynDimEliminationInfo]:
        elimination_infos: list[DynDimEliminationInfo] = []

        # NOTE: (dim_size, elim_op, dim, in_idx)
        for dyn_dim_snk, dyn_dim_src, dyn_dim_dep in new_dg.get_all_edges():
            dyn_dim_src_var, snk_side_spatial_dim = self._get_var_and_snk_side_spatial_dim(
                new_dg, dyn_dim_snk, dyn_dim_src, dyn_dim_dep
            )
            if dyn_dim_src_var is None:
                continue

            assert snk_side_spatial_dim is not None

            dynamic_dim_size = new_dg.get_input_shape(dyn_dim_snk, dyn_dim_dep.sink_in_idx).at(
                snk_side_spatial_dim
            )

            # NOTE: This checks if the dynamic dim has already been padded by a previous round
            # of incrementalization. (Or indeed from AD ANY padding.)
            if (
                isinstance(dyn_dim_snk, top.PadOp)
                and not dyn_dim_snk.is_static()
                and dyn_dim_snk.dim == snk_side_spatial_dim
            ):
                log.debug("Skipping pad for %s because it is already padded", dyn_dim_snk)
                continue

            if isinstance(dynamic_dim_size, (int, ie.ConstInt)):
                # NOTE: dynamic access produces constant size, so we don't need to do anything.
                continue

            assert isinstance(dynamic_dim_size, ie.IntIndexValue), (
                f"dynamic_dim_size: {type(dynamic_dim_size)}, {dynamic_dim_size}"
            )

            elimination_ops, masks_needed = find_dim_elimination_points(
                new_dg, dyn_dim_snk, dyn_dim_dep, dyn_dim_src, snk_side_spatial_dim
            )

            if len(elimination_ops) == 0:
                # No elimination points found, skip. Maybe the user wants to sink a dynamic dim.
                continue

            for elim_op, elim_op_in_dim, in_id in elimination_ops:
                elimination_infos.append(
                    DynDimEliminationInfo(
                        elim_op=elim_op,
                        in_id=in_id,
                        elim_op_in_dim=elim_op_in_dim,
                        dyn_dim_size=dynamic_dim_size,
                        dyn_src_op=dyn_dim_src,
                        dyn_src_op_dom_var=dyn_dim_src_var,
                        masks_needed=masks_needed,
                    )
                )

        return elimination_infos
