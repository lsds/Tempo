import dataclasses
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import tempo.core.global_objects as glob
from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.datatypes import OpInId, OpOutId
from tempo.core.dependence_graph import PDG, DependencyData
from tempo.core.symbolic_tensor import (
    SymbolicTensor,
    _get_symbolic_tensor_for_op_output,
    lift_to_symbolic_tensor,
    translate_ie_to_st,
)
from tempo.core.tensor_ops import TensorOp
from tempo.transformations.compilation_pass import CompilationCtx, Transformation
from tempo.transformations.optimizer.algebraic.optimizer import AlgebraicOptimizer
from tempo.transformations.vectorization.core import OpVecCtx
from tempo.transformations.vectorization.vec_rules import OP_VEC_RULES, GoToBackOfQueueError
from tempo.transformations.vectorization.vectorizibility import (
    filter_ops_to_vectorize,
    get_ops_to_vectorize,
)
from tempo.utils import logger
from tempo.utils.dg_utils import (
    is_index_src_tensor_param,
    is_non_dimensional_param,
)

log = logger.get_logger(__name__)


def _promote_index_ops(
    dg: PDG,
    ops_to_vectorize: Set[TensorOp],
) -> None:
    """
    INDEX SELECT CASES

    | Ops Vectorized | Can use `index_select`? | Needs reshape of index and output?
    | -------------- | ----------------------- | ------------------------
    | index          | Yes                     | ️ Yes, if index is ≥1D
    | tensor         | Yes (needs dim++)       |  No
    | tensor, index  | No                      |  N/A

    INDEX ADD CASES
    We always promote to scatteradd.

    # TODO: Other ops that will likely need special cases:
    # - index slice
    # - conv
    """
    for op in set(ops_to_vectorize):
        if type(op) is top.IndexSelectOp or type(op) is top.IndexAddOp:
            deps = dg.get_flat_direct_dependencies(op)

            TENSOR_PARAM_IDX = 0
            INDEX_PARAM_IN_IDX = 1
            SRC_TENSOR_IN_IDX = 2

            ten = deps[TENSOR_PARAM_IDX]
            ind = deps[INDEX_PARAM_IN_IDX]

            # ten_vec = ten[0] in ops_to_vectorize
            # ind_vec = ind[0] in ops_to_vectorize

            tensor_s: SymbolicTensor = _get_symbolic_tensor_for_op_output(  # type: ignore
                dg, ten[0], ten[1].src_out_idx
            ).symbolic_index(ten[1].expr)

            index_s: SymbolicTensor = _get_symbolic_tensor_for_op_output(
                dg, ind[0], ind[1].src_out_idx
            ).symbolic_index(ind[1].expr)

            dg_ops_start = set(dg.nodes)

            new_s: Optional[SymbolicTensor] = None
            if type(op) is top.IndexAddOp:
                # TODO: There are probably opportunities for nuance like below, but these
                # ops are rare.
                src = deps[SRC_TENSOR_IN_IDX]
                src_s = _get_symbolic_tensor_for_op_output(
                    dg, src[0], src[1].src_out_idx
                ).symbolic_index(src[1].expr)
                new_s = tensor_s._scatter_based_index_add(dim=op.dim, index=index_s, src=src_s)
            # else:
            #    if ten_vec and ind_vec:
            #        # NOTE: _gather_based_index_select handles the broadcasting
            #        # NOTE: This gather will now need to be vectorized.
            #        new_s = tensor_s._gather_based_index_select(dim=op.dim, index=index_s)

            if new_s:
                dg.move_dependents(op, new_s.op)
                dg.remove_op(op)
                ops_to_vectorize.remove(op)

            # NOTE: we do it this way because gather may register many unsq and expand
            dg_ops_end = set(dg.nodes)
            new_ops = dg_ops_end - dg_ops_start
            for new_op in new_ops:
                ops_to_vectorize.add(new_op)


class Vectorize(Transformation):
    def __init__(self, ctx: CompilationCtx):
        super().__init__(ctx)

        self.op_vectorizations: Dict[
            TensorOp, Tuple[List[ie.Symbol], List[ie.IntIndexValueLike]]
        ] = defaultdict(lambda: ([], []))

    def _vectorize_all_ops(
        self,
        dg: PDG,
        dim: ie.Symbol,
        size: ie.IntIndexValueLike,
        op_mapping: Dict[TensorOp, TensorOp],
        op_vectorizations: Dict[TensorOp, Tuple[List[ie.Symbol], List[ie.IntIndexValueLike]]],
        ops_to_vectorize: Set[TensorOp],
    ) -> None:
        remaining_ops = list(ops_to_vectorize)

        while remaining_ops:
            op = remaining_ops.pop(0)  # Get from front of list

            # Rule to vectorize this op (from the dictionary of rules)
            # TensorOp We know a rule exists in the dictionary,
            # as the case where there is no rule has already
            # been handled in _run
            vec_rule = OP_VEC_RULES[type(op)]

            # Applying the vectorization rule to the op
            op_vec_ctx = OpVecCtx(
                dg,
                op,
                dim,
                size,
                op_mapping,
                op_vectorizations,
                ops_to_vectorize,
            )

            try:
                new_op = vec_rule(op_vec_ctx)
                # new_op.tags[f"VECTORIZED_{dim}"] = True
                # print(f"Vectorizing {op} over {dim} with size {size} to {new_op}")
                assert new_op is not None
            except GoToBackOfQueueError:
                remaining_ops.append(op)  # Add to back of list
                continue

    def vectorize_edges(  # noqa: C901
        self,
        dg: PDG,
        op_mapping: Dict[TensorOp, TensorOp],
        dim: ie.Symbol,
        size: ie.IntIndexValueLike,
        ops_to_vectorize: Set[TensorOp],
    ) -> None:
        for snk, src, data in dg.get_all_edges():
            for k, v in snk.tags.items():
                glob.active_tags.append((k, v))
            for k, v in src.tags.items():
                glob.active_tags.append((k, v))
            # Get the vectorized versions of each operation from the dictionary
            # If the op has not been vectorized, new_x = op_x
            new_snk = op_mapping.get(snk, snk)
            new_src = op_mapping.get(src, src)

            src_vectorized = self._has_been_vectorized(new_src, dim)
            snk_vectorized = self._has_been_vectorized(new_snk, dim)

            # There is a total of 5 cases:
            # - 1. snk and src are both vec -> simply remove dim from edge expr
            # - 2. neither are vec -> do nothing
            # - 3. src is vec, snk is not -> spatially index src at symbolic index value
            # - snk is vec, src is not:
            #   - src has dim -> fully symbolic index src (0:D), permuting any dims back if needed
            #   - src does not have dim -> expand src by adding (D,) to its shape

            if src_vectorized:
                # NOTE: has to be the old src, because new_src does not have dim in domain anymore
                dim_idx = src.domain.find_variable_index(dim)
                if snk_vectorized:  # Both were vectorized
                    self.vectorized_src_and_snk(
                        dg, dim, dim_idx, size, new_snk, new_src, data, ops_to_vectorize
                    )
                else:  # src is vectorized, but snk is not
                    self.vectorized_src_only(
                        dg, dim, dim_idx, size, new_snk, new_src, data, ops_to_vectorize
                    )
            else:  # src is not vectorized
                if snk_vectorized:  # src is not vectorized, but snk is
                    # Was the src not vectorized because it does not have the dimension, or
                    # due to another reason?
                    if src.domain.has_dim(dim):
                        # NOTE: has to be the old src,
                        # because new_src does not have dim in domain anymore
                        dim_idx = src.domain.find_variable_index(dim)
                        self.vectorized_snk_only_src_has_dim(
                            dg,
                            dim,
                            dim_idx,
                            size,
                            new_snk,
                            new_src,
                            data,
                            ops_to_vectorize,
                        )
                    else:
                        self.vectorized_snk_only_src_not_has_dim(
                            dg,
                            dim,
                            -1,
                            size,
                            snk,
                            new_snk,
                            new_src,
                            data,
                            ops_to_vectorize,
                        )
                else:  # If neither are vectorized, the edge connecting A and B remains unchanged
                    assert new_src.op_id == src.op_id
                    assert new_snk.op_id == snk.op_id

            for _ in snk.tags:
                glob.active_tags.pop()
            for _ in src.tags:
                glob.active_tags.pop()
            # If neither are vectorized, the edge connecting A and B remains unchanged

    # A --> B where A could not be vectorized, but B has been
    def vectorized_src_only(
        self,
        dg: PDG,
        dim: ie.Symbol,
        dim_idx: int,
        size: ie.IntIndexValueLike,
        snk: TensorOp,
        src: TensorOp,
        dep_data: DependencyData,
        ops_to_vectorize: Set[TensorOp],
    ) -> None:
        # If, for some reason, snk can't be vectorized, we need to make it access
        # it's vectorized dependency src at the current dim value.
        # We do this with a spatial index operation based on the symbolic index of the dimension

        # The only special case is when the expression was a 0:D,
        # in which case we don't need to index at all.
        dim_index_expr = dep_data.expr[dim_idx].partial_eval(dg.static_bounds)  # type: ignore
        expected = ie.slice_(ie.ConstInt(0), dim.as_bound().partial_eval(dg.static_bounds))

        # TODO maybe need tp change to src.ubound[dim_idx] instead of dim.as_bound()
        if dim_index_expr.struct_eq(expected):
            new_e = dep_data.expr.skip_idx(dim_idx)

            new_dep_data = DependencyData(
                new_e, dep_data.src_out_idx, dep_data.sink_in_idx, dep_data.cond
            )

            # TODO this check should not be needed (or at least not be len - 1)
            if dim_idx != len(dep_data.expr) - 1:
                self.insert_edge_permuting_if_needed(
                    dg, snk, src, dep_data, new_dep_data, dim, dim_idx, size
                )
            else:
                dg.add_edge(snk, src, new_dep_data)
        elif dim_index_expr.is_constant():
            # We insert an index_op to perform the indexing
            new_e = dep_data.expr.skip_idx(dim_idx)
            symb_t = _get_symbolic_tensor_for_op_output(dg, src, dep_data.src_out_idx)
            # NOTE: 0 because the newly vectorized dim will be the first dim of src
            indexed = symb_t.index_select(dim=0, index=lift_to_symbolic_tensor(dim_index_expr))
            new_dep_data = DependencyData(new_e, OpOutId(0), dep_data.sink_in_idx, dep_data.cond)
            if dim_idx != len(dep_data.expr) - 1:
                self.insert_edge_permuting_if_needed(
                    dg, snk, indexed.op, dep_data, new_dep_data, dim, dim_idx, size
                )
            else:
                dg.add_edge(snk, indexed.op, new_dep_data)
        else:
            # NOTE: recently changed, adding replace_idx.
            new_e = dep_data.expr.replace_idx(dim_idx, dim)
            symb_t = _get_symbolic_tensor_for_op_output(dg, src, dep_data.src_out_idx)

            # NOTE: 0 because the newly vectorized dim will be the first dim
            if isinstance(dim_index_expr, ie.Slice):
                indexed = symb_t.index_slice(
                    dim=0,
                    start=dim_index_expr.start,
                    length=dim_index_expr.stop - dim_index_expr.start,
                )
            else:
                index_st = translate_ie_to_st(dim_index_expr)
                indexed = symb_t.index_select(dim=0, index=index_st)
            if len(new_e) != len(indexed.op.domain):
                print(f"Index expr={new_e} does not match {indexed.op.domain}")
                print(f"Dim index expr={dim_index_expr}")

                print(f"{snk=}. {snk.domain=} Traceback:")
                print(f"{snk.creation_traceback}")
                print()
                print(f"{dep_data}. Traceback:")
                print(f"{dep_data.creation_traceback}")
                print()
                print(f"{src=}. {src.domain=} Traceback:")
                print(f"{src.creation_traceback}")

                raise ValueError("Index expr does not match indexed op domain")

            # Edge from op to new op
            src = indexed.op

            new_dep_data = DependencyData(
                new_e,
                OpOutId(0),
                dep_data.sink_in_idx,
                dep_data.cond,
            )
            dg.add_edge(snk, src, new_dep_data)

    def insert_edge_permuting_if_needed(
        self,
        dg: PDG,
        new_snk: TensorOp,
        new_src: TensorOp,
        old_dep_data: DependencyData,
        new_dep_data: DependencyData,
        dim: ie.Symbol,
        dim_idx: int,
        size: ie.IntIndexValueLike,
    ) -> None:
        # TODO: method needs documentation and likely can be simplified
        slices_before_vec_dim = sum(not m.is_point() for m in old_dep_data.expr[:dim_idx])
        slices_after_vec_dim = sum(not m.is_point() for m in old_dep_data.expr[dim_idx + 1 :])
        slices_total = slices_before_vec_dim + slices_after_vec_dim

        src_vectorized = self._has_been_vectorized(new_src, dim)
        snk_vectorized = self._has_been_vectorized(new_snk, dim)

        # NOTE: If the source was vec, but there are no slices at all in the old expr,
        # then we don't need to permute the dims because the new dim is just prepended.
        # TODO if there are slices after, why does that matter?
        if src_vectorized and slices_total == 0:
            dg.add_edge(new_snk, new_src, new_dep_data)
            return

        # NOTE: If the source was not vec, but there are no slices before the vec dim,
        # then again we don't need to permute the dims because the new dim is just prepended.
        if (not src_vectorized) and slices_before_vec_dim == 0:
            dg.add_edge(new_snk, new_src, new_dep_data)
            return

        stensor = _get_symbolic_tensor_for_op_output(dg, new_src, old_dep_data.src_out_idx)
        stensor = stensor.symbolic_index(new_dep_data.expr)

        # For each dimension in the tensor, assign each a number starting from 0
        permute_dims = list(range(len(stensor.shape)))

        curr_pos = slices_total if src_vectorized else slices_before_vec_dim
        new_pos = 0 if snk_vectorized else slices_before_vec_dim

        permute_dims.remove(curr_pos)
        permute_dims.insert(new_pos, curr_pos)

        # Permute the symbolic tensor based on the permutation we just created
        stensor = stensor.permute(tuple(permute_dims))

        # Adding the vectorization information to op_vectorizations for this operation
        if not self._has_been_vectorized(stensor.op, dim):
            self.op_vectorizations[stensor.op][0].append(dim)
            self.op_vectorizations[stensor.op][1].append(size)

        # Add in the new edge between the dependent operation and the new (vectorized) op
        # now that the new op has been permuted
        dg.add_edge(
            new_snk,
            stensor.op,
            DependencyData(
                stensor.domain.basis_expr,
                OpOutId(0),
                old_dep_data.sink_in_idx,
                old_dep_data.cond,
            ),
        )

        assert len(permute_dims) == len(dg.get_input_shape(stensor.op, OpInId(0)))

    # A --> B where A is vectorized and B is not
    def vectorized_snk_only_src_has_dim(
        self,
        dg: PDG,
        dim: ie.Symbol,
        dim_idx: int,
        size: ie.IntIndexValueLike,
        snk: TensorOp,
        src: TensorOp,
        dep_data: DependencyData,
        ops_to_vectorize: Set[TensorOp],
    ) -> None:
        e = dep_data.expr

        # If the sink is vectorized but the src is not (and the src did have the dimension),
        # then the sink needs to access the src on the entire dim at once.
        # This indexing can cause a permutation of dims, in which case we permute them back

        # TODO: this only works if the expression between these was d. For now, this must have
        # been the case, because we check that the edges are all d or 0:D.
        # To support other cases, we will need smarter processing of these indexes
        if not e.members[dim_idx].struct_eq(dim):
            print(
                "vectorized_snk_only_src_has_dim: snk=%s, src=%s, dep_data=%s, dim=%s",
                snk,
                src,
                dep_data,
                dim,
            )
            print("dim_idx=%d, size=%s" % (dim_idx, size))
            print()
            print("snk traceback:")
            print(snk.creation_traceback)
            print()
            print("src traceback:")
            print(src.creation_traceback)
            print()
            print("dep_data traceback:")
            print(dep_data.creation_traceback)
            print()
            raise ValueError("Expected %s but got %s" % (dim, e.members[dim_idx]))

        new_e = e.replace_idx(
            dim_idx, ie.slice_(ie.ConstInt(0), dim.as_bound().partial_eval(dg.static_bounds))
        )
        new_dep_data = DependencyData(
            new_e, dep_data.src_out_idx, dep_data.sink_in_idx, dep_data.cond
        )

        self.insert_edge_permuting_if_needed(
            dg, snk, src, dep_data, new_dep_data, dim, dim_idx, size
        )

    def vectorized_snk_only_src_not_has_dim(
        self,
        dg: PDG,
        dim: ie.Symbol,
        dim_idx: int,
        size: ie.IntIndexValueLike,
        old_snk: TensorOp,
        vec_snk: TensorOp,
        src: TensorOp,
        dep_data: DependencyData,
        ops_to_vectorize: Set[TensorOp],
    ) -> None:
        # case = isinstance(snk, top.CatOp) and isinstance(src, top.ConstOp)
        # I think, if is_index_src_tensor_param is true and the index param has been vec,
        # then we don't need to expand?
        # if self._is_index_src_tensor_param(dg, src):
        #    dg.add_edge(snk, src, dep_data)
        #    return

        if is_non_dimensional_param(dg, src):
            # A special case where the src is an index of an index_select or index_add
            # and it was not vectorized because it does not have the dimension.
            # In this case, we just move the dim parameter of the index forward by one, but use
            # the same index, as it is constant.
            dg.add_edge(vec_snk, src, dep_data)
            return
        if isinstance(vec_snk, top.IndexSelectOp):
            deps = dg.get_flat_direct_dependencies(old_snk)
            tensor_vec = deps[0][0] in ops_to_vectorize
            index_vec = deps[1][0] in ops_to_vectorize
            only_index_vec = index_vec and not tensor_vec
            if is_index_src_tensor_param(dg, src) and only_index_vec:
                dg.add_edge(vec_snk, src, dep_data)
                return

        symb_t = _get_symbolic_tensor_for_op_output(dg, src, dep_data.src_out_idx)
        symb_t = symb_t.symbolic_index(dep_data.expr)
        expanded = symb_t.unsqueeze(0).expand(symb_t.shape.prepend_dim(size))
        self.op_vectorizations[expanded.op][0].append(dim)
        self.op_vectorizations[expanded.op][1].append(size)

        new_dep_data = DependencyData(
            expanded.domain.basis_expr, OpOutId(0), dep_data.sink_in_idx, dep_data.cond
        )

        dg.add_edge(vec_snk, expanded.op, new_dep_data)
        # TODO: unsure if permuting is ever needed. If it ever is, we need to change
        # the before_check, because it probably needs to check for slices_after, not slices_before
        # in this case.
        # self.insert_edge_permuting_if_needed(
        #    dg, snk, expanded.op, dep_data, new_dep_data, dim, dim_idx, size
        # )

    # snk --> src where both are vectorized
    def vectorized_src_and_snk(
        self,
        dg: PDG,
        dim: ie.Symbol,
        dim_idx: int,
        size: ie.IntIndexValueLike,
        snk: TensorOp,
        src: TensorOp,
        dep_data: DependencyData,
        ops_to_vectorize: Set[TensorOp],
    ) -> None:
        e = dep_data.expr

        # Skip over the dimension corresponding to dim_idx as we can no longer index on it
        # (it has been vectorized)
        new_e = e.skip_idx(dim_idx)

        new_dep_data = DependencyData(
            new_e, dep_data.src_out_idx, dep_data.sink_in_idx, dep_data.cond
        )

        self.insert_edge_permuting_if_needed(
            dg, snk, src, dep_data, new_dep_data, dim, dim_idx, size
        )

    def _has_been_vectorized(
        self,
        op: TensorOp,
        dim: ie.Symbol,
    ) -> bool:
        return any(d.struct_eq(dim) for d in self.op_vectorizations[op][0])

    def _run(self) -> Tuple[PDG, bool]:  # noqa: C901
        new_dg = self.copy_dg()
        glob.set_active_dg(new_dg)
        new_ctx: CompilationCtx = dataclasses.replace(self.ctx, dg=new_dg)
        self.ctx: CompilationCtx = new_ctx

        vectorizations_done = 0
        for dim in reversed(list(new_dg.universe.variables)):
            dim_vectorizations_done = 0
            # Mapping between the original ops in the graph, and their new vectorized versions
            # for this dimension
            op_mapping: Dict[TensorOp, TensorOp] = {}

            # dim_idx = new_dg.universe.find_variable_index(dim)
            # if dim_idx != 2:
            #    continue

            ops_to_vectorize = get_ops_to_vectorize(new_dg, dim)

            assert all(o.domain.has_dim(dim) for o in ops_to_vectorize), (
                f"Some ops in {ops_to_vectorize} do not have {dim}"
            )

            ops_to_vectorize = filter_ops_to_vectorize(new_ctx, ops_to_vectorize, dim)

            if len(ops_to_vectorize) == 0:
                continue

            # TODO: remove once we move the scatter add code too
            _promote_index_ops(new_dg, ops_to_vectorize)

            bound = dim.as_bound()

            # TODO if we try vectorizing dynamic bounds, this will have to change.
            size = bound.partial_eval(new_dg.static_bounds)

            self._vectorize_all_ops(
                new_dg, dim, size, op_mapping, self.op_vectorizations, ops_to_vectorize
            )
            dim_vectorizations_done += len(ops_to_vectorize)

            # Now that all ops are vectorized on this dimension, adjust the edges between them
            self.vectorize_edges(new_dg, op_mapping, dim, size, ops_to_vectorize)

            # Now that all the edges have been re-wired to the vectorized ops,
            # we can safely delete the ops which have been replaced by their vectorized equivalents
            # (these ops served as the keys in the old op --> new op mapping)
            # This also deletes all edges they are involved in.
            for op in op_mapping.keys():
                new_dg.remove_op(op)

            new_ctx, _, _ = AlgebraicOptimizer(new_ctx).run()
            new_dg = new_ctx.dg
            glob.set_active_dg(new_dg)

            vectorizations_done += dim_vectorizations_done
            log.info(
                "Performed %s vectorizations for dimension %s.",
                dim_vectorizations_done,
                dim,
            )
        # Returning the new dg with grouped subgraphs, and whether or not the
        # original dg was modified
        log.info("Performed %s vectorizations in total.", vectorizations_done)
        return new_dg, vectorizations_done > 0
