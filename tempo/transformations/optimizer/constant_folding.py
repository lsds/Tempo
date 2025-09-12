from functools import partial
from pathlib import Path

import numpy as np

from tempo.core import tensor_ops as top
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import OpId, OpOutId, TempoTensorProtocol
from tempo.core.dependence_graph import PDG, OpData
from tempo.core.dtype import dtypes
from tempo.core.global_objects import TemporaryActiveDgCtxManager
from tempo.core.symbolic_tensor import SymbolicTensor, get_symbolic_tensor_for_op_output
from tempo.transformations.compilation_pass import CompilationCtx, Transformation
from tempo.utils import logger

log = logger.get_logger(__name__)


def _get_folding_execution_config(original_exec_cfg: ExecutionConfig) -> ExecutionConfig:
    return ExecutionConfig(
        path=str(Path(original_exec_cfg.path) / "const_folding"),
        dev=original_exec_cfg.dev,
        backend=original_exec_cfg.backend,
        executor_debug_mode=False,
        deterministic=False,
        seed=0,
        visualize_pipeline_stages=False,
        validate_pipeline_stages=False,
        render_schedule=False,
        gc_bytes_min=0,
        enable_constant_folding=False,
        enable_algebraic_optimizer=False,
        enable_gc=False,
        enable_swap=False,
        enable_parallel_block_detection=False,
        enable_dead_code_elim=False,
        enable_vectorization=False,
        enable_duplicate_code_elim=False,
        enable_incrementalization=False,
        enable_hybrid_tensorstore=False,
        enable_donation_analysis=False,
        enable_broadcast_elim=False,
        enable_domain_reduction=False,
        enable_fold_pads_into_storage=False,
        enable_inplace_write=False,
        enable_symbol_prealloc_store=False,
        enable_statifying_incrementalization=False,
        enable_pad_mask_removal=False,
        torch_pinned_prealloc_size_bytes=0,
        torch_pinned_memory_enabled=False,
        visualize_debug_stages=False,
        enable_codegen_thunk_warmup=False,
        # NOTE: we enable these for faster compilation and const folding
        enable_codegen_dataflows=True,
        enable_dataflow_grouping=True,
        enable_codegen_dynamic_dataflows=True,
    )


class ConstantFolding(Transformation):
    """This transformations merges and folds constants in the DG."""

    def __init__(self, ctx: CompilationCtx) -> None:
        super().__init__(ctx)

    def _run(self) -> tuple[PDG, bool]:
        new_dg = self.ctx.dg

        nodes_to_fold, all_nodes_in_folding, unif_consts = self._get_folding_nodes(new_dg)
        removed = len(all_nodes_in_folding) - len(nodes_to_fold)

        if len(nodes_to_fold) == 0 or nodes_to_fold == all_nodes_in_folding:
            return new_dg, False

        log.info(
            "Found %s uniform consts out of %s nodes to fold (%s%%)",
            len(unif_consts),
            len(nodes_to_fold),
            round(len(unif_consts) / len(nodes_to_fold) * 100, 2),
        )

        self._fold_all(new_dg, nodes_to_fold, all_nodes_in_folding, unif_consts)
        log.info(
            "Evaluated %s down to %s nodes. Removed %s nodes.",
            len(all_nodes_in_folding),
            len(nodes_to_fold),
            removed,
        )
        return new_dg, True

    def _get_folding_nodes(
        self, dg: PDG
    ) -> tuple[set[top.TensorOp], set[top.TensorOp], set[top.TensorOp]]:
        # Start with the set of constant nodes that are valid folding nodes
        initial_consts = {node for node in dg.nodes if self._is_valid_initial_const_node(node)}

        # Initialize frontiers
        all_nodes: set[top.TensorOp] = set()
        frontier = initial_consts.copy()  # Nodes to explore

        while frontier:
            const = frontier.pop()

            # Check if the current node should be folded
            if self._can_fold_node(const, dg, all_nodes):
                all_nodes.add(const)
                # folding_candidates_and_children.update(new_dg.get_flat_direct_dependencies(const))

                # Add valid dependents to the frontier to continue folding
                for parent, _ in dg.get_direct_dependents(const):
                    frontier.add(parent)

        # Any node which has a dependent which is not in all_nodes.
        nodes_to_fold = set()
        for node in all_nodes:
            # UDF sinks should not be fold targets. They should just execute.
            if node.num_outputs == 0:
                nodes_to_fold.add(node)
                continue
            for dependent, _ in dg.get_flat_direct_dependents(node):
                if dependent not in all_nodes:
                    nodes_to_fold.add(node)

        nodes_to_fold_ = set()
        all_nodes_ = set()
        induced_subgraph = dg.induced_subgraph(OpId(-1), all_nodes)
        for node in set(nodes_to_fold):
            if len(induced_subgraph.get_flat_direct_dependencies(node)) > 0 or not (
                isinstance(node, top.ConstOp)
            ):
                nodes_to_fold_.add(node)
                all_nodes_.add(node)
                succ = list(induced_subgraph.recursive_dependency_nodes(node))
                all_nodes_.update(succ)

        nodes_to_fold = nodes_to_fold_
        all_nodes = all_nodes_

        induced_subgraph = dg.induced_subgraph(OpId(-1), all_nodes)
        unif_consts = set()
        # Find the nodes which are computed using elementwise ops from uniform consts.
        # NOTE: These will result in uniform consts.
        for node in nodes_to_fold:
            succ = list(induced_subgraph.recursive_dependency_nodes(node))
            leafs = [s for s in succ if len(induced_subgraph.get_flat_direct_dependencies(s)) == 0]
            unif_const_leafs = [l for l in leafs if isinstance(l, top.ConstOp) and l.is_uniform]
            if set(unif_const_leafs) == set(leafs) and len(unif_const_leafs) > 0:
                nodes_between = set()
                for l in leafs:
                    nodes_between.update(induced_subgraph.get_ops_between(node, l))

                # NOTE: These ops can affect the uniformness of the consts
                if all(
                    not isinstance(n, (top.CumSumOp, top.ConvOp, top.IndexAddOp, top.CatOp))
                    for n in nodes_between
                ):
                    unif_consts.add(node)

        return nodes_to_fold, all_nodes, unif_consts

    def _can_fold_node(
        self,
        node: top.TensorOp,
        new_dg: PDG,
        all_folding_nodes: set[top.TensorOp],
    ) -> bool:
        if not node.domain.is_empty():
            return False  # Skip nodes with a domain (varying with symbolic dimensions)

        children = [c for c, _ in new_dg.get_flat_direct_dependencies(node)]
        return all(c in all_folding_nodes for c in children)

    def _is_valid_initial_const_node(self, node: top.TensorOp) -> bool:
        return node.domain.is_empty() and (
            isinstance(node, top.ConstOp)
            or (isinstance(node, top.UDFOp) and node.num_inputs == 0)
            or (isinstance(node, top.RandOp))
            or (isinstance(node, top.EvalSymbolOp) and node.symbol in self.ctx.dg.static_bounds)
        )

    def _fold_all(
        self,
        dg: PDG,
        nodes_to_fold: set[top.TensorOp],
        all_nodes: set[top.TensorOp],
        unif_consts: set[top.TensorOp],
    ) -> bool:
        """Compile and execute a subgraph of the DG to replace a node with a constant if possible.

        Args:
            dg (DependenceGraph): The dependence graph.
            all_nodes (Set[TensorOp]): The nodes to fold and their dependencies.

        """
        # Reorder these imports to avoid circular imports
        from tempo.transformations.iterate_compilation_pass import Pipeline
        from tempo.transformations.scheduling.schedule_execution_transform import (
            ScheduleExecution,
        )

        pipeline = partial(Pipeline, passes=[partial(ScheduleExecution, quiet=True)], quiet=True)

        exec_config = _get_folding_execution_config(self.ctx.exec_cfg)

        from tempo.transformations.frontend_compiler import compile_frontend

        sub_graph = dg.induced_subgraph(OpId(-1), all_nodes)

        res_map: dict[tuple[OpId, OpOutId], TempoTensorProtocol] = {}
        with TemporaryActiveDgCtxManager(sub_graph):
            for node in nodes_to_fold:
                for o in range(node.num_outputs):
                    output = OpOutId(o)
                    node_symb_t = get_symbolic_tensor_for_op_output(sub_graph, node, output)
                    SymbolicTensor.sink_udf(
                        lambda x, node=node, output=output: res_map.setdefault(  # type: ignore
                            (node.op_id, output), x
                        ),
                        node_symb_t,
                    )

        sub_graph.bound_defs = {**dg.static_bounds}

        ctx, _ = compile_frontend(sub_graph, exec_config, pipeline, True)
        from tempo.runtime.backend_compiler import compile_backend

        executor, ctx = compile_backend(ctx, True)
        bend = executor.backend

        executor.execute()
        # log.debug("=== Done Compiling and Executing subgraph ===")

        for node in nodes_to_fold:
            opdata = dg.ops_by_id[node.op_id]

            for output_ in range(opdata.num_outputs):
                outid = OpOutId(output_)
                shape = opdata.output_shapes[outid]
                dtype = opdata.output_dtypes[outid]
                domain = node.domain.copy()

                # tensorid = TensorId(node.op_id, outid)
                # tensor = executor.get_spatial_tensor(tensorid)
                tensor = res_map[(node.op_id, output)]

                val = bend.to_numpy(tensor)
                is_uniform = node in unif_consts

                if not is_uniform and val.size < 10_000:
                    min_val, max_val = np.min(val), np.max(val)
                    is_uniform = min_val == max_val

                if is_uniform:
                    val = np.asarray(val.flat[0].item(), dtype=dtypes.to_np(dtype))
                    # val = np.asarray(min_val, dtype=dtypes.to_np(dtype))

                # we can add the new const node
                new_out_id = OpOutId(0)
                const = top.ConstOp(
                    dg.get_next_op_id(),
                    domain,
                    node.tags,
                    shape,
                    dtype,
                    val,
                    is_uniform=is_uniform,
                )
                new_opdata = OpData(const, {new_out_id: shape}, {new_out_id: dtype})
                dg.insert_op(new_opdata)

                # The new const has no dependencies as it is a constant
                dg.move_dependents(node, const)

        for op in all_nodes:
            dg.remove_op(op)
        return True
