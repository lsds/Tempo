from functools import partial
from typing import Set, Tuple

import numpy as np

from tempo.core import tensor_ops as top
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import OpId, OpOutId, TensorId
from tempo.core.dependence_graph import PDG, OpData
from tempo.core.dtype import dtypes
from tempo.transformations.compilation_pass import CompilationCtx, Transformation
from tempo.utils import logger

log = logger.get_logger(__name__)


class ConstantFolding(Transformation):
    """This transformations merges and folds constants in the DG."""

    def __init__(self, ctx: CompilationCtx) -> None:
        super().__init__(ctx)

    def _get_folding_execution_config(self, original_exec_cfg: ExecutionConfig) -> ExecutionConfig:
        return ExecutionConfig(
            path="./const",
            dev="cpu",
            executor_debug_mode=False,
            backend=original_exec_cfg.backend,
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
            enable_codegen_dataflows=False,
            enable_dataflow_grouping=False,
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
        )

    def _run(self) -> Tuple[PDG, bool]:
        new_dg = self.ctx.dg

        nodes_to_fold, all_nodes_in_folding = self._get_folding_nodes(new_dg)
        removed = len(all_nodes_in_folding) - len(nodes_to_fold)

        if len(nodes_to_fold) == 0 or nodes_to_fold == all_nodes_in_folding:
            return new_dg, False

        self._fold_all(new_dg, nodes_to_fold, all_nodes_in_folding)
        log.info(
            "Evaluated %s down to %s nodes. Removed %s nodes.",
            len(all_nodes_in_folding),
            len(nodes_to_fold),
            removed,
        )
        return new_dg, True

    def _get_folding_nodes(self, dg: PDG) -> Tuple[Set[top.TensorOp], Set[top.TensorOp]]:
        # Start with the set of constant nodes that are valid folding nodes
        initial_consts = {node for node in dg.nodes if self._is_valid_const_node(node)}

        # Initialize frontiers
        all_nodes: Set[top.TensorOp] = set()
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
            for dependent, _ in dg.get_flat_direct_dependents(node):
                if dependent not in all_nodes:
                    nodes_to_fold.add(node)
        return nodes_to_fold, all_nodes

    def _can_fold_node(
        self,
        node: top.TensorOp,
        new_dg: PDG,
        all_folding_nodes: Set[top.TensorOp],
    ) -> bool:
        if not node.domain.is_empty():
            return False  # Skip nodes with a domain (varying with symbolic dimensions)

        children = [c for c, _ in new_dg.get_flat_direct_dependencies(node)]
        return all(c in all_folding_nodes for c in children)

    def _is_valid_const_node(self, node: top.TensorOp) -> bool:
        return node.domain.is_empty() and (
            isinstance(node, top.ConstOp)
            or (isinstance(node, top.RandOp))
            or (isinstance(node, top.EvalSymbolOp) and node.symbol in self.ctx.dg.static_bounds)
        )

    def _fold_all(
        self,
        dg: PDG,
        nodes_to_fold: Set[top.TensorOp],
        all_nodes: Set[top.TensorOp],
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

        exec_config = self._get_folding_execution_config(self.ctx.exec_cfg)

        from tempo.transformations.frontend_compiler import compile_frontend

        sub_graph = dg.induced_subgraph(OpId(-1), all_nodes)
        sub_graph.bound_defs = {**dg.static_bounds}

        ctx, _ = compile_frontend(sub_graph, exec_config, pipeline, True)
        from tempo.runtime.backend_compiler import compile_backend

        executor = compile_backend(ctx, True)

        executor.execute()
        # log.debug("=== Done Compiling and Executing subgraph ===")

        for node in nodes_to_fold:
            opdata = dg.ops_by_id[node.op_id]

            for output in range(opdata.num_outputs):
                outid = OpOutId(output)
                shape = opdata.output_shapes[outid]
                dtype = opdata.output_dtypes[outid]
                domain = node.domain.copy()

                tensorid = TensorId(node.op_id, outid)
                tensor = executor.get_spatial_tensor(tensorid)

                val = np.array(tensor, dtype=dtypes.to_np(dtype))
                min_val, max_val = np.min(val), np.max(val)
                is_uniform = min_val == max_val
                if is_uniform:
                    val = np.array(min_val, dtype=dtypes.to_np(dtype))

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
