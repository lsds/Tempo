import dataclasses
import os
import time
from typing import Optional, Set

import optree
from graphviz import Digraph

from tempo.core import tensor_ops as top
from tempo.core.analysis_ctx import AnalysisCtx
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import OpOutId, TensorId
from tempo.core.dependence_graph import PDG, DependencyData, OpData
from tempo.core.op_tags import REGION_TAG
from tempo.core.shape import Shape
from tempo.core.utils import bytes_to_human_readable
from tempo.utils import logger
from tempo.utils.memory_estimator import MemoryEstimator

log = logger.get_logger(__name__)


def raise_error_with_pdg_render(dg: PDG, user_msg: str, path: str = ".") -> None:
    exec_cfg = ExecutionConfig.default()
    from tempo.core.isl_context_factory import get_isl_context

    isl_ctx = get_isl_context(exec_cfg)
    ctx = CompilationCtx(dg, AnalysisCtx(isl_ctx), exec_cfg)
    r = DGRenderer(ctx, path)
    r.render()

    msg = f"Error: {user_msg}. \n Error PDG state rendered at {path}."

    raise ValueError(msg)


def raise_error_with_ctx_render(ctx: CompilationCtx, user_msg: str, path: str = ".") -> None:
    complete_path = f"{path}/error_state"
    r = DGRenderer(ctx, complete_path)
    r.render()

    msg = f"Error: {user_msg}. \n Error PDG state rendered at {complete_path}."

    raise ValueError(msg)


class DGRenderer:
    def __init__(
        self,
        ctx: CompilationCtx,
        out_fname: Optional[str] = None,
    ):
        dg = ctx.dg
        self.dg = dg
        self.ctx = ctx
        self.mem_est = MemoryEstimator(ctx)

        self.highlight_ops: Set[top.TensorOp] = set()

        if out_fname is None:
            # Use the current time to make a directory
            time_str = "./dgs/" + time.strftime("%Y%m%d-%H%M")
            out_fname = f"{time_str}/dg"

        self.location, self.name = str(out_fname).rsplit("/", 1)

        # Expand location to absolute path
        self.location = os.path.abspath(self.location)

        self.max_size_mb = 0.0

        for node in dg.nodes:
            try:
                op_size_bytes = self.mem_est.estimate_op_size_bytes(node.op_id)
            except Exception as _:
                # log.error("Error estimating op size for %s", node.op_id)
                op_size_bytes = None
            if op_size_bytes is not None:
                op_size_mb = op_size_bytes / 1024 / 1024
                self.max_size_mb = max(self.max_size_mb, op_size_mb)

    def render(self) -> None:
        try:
            # Create the directory if it doesn't exist
            os.makedirs(self.location, exist_ok=True)

            self._render(self.dg, self.location, self.name)
            for op in self.dg.nodes:
                if isinstance(op, top.ExecDataflowOp):
                    old_mem_est = self.mem_est
                    self.mem_est = MemoryEstimator(
                        dataclasses.replace(self.ctx, dg=op.dataflow.subgraph)
                    )
                    subg: PDG = op.dataflow.subgraph  # type: ignore
                    self._render(
                        subg,
                        f"{self.location}/{self.name}_subgraphs",
                        f"{op.op_id}",
                        annotation=f"irouter={op.dataflow.irouter}\norouter={op.dataflow.orouter}",
                    )
                    self.mem_est = old_mem_est

        except Exception as e:
            log.error("Skipping rendering DG to dot due to error: %s", e)
            raise e

    def _render(
        self,
        dg: PDG,
        location: str,
        name: str,
        annotation: Optional[str] = None,
    ) -> None:
        # create the directory
        os.makedirs(location, exist_ok=True)
        dot = Digraph(comment=name)
        self._visualize_dg(dg, dot)  # annotation is not None
        dot.attr(label=annotation)

        # Save the DOT source to a file
        dot_file = f"{location}/{name}.dot"
        log.debug("Saving DOT source to %s", dot_file)
        dot.save(dot_file)

        # dot.render('example_graph', format='png', cleanup=True)

    def _visualize_dg(self, dg: PDG, dot: Digraph) -> None:
        for op_data in dg.ops_by_id.values():
            try:
                self._render_op(dg, dot, op_data)
            except Exception as e:
                log.error("Error rendering op %s: %s", op_data.op, e)
                print(op_data.op.creation_traceback)
                raise e
        for u, v, dep_data in dg.get_all_edges(include_control=True):
            self._render_edge(dg, dot, u, v, dep_data)

    def _render_edge(
        self,
        dg: PDG,
        dot: Digraph,
        sink: top.TensorOp,
        src: top.TensorOp,
        dep_data: DependencyData,
    ) -> None:
        analysis_ctx = self.ctx.analysis_ctx
        is_dataflow = dg.parent_graph is not None
        is_control = dep_data.is_control_edge

        # is_cycle_member = False
        # if dep_data.is_unconditional_basis():
        #    src_dependencies = dg.get_flat_direct_dependencies(src)
        #    for dep_op, dep_data_ in src_dependencies:
        #        if (
        #            dep_op == sink
        #            and dep_data_.expr.is_basis()
        #            and dep_data_.cond is None
        #        ):
        #            is_cycle_member = True
        #            break

        label = str(dep_data)  # if not is_dataflow else ""
        is_wrong = len(dep_data.expr) != len(src.domain)  # or is_cycle_member
        is_basic_conn = dep_data.expr.struct_eq(src.domain.basis_expr)
        donated = False
        copied = False
        if not is_dataflow and not is_control:
            if analysis_ctx._donatable_args is not None:
                donated = int(dep_data.sink_in_idx) in analysis_ctx._donatable_args[sink.op_id]
                # donated = dg.analysis_ctx._tensor_is_donated_to
            if analysis_ctx._needed_merge_copies is not None:
                if sink.op_id in analysis_ctx._needed_merge_copies:
                    copies = analysis_ctx._needed_merge_copies[sink.op_id]
                    copied = copies[int(dep_data.sink_in_idx)]

        dot.edge(
            str(sink.op_id),  # if not is_dataflow else str(src.op_id),
            str(src.op_id),  # if not is_dataflow else str(sink.op_id),
            label=label,
            style=(
                "dotted" if is_control else ("dashed" if dep_data.cond is not None else "solid")
            ),
            color=("red" if is_wrong else ("black" if is_basic_conn else "pink")),
            # thickness
            penwidth=("5" if (not is_basic_conn or is_wrong) else "1"),
            # Add  a "D" next to sink if the argument is donatable
            taillabel=("D" if donated else ""),
            headlabel=("C" if copied else ""),  # TODO headlabel?
        )

    def memory_size_to_color(self, size_mb: float) -> str:
        # Clamp the value of size_mb to be between 0 and max_size_mb
        clamped_size_mb = min(max(size_mb, 0), self.max_size_mb)

        # Calculate the intensity for green and blue channels to decrease from FF to 00
        # as the memory size increases to the max_size_mb.
        gb_intensity = 255 - int((clamped_size_mb / self.max_size_mb) * 255)

        # Generate the hex RGB color code
        color_code = f"#FF{gb_intensity:02X}{gb_intensity:02X}"

        return color_code

    def memory_size_to_pen_width(self, size_mb: float) -> str:
        clamped_size_mb = min(max(size_mb, 0), self.max_size_mb)

        # We return a pen width between 1 and 5, based on the memory size
        pen_width = 1 + int((clamped_size_mb / self.max_size_mb) * 4)

        return str(pen_width * 2)

    def _render_op(self, dg: PDG, dot: Digraph, op_data: OpData) -> None:
        analysis_ctx = self.ctx.analysis_ctx
        is_dataflow = dg.parent_graph is not None
        op = op_data.op
        op_str = str(op)

        try:
            op_size_bytes = self.mem_est.estimate_op_size_bytes(op.op_id)
        except Exception:
            op_size_bytes = 0

        label = f"{op_str}\n"
        try:
            label += f"in_shapes: {str(dg.get_input_shapes_list(op_data.op))}\n"
        except Exception:
            label += "in_shapes: UNKNOWN\n"
        label += f"out_shapes: {str(op_data.output_shapes)}\n"
        label += f"out_dtypes: {str(op_data.output_dtypes)}\n"
        label += f"domain: {str(op.domain)}\n"  # if not is_dataflow else ""
        label += (
            f"isl domain: {str(analysis_ctx.get_or_make_domain(op))}\n" if not is_dataflow else ""
        )
        tags_flattened = op.flat_tags
        label += f"tags: {str(tags_flattened)}\n"
        label += f"total memory:{bytes_to_human_readable(op_size_bytes)}\n"
        out_tensor_point_sizes = [
            bytes_to_human_readable(
                self.mem_est.estimate_tensor_point_size_bytes(op.op_id, OpOutId(out))
            )
            for out in range(op.num_outputs)
        ]
        label += f"Out tensor point sizes: {str(out_tensor_point_sizes)}\n"
        if not is_dataflow:
            if analysis_ctx._tensor_storage_classes is not None:
                storages = {}
                for out in range(op.num_outputs):
                    storage = analysis_ctx._tensor_storage_classes[TensorId(op.op_id, OpOutId(out))]
                    storages[out] = storage
                label += f"storage: {storages}\n"
            if analysis_ctx._device_assignment is not None:
                dev = analysis_ctx._device_assignment[op.op_id]
                label += f"device: {dev}\n"

        op_size_mb = op_size_bytes / (2**20)
        color_rgb = self._pick_node_color(op, op_size_mb)

        # TODO: add more checks
        wrong = False
        try:
            if isinstance(op, top.ElementWiseOp):
                if not Shape.can_broadcast(*dg.get_input_shapes_list(op)):
                    wrong = True
                    label += "Broadcast mismatch\n"

            dynamic = any(s.is_dynamic() for s in dg.get_input_shapes_list(op)) or any(
                s.is_dynamic() for s in op_data.output_shapes.values()
            )
        except Exception:
            dynamic = False

        dot.node(
            str(op.op_id),
            label,
            color="black" if not wrong else "red",
            fillcolor=color_rgb,
            style="filled" + (",dashed" if dynamic else ""),
            penwidth=self.memory_size_to_pen_width(op_size_mb),
        )

    def _pick_node_color(self, op: top.TensorOp, op_size_mb: float) -> str:
        render_mode: str = os.environ.get("DG_RENDER_MODE", "regions")

        if render_mode == "highlight":
            if op in self.highlight_ops:
                color_rgb = "#e3e0e7"
            else:
                color_rgb = "#ffffff"

        if render_mode == "memory":
            # color is a linear function of the size, going from white to red
            color_rgb = self.memory_size_to_color(op_size_mb)
        elif render_mode == "regions":
            if REGION_TAG in op.tags:
                region_tags = sorted(set(optree.tree_flatten(op.tags[REGION_TAG])[0]))
                color_red = hash(str(region_tags) + "red") % 55 + 200
                color_green = hash(str(region_tags) + "green") % 55 + 200
                color_blue = hash(str(region_tags) + "blue") % 55 + 200
                color_rgb = f"#{color_red:02X}{color_green:02X}{color_blue:02X}"
            else:
                color_rgb = "#ffffff"
        else:
            # color_rgb = "#ffffff"
            if op in self.highlight_ops:
                color_rgb = "#e3e0e7"
            else:
                color_rgb = "#ffffff"
        return color_rgb
