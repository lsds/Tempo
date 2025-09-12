from collections import defaultdict
from collections.abc import Mapping
from itertools import product

from tempo.core import index_expr as ie
from tempo.core.compilation_ctx import CompilationCtx
from tempo.core.datatypes import OpId, OpInId, OpOutId
from tempo.core.shape import Shape
from tempo.utils import isl as isl_utils
from tempo.utils.logger import get_logger

log = get_logger(__name__)


# TODO: This class is very useful, but is not clean right now. Come back and improve it.


class MemoryEstimator:
    """Provides a way to estimate the memory usage of a given op or tensor."""

    def __init__(self, ctx: CompilationCtx):
        self.ctx = ctx
        self.dg = ctx.dg

        # Setup a dict of bound size estimates
        self.bound_size_estimates: Mapping[ie.Symbol, int] = defaultdict(
            lambda: ctx.exec_cfg.default_dim_upper_bound_size
        )
        for bound, bound_def in ctx.dg.bound_defs.items():
            if isinstance(bound_def, int):
                self.bound_size_estimates[bound] = bound_def
            elif isinstance(bound_def, ie.ConstInt):
                self.bound_size_estimates[bound] = bound_def.const

        # TODO: there likely is a better solution
        self.all_vars_and_bounds = dict(self.bound_size_estimates)
        for var in self.dg.universe.variables:
            self.all_vars_and_bounds[var] = ctx.exec_cfg.default_dim_upper_bound_size

    def estimate_tensor_expr_size_bytes(
        self,
        expr: ie.IndexExpr,
        op_id: OpId,
        out_idx: OpOutId | None = None,
        in_idx: OpInId | None = None,
    ) -> tuple[int, bool]:
        point_size = self.estimate_tensor_point_size_bytes(op_id, out_idx, in_idx)
        approx = False
        if expr.is_point():
            num = 1
        else:
            num_ = isl_utils.simplify_shape(
                Shape.from_(expr.evaluate_shape(self.dg.static_bounds)),
                self.dg.static_bounds,
            ).prod()
            if isinstance(num_, int):
                ...
            else:
                if isinstance(num_, ie.ConstInt):
                    num = num_.const
                else:
                    approx = True

                    num = num_.evaluate(self.all_vars_and_bounds)

        return point_size * num, approx

    def try_estimate_tensor_point_size_bytes(  # noqa: C901
        self,
        op_id: OpId,
        out_idx: OpOutId | None = None,
        in_idx: OpInId | None = None,
    ) -> int:
        try:
            return self.estimate_tensor_point_size_bytes(op_id, out_idx, in_idx)
        except Exception:
            return 0

    def estimate_tensor_point_size_bytes(  # noqa: C901
        self,
        op_id: OpId,
        out_idx: OpOutId | None = None,
        in_idx: OpInId | None = None,
    ) -> int:
        op_data = self.dg.ops_by_id[op_id]

        if out_idx is None and in_idx is None:
            raise ValueError("Either out_idx or in_idx must be provided")

        if out_idx is not None:
            assert in_idx is None
            shape = op_data.output_shapes[out_idx]
            dtype = op_data.output_dtypes[out_idx]

        if in_idx is not None:
            assert out_idx is None
            shape = self.dg.get_input_shape(op_data.op, in_idx)
            dtype = self.dg.get_input_dtype(op_data.op, in_idx)

        if shape.is_static():
            return shape.as_static().prod() * dtype.repr_bytes  # type: ignore
        else:
            # We err on the side of caution and assume that the tensor is large
            statically_known_portion = 1
            dynamic_portion: ie.IntIndexValue = ie.ConstInt(1)
            for dim in shape._shape:
                if isinstance(dim, int):
                    statically_known_portion *= dim
                else:
                    dynamic_portion *= dim  # type: ignore

            statically_known_portion *= dtype.repr_bytes

            # Since we do not know the exact size of the dynamic portion, we estimate it by
            # sampling the size of the dynamic portion at 5 different points for each variable
            # and test all possible combinations of these values. We then take the maximum size.
            variable_values = {
                var: [
                    0,
                    (bound_estimate := self.bound_size_estimates[var.as_bound()]) // 4,
                    bound_estimate // 2,
                    bound_estimate * 3 // 4,
                    bound_estimate,
                ]
                for var in dynamic_portion.vars_used()
            }

            dynamic_portion_eval = max(
                dynamic_portion.evaluate(
                    {
                        **self.bound_size_estimates,
                        **dict(zip(variable_values.keys(), values, strict=True)),
                    }
                )
                for values in product(*variable_values.values())
            )
            size = statically_known_portion * dynamic_portion_eval
            return size

    def estimate_op_size_bytes(self, op_id: OpId) -> int:
        try:
            return self.estimate_op_size_bytes_out(op_id) + self.estimate_op_size_bytes_in(op_id)
        except Exception as e:
            # log.debug("Error estimating op size for %s: %s", op_id, e)
            # print(f"Error estimating op size for {op_id}: {e}")
            raise e

    def estimate_op_size_bytes_out(self, op_id: OpId) -> int:
        num_outputs = self.dg.ops_by_id[op_id].num_outputs
        output_sizes = sum(
            self.estimate_tensor_point_size_bytes(op_id, out_idx=OpOutId(i))
            for i in range(num_outputs)
        )
        return output_sizes

    def estimate_op_size_bytes_in(self, op_id: OpId) -> int:
        # NOTE: it may be MergeOp
        op_data = self.dg.ops_by_id[op_id]
        op = op_data.op

        from tempo.core import tensor_ops as top

        if not isinstance(op, top.MergeOp):
            num_inputs = op.num_inputs
            input_sizes = sum(
                self.estimate_tensor_point_size_bytes(op_id, in_idx=OpInId(i))
                for i in range(num_inputs)
            )
        else:
            if op_data.uncommitted_branch_conds:
                input_sizes = 0
            else:
                num_inputs = op.num_inputs
                # NOTE: They should all be the same size, tbh
                input_sizes = max(
                    self.estimate_tensor_point_size_bytes(op_id, in_idx=OpInId(i))
                    for i in range(num_inputs)
                )

        return input_sizes

    def get_max_tensor_out_bytes(self) -> int:
        max_bytes = 0

        for n in self.dg.nodes:
            for out_idx in range(n.num_outputs):
                max_bytes = max(
                    max_bytes, self.estimate_tensor_point_size_bytes(n.op_id, OpOutId(out_idx))
                )
        return max_bytes
