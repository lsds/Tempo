import math
from collections.abc import Sequence
from typing import Any

from tempo.api.recurrent_tensor import AutodiffFn
from tempo.core import index_expr as ie
from tempo.core import tensor_ops as top
from tempo.core.dtype import DataType, dtypes
from tempo.core.shape import Shape
from tempo.core.symbolic_tensor import SymbolicTensor, translate_ie_to_st
from tempo.utils import dg_utils
from tempo.utils.common import argsort
from tempo.utils.dg_utils import get_padding_for_slice

# NOTE: Many derivatives here are taken from pytorch's derivatives.yaml file
# https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml
# NOTE: Others are sourced from tinygrad mlops


class TemporalIndex(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        self.x = args[0]
        self.e = kwargs["e"]
        self.y = self.x.temporal_index(self.e)
        return (self.y,)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        snk_dom = self.y.domain
        src_dom = self.x.domain
        e = self.e
        from tempo.utils import isl as isl_utils

        rev_e = isl_utils.reverse_dependence_expr(e, snk_dom, src_dom)

        # NOTE: Each fwd slice creates two cases in grad accumulation:
        # one where it is in the slice
        # and one where it is not.
        fwd_slices_and_vars = [
            (i, v)
            for i, v in zip(e.members, src_dom.variables, strict=True)
            if isinstance(i, ie.Slice)
        ]

        # NOTE: applying e to src can remove dims from domain, resulting in snk_dom with fewer dims.
        # This can mean that rev_e has fewer members than e.
        # We need to align the two by filling in Nans.
        none_padded_rev_e_members: list[ie.IndexAtom | None] = []
        rev_e_idx = 0
        for v in src_dom.variables:
            if v in snk_dom:
                # This dimension is preserved, use the corresponding rev_e member
                none_padded_rev_e_members.append(rev_e.members[rev_e_idx])
                rev_e_idx += 1
            else:
                # This dimension is removed, pad with None
                none_padded_rev_e_members.append(None)

        # NOTE: a completely different approach could be:
        # 1) For each fwd slice, introduce a tau into universe, with bounds m_rev.start, m_rev.stop.
        # 2) Each tau temporally indexes grad_output at tau, and spatially indexes at correct idx
        # 3) Sum over tau to get grad at x

        slice_count = 0
        pads = []
        for i, m in enumerate(e.members):
            if isinstance(m, ie.Slice):
                # NOTE: Because we are going to dynamically index an already dynamic tensor,
                # we instead pad it's dynamic dimension in order to statify it.
                src_dom_var = src_dom.variables[i]

                # TODO: max val is still problematic for 0:t+1 or t:T cases as it returns infty.
                #  We will need a better solution than this if..
                if dg_utils.is_window_access(m):
                    max_val = isl_utils.int_index_val_max(
                        m.evaluate_shape({})[0]  # , src_dom
                    )
                else:
                    max_val = src_dom_var.as_bound()

                assert max_val is not None, "max_val should not be None"

                padding = get_padding_for_slice(m, max_val, src_dom_var)
                # print(f"padding: {padding}")

                if padding is not None:
                    grad_output = grad_output.pad_dim(padding, dim=slice_count, mode="any")
                    pads.append(padding)
                else:
                    pads.append((0, 0))
                slice_count += 1

        # print(f"Grad output: {grad_output}")
        parent_grad_: SymbolicTensor = grad_output.temporal_index(rev_e)
        # print(f"Parent grad (temp indexed): {parent_grad_}")

        # NOTE: We have grabbed all the symbolic indexes needed, but now we have to spatially
        # index them to recover the grad.
        for i, (m, m_rev) in enumerate(zip(e.members, none_padded_rev_e_members, strict=False)):
            if isinstance(m, ie.Slice):
                dom_var = src_dom.variables[i]
                # NOTE: if constant, eliminates dim and does not introduce dynamic dim.
                #  Which means we can just index_select t
                if m.is_constant():
                    # TODO: this dom_var.as_bound() will fail when the snk does not execute
                    # T times, but some reduced amount due to,e.g, a merge.
                    # We need to fix this.
                    # TODO: I also simply do not like the * approach.
                    # It doesnt even make sense to me.
                    # selected_grad = parent_grad_.index_select(0, dom_var)
                    # bound_expanded = SymbolicTensor.lift(dom_var.as_bound()).broadcast_to_shape(
                    #    selected_grad.shape
                    # )
                    # parent_grad_ = selected_grad * bound_expanded
                    grad_mul = (
                        SymbolicTensor.lift(dom_var.as_bound()).broadcast_to_shape(
                            parent_grad_.shape
                        )
                        * parent_grad_
                    )
                    parent_grad_ = grad_mul.index_select(0, dom_var)
                else:
                    assert isinstance(m_rev, ie.Slice), (
                        f"If m is dynamic slice, m_rev must also be slice, got {m_rev}"
                    )
                    # NOTE: if not constant, introduces dynamic-size dim, where the position
                    # of the t grads can change. We use custom gather logic to handle this.
                    # positions = SymbolicTensor.arange(0, m_rev.stop - m_rev.start)
                    i_positions = ie.Slice(ie.ConstInt(0), m_rev.stop - m_rev.start)
                    # i_positions = ie.Slice(
                    #    ie.ConstInt(0), max_vals[i]
                    # )
                    # print(f"i_positions: {i_positions}")
                    i_positions = isl_utils.simplify_expr_atom(i_positions)
                    # print(f"i_positions post-simplify: {i_positions}")
                    # print(f"m_rev: {m_rev}")
                    pad_offset = ie.lift_to_int_ie(pads[i][0])
                    # print(f"pad_offset: {pad_offset}")
                    simplified_pad_offset = isl_utils.simplify_expr_atom(pad_offset)

                    from tempo.core import global_objects as glob

                    dg = glob.get_active_dg()
                    tau, TAU = dg.extend_universe("tau", m_rev.stop - m_rev.start)
                    # tau, TAU = dg.extend_universe("tau", m_rev.stop)

                    expr = (
                        dom_var
                        - m.start.remap({dom_var: m_rev.start + tau})
                        + simplified_pad_offset.remap({dom_var: m_rev.start + tau})
                    )

                    # print(f"    idxs expr: {expr}")
                    simplified_expr = isl_utils.simplify_int_index_value(
                        expr,
                    )
                    # print(f"    idxs simplified_expr: {simplified_expr}")
                    simplified_expr_mapped = simplified_expr.remap(
                        {tau: i_positions}  # type: ignore
                    )
                    # print(f"    idxs simplified_expr_mapped: {simplified_expr_mapped}")
                    dg.remove_universe_dim(tau)

                    idxs = translate_ie_to_st(simplified_expr_mapped)
                    # print(f"    idxs: {idxs}, type: {type(idxs)}")
                    # NOTE: We select across the spatialized fwd time dimension,
                    # then sum over the bwd time dim
                    # NOTE: When idx is a scalar, a more efficient approach is to use index_select
                    # Given the padding, this should work for both 0:t+1 and t:T common cases
                    if idxs.shape.is_scalar():
                        parent_grad_ = parent_grad_.index_select(dim=1, index=idxs).sum(0)
                    else:
                        # NOTE: This is needed right now because statify may create invalid
                        # idxs for backward windows.
                        # TODO: Eventually, remove the need for this. One solution is that
                        # during statify inc, encountering a gather causes us to emit a max
                        # like this. Unsure.
                        idxs = SymbolicTensor.max(
                            SymbolicTensor.stack(idxs, idxs.full_like_self(0))
                        )[0]

                        for i in range(len(parent_grad_.shape)):
                            idxs = idxs.unsqueeze(i) if i != 0 else idxs
                        parent_grad_shape_except_1 = tuple(
                            (v if i != 1 else 1) for i, v in enumerate(parent_grad_.shape)
                        )
                        # print(f"Idxs shape: {idxs.shape}")
                        # print(f"parent_grad_shape_except_1: {parent_grad_shape_except_1}")
                        # print(f"    parent_grad_shape_except_1: {parent_grad_shape_except_1}")
                        gather_idxs = idxs.expand(Shape.from_(parent_grad_shape_except_1))

                        parent_grad_ = (
                            parent_grad_.gather(dim=1, index=gather_idxs).squeeze(1).sum(0)
                        )

        # Create condition for when all slices are active
        all_slices_active: ie.BooleanIndexValue = ie.ConstBool(True)
        for slice_expr, v in fwd_slices_and_vars:
            # TODO wouldnt this if be unnecessary given the simplify below?
            if not slice_expr.struct_eq(ie.slice_(0, v.as_bound())):
                all_slices_active = (
                    all_slices_active & (slice_expr.start <= v) & (v < slice_expr.stop)
                )
                # TODO add support for step by using %: (v % slice_expr.step == 0)
        all_slices_active = isl_utils.simplify_boolean_index_expr(src_dom, all_slices_active)

        # NOTE: Manual merge op registration
        # TODO make it pretty
        parent_grad = SymbolicTensor.merge_like(self.x)
        merge_op: top.MergeOp = parent_grad.op  # type: ignore

        from tempo.core import global_objects as glob

        dg = glob.get_active_dg()
        merge_op_data = dg.ops_by_id[merge_op.op_id]
        merge_op_data.uncommitted_branch_conds.append(
            (all_slices_active, parent_grad_.tensor_id, parent_grad_.domain.basis_expr)
        )
        zero_grad = SymbolicTensor.zeros(self.x.shape, self.x.dtype)
        merge_op_data.uncommitted_branch_conds.append(
            (ie.ConstBool(True), zero_grad.tensor_id, zero_grad.domain.basis_expr)
        )

        merge_op_data.op.num_inputs_[0] = merge_op_data.op.num_inputs_[0] + 2  # type: ignore

        return (parent_grad,)


class Zero(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x = args[0]
        return (x.full_like_self(0.0),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        return (grad_output.full_like_self(0.0) if self.needs_input_grad[0] else None,)


class MatMul(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x, y = args
        self.x, self.y = x, y
        return (x.matmul(y),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        return (
            (grad_output.matmul(self.y.transpose(-2, -1)) if self.needs_input_grad[0] else None),
            (self.x.transpose(-2, -1).matmul(grad_output) if self.needs_input_grad[1] else None),
        )


class Add(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x, y = args
        return (x.add(y),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        return (
            grad_output if self.needs_input_grad[0] else None,
            grad_output if self.needs_input_grad[1] else None,
        )


class Sub(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x, y = args
        return (x.sub(y),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        # NOTE the negate below is meant to be a graph op
        return (
            grad_output if self.needs_input_grad[0] else None,
            -grad_output if self.needs_input_grad[1] else None,
        )


class Mul(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x, y = args
        self.x, self.y = x, y
        return (x * y,)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        return (
            self.y * grad_output if self.needs_input_grad[0] else None,
            self.x * grad_output if self.needs_input_grad[1] else None,
        )


class Div(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x, y = args
        self.x, self.y = x, y
        return (x.divide(y),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        return (
            grad_output / self.y if self.needs_input_grad[0] else None,
            (-self.x * grad_output / (self.y * self.y) if self.needs_input_grad[1] else None),
        )


class Negate(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x = args[0]
        return (-x,)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        return (-grad_output if self.needs_input_grad[0] else None,)


class Sin(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x = args[0]
        self.x = x
        return (x.sin(),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        return (
            (
                (self.x.full_like_self(math.pi / 2) - self.x).sin() * grad_output
                if self.needs_input_grad[0]
                else None
            ),
        )


# class Conv(AutodiffFn):
#    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
#        input_, weight = args
#        self.input_ = input_
#        self.weight = weight
#
#        self.stride: Tuple[int, ...] = kwargs.get("stride", ())
#        self.padding: Tuple[int, ...] = kwargs.get("padding", ())
#        self.dilation: Tuple[int, ...] = kwargs.get("dilation", ())
#        self.transposed: bool = kwargs.get("transposed", False)
#        self.output_padding: Tuple[int, ...] = kwargs.get("output_padding", ())
#        self.groups: int = kwargs.get("groups", 0)
#        self.n_dims: int = kwargs.get("n_dims", 0)
#
#        assert not self.transposed or sum(self.output_padding) == 0,
# "output_padding is only supported for transposed convolutions"
#
#        self.output = input_.conv(
#                weight,
#                self.stride,
#                self.padding,
#                self.dilation,
#                self.transposed,
#                self.output_padding,
#                self.groups,
#                self.n_dims,
#            )
#        return (
#            self.output,
#        )
#
#    def backward(self, grad_output: SymbolicTensor) -> Sequence[Optional[SymbolicTensor]]:
#        #TODO this is wrong and is causing incorrect shapes. Fix it.
# For the weight grad, you may need to pad the input. Yo
#  u can use conv padding or the pad function of symbolic tensros.
#        input_grad = grad_output.conv(
#            self.weight,
#            self.dilation,
#            self.padding,
#            self.stride,
#            not self.transposed,
#            self.output_padding,
#            self.groups,
#            self.n_dims,
#        ) if self.needs_input_grad[0] else None
#
#        # Compute weight gradient using input and grad_output
#        pad_amount = tuple((d * (k - 1), d * (k - 1)) for d, k in zip(self.dilation,
#  self.weight.shape[2:], strict=False))
#        padded_input = self.input_.pad(pad_amount)
#        weight_grad = padded_input.conv(
#            grad_output,
#            self.stride,
#            self.padding,
#            self.dilation,
#            True,  # Transposed convolution
#            self.output_padding,
#            self.groups,
#            self.n_dims,
#        )
#
#
#        input_grad_shape_matches = input_grad.shape == self.input_.shape
#        weight_grad_shape_matches = weight_grad.shape == self.weight.shape
#
#        print("="*100)
#        print(f"output.shape: {self.output.shape}")
#        print(f"grad_output.shape: {grad_output.shape}")
#        print(f"self.input_.shape: {self.input_.shape}")
#
#        if not input_grad_shape_matches:
#            print("input_grad.shape != self.input_.shape")
#        print(f"input_grad.shape: {input_grad.shape}")
#        print(f"self.input_.shape: {self.input_.shape}")
#
#        if not weight_grad_shape_matches:
#            print("weight_grad.shape != self.weight.shape")
#        print(f"weight_grad.shape: {weight_grad.shape}")
#        print(f"self.weight.shape: {self.weight.shape}")
#
#        if not input_grad_shape_matches or not weight_grad_shape_matches:
#            raise ValueError("Shape mismatch")
#
#        return (
#            input_grad if self.needs_input_grad[0] else None,
#            weight_grad if self.needs_input_grad[1] else None,
#        )


# https://pythonandml.github.io/dlbook/content/convolutional_neural_networks/backpropagation_convolution.html
class Conv(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        input_, weight = args
        self.input_ = input_
        self.weight = weight

        # --- retrieve convolution parameters
        self.stride: tuple[int, ...] = kwargs.get("stride", ())
        self.transposed: bool = kwargs.get("transposed", False)
        # self.groups: int = kwargs.get("groups", 1)
        self.n_dims: int = kwargs.get("n_dims", 2)  # e.g. 2D, 3D, etc.

        # For now, forbid transposed. Also forbid output_padding (only relevant if transposed).
        assert not self.transposed, "Transposed convolution not supported yet."

        # Actual forward
        self.output = input_.conv(
            weight,
            stride=self.stride,
            transposed=self.transposed,
            # groups=self.groups,
            n_dims=self.n_dims,
        )
        return (self.output,)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        # ---- 1) grad w.r.t. input

        # dilated_grad_output = grad_output.dilate((s - 1 for s in self.stride))
        if self.needs_input_grad[0]:
            ## flip weight in the spatial dims: [out_channels, in_channels, *kernel_shape]
            ## Typically the first two dims are the channel dims, so flip from dim=2 onward
            # flipped_weight = self.weight.flip(dim=tuple(range(2, 2 + self.n_dims)))

            ## Need to pad and dilate the grad_output
            ##TODO is this -1 everywhere correct?
            # padded_grad_output = dilated_grad_output.pad((s-1,s-1) for s in self.stride)

            # input_grad = padded_grad_output.conv(
            #    flipped_weight,
            #    stride=(1,) * self.n_dims,
            #    transposed=False,
            #    groups=self.groups,
            #    n_dims=self.n_dims,
            # )
            input_grad = grad_output.conv(
                self.weight,
                stride=(1,) * self.n_dims,
                transposed=not self.transposed,
                n_dims=self.n_dims,
            )
        else:
            input_grad = None

        # ---- 2) grad w.r.t. weight
        if self.needs_input_grad[1]:
            raw_weight_grad = self.input_.conv(
                grad_output,
                stride=(1,) * self.n_dims,
                transposed=False,
                # groups=self.groups,
                n_dims=self.n_dims,
            )

            # TODO pretty sure this is wrong, we don't want to sum over batch dim
            # Typically, raw_weight_grad has shape [N, out_channels, in_channels, *kernel_dims]
            # or something similar, so we sum over N (batch) to get
            #  [out_channels, in_channels, *kernel_dims].
            # The exact shape that your `conv` returns may vary.
            # We assume the first dimension is batch.
            # So do a SumOp over dim=0 to reduce batch.
            # If the conv you have already lumps batch into the "output channels,"
            # you might skip this step.
            # weight_grad = raw_weight_grad.sum(reduce_dims=(0,), keepdim=False)
            weight_grad = raw_weight_grad
        else:
            weight_grad = None

        return (
            input_grad if self.needs_input_grad[0] else None,
            weight_grad if self.needs_input_grad[1] else None,
        )


class Relu(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x = args[0]
        x = x.unsqueeze(dim=0)
        x = SymbolicTensor.cat(x, x.full_like_self(0.0))
        self.ret, _ = x.max(dim=0, keepdim=False)
        return (self.ret,)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        return (
            (
                (self.ret.full_like_self(0.0) < self.ret).to_dtype(grad_output.dtype) * grad_output
                if self.needs_input_grad[0]
                else None
            ),
        )


class Sigmoid(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x = args[0]
        ones = x.full_like_self(1.0)
        self.ret = ones / (ones + (-x).exp())

        return (self.ret,)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        return (
            (
                self.ret * (self.ret.full_like_self(1) - self.ret) * grad_output
                if self.needs_input_grad[0]
                else None
            ),
        )


class Ln(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        self.antilogarithm = args[0]
        ret = SymbolicTensor.ln(self.antilogarithm)
        return (ret,)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        return (grad_output / self.antilogarithm if self.needs_input_grad[0] else None,)


class Exp(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        self.exponent = args[0]
        self.ret = self.exponent.exp()
        return (self.ret,)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        return (grad_output * self.ret if self.needs_input_grad[0] else None,)


class Pow(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        base, exponent = args
        self.base = base
        self.exponent = exponent
        self.ret = base.pow_(exponent)
        return (self.ret,)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        base = self.base
        # base = self.base + self.base.full_like_self(1.0e-8)

        return (
            (
                grad_output * self.exponent * base.pow_(self.exponent - 1)
                if self.needs_input_grad[0]
                else None
            ),
            grad_output * self.ret * base.ln() if self.needs_input_grad[1] else None,
        )


class Sqrt(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x = args[0]
        self.ret = x.sqrt()
        # assert False
        return (self.ret,)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        # TODO do we need the EPSILON here?
        return (
            (
                grad_output
                / (
                    self.ret.full_like_self(2) * self.ret
                    # + self.ret.full_like_self(1.0e-8)
                )
                if self.needs_input_grad[0]
                else None
            ),
        )


class Sum(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x = args[0]
        self.input_shape = x.shape
        self.keepdim: bool = kwargs.get("keepdim", False)
        self.reduce_dims: tuple[int, ...] = kwargs.get("reduce_dims", tuple(range(len(x.shape))))
        res = x.sum(dims=self.reduce_dims, keepdim=self.keepdim)
        return (res,)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        if not self.keepdim:
            for dim in sorted(self.reduce_dims, reverse=False):
                grad_output = grad_output.unsqueeze(dim=dim)
        return (grad_output.expand(self.input_shape) if self.needs_input_grad[0] else None,)


class CumSum(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x = args[0]
        self.dim = kwargs.get("dim", -1)
        return (x.cumsum(dim=self.dim),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        # https://github.com/pytorch/pytorch/blob/aeb5fd52c74b1c673e8565b9593cbdd5ed6f04f2/
        # torch/csrc/autograd/FunctionsManual.cpp#L883C13-L883C13
        return (
            (
                grad_output.flip(dim=self.dim).cumsum(dim=self.dim).flip(dim=self.dim)
                if self.needs_input_grad[0]
                else None
            ),
        )


class Where(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        condition, x, y = args
        self.condition = condition
        return (condition.where(x, y),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        return (
            None,
            (
                self.condition.where(grad_output, grad_output.full_like_self(0.0))
                if self.needs_input_grad[1]
                else None
            ),
            (
                self.condition.where(grad_output.full_like_self(0.0), grad_output)
                if self.needs_input_grad[2]
                else None
            ),
        )


class IndexSelect(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        self.tensor = args[0]
        self.index = args[1]
        self.dim: int = kwargs.get("dim", 0)
        return (self.tensor.index_select(self.dim, self.index),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        return (
            (
                self.tensor.full_like_self(0.0).index_add(self.dim, grad_output, self.index)
                if self.needs_input_grad[0]
                else None
            ),
            None,
        )


class IndexAdd(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        sink, src, index = args
        self.src = src
        self.index = index

        self.dim: int = kwargs.get("dim", 0)
        self.alpha: float = kwargs.get("alpha", 1.0)

        assert self.alpha == 1.0, "Backward not implemented for alpha != 1.0"

        return (sink.index_add(self.dim, src, self.index, self.alpha),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        return (
            (grad_output if self.needs_input_grad[0] else None),
            (grad_output.index_select(self.dim, self.index) if self.needs_input_grad[1] else None),
            None,
        )


class Gather(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        src, index = args
        dim: int = kwargs.get("dim", 0)
        self.dim = dim
        self.index = index
        self.src = src
        return (src.gather(dim, index),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        return (
            (
                self.src.full_like_self(0.0).scatter_add(self.dim, self.index, grad_output)
                if self.needs_input_grad[0]
                else None
            ),
            None,
        )


class ScatterAdd(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        sink, index, src = args
        dim: int = kwargs.get("dim", -1)
        self.dim = dim
        self.index = index
        return (sink.scatter_add(dim, index, src),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        return (
            grad_output if self.needs_input_grad[0] else None,
            None,
            (grad_output.gather(self.dim, self.index) if self.needs_input_grad[2] else None),
        )


class Expand(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x = args[0]
        shape: Shape = kwargs.get("shape", Shape.scalar())

        self.input_shape = x.shape

        assert len(shape) == len(x.shape), "Number of dimensions must match"

        assert (
            shape.is_scalar()
            or x.shape.is_scalar()
            or all(
                (Shape.dim_is_equal(x.shape, shape, dim=i) or Shape.dim_is_one(x.shape, dim=i))
                for i in range(len(shape))
            )
        ), f"Shapes must be compatible for expansion, got {x.shape} and {shape}"

        self.dims_expanded = tuple(
            i for i in range(len(shape)) if not Shape.dim_is_equal(x.shape, shape, dim=i)
        )
        return (x.expand(shape),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        if self.needs_input_grad[0]:
            grad_output = grad_output.sum(dims=self.dims_expanded, keepdim=True)
            return (grad_output,)
        else:
            return (None,)


class Cast(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x = args[0]
        dtype: DataType = kwargs.get("dtype", dtypes.float32)
        self.input_dtype = x.dtype
        return (x.to_dtype(dtype),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        return ((grad_output.to_dtype(self.input_dtype) if self.needs_input_grad[0] else None),)


class Flip(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x = args[0]
        dim: int = kwargs.get("dim", -1)
        self.input_shape = x.shape
        self.dim = dim
        return (x.flip(dim),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        return (grad_output.flip(dim=self.dim) if self.needs_input_grad[0] else None,)


class Reshape(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x = args[0]
        shape: Shape = kwargs.get("shape", Shape.scalar())
        self.input_shape = x.shape
        return (x.reshape(shape),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        return (grad_output.reshape(self.input_shape) if self.needs_input_grad[0] else None,)


class Permute(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x = args[0]
        dims: tuple[int, ...] = kwargs.get("dims", ())
        self.dims = dims
        return (x.permute(dims),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        return (
            (
                grad_output.permute(argsort(self.dims))  # type: ignore
                if self.needs_input_grad[0]
                else None
            ),
        )


class Ident(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x = args[0]
        return (x.ident(),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        if not self.needs_input_grad[0]:
            return (None,)

        return (grad_output,)


class Squeeze(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x = args[0]
        dim: int = kwargs.get("dim", ())
        self.dim = dim
        return (x.squeeze(dim),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        if not self.needs_input_grad[0]:
            return (None,)

        grad_output = grad_output.unsqueeze(dim=self.dim)
        return (grad_output,)


class Unsqueeze(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x = args[0]
        dim: int = kwargs.get("dim", -1)
        self.dim = dim
        return (x.unsqueeze(dim),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        if not self.needs_input_grad[0]:
            return (None,)

        return (grad_output.squeeze(dim=self.dim),)


class Max(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x = args[0]
        dim: int = kwargs.get("dim", -1)
        keepdim: bool = kwargs.get("keepdim", False)
        self.dim = dim
        self.keepdim = keepdim
        vals, idxs = x.max(dim, keepdim)

        self.x = x
        self.idxs = idxs
        if not keepdim:  # TODO does keepdim matter during backward? unsure
            self.idxs = idxs.unsqueeze(dim)
        return vals, idxs

    # TODO does this not need multiple grad inputs?
    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        if not self.keepdim:
            grad_output = grad_output.unsqueeze(dim=self.dim)
        return (
            (
                # TODO: why not gather? Because we need to sum multiple gradients?
                self.x.full_like_self(0.0).scatter_add(self.dim, self.idxs, grad_output)
                if self.needs_input_grad[0]
                else None
            ),
        )


class Cat(AutodiffFn):
    # https://github.com/pytorch/pytorch/blob/e72b0be2e1c6f63d410cb176fe2380b9990c8f2b/
    # tools/autograd/derivatives.yaml#L195C12-L195C12
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        dim: int = kwargs.get("dim", -1)
        self.dim = dim
        self.saved_tensors = args
        return (SymbolicTensor.cat(*args, dim=dim),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        # Calculate gradients for each input tensor
        grad_inputs: list[SymbolicTensor | None] = []

        # Keep track of the current index in the grad_output tensor
        grad_index: int = 0

        for need_grad, tensor in zip(self.needs_input_grad, self.saved_tensors, strict=False):
            if need_grad:
                # Determine the size of the gradient for this tensor
                grad_size = tensor.size(self.dim)
                assert isinstance(grad_size, int), "Grad does not yet support shape variance"

                lb = grad_index
                grad_input = grad_output.index_slice(self.dim, start=lb, length=grad_size)
                grad_inputs.append(grad_input)

                # Update the grad_index for the next tensor
                grad_index += grad_size
            else:
                # Determine the size of the gradient for this tensor
                grad_size = tensor.size(self.dim)
                assert not isinstance(grad_size, tuple)
                assert isinstance(grad_size, int), "Grad does not yet support shape variance"
                # No gradient required for this tensor
                grad_inputs.append(None)
                # Update the grad_index for the next tensor
                grad_index = grad_index + grad_size

        return tuple(grad_inputs)


class Pad(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x = args[0]
        padding = kwargs["padding"]
        dim = kwargs["dim"]
        value = kwargs.get("value", 0.0)
        mode = kwargs.get("mode", "constant")
        self.padding = padding
        self.dim = dim
        self.value = value
        self.mode = mode
        # TODO does this work for every mode? Supposedly yes...
        return (x.pad_dim(padding=padding, dim=dim, value=value, mode=mode),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        if not self.needs_input_grad[0]:
            return (None,)

        # For each dimension, slice out the non-padded region
        result = grad_output
        size = result.shape.at(self.dim)
        result = result.index_slice(
            self.dim, start=self.padding[0], length=size - (self.padding[0] + self.padding[1])
        )
        return (result,)


class Slice(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x = args[0]
        self.dim = kwargs["dim"]
        self.start = kwargs["start"]
        self.length = kwargs["length"]
        self.input_shape = x.shape
        return (x.index_slice(dim=self.dim, start=self.start, length=self.length),)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        if not self.needs_input_grad[0]:
            return (None,)

        padding = (self.start, self.input_shape.at(self.dim) - (self.start + self.length))

        return (grad_output.pad_dim(padding=padding, dim=self.dim),)


class Split(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x = args[0]
        self.dim = kwargs["dim"]
        self.num_splits = kwargs["num_splits"]

        self.grads_collected: list[SymbolicTensor] = []

        return x.split(self.dim, self.num_splits)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        self.grads_collected.append(grad_output)
        if len(self.grads_collected) == self.num_splits:
            return (SymbolicTensor.cat(*self.grads_collected, dim=self.dim),)
        else:
            return (None,)


class Sort(AutodiffFn):
    def forward(self, *args: SymbolicTensor, **kwargs: Any) -> Sequence[SymbolicTensor]:
        x = args[0]
        dim = kwargs["dim"]
        stable = kwargs.get("stable", False)
        descending = kwargs.get("descending", False)
        self.dim = dim
        self.stable = stable
        self.descending = descending

        sorted_vals, indices = x.sort(dim, stable, descending)
        self.indices = indices
        return (sorted_vals, indices)

    def backward(self, grad_output: SymbolicTensor) -> Sequence[SymbolicTensor | None]:
        # Use gather to place gradients back at their original positions
        # The indices tell us where each element in the sorted output came from
        return (grad_output.gather(self.dim, self.indices) if self.needs_input_grad[0] else None,)
