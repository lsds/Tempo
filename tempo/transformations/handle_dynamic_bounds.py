from tempo.utils import logger

log = logger.get_logger(__name__)

# TODO:
# - We could have users specify dynamic bounds as functions earlier, like:
#     (b, B), (i, I), (t, T) = ctx.symbols(3)
#     ctx.mark_dynamic(T, (B, I))

#     - This is so that when creating the graph, we already set the correct domains on everything.

# - Mean will need adjustments, since divisor is not B*T, but B*(mean(T)) or T[0:B, i].sum(0)
# We would want T to be treated like other recurrent tensors, being for a "single sample/timestep",
# then getting vectorized

# - How do we support min(t+c, T)? -> We can't wait for T to be resolved.
#      We need to run it online and we need to make sure evaluate it correctly.
#      I believe schedule-wise we are good.
#      Runtime-wise, we will need support from the thunk launchers,
#      which will need to gather inputs from the tensor store based on T's.
# - How do we support t:min(t+c, T)? -> These yield a dynamic shaped tensor.
#  Like other slice accesses, we will need to mask for certain operations.

# ---

# It seems like a good idea to have users be specifically aware that
#     dynamic bounds are functions of other dims.
# This will make it easier to reason about the semantics of the program.
# Something cannot be dynamic if it is not a function of something else.

# Suppose then that we find a good way to do this.
# What would users write for the mean thing? B * T[0:B, i].mean() is one option.
#      Another is T[0:B, i].sum(0).
# Basically, T evaluates to the b,i-specific value.

# For forward accesses, we can add control edges to ensure that T is always evaluated
#      before anything that depends on it.
# For backward accesses, we can do the same, actually.

# TODO but these cases are a seperate matter.
# Dynamic point queries will require "Gathers" in the tensor store.
# Dynamic slice queries will require masks.
#  - The worst case are dynamic slice queries which use (T - n):(T) or something,
#      as these will require masking + gathering.


# ==== dimmed ops ====:

# Doesnt even make much sense for these ops, since the user can't know the bounds,
#      why would they select some indexes?
# GatherOp
# ScatterAddOp
# IndexSelectOp
# IndexAddOp

# IndexSliceOp
# PadOp

# FlipOp
# CatOp
# SqueezeOp
# UnsqueezeOp
# ExpandOp
# ReshapeOp
# PermuteOp


# ==== elementwise ops ====:
# CastOp
# SqrtOp
# NegOp
# NotOp
# LnOp
# ExpOp
# SinOp
# AddOp
# SubOp
# MulOp
# DivOp
# PowOp
# OrOp
# AndOp
# EqualOp
# LessThanOp
# WhereOp

# ==== reduction ops ====:
# SumOp
# MaxOp

# === Scans ===:
# CumSumOp

# === Composite ops ===:
# MatMulOp
# ConvOp

# === Not a problem ===:
# IdentOp
# RandOp
# ConstOp

# === May need special handling ===:
# EvalSymbolOp
# MergeOp


# One solution is masking with Nans on future accesses.
# Then, we always use Nan Operator versions in backend.
# https://github.com/pytorch/pytorch/issues/61474#issuecomment-1735537507
# While this approach is cool, it has issues with detecting actual nans.


# class HandleDynamicBounds(Transformation):
#    """This transformation ensures correct semantics when using dynamic dimensions.
#
#    This is done by masking reductions across dynamic dimensions, and altering the
#    evaluation of dynamic bound symbols.
#    """
#
#    def __init__(self, ctx: CompilationCtx) -> None:
#        super().__init__(ctx)
#
#    def _run(self) -> Tuple[PDG, bool]:
#        new_dg = self.ctx.dg
#
#        for dyn_bound, def_ in new_dg.dynamic_bounds.items():
#            assert isinstance(def_, TensorId)  # TODO: ensure this is true.
#
#            def_op = new_dg.ops_by_id[def_].op
#            def_st = _get_symbolic_tensor_for_op_output(new_dg, def_op, def_.output_id)
#
#            dyn_var = dyn_bound.as_var()
#
#            mask_domain = def_st.domain
#            dependency_domain = mask_domain.remove_dim(dyn_var)
#
#            # valid_mask = SymbolicTensor.merge((), dtypes.bool_, domain=dependency_domain)
#            # valid_mask[(*dependency_domain[:-1], 0)] = True
#            # valid_mask[dependency_domain] = ~def_st[(*dependency_domain[:-1], dyn_var - 1)]
#
#            # TODO
#            # NOTE: Assumption: once True, always True for all future steps.
#            # Otherwise, we need a recursive definition.
#            valid_mask = def_st.logical_not()
#
#            dyn_var_idx = mask_domain.find_variable_index(dyn_var)
#
#            # index with b, i, 0:T, then sum over T.
#            # TODO: We are using dyn_bound????
#            expr = mask_domain.basis_expr.replace_idx(dyn_var_idx, ie.IndexSlice(0, dyn_bound))
#
#            # Runtime value of T, once all are done
#            tot_steps = valid_mask.symbolic_index(expr).cumsum(dim=0)  # dom: b, i
#
#            for op in new_dg.nodes:
#                if not op.domain.has_dim(dyn_var):
#                    continue
#                idx_in_op = op.domain.find_variable_index(dyn_var)
#                for dependency_op, dependency_data in new_dg.get_flat_direct_dependents(op):
#                    # TODO: We really only care if it is something that indexes into the future.
#                    # NOTE: Is this right? I suppose we can restrict ourselves to that case,
#  but its not the general solution.
#                    # What about stuff that accesses into min(t+3, T)? ->
#  Need to gather the inputs when t > T -2 or something.
#                    # What about stuff that accesses into t:min(t+3, T)? -> Need to slice + mask?
#                    if not dependency_data.expr.members[idx_in_op].equivalent(dyn_var):
#                        new_data = dependency_data.copy()
#                        new_data[dyn_var] = new_data[dyn_var] & valid_mask[dependency_domain]
#                        new_dg.add_edge(dependency_op, op, new_data)
#
#        log.info("Hello %s", 3)
#        return new_dg, True
#
