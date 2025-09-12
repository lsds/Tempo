from tempo.api import autodiff as ad
from tempo.api.recurrent_tensor import RecurrentTensor
from tempo.core import index_expr as ie
from tempo.core.domain import Domain
from tempo.utils.logger import get_logger

logger = get_logger(__name__)


def propagate_grad_backwards_simple(  # noqa: C901
    tensor: RecurrentTensor,
) -> None:
    assert tensor.grad is not None, "Gradient should not be None."
    assert tensor._ctx is not None, "Context should not be None."
    tensor_out_grad = tensor.grad
    tensor_in_grads = [
        RecurrentTensor(g, requires_grad=False) if g is not None else None
        for g in tensor._ctx.backward(tensor_out_grad._underlying)
    ]

    for parent, tensor_in_grad in zip(tensor._ctx.parents, tensor_in_grads, strict=True):
        if tensor_in_grad is not None and parent.requires_grad:
            tensor_in_grad = _sum_over_extra_dims(parent, tensor_in_grad)
            if not parent.is_like(tensor_in_grad):
                msg = f"Error setting parent {parent} with grad {tensor_in_grad}. "
                msg += f"\t{tensor_in_grad.spatial_shape=} != {parent.spatial_shape=} OR\n"
                msg += f"\t{tensor_in_grad.dtype=} != {parent.dtype=} OR \n"
                msg += f"\t{tensor_in_grad.unindexed_domain=} != {parent.unindexed_domain=}.\n"
                msg += f"{tensor_in_grad.shape=} !=? {parent.shape=}"
                raise ValueError(msg)
            parent.grad = tensor_in_grad if parent.grad is None else (parent.grad + tensor_in_grad)

    if not isinstance(tensor._ctx, ad.Split):
        tensor._ctx = None
    else:
        if tensor_in_grads[0] is not None:
            # NOTE: only clear split ctx when it has collected all grads.
            tensor._ctx = None


def _sum_over_extra_dims(
    parent: RecurrentTensor, tensor_in_grad: RecurrentTensor
) -> RecurrentTensor:
    """
    If the domain of grad is bigger than parent, we need to sum the contributions of grad.
    This can happen when in the fwd we have x(dom=(d0)) <--d0-- y(dom=(d0, d1)). Since the index
    is basis expression, it does not register a TemporalIndex ctx.

    #TODO:
    1. For now, this workaround is
    sufficient, but in the future, we really should fix this by having every RT method
    register a TemporalIndex ctx when src and snk domains are different.
    We could piggyback on lift methods.
    2. Additionally, summing 0:D1 is incorrect for an RT whose domain is (0<d0<D0, 0<d1<UB<D1),
    which can happen when y is within some merge. Or perhaps it is correct iff the merge gradient
    is also zeros for all non-defined points.
    """
    if parent.domain.is_contained_in(
        tensor_in_grad.domain
    ) and not tensor_in_grad.domain.is_contained_in(parent.domain):
        diff_dom = Domain.difference(tensor_in_grad.domain, parent.domain)

        # Build expression to sum over all diff_dom variables at once
        grad_dom_vars = tensor_in_grad.domain.variables
        expr: list[slice | ie.IndexAtom] = []
        sum_count = 0
        for i in range(len(grad_dom_vars)):
            if grad_dom_vars[i] in diff_dom:
                expr.append(ie.slice_(0, grad_dom_vars[i].as_bound()))
                sum_count += 1
            else:
                expr.append(slice(None))

                # Sum over all diff_dom dimensions at once
        tensor_in_grad = tensor_in_grad[expr].sum(tuple(range(sum_count)))

    return tensor_in_grad
