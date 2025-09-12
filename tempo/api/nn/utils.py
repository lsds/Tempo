from collections.abc import Sequence

from tempo.api.recurrent_tensor import MaybeRecurrentTensor, RecurrentTensor


def clip_norm(
    grads: Sequence[RecurrentTensor],
    max_norm: MaybeRecurrentTensor,
    norm_type: MaybeRecurrentTensor = 2.0,
    eps: MaybeRecurrentTensor = 1e-6,
) -> tuple[Sequence[RecurrentTensor], RecurrentTensor]:
    """This assumes grads are already reduced across the batch and time dimensions.

    Args:
        grads (Iterable[RecurrentTensor]): _description_
        max_norm (MaybeRecurrentTensor): _description_
        norm_type (MaybeRecurrentTensor, optional): _description_. Defaults to 2.0.
        eps (MaybeRecurrentTensor, optional): _description_. Defaults to 1e-6.

    Returns:
        Tuple[Iterable[RecurrentTensor], RecurrentTensor]: _description_

    """
    max_norm = RecurrentTensor.lift(max_norm)
    norm_type = RecurrentTensor.lift(norm_type)

    norms = [grad.l_norm(n=norm_type) for grad in grads]
    total_norm = RecurrentTensor.stack(*norms).l_norm(n=norm_type)

    clip_coef = max_norm / (total_norm + eps)
    clip_coef = clip_coef.clip(ub=1.0)

    return [g * clip_coef for g in grads], total_norm
