from tempo.api import min  # noqa: A004
from tempo.api.recurrent_tensor import MaybeRecurrentTensor, RecurrentTensor
from tempo.api.rl.datatypes import RewardsRecurrentTensor, ValueRecurrentTensor
from tempo.core import index_expr as ie


def n_step_returns(
    r: RecurrentTensor | RewardsRecurrentTensor,
    t: ie.Symbol,
    T: ie.IntIndexValue | None = None,
    n: int = 5,
    gamma: MaybeRecurrentTensor = 0.98,
) -> RecurrentTensor:
    r = r.ident()

    if T is None:
        T = r.domain.get_corresponding_ubound(t)

    gamma = RecurrentTensor.lift(gamma)

    if n == 1:
        return r
    else:
        r_t_idx = r.domain.find_variable_index(t)
        r_index_expr = (*((slice(None),) * r_t_idx), slice(t, min(t + n, T)))
        return r[r_index_expr].discounted_sum(gamma)


def bootstrapped_n_step_returns(
    r: RecurrentTensor | RewardsRecurrentTensor,
    v: RecurrentTensor,
    t: ie.Symbol,
    T: ie.IntIndexValue | None = None,
    n: int = 5,
    gamma: MaybeRecurrentTensor = 0.98,
) -> RecurrentTensor:
    r = r.ident()
    v = v.ident()

    if T is None:
        T = r.domain.get_corresponding_ubound(t)

    v_t_idx = v.domain.find_variable_index(t)
    v_index_expr_prefix = (slice(None),) * v_t_idx
    if n == 1:
        discounted_rewards = r
        discounted_v_or_q_in_n_steps = gamma * v[(*v_index_expr_prefix, t + 1)]
    else:
        discounted_rewards = n_step_returns(r, t, T, n, gamma)

        v_or_q_discount = RecurrentTensor.pow_(gamma, (RecurrentTensor.min(n, (T - t)))[0])

        v_in_n_steps = v[(*v_index_expr_prefix, min(t + n, T - 1))]
        discounted_v_or_q_in_n_steps = v_or_q_discount * v_in_n_steps

    res = RecurrentTensor.placeholder(r.shape, r.dtype, r.domain.union(v.domain))
    res[t == (T - 1)] = discounted_rewards  # type: ignore
    res[True] = discounted_rewards + discounted_v_or_q_in_n_steps

    return res


def n_step_td_residual(
    r: RecurrentTensor | RewardsRecurrentTensor,
    v: RecurrentTensor,
    t: ie.Symbol,
    T: ie.IntIndexValue | None = None,
    n: int = 5,
    gamma: MaybeRecurrentTensor = 0.98,
) -> RecurrentTensor:
    r = r.ident()
    v = v.ident()

    target = bootstrapped_n_step_returns(r, v, t, T, n, gamma)
    return target.detach() - v


def gae(
    r: RewardsRecurrentTensor | RecurrentTensor,
    v: ValueRecurrentTensor | RecurrentTensor,
    t: ie.Symbol,
    T: ie.IntIndexValue | None = None,
    gamma: MaybeRecurrentTensor = 0.98,
    lambda_: MaybeRecurrentTensor = 0.95,
) -> RecurrentTensor:
    """Implementation of Generalized Advantage Estimation from
     https://arxiv.org/pdf/1506.02438
    For the case where lambda_ = 1.0 and segmented execution
     is being used, we use the truncated implementation
    provided in https://arxiv.org/pdf/1707.06347
    """
    r = r.ident()
    v = v.ident()

    if T is None:
        T = r.domain.get_corresponding_ubound(t)

    true_T = r.domain.get_corresponding_ubound(t)

    if lambda_ == 1.0:
        """
        In this case, GAE is just the normal advantage, ie, G - V.
        If segmented execution, we need to add a bootstrapping term
        as G will be truncated.
        """

        r_t_idx = r.domain.find_variable_index(t)
        r_index_expr = (*((slice(None),) * r_t_idx), slice(t, T))
        g = r[r_index_expr].discounted_sum(gamma)

        # NOTE: if T is not the true end of the episode, we need to add a bootstrapping term
        if not true_T.struct_eq(T):
            v_t_idx = v.domain.find_variable_index(t)
            v_index_expr = (*((slice(None),) * v_t_idx), T - 1)
            bootstrap = gamma ** (T - t) * v[v_index_expr]
            return g + bootstrap - v
        return g - v
    else:
        residuals = n_step_td_residual(r, v, t, T, 1, gamma).detach()

        if lambda_ == 0.0:
            return residuals

        residuals_t_idx = residuals.domain.find_variable_index(t)
        residuals_index_expr = (*((slice(None),) * residuals_t_idx), slice(t, T))

        gae = residuals[residuals_index_expr].discounted_sum(RecurrentTensor.lift(gamma) * lambda_)
        return gae
