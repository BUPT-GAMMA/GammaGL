import tensorlayerx as tlx
from defog_utils import backend_one_hot


def p_xt_g_x1(X1, E1, t, limit_dist):
    r"""Compute the conditional probability ``p(x_t | x_1)`` under linear flow.

    Linear interpolation: ``p(x_t | x_1) = t * onehot(x_1) + (1-t) * limit_dist``.

    Parameters
    ----------
    X1 : tensor
        Integer node labels, shape ``(bs, n)``.
    E1 : tensor
        Integer edge labels, shape ``(bs, n, n)``.
    t : tensor
        Time values in [0,1], shape ``(bs, 1)``.
    limit_dist : PlaceHolder
        Noise distribution with ``.X`` of shape ``(dx,)`` and ``.E`` of shape ``(de,)``.

    Returns
    -------
    tuple
        ``(prob_X, prob_E)`` with shapes ``(bs, n, dx)`` and ``(bs, n, n, de)``.
    """
    dx = len(limit_dist.X)
    de = len(limit_dist.E)

    # t_time shape: (bs, 1, 1) for broadcasting with X
    t_squeezed = tlx.reshape(t, [-1])  # (bs,)
    t_x = tlx.reshape(t_squeezed, [-1, 1, 1])       # (bs, 1, 1)
    t_e = tlx.reshape(t_squeezed, [-1, 1, 1, 1])    # (bs, 1, 1, 1)

    X1_onehot = backend_one_hot(X1, dx)  # (bs, n, dx)
    E1_onehot = backend_one_hot(E1, de)  # (bs, n, n, de)

    limit_X = tlx.reshape(tlx.cast(limit_dist.X, tlx.float32), [1, 1, dx])    # (1, 1, dx)
    limit_E = tlx.reshape(tlx.cast(limit_dist.E, tlx.float32), [1, 1, 1, de]) # (1, 1, 1, de)

    prob_X = t_x * X1_onehot + (1.0 - t_x) * limit_X
    prob_E = t_e * E1_onehot + (1.0 - t_e) * limit_E

    prob_X = tlx.clip_by_value(prob_X, clip_value_min=0.0, clip_value_max=1.0)
    prob_E = tlx.clip_by_value(prob_E, clip_value_min=0.0, clip_value_max=1.0)

    return prob_X, prob_E


def dt_p_xt_g_x1(X1, E1, limit_dist):
    r"""Compute the time derivative ``d/dt p(x_t | x_1)``.

    Since the interpolation is linear, the derivative is constant:
    ``d/dt p(x_t | x_1) = onehot(x_1) - limit_dist``.

    Parameters
    ----------
    X1 : tensor
        Integer node labels, shape ``(bs, n)``.
    E1 : tensor
        Integer edge labels, shape ``(bs, n, n)``.
    limit_dist : PlaceHolder
        Noise distribution.

    Returns
    -------
    tuple
        ``(dX, dE)`` with shapes ``(bs, n, dx)`` and ``(bs, n, n, de)``.
    """
    dx = len(limit_dist.X)
    de = len(limit_dist.E)

    X1_onehot = backend_one_hot(X1, dx)  # (bs, n, dx)
    E1_onehot = backend_one_hot(E1, de)  # (bs, n, n, de)

    limit_X = tlx.reshape(tlx.cast(limit_dist.X, tlx.float32), [1, 1, dx])
    limit_E = tlx.reshape(tlx.cast(limit_dist.E, tlx.float32), [1, 1, 1, de])

    dX = X1_onehot - limit_X
    dE = E1_onehot - limit_E

    return dX, dE
