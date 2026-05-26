import numpy as np
import tensorlayerx as tlx
from defog_utils import (PlaceHolder, backend_multinomial, backend_one_hot,
                         backend_triu)


def assert_correctly_masked(variable, node_mask):
    r"""Debug assertion: masked positions should be near zero.

    Parameters
    ----------
    variable : tensor
        Tensor to check.
    node_mask : tensor
        Boolean mask ``(bs, n)``.
    """
    x_mask = tlx.expand_dims(tlx.cast(node_mask, tlx.float32), axis=-1)
    inv_mask = 1.0 - x_mask
    masked_vals = variable * inv_mask
    max_val = float(tlx.convert_to_numpy(tlx.reduce_max(tlx.abs(masked_vals))))
    assert max_val < 1e-4, f"Masked values not zero: max={max_val}"


def sample_discrete_feature_noise(limit_dist, node_mask):
    r"""Sample noise from the limit distribution.

    Parameters
    ----------
    limit_dist : PlaceHolder
        Noise distribution with ``.X`` shape ``(dx,)``, ``.E`` shape ``(de,)``.
    node_mask : tensor
        Boolean mask ``(bs, n)``.

    Returns
    -------
    PlaceHolder
        Sampled one-hot noise ``(X, E, y)`` masked.
    """
    nm = tlx.cast(node_mask, tlx.float32)
    bs = nm.shape[0]
    n = nm.shape[1]

    dx = len(limit_dist.X)
    de = len(limit_dist.E)

    # Sample nodes
    x_probs = tlx.cast(limit_dist.X, tlx.float32)
    x_probs = tlx.tile(tlx.reshape(x_probs, [1, dx]), [bs * n, 1])  # (bs*n, dx)
    U_X = backend_multinomial(x_probs, 1)  # (bs*n, 1)
    U_X = tlx.reshape(U_X, [bs, n])  # (bs, n)

    # Sample edges
    e_probs = tlx.cast(limit_dist.E, tlx.float32)
    e_probs = tlx.tile(tlx.reshape(e_probs, [1, de]), [bs * n * n, 1])
    U_E = backend_multinomial(e_probs, 1)
    U_E = tlx.reshape(U_E, [bs, n, n])

    # Convert to one-hot
    U_X_oh = backend_one_hot(U_X, dx)  # (bs, n, dx)
    U_E_oh = backend_one_hot(U_E, de)  # (bs, n, n, de)

    # Symmetrize E: keep upper triangle, add transpose
    U_E_oh = backend_triu(U_E_oh, diagonal=1)
    U_E_oh = U_E_oh + tlx.transpose(U_E_oh, perm=[0, 2, 1, 3])

    U_y = tlx.zeros([bs, 0], dtype=tlx.float32)

    return PlaceHolder(X=U_X_oh, E=U_E_oh, y=U_y).mask(node_mask)


def sample_discrete_features(probX, probE, node_mask, mask=False):
    r"""Sample discrete features (integer class labels) from probability distributions.

    Parameters
    ----------
    probX : tensor
        Node class probabilities ``(bs, n, dx)``.
    probE : tensor
        Edge class probabilities ``(bs, n, n, de)``.
    node_mask : tensor
        Boolean mask ``(bs, n)``.
    mask : bool
        If True, zero out masked positions after sampling.

    Returns
    -------
    PlaceHolder
        Sampled integer indices ``X: (bs, n)``, ``E: (bs, n, n)``.
    """
    bs = probX.shape[0]
    n = probX.shape[1]
    dx = probX.shape[2]
    de = probE.shape[3]

    # Handle masked nodes: set uniform probability
    nm = tlx.cast(node_mask, tlx.float32)
    nm_exp = tlx.expand_dims(nm, axis=-1)  # (bs, n, 1)
    uniform_x = tlx.ones_like(probX) / dx
    probX = probX * nm_exp + uniform_x * (1.0 - nm_exp)

    # Sample nodes
    probX_flat = tlx.reshape(probX, [bs * n, dx])
    X_t = backend_multinomial(probX_flat, 1)  # (bs*n, 1)
    X_t = tlx.reshape(X_t, [bs, n])

    # Handle masked edges: set uniform probability for invalid pairs and diagonal
    inv_edge_mask = 1.0 - tlx.expand_dims(nm, axis=2) * tlx.expand_dims(nm, axis=1)
    inv_edge_mask = tlx.expand_dims(inv_edge_mask, axis=-1)  # (bs, n, n, 1)

    eye_n = tlx.eye(n)
    diag_mask = tlx.reshape(eye_n, [1, n, n, 1])
    diag_mask = tlx.tile(diag_mask, [bs, 1, 1, 1])

    uniform_e = tlx.ones_like(probE) / de
    probE = probE * (1.0 - inv_edge_mask) + uniform_e * inv_edge_mask
    probE = probE * (1.0 - diag_mask) + uniform_e * diag_mask

    # Sample edges
    probE_flat = tlx.reshape(probE, [bs * n * n, de])
    E_t = backend_multinomial(probE_flat, 1)
    E_t = tlx.reshape(E_t, [bs, n, n])

    # Symmetrize: keep upper triangle, mirror to lower
    E_t_int = tlx.cast(E_t, tlx.int64)
    if tlx.BACKEND == 'torch':
        import torch
        E_t_int = torch.triu(E_t_int, diagonal=1)
    else:
        E_t_np = tlx.convert_to_numpy(E_t_int)
        E_t_np = np.triu(E_t_np, k=1)
        E_t_int = tlx.convert_to_tensor(E_t_np, dtype=tlx.int64)

    E_t_int = E_t_int + tlx.transpose(E_t_int, perm=[0, 2, 1])

    if mask:
        nm_int = tlx.cast(node_mask, tlx.int64)
        X_t = tlx.cast(X_t, tlx.int64) * nm_int
        em = tlx.expand_dims(nm_int, axis=2) * tlx.expand_dims(nm_int, axis=1)
        E_t_int = E_t_int * em

    X_t = tlx.cast(X_t, tlx.int64)
    E_t_int = tlx.cast(E_t_int, tlx.int64)
    U_y = tlx.zeros([bs, 0], dtype=tlx.float32)

    return PlaceHolder(X=X_t, E=E_t_int, y=U_y)
