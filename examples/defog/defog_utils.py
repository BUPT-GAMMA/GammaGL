import math
import numpy as np
import tensorlayerx as tlx


def _to_dense_batch(x, batch):
    r"""Backend-agnostic conversion from sparse batched node features to dense.

    Parameters
    ----------
    x : tensor
        Node features ``(N_total, dx)``.
    batch : tensor
        Batch assignment vector ``(N_total,)``.

    Returns
    -------
    tuple
        ``(X, mask)`` where X is ``(bs, n_max, dx)`` and mask is ``(bs, n_max)`` bool.
    """
    x_np = tlx.convert_to_numpy(x)
    if batch is None:
        X = x_np[np.newaxis, :, :]
        mask = np.ones((1, x_np.shape[0]), dtype=bool)
        return tlx.convert_to_tensor(X.astype(np.float32)), \
               tlx.convert_to_tensor(mask)

    batch_np = tlx.convert_to_numpy(batch).astype(np.int64)
    batch_size = int(batch_np.max()) + 1
    num_nodes = np.bincount(batch_np, minlength=batch_size)
    max_num_nodes = int(num_nodes.max())
    dx = x_np.shape[-1]

    cum_nodes = np.concatenate([[0], np.cumsum(num_nodes)[:-1]])

    X = np.zeros((batch_size, max_num_nodes, dx), dtype=np.float32)
    mask = np.zeros((batch_size, max_num_nodes), dtype=bool)

    for i in range(len(batch_np)):
        b = batch_np[i]
        local_idx = i - cum_nodes[b]
        X[b, local_idx] = x_np[i]
        mask[b, local_idx] = True

    return tlx.convert_to_tensor(X), tlx.convert_to_tensor(mask)


def _to_dense_adj(edge_index, batch, edge_attr=None, max_num_nodes=None):
    r"""Backend-agnostic conversion from sparse edge_index to dense adjacency.

    Parameters
    ----------
    edge_index : tensor
        Edge indices ``(2, E_total)``.
    batch : tensor
        Batch assignment vector ``(N_total,)``.
    edge_attr : tensor, optional
        Edge attributes ``(E_total, de)``.
    max_num_nodes : int, optional
        Maximum number of nodes (for padding).

    Returns
    -------
    tensor
        Dense adjacency ``(bs, n_max, n_max, de)`` or ``(bs, n_max, n_max)``.
    """
    ei_np = tlx.convert_to_numpy(edge_index).astype(np.int64)
    batch_np = tlx.convert_to_numpy(batch).astype(np.int64) if batch is not None else None

    if batch_np is None:
        num_nodes = int(ei_np.max()) + 1 if ei_np.size > 0 else 0
        batch_np = np.zeros(num_nodes, dtype=np.int64)

    batch_size = int(batch_np.max()) + 1
    num_nodes_per = np.bincount(batch_np, minlength=batch_size)
    cum_nodes = np.concatenate([[0], np.cumsum(num_nodes_per)[:-1]])

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes_per.max())

    src = ei_np[0]
    dst = ei_np[1]
    batch_src = batch_np[src]
    src_local = src - cum_nodes[batch_src]
    dst_local = dst - cum_nodes[batch_src]

    if edge_attr is not None:
        ea_np = tlx.convert_to_numpy(edge_attr).astype(np.float32)
        de = ea_np.shape[-1]
        adj = np.zeros((batch_size, max_num_nodes, max_num_nodes, de), dtype=np.float32)
        for i in range(len(src)):
            adj[batch_src[i], src_local[i], dst_local[i]] = ea_np[i]
    else:
        adj = np.zeros((batch_size, max_num_nodes, max_num_nodes), dtype=np.float32)
        for i in range(len(src)):
            adj[batch_src[i], src_local[i], dst_local[i]] = 1.0

    return tlx.convert_to_tensor(adj)


# ============================================================
# Backend compatibility helpers
# ============================================================

def backend_multinomial(probs, num_samples=1):
    r"""Backend-agnostic multinomial sampling.

    Parameters
    ----------
    probs : tensor
        Probability distributions, shape ``(..., num_classes)``.
    num_samples : int
        Number of samples to draw per row.

    Returns
    -------
    tensor
        Sampled indices.
    """
    if tlx.BACKEND == 'torch':
        import torch
        probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
        probs = torch.clamp(probs, min=0.0)
        row_sums = probs.sum(dim=-1, keepdim=True)
        fallback = torch.full_like(probs, 1.0 / probs.shape[-1])
        probs = torch.where(row_sums > 0, probs, fallback)
        return torch.multinomial(probs, num_samples, replacement=True)
    elif tlx.BACKEND == 'tensorflow':
        import tensorflow as tf
        logits = tf.math.log(tf.cast(probs, tf.float32) + 1e-30)
        return tf.random.categorical(logits, num_samples, dtype=tf.int64)
    elif tlx.BACKEND == 'paddle':
        import paddle
        return paddle.multinomial(probs.astype('float32'), num_samples, replacement=True)
    elif tlx.BACKEND == 'mindspore':
        import mindspore
        return mindspore.ops.multinomial(probs.astype(mindspore.float32), num_samples, replacement=True)
    else:
        probs_np = tlx.convert_to_numpy(probs).astype(np.float64)
        results = []
        for row in probs_np:
            row = np.clip(row, 0, None)
            s = row.sum()
            if s <= 0:
                row = np.ones_like(row) / len(row)
            else:
                row = row / s
            results.append(np.random.choice(len(row), size=num_samples, replace=True, p=row))
        return tlx.convert_to_tensor(np.array(results), dtype=tlx.int64)


def backend_one_hot(indices, num_classes):
    r"""Backend-agnostic one-hot encoding.

    Parameters
    ----------
    indices : tensor
        Integer tensor of class indices.
    num_classes : int
        Total number of classes.

    Returns
    -------
    tensor
        One-hot encoded float tensor.
    """
    if tlx.BACKEND == 'torch':
        import torch
        import torch.nn.functional as F
        return F.one_hot(indices.long(), num_classes=num_classes).float()
    elif tlx.BACKEND == 'tensorflow':
        import tensorflow as tf
        return tf.one_hot(tf.cast(indices, tf.int64), depth=num_classes, dtype=tf.float32)
    elif tlx.BACKEND == 'paddle':
        import paddle
        import paddle.nn.functional as F
        return F.one_hot(indices.astype('int64'), num_classes=num_classes).astype('float32')
    elif tlx.BACKEND == 'mindspore':
        import mindspore
        return mindspore.ops.one_hot(indices.astype(mindspore.int64),
                                     num_classes,
                                     mindspore.Tensor(1.0, mindspore.float32),
                                     mindspore.Tensor(0.0, mindspore.float32))
    else:
        idx_np = tlx.convert_to_numpy(indices).astype(np.int64)
        result = np.eye(num_classes, dtype=np.float32)[idx_np]
        return tlx.convert_to_tensor(result)


def backend_nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0):
    r"""Replace NaN / Inf values in a tensor."""
    if tlx.BACKEND == 'torch':
        import torch
        return torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)
    else:
        is_nan = tlx.is_nan(tensor)
        result = tlx.where(is_nan, tlx.zeros_like(tensor) + nan, tensor)
        result = tlx.where(result > 1e30, tlx.zeros_like(result) + posinf, result)
        result = tlx.where(result < -1e30, tlx.zeros_like(result) + neginf, result)
        return result


def backend_triu(matrix, diagonal=0):
    r"""Backend-agnostic upper triangular extraction.

    For edge tensors of shape ``(bs, n, n, de)``, the upper triangle must be
    taken over the node-node axes, not over the last feature axis.
    """
    if len(matrix.shape) == 4:
        n = matrix.shape[1]
        tri_mask = np.triu(np.ones((n, n), dtype=np.float32), k=diagonal)
        tri_mask = tlx.convert_to_tensor(tri_mask, dtype=matrix.dtype)
        tri_mask = tlx.reshape(tri_mask, [1, n, n, 1])
        return matrix * tri_mask

    if tlx.BACKEND == 'torch':
        import torch
        return torch.triu(matrix, diagonal=diagonal)
    elif tlx.BACKEND == 'tensorflow':
        import tensorflow as tf
        return tf.linalg.band_part(matrix, 0, -1) if diagonal == 0 else _triu_with_offset(matrix, diagonal)
    elif tlx.BACKEND == 'paddle':
        import paddle
        return paddle.triu(matrix, diagonal=diagonal)
    else:
        arr = tlx.convert_to_numpy(matrix)
        result = np.triu(arr, k=diagonal)
        return tlx.convert_to_tensor(result, dtype=matrix.dtype)


def _triu_with_offset(matrix, diagonal):
    """Helper for TF triu with offset."""
    import tensorflow as tf
    shape = matrix.shape
    rows = shape[-2]
    cols = shape[-1]
    row_idx = tf.range(rows)
    col_idx = tf.range(cols)
    mask = col_idx[None, :] >= row_idx[:, None] + diagonal
    mask = tf.cast(mask, matrix.dtype)
    return matrix * mask


def count_nonzero(tensor, axis=-1):
    r"""Count nonzero elements along an axis."""
    nonzero_mask = tlx.cast(tensor != 0, tlx.float32)
    return tlx.reduce_sum(nonzero_mask, axis=axis)


def gather_last_dim(tensor, index):
    r"""Gather values along the last dimension.

    Equivalent to ``torch.gather(tensor, -1, index)``.
    """
    if tlx.BACKEND == 'torch':
        import torch
        return torch.gather(tensor, -1, index)
    elif tlx.BACKEND == 'paddle':
        import paddle
        return paddle.take_along_axis(tensor, index, axis=-1)
    elif tlx.BACKEND == 'tensorflow':
        import tensorflow as tf
        ndim = len(tensor.shape)
        return tf.gather(tensor, tf.squeeze(index, axis=-1), batch_dims=ndim - 1)[..., tf.newaxis] \
            if index.shape[-1] == 1 else _tf_gather_last(tensor, index)
    else:
        t_np = tlx.convert_to_numpy(tensor)
        i_np = tlx.convert_to_numpy(index)
        result = np.take_along_axis(t_np, i_np, axis=-1)
        return tlx.convert_to_tensor(result, dtype=tensor.dtype)


def _tf_gather_last(tensor, index):
    """TF helper for gather along last dim with arbitrary index shape."""
    import tensorflow as tf
    ndim = len(tensor.shape)
    idx_shape = index.shape
    flat_tensor = tf.reshape(tensor, [-1, tensor.shape[-1]])
    batch_size = 1
    for d in tensor.shape[:-1]:
        batch_size *= d
    offsets = tf.range(batch_size) * tensor.shape[-1]
    flat_index = tf.reshape(index, [-1]) + tf.repeat(offsets, idx_shape[-1])
    flat_result = tf.gather(tf.reshape(tensor, [-1]), flat_index)
    return tf.reshape(flat_result, idx_shape)


# ============================================================
# PlaceHolder
# ============================================================

class PlaceHolder:
    r"""Lightweight container for graph data triplet ``(X, E, y)``.

    Parameters
    ----------
    X : tensor
        Node features, shape ``(bs, n, dx)``.
    E : tensor
        Edge features, shape ``(bs, n, n, de)``.
    y : tensor
        Global features, shape ``(bs, dy)``.
    """
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x):
        self.X = tlx.cast(self.X, x.dtype)
        self.E = tlx.cast(self.E, x.dtype)
        if self.y is not None:
            self.y = tlx.cast(self.y, x.dtype)
        return self

    def mask(self, node_mask, collapse=False):
        r"""Apply node mask to zero out padding positions.

        Parameters
        ----------
        node_mask : tensor
            Boolean mask of shape ``(bs, n)``.
        collapse : bool
            If True, return argmax indices with -1 for masked positions.
        """
        x_mask = tlx.expand_dims(tlx.cast(node_mask, tlx.float32), axis=-1)  # (bs, n, 1)
        e_mask1 = tlx.expand_dims(x_mask, axis=2)   # (bs, n, 1, 1)
        e_mask2 = tlx.expand_dims(x_mask, axis=1)   # (bs, 1, n, 1)

        if collapse:
            self.X = tlx.argmax(self.X, axis=-1)
            self.E = tlx.argmax(self.E, axis=-1)
            nm_int = tlx.cast(node_mask, self.X.dtype)
            self.X = self.X * nm_int + (1 - nm_int) * (-1)
            em = tlx.cast(tlx.squeeze(e_mask1 * e_mask2, axis=-1), self.E.dtype)
            self.E = self.E * em + (1 - em) * (-1)
        else:
            self.X = self.X * x_mask
            self.E = self.E * (e_mask1 * e_mask2)
        return self

    def split(self, node_mask):
        r"""Split a batched PlaceHolder into a list of individual graphs."""
        bs = node_mask.shape[0]
        n_nodes = tlx.reduce_sum(tlx.cast(node_mask, tlx.int64), axis=1)
        n_nodes = tlx.convert_to_numpy(n_nodes)
        graphs = []
        X_np = tlx.convert_to_numpy(self.X)
        E_np = tlx.convert_to_numpy(self.E)
        for i in range(bs):
            n = int(n_nodes[i])
            xi = tlx.convert_to_tensor(X_np[i, :n])
            ei = tlx.convert_to_tensor(E_np[i, :n, :n])
            graphs.append((xi, ei))
        return graphs

    def __repr__(self):
        x_shape = self.X.shape if hasattr(self.X, 'shape') else self.X
        e_shape = self.E.shape if hasattr(self.E, 'shape') else self.E
        y_shape = self.y.shape if (self.y is not None and hasattr(self.y, 'shape')) else self.y
        return f"PlaceHolder(X={x_shape}, E={e_shape}, y={y_shape})"


# ============================================================
# Dense conversion utilities
# ============================================================

def to_dense(x, edge_index, edge_attr, batch, num_nodes=None):
    r"""Convert sparse graph to dense representation.

    Parameters
    ----------
    x : tensor
        Node features ``(N_total, dx)``.
    edge_index : tensor
        Edge indices ``(2, E_total)``.
    edge_attr : tensor
        Edge attributes ``(E_total, de)``.
    batch : tensor
        Batch assignment vector ``(N_total,)``.

    Returns
    -------
    PlaceHolder, node_mask
        Dense graph data and boolean node mask.
    """
    X, node_mask = _to_dense_batch(x, batch)

    max_num_nodes = X.shape[1]

    # Remove self-loops
    src = edge_index[0]
    dst = edge_index[1]
    if tlx.BACKEND == 'torch':
        import torch
        mask = src != dst
        edge_index_clean = edge_index[:, mask]
        edge_attr_clean = edge_attr[mask] if edge_attr is not None else None
    else:
        mask_np = tlx.convert_to_numpy(src) != tlx.convert_to_numpy(dst)
        mask_t = tlx.convert_to_tensor(mask_np)
        idx = tlx.convert_to_numpy(tlx.where(mask_t)).flatten()
        ei_np = tlx.convert_to_numpy(edge_index)
        edge_index_clean = tlx.convert_to_tensor(ei_np[:, idx], dtype=edge_index.dtype)
        if edge_attr is not None:
            ea_np = tlx.convert_to_numpy(edge_attr)
            edge_attr_clean = tlx.convert_to_tensor(ea_np[idx], dtype=edge_attr.dtype)
        else:
            edge_attr_clean = None

    E = _to_dense_adj(edge_index_clean, batch, edge_attr_clean,
                       max_num_nodes=max_num_nodes)

    if len(E.shape) == 3:
        E = tlx.expand_dims(E, axis=-1)

    E = encode_no_edge(E)

    # Apply node_mask to zero out padding positions (matches original DeFoG PlaceHolder.mask)
    x_mask = tlx.expand_dims(tlx.cast(node_mask, E.dtype), axis=-1)
    e_mask = tlx.expand_dims(x_mask, axis=2) * tlx.expand_dims(x_mask, axis=1)
    E = E * e_mask
    X = X * x_mask

    node_mask = tlx.cast(node_mask, tlx.bool)
    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    r"""Encode 'no-edge' as the first channel (index 0).

    Parameters
    ----------
    E : tensor
        Edge features ``(bs, n, n, de)``.

    Returns
    -------
    tensor
        Modified E with ``E[:,:,:,0] = 1`` where no edge exists.
    """
    if len(E.shape) != 4:
        return E
    if E.shape[-1] == 0:
        return E

    E_np = tlx.convert_to_numpy(E)
    no_edge = np.sum(E_np, axis=3) == 0
    E_np[:, :, :, 0][no_edge] = 1.0

    n = E_np.shape[1]
    for i in range(n):
        E_np[:, i, i, :] = 0.0

    return tlx.convert_to_tensor(E_np, dtype=E.dtype)


# ============================================================
# EMA (Exponential Moving Average)
# ============================================================

class EMA:
    r"""Exponential Moving Average of model parameters.

    Maintains shadow copies of all trainable parameters and updates them
    after each training step: ``shadow = decay * shadow + (1 - decay) * param``.

    During sampling/evaluation, the EMA weights are swapped in place of the
    original weights, then restored afterwards.

    Parameters
    ----------
    model : tlx.nn.Module
        The model whose parameters to track.
    decay : float
        EMA decay factor (e.g. 0.999). Higher = slower update.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow_params = {}
        self._backup_params = None
        # Initialize shadow parameters as clones of model params
        for name, param in _named_parameters(model):
            self.shadow_params[name] = _clone_detach(param)

    def update(self, model):
        """Update shadow parameters after a training step."""
        for name, param in _named_parameters(model):
            if name in self.shadow_params:
                new_val = self.decay * self.shadow_params[name] + (1.0 - self.decay) * _detach(param)
                self.shadow_params[name] = new_val
            else:
                self.shadow_params[name] = _clone_detach(param)

    def swap_in(self, model):
        """Replace model parameters with EMA shadow parameters."""
        self._backup_params = {}
        for name, param in _named_parameters(model):
            self._backup_params[name] = _clone_detach(param)
            _copy_param(param, self.shadow_params[name])

    def swap_out(self, model):
        """Restore original model parameters from backup."""
        if self._backup_params is not None:
            for name, param in _named_parameters(model):
                _copy_param(param, self._backup_params[name])
            self._backup_params = None

    def state_dict(self):
        """Return state for saving."""
        import pickle, io
        buf = io.BytesIO()
        shadow_np = {k: tlx.convert_to_numpy(v) for k, v in self.shadow_params.items()}
        pickle.dump({'shadow_params': shadow_np, 'decay': self.decay}, buf)
        return buf.getvalue()

    def load_state_dict(self, state_bytes):
        """Load state from saved bytes."""
        import pickle, io
        buf = io.BytesIO(state_bytes)
        data = pickle.loads(buf.read())

        if isinstance(data, dict) and 'shadow_params' in data:
            self.decay = data.get('decay', self.decay)
            shadow_params = data['shadow_params']
        else:
            shadow_params = data

        for k, v in shadow_params.items():
            self.shadow_params[k] = tlx.convert_to_tensor(v)


def _named_parameters(model):
    """Iterate over (name, param) for all trainable parameters."""
    if tlx.BACKEND == 'torch':
        return model.named_parameters()
    else:
        # Fallback: use trainable_weights with index
        params = model.trainable_weights
        return [(f'param_{i}', p) for i, p in enumerate(params)]


def _clone_detach(tensor):
    """Clone and detach a tensor."""
    if tlx.BACKEND == 'torch':
        import torch
        return tensor.clone().detach()
    else:
        return tlx.convert_to_tensor(tlx.convert_to_numpy(tensor).copy())


def _detach(tensor):
    """Detach a tensor (remove from computation graph)."""
    if tlx.BACKEND == 'torch':
        return tensor.detach()
    else:
        return tlx.convert_to_tensor(tlx.convert_to_numpy(tensor))


def _copy_param(param, value):
    """Copy value into param in-place."""
    if tlx.BACKEND == 'torch':
        param.data.copy_(value)
    else:
        np_val = tlx.convert_to_numpy(value)
        tlx.assign(param, tlx.convert_to_tensor(np_val))
