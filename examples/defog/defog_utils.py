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

class PlaceHolder:
    def __init__(self, X, E, y=None):
        self.X = X
        self.E = E
        self.y = y

    def mask(self, node_mask):
        X, E = apply_node_mask(self.X, self.E, node_mask)
        return PlaceHolder(X=X, E=E, y=self.y)

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

def apply_node_mask(X, E, node_mask):
    """Zero-out features of padded nodes (node_mask=False)."""
    x_mask = tlx.expand_dims(tlx.cast(node_mask, X.dtype), axis=-1)
    e_mask = tlx.expand_dims(x_mask, axis=2) * tlx.expand_dims(x_mask, axis=1)
    
    X_masked = X * x_mask
    E_masked = E * e_mask
    return X_masked, E_masked


def _to_dense_adj(edge_index, batch, edge_attr=None, max_num_nodes=None):
    """Robust conversion of sparse adjacency to dense using numpy."""
    batch_np = tlx.convert_to_numpy(batch)
    ei_np = tlx.convert_to_numpy(edge_index)
    ea_np = tlx.convert_to_numpy(edge_attr) if edge_attr is not None else None
    
    bs = int(np.max(batch_np)) + 1 if len(batch_np) > 0 else 1
    if max_num_nodes is None:
        nodes_per_graph = np.bincount(batch_np, minlength=bs)
        max_num_nodes = int(np.max(nodes_per_graph))
        
    de = 1 if ea_np is None else (ea_np.shape[-1] if len(ea_np.shape) > 1 else 1)
    adj = np.zeros((bs, max_num_nodes, max_num_nodes, de), dtype=np.float32)
    
    # Compute node offsets within each graph
    cum_nodes = np.zeros(bs + 1, dtype=np.int64)
    nodes_per_graph = np.bincount(batch_np, minlength=bs)
    cum_nodes[1:] = np.cumsum(nodes_per_graph)
    
    if len(ei_np[0]) > 0:
        graph_idx = batch_np[ei_np[0]]
        src = ei_np[0] - cum_nodes[graph_idx]
        dst = ei_np[1] - cum_nodes[graph_idx]
        
        valid = (src < max_num_nodes) & (dst < max_num_nodes)
        graph_idx = graph_idx[valid]
        src = src[valid]
        dst = dst[valid]
        
        if ea_np is not None:
            vals = ea_np[valid]
            if len(vals.shape) == 1:
                vals = vals.reshape(-1, 1)
        else:
            vals = np.ones((len(src), 1), dtype=np.float32)
            
        adj[graph_idx, src, dst, :] = vals
        
    if ea_np is None or (len(ea_np.shape) == 1):
        adj = adj.squeeze(-1)
        
    return tlx.convert_to_tensor(adj)


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
    mask = src != dst
    edge_index_clean = edge_index[:, mask]
    edge_attr_clean = edge_attr[mask] if edge_attr is not None else None

    E = _to_dense_adj(edge_index_clean, batch, edge_attr_clean,
                       max_num_nodes=max_num_nodes)

    if len(E.shape) == 3:
        E = tlx.expand_dims(E, axis=-1)

    E = encode_no_edge(E)

    # Apply node_mask to zero out padding positions
    X, E = apply_node_mask(X, E, node_mask)

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
        for name, param in model.named_parameters():
            self.shadow_params[name] = param.clone().detach()

    def update(self, model):
        """Update shadow parameters after a training step."""
        for name, param in model.named_parameters():
            if name in self.shadow_params:
                new_val = self.decay * self.shadow_params[name] + (1.0 - self.decay) * param.detach()
                self.shadow_params[name] = new_val
            else:
                self.shadow_params[name] = param.clone().detach()

    def swap_in(self, model):
        """Replace model parameters with EMA shadow parameters."""
        self._backup_params = {}
        for name, param in model.named_parameters():
            self._backup_params[name] = param.clone().detach()
            param.data.copy_(self.shadow_params[name])

    def swap_out(self, model):
        """Restore original model parameters from backup."""
        if self._backup_params is not None:
            for name, param in model.named_parameters():
                param.data.copy_(self._backup_params[name])
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



