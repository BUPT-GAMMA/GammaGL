import math
import tensorlayerx as tlx


def masked_softmax(x, mask, dim=-1):
    r"""Softmax with masking: sets masked positions to ``-inf`` before softmax.

    Parameters
    ----------
    x : tensor
        Input tensor.
    mask : tensor
        Boolean or float mask (1 = valid, 0 = masked).
    dim : int
        Dimension along which to compute softmax.

    Returns
    -------
    tensor
        Softmax output with masked positions zeroed.
    """
    if tlx.BACKEND == 'torch':
        import torch
        mask_sum = torch.sum(mask)
    else:
        mask_sum = float(tlx.convert_to_numpy(tlx.reduce_sum(tlx.cast(mask, tlx.float32))))

    if mask_sum == 0:
        return x

    mask_float = tlx.cast(mask, x.dtype)
    while len(mask_float.shape) < len(x.shape):
        mask_float = tlx.expand_dims(mask_float, axis=-1)

    neg_inf = tlx.zeros_like(x) - 1e9
    x_masked = tlx.where(mask_float > 0.5, x, neg_inf)
    return tlx.softmax(x_masked, axis=dim)


class Xtoy(tlx.nn.Module):
    r"""Aggregate node features to global features via statistics.

    Computes ``[mean, min, max, std]`` over the node dimension, concatenates
    them, and applies a linear projection.

    Parameters
    ----------
    dx : int
        Input node feature dimension.
    dy : int
        Output global feature dimension.
    name : str, optional
        Module name.
    """
    def __init__(self, dx, dy, name=None):
        super().__init__(name=name)
        self.lin = tlx.layers.Linear(in_features=4 * dx, out_features=dy, W_init=tlx.initializers.HeUniform(a=math.sqrt(5)))

    def forward(self, X):
        """
        Parameters
        ----------
        X : tensor
            Node features of shape ``(bs, n, dx)``.

        Returns
        -------
        tensor
            Global features of shape ``(bs, dy)``.
        """
        m = tlx.reduce_mean(X, axis=1)       # (bs, dx)
        mi = tlx.reduce_min(X, axis=1)       # (bs, dx)
        ma = tlx.reduce_max(X, axis=1)       # (bs, dx)

        if tlx.BACKEND == 'torch':
            import torch
            std = torch.std(X, dim=1)
        else:
            mean_val = tlx.reduce_mean(X, axis=1, keepdims=True)
            diff_sq = (X - mean_val) ** 2
            std = tlx.sqrt(tlx.reduce_mean(diff_sq, axis=1) + 1e-12)

        z = tlx.concat([m, mi, ma, std], axis=-1)  # (bs, 4*dx)
        return self.lin(z)


class Etoy(tlx.nn.Module):
    r"""Aggregate edge features to global features via statistics.

    Parameters
    ----------
    d : int
        Input edge feature dimension.
    dy : int
        Output global feature dimension.
    name : str, optional
        Module name.
    """
    def __init__(self, d, dy, name=None):
        super().__init__(name=name)
        self.lin = tlx.layers.Linear(in_features=4 * d, out_features=dy, W_init=tlx.initializers.HeUniform(a=math.sqrt(5)))

    def forward(self, E):
        """
        Parameters
        ----------
        E : tensor
            Edge features of shape ``(bs, n, n, de)``.

        Returns
        -------
        tensor
            Global features of shape ``(bs, dy)``.
        """
        m = tlx.reduce_mean(E, axis=(1, 2))    # (bs, de)

        mi = tlx.reduce_min(tlx.reduce_min(E, axis=2), axis=1)
        ma = tlx.reduce_max(tlx.reduce_max(E, axis=2), axis=1)

        if tlx.BACKEND == 'torch':
            import torch
            std = torch.std(E, dim=(1, 2))
        else:
            mean_val = tlx.reduce_mean(E, axis=(1, 2), keepdims=True)
            diff_sq = (E - mean_val) ** 2
            std = tlx.sqrt(tlx.reduce_mean(diff_sq, axis=(1, 2)) + 1e-12)

        z = tlx.concat([m, mi, ma, std], axis=-1)  # (bs, 4*de)
        return self.lin(z)


class NodeEdgeBlock(tlx.nn.Module):
    r"""Self-attention block with FiLM conditioning from edge and global features.

    This block implements multi-head attention where edge features modulate
    attention scores via FiLM (Feature-wise Linear Modulation), and global
    features condition both node and edge updates.

    Parameters
    ----------
    dx : int
        Node feature dimension.
    de : int
        Edge feature dimension.
    dy : int
        Global feature dimension.
    n_head : int
        Number of attention heads.
    name : str, optional
        Module name.
    """
    def __init__(self, dx, de, dy, n_head, name=None):
        super().__init__(name=name)
        assert dx % n_head == 0, f"dx ({dx}) must be divisible by n_head ({n_head})"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.n_head = n_head
        self.df = dx // n_head

        self.dropout_attn = tlx.layers.Dropout(p=0.1)

        # QKV projections
        self.q = tlx.layers.Linear(in_features=dx, out_features=dx, W_init=tlx.initializers.HeUniform(a=math.sqrt(5)))
        self.k = tlx.layers.Linear(in_features=dx, out_features=dx, W_init=tlx.initializers.HeUniform(a=math.sqrt(5)))
        self.v = tlx.layers.Linear(in_features=dx, out_features=dx, W_init=tlx.initializers.HeUniform(a=math.sqrt(5)))

        # FiLM: E -> attention (multiply and add)
        self.e_add = tlx.layers.Linear(in_features=de, out_features=dx, W_init=tlx.initializers.HeUniform(a=math.sqrt(5)))
        self.e_mul = tlx.layers.Linear(in_features=de, out_features=dx, W_init=tlx.initializers.HeUniform(a=math.sqrt(5)))

        # FiLM: y -> E
        self.y_e_mul = tlx.layers.Linear(in_features=dy, out_features=dx, W_init=tlx.initializers.HeUniform(a=math.sqrt(5)))
        self.y_e_add = tlx.layers.Linear(in_features=dy, out_features=dx, W_init=tlx.initializers.HeUniform(a=math.sqrt(5)))

        # FiLM: y -> X
        self.y_x_mul = tlx.layers.Linear(in_features=dy, out_features=dx, W_init=tlx.initializers.HeUniform(a=math.sqrt(5)))
        self.y_x_add = tlx.layers.Linear(in_features=dy, out_features=dx, W_init=tlx.initializers.HeUniform(a=math.sqrt(5)))

        # Global feature processing
        self.y_y = tlx.layers.Linear(in_features=dy, out_features=dy, W_init=tlx.initializers.HeUniform(a=math.sqrt(5)))
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output projections
        self.x_out = tlx.layers.Linear(in_features=dx, out_features=dx, W_init=tlx.initializers.HeUniform(a=math.sqrt(5)))
        self.e_out = tlx.layers.Linear(in_features=dx, out_features=de, W_init=tlx.initializers.HeUniform(a=math.sqrt(5)))
        self.y_out = tlx.nn.Sequential(
            tlx.layers.Linear(in_features=dy, out_features=dy, W_init=tlx.initializers.HeUniform(a=math.sqrt(5))),
            tlx.layers.ReLU(),
            tlx.layers.Linear(in_features=dy, out_features=dy, W_init=tlx.initializers.HeUniform(a=math.sqrt(5))),
        )

    def forward(self, X, E, y, node_mask):
        """
        Parameters
        ----------
        X : tensor
            Node features ``(bs, n, dx)``.
        E : tensor
            Edge features ``(bs, n, n, de)``.
        y : tensor
            Global features ``(bs, dy)``.
        node_mask : tensor
            Boolean mask ``(bs, n)``.

        Returns
        -------
        tuple
            ``(newX, newE, new_y)`` with same shapes as inputs.
        """
        bs, n, _ = X.shape
        x_mask = tlx.expand_dims(tlx.cast(node_mask, X.dtype), axis=-1)  # (bs, n, 1)
        e_mask1 = tlx.expand_dims(x_mask, axis=2)  # (bs, n, 1, 1)
        e_mask2 = tlx.expand_dims(x_mask, axis=1)  # (bs, 1, n, 1)

        # Q, K, V
        Q = self.q(X) * x_mask  # (bs, n, dx)
        K = self.k(X) * x_mask
        V = self.v(X) * x_mask

        # Reshape for multi-head: (bs, n, n_head, df)
        Q = tlx.reshape(Q, [bs, n, self.n_head, self.df])
        K = tlx.reshape(K, [bs, n, self.n_head, self.df])

        # Attention scores: (bs, n, 1, n_head, df) * (bs, 1, n, n_head, df)
        Q_exp = tlx.expand_dims(Q, axis=2)  # (bs, n, 1, n_head, df)
        K_exp = tlx.expand_dims(K, axis=1)  # (bs, 1, n, n_head, df)
        Y = Q_exp * K_exp / math.sqrt(self.df)  # (bs, n, n, n_head, df)

        # FiLM: Edge modulation of attention
        E1 = self.e_mul(E) * (e_mask1 * e_mask2)  # (bs, n, n, dx)
        E1 = tlx.reshape(E1, [bs, n, n, self.n_head, self.df])
        E2 = self.e_add(E) * (e_mask1 * e_mask2)
        E2 = tlx.reshape(E2, [bs, n, n, self.n_head, self.df])
        Y = Y * (E1 + 1) + E2

        # New edge features
        newE = tlx.reshape(Y, [bs, n, n, self.dx])  # flatten heads

        # FiLM: y -> E
        ye1 = tlx.reshape(self.y_e_add(y), [bs, 1, 1, self.dx])
        ye2 = tlx.reshape(self.y_e_mul(y), [bs, 1, 1, self.dx])
        newE = ye1 + (ye2 + 1) * newE
        newE = self.e_out(newE) * (e_mask1 * e_mask2)  # (bs, n, n, de)

        # Softmax attention
        softmax_mask = tlx.cast(
            tlx.expand_dims(x_mask, axis=1),  # (bs, 1, n, 1)
            tlx.float32
        )
        softmax_mask = tlx.tile(
            tlx.reshape(softmax_mask, [bs, 1, n, 1]),
            [1, n, 1, self.n_head]
        )  # (bs, n, n, n_head)
        attn = masked_softmax(Y, softmax_mask, dim=2)  # (bs, n, n, n_head, df)

        # Value aggregation
        V = tlx.reshape(V, [bs, n, self.n_head, self.df])
        V_exp = tlx.expand_dims(V, axis=1)  # (bs, 1, n, n_head, df)
        weighted_V = attn * V_exp  # (bs, n, n, n_head, df)
        weighted_V = tlx.reduce_sum(weighted_V, axis=2)  # (bs, n, n_head, df)
        weighted_V = tlx.reshape(weighted_V, [bs, n, self.dx])  # (bs, n, dx)

        # FiLM: y -> X
        yx1 = tlx.reshape(self.y_x_add(y), [bs, 1, self.dx])
        yx2 = tlx.reshape(self.y_x_mul(y), [bs, 1, self.dx])
        newX = yx1 + (yx2 + 1) * weighted_V
        newX = self.x_out(newX) * x_mask  # (bs, n, dx)

        # Global feature update
        y_out = self.y_y(y)
        x_y = self.x_y(X)
        e_y = self.e_y(E)
        new_y = y_out + x_y + e_y
        new_y = self.y_out(new_y)

        return newX, newE, new_y


class XEyTransformerLayer(tlx.nn.Module):
    r"""A single transformer layer that jointly updates node, edge, and global features.

    Uses ``NodeEdgeBlock`` for self-attention followed by feed-forward networks
    (FFN) for each feature type, with residual connections and layer normalization.

    Parameters
    ----------
    dx : int
        Node feature dimension.
    de : int
        Edge feature dimension.
    dy : int
        Global feature dimension.
    n_head : int
        Number of attention heads.
    dim_ffX : int
        FFN hidden dimension for nodes.
    dim_ffE : int
        FFN hidden dimension for edges.
    dim_ffy : int
        FFN hidden dimension for global features.
    dropout : float
        Dropout rate.
    layer_norm_eps : float
        Epsilon for layer normalization.
    name : str, optional
        Module name.
    """
    def __init__(self, dx, de, dy, n_head, dim_ffX=2048, dim_ffE=128,
                 dim_ffy=2048, dropout=0.1, layer_norm_eps=1e-5, name=None):
        super().__init__(name=name)

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head)

        # FFN for X
        self.linX1 = tlx.layers.Linear(in_features=dx, out_features=dim_ffX, W_init=tlx.initializers.HeUniform(a=math.sqrt(5)))
        self.linX2 = tlx.layers.Linear(in_features=dim_ffX, out_features=dx, W_init=tlx.initializers.HeUniform(a=math.sqrt(5)))
        self.normX1 = tlx.layers.LayerNorm(normalized_shape=dx, epsilon=layer_norm_eps)
        self.normX2 = tlx.layers.LayerNorm(normalized_shape=dx, epsilon=layer_norm_eps)
        self.dropoutX1 = tlx.layers.Dropout(p=dropout)
        self.dropoutX2 = tlx.layers.Dropout(p=dropout)
        self.dropoutX3 = tlx.layers.Dropout(p=dropout)

        # FFN for E
        self.linE1 = tlx.layers.Linear(in_features=de, out_features=dim_ffE, W_init=tlx.initializers.HeUniform(a=math.sqrt(5)))
        self.linE2 = tlx.layers.Linear(in_features=dim_ffE, out_features=de, W_init=tlx.initializers.HeUniform(a=math.sqrt(5)))
        self.normE1 = tlx.layers.LayerNorm(normalized_shape=de, epsilon=layer_norm_eps)
        self.normE2 = tlx.layers.LayerNorm(normalized_shape=de, epsilon=layer_norm_eps)
        self.dropoutE1 = tlx.layers.Dropout(p=dropout)
        self.dropoutE2 = tlx.layers.Dropout(p=dropout)
        self.dropoutE3 = tlx.layers.Dropout(p=dropout)

        # FFN for y
        self.lin_y1 = tlx.layers.Linear(in_features=dy, out_features=dim_ffy, W_init=tlx.initializers.HeUniform(a=math.sqrt(5)))
        self.lin_y2 = tlx.layers.Linear(in_features=dim_ffy, out_features=dy, W_init=tlx.initializers.HeUniform(a=math.sqrt(5)))
        self.norm_y1 = tlx.layers.LayerNorm(normalized_shape=dy, epsilon=layer_norm_eps)
        self.norm_y2 = tlx.layers.LayerNorm(normalized_shape=dy, epsilon=layer_norm_eps)
        self.dropout_y1 = tlx.layers.Dropout(p=dropout)
        self.dropout_y2 = tlx.layers.Dropout(p=dropout)
        self.dropout_y3 = tlx.layers.Dropout(p=dropout)

    def forward(self, X, E, y, node_mask):
        """
        Parameters
        ----------
        X : tensor
            Node features ``(bs, n, dx)``.
        E : tensor
            Edge features ``(bs, n, n, de)``.
        y : tensor
            Global features ``(bs, dy)``.
        node_mask : tensor
            Boolean mask ``(bs, n)``.

        Returns
        -------
        tuple
            ``(X, E, y)`` with same shapes as inputs.
        """
        # Self-attention
        newX, newE, new_y = self.self_attn(X, E, y, node_mask)

        # Residual + LayerNorm (block 1)
        X = self.normX1(X + self.dropoutX1(newX))
        E = self.normE1(E + self.dropoutE1(newE))
        y = self.norm_y1(y + self.dropout_y1(new_y))

        # FFN
        ff_X = self.linX2(self.dropoutX2(tlx.relu(self.linX1(X))))
        ff_E = self.linE2(self.dropoutE2(tlx.relu(self.linE1(E))))
        ff_y = self.lin_y2(self.dropout_y2(tlx.relu(self.lin_y1(y))))

        # Residual + LayerNorm (block 2)
        X = self.normX2(X + self.dropoutX3(ff_X))
        E = self.normE2(E + self.dropoutE3(ff_E))
        y = self.norm_y2(y + self.dropout_y3(ff_y))

        return X, E, y
