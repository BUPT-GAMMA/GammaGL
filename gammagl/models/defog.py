import math
import tensorlayerx as tlx
from gammagl.layers.attention.defog_layer import XEyTransformerLayer


def _timestep_embedding(timesteps, dim, max_period=10000):
    r"""Sinusoidal timestep embedding (internal helper)."""
    import numpy as np
    half = dim // 2
    freqs = np.exp(-math.log(max_period) * np.arange(0, half, dtype=np.float32) / half)
    freqs = tlx.convert_to_tensor(freqs)

    ts = tlx.cast(tlx.reshape(timesteps, [-1, 1]), tlx.float32)
    freqs = tlx.reshape(freqs, [1, -1])
    args = ts * freqs

    cos_part = tlx.cos(args)
    sin_part = tlx.sin(args)
    embedding = tlx.concat([cos_part, sin_part], axis=-1)

    if dim % 2 == 1:
        zeros_pad = tlx.zeros([embedding.shape[0], 1], dtype=tlx.float32)
        embedding = tlx.concat([embedding, zeros_pad], axis=-1)

    return embedding


class DeFoGModel(tlx.nn.Module):
    r"""Graph Transformer denoiser network for DeFoG discrete flow matching.

    From the `"DeFoG: Discrete Flow Matching for Graph Generation"
    <https://arxiv.org/abs/2410.04263>`_ paper (ICML 2025).

    The model takes a noisy graph representation ``(X, E, y)`` and predicts the
    clean graph. It uses a stack of ``XEyTransformerLayer`` blocks with FiLM
    conditioning between node, edge, and global features. A 64-dimensional
    sinusoidal timestep embedding is concatenated to the global features before
    the input MLP.

    Parameters
    ----------
    n_layers : int
        Number of ``XEyTransformerLayer`` blocks.
    input_dims : dict
        Input feature dimensions with keys ``'X'``, ``'E'``, ``'y'``.
    hidden_mlp_dims : dict
        Hidden MLP dimensions with keys ``'X'``, ``'E'``, ``'y'``.
    hidden_dims : dict
        Hidden transformer dimensions with keys ``'dx'``, ``'de'``, ``'dy'``,
        ``'n_head'``, ``'dim_ffX'``, ``'dim_ffE'``, ``'dim_ffy'``.
    output_dims : dict
        Output dimensions with keys ``'X'``, ``'E'``, ``'y'``.
    name : str, optional
        Model name.
    """
    def __init__(self, n_layers, input_dims, hidden_mlp_dims, hidden_dims,
                 output_dims, name=None):
        super().__init__(name=name)
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']

        # Input MLPs
        self.mlp_in_X = tlx.nn.Sequential(
            tlx.layers.Linear(in_features=input_dims['X'],
                              out_features=hidden_mlp_dims['X'], W_init=tlx.initializers.HeUniform(a=math.sqrt(5))),
            tlx.layers.ReLU(),
            tlx.layers.Linear(in_features=hidden_mlp_dims['X'],
                              out_features=hidden_dims['dx'], W_init=tlx.initializers.HeUniform(a=math.sqrt(5))),
            tlx.layers.ReLU(),
        )
        self.mlp_in_E = tlx.nn.Sequential(
            tlx.layers.Linear(in_features=input_dims['E'],
                              out_features=hidden_mlp_dims['E'], W_init=tlx.initializers.HeUniform(a=math.sqrt(5))),
            tlx.layers.ReLU(),
            tlx.layers.Linear(in_features=hidden_mlp_dims['E'],
                              out_features=hidden_dims['de'], W_init=tlx.initializers.HeUniform(a=math.sqrt(5))),
            tlx.layers.ReLU(),
        )
        # +64 for timestep embedding dimension
        self.mlp_in_y = tlx.nn.Sequential(
            tlx.layers.Linear(in_features=input_dims['y'] + 64,
                              out_features=hidden_mlp_dims['y'], W_init=tlx.initializers.HeUniform(a=math.sqrt(5))),
            tlx.layers.ReLU(),
            tlx.layers.Linear(in_features=hidden_mlp_dims['y'],
                              out_features=hidden_dims['dy'], W_init=tlx.initializers.HeUniform(a=math.sqrt(5))),
            tlx.layers.ReLU(),
        )

        # Transformer layers
        self.tf_layers = tlx.nn.ModuleList()
        for i in range(n_layers):
            self.tf_layers.append(
                XEyTransformerLayer(
                    dx=hidden_dims['dx'],
                    de=hidden_dims['de'],
                    dy=hidden_dims['dy'],
                    n_head=hidden_dims['n_head'],
                    dim_ffX=hidden_dims['dim_ffX'],
                    dim_ffE=hidden_dims['dim_ffE'],
                    dim_ffy=hidden_dims.get('dim_ffy', 2048),
                )
            )

        # Output MLPs
        self.mlp_out_X = tlx.nn.Sequential(
            tlx.layers.Linear(in_features=hidden_dims['dx'],
                              out_features=hidden_mlp_dims['X'], W_init=tlx.initializers.HeUniform(a=math.sqrt(5))),
            tlx.layers.ReLU(),
            tlx.layers.Linear(in_features=hidden_mlp_dims['X'],
                              out_features=output_dims['X'], W_init=tlx.initializers.HeUniform(a=math.sqrt(5))),
        )
        self.mlp_out_E = tlx.nn.Sequential(
            tlx.layers.Linear(in_features=hidden_dims['de'],
                              out_features=hidden_mlp_dims['E'], W_init=tlx.initializers.HeUniform(a=math.sqrt(5))),
            tlx.layers.ReLU(),
            tlx.layers.Linear(in_features=hidden_mlp_dims['E'],
                              out_features=output_dims['E'], W_init=tlx.initializers.HeUniform(a=math.sqrt(5))),
        )
        self.mlp_out_y = tlx.nn.Sequential(
            tlx.layers.Linear(in_features=hidden_dims['dy'],
                              out_features=hidden_mlp_dims['y'], W_init=tlx.initializers.HeUniform(a=math.sqrt(5))),
            tlx.layers.ReLU(),
            tlx.layers.Linear(in_features=hidden_mlp_dims['y'],
                              out_features=output_dims['y'], W_init=tlx.initializers.HeUniform(a=math.sqrt(5))),
        )

    def forward(self, X, E, y, node_mask):
        r"""Forward pass of the Graph Transformer.

        Parameters
        ----------
        X : tensor
            Node features ``(bs, n, input_dims['X'])``.
        E : tensor
            Edge features ``(bs, n, n, input_dims['E'])``.
        y : tensor
            Global features ``(bs, input_dims['y'])``. The last element
            of each row is expected to be the timestep ``t``.
        node_mask : tensor
            Boolean mask ``(bs, n)``.

        Returns
        -------
        tuple
            ``(X, E, y)`` predicted clean graph logits.
        """
        bs = X.shape[0]
        n = X.shape[1]

        # Diagonal mask for edges (zero out self-loops)
        eye_n = tlx.cast(tlx.eye(n), tlx.bool)
        diag_mask = tlx.cast(
            ~tlx.tile(tlx.reshape(eye_n, [1, n, n, 1]), [bs, 1, 1, 1]),
            X.dtype
        )

        # Skip connections from input
        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]

        # Input MLP for E + symmetrize
        new_E = self.mlp_in_E(E)
        new_E = (new_E + tlx.transpose(new_E, perm=[0, 2, 1, 3])) / 2.0

        # Timestep embedding: extract last element of y as timestep
        t = y[:, -1:]  # (bs, 1)
        time_emb = _timestep_embedding(t, 64)  # (bs, 64)
        y_with_t = tlx.concat([y, time_emb], axis=-1)  # (bs, dy + 64)

        # Apply input MLPs and mask
        new_X = self.mlp_in_X(X)
        new_y = self.mlp_in_y(y_with_t)

        # Mask
        x_mask = tlx.expand_dims(tlx.cast(node_mask, new_X.dtype), axis=-1)
        e_mask = tlx.expand_dims(x_mask, axis=2) * tlx.expand_dims(x_mask, axis=1)
        new_X = new_X * x_mask
        new_E = new_E * e_mask

        X, E, y = new_X, new_E, new_y

        # Transformer layers
        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        # Output MLPs
        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)

        # Skip connections
        X = X + X_to_out
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out

        # Symmetrize E
        E = (E + tlx.transpose(E, perm=[0, 2, 1, 3])) / 2.0

        # Final masking
        X = X * x_mask
        E = E * e_mask

        return X, E, y
