import tensorlayerx as tlx
from gammagl.layers.pool.glob import (
    global_sum_pool,
    global_mean_pool,
    global_max_pool,
)


class MoleculeReadout(tlx.nn.Module):
    r"""Global readout module for molecular property prediction.

    Combines multiple pooling strategies (sum, mean, max) to produce a
    rich molecular-level representation from node-level embeddings, then
    passes the concatenated result through an MLP predictor.

    Parameters
    ----------
    in_channels : int
        Dimensionality of the input node features.
    hidden_channels : int
        Hidden dimension of the prediction MLP.
    out_channels : int
        Output dimension (e.g. number of property classes or regression targets).
    dropout : float, optional
        Dropout rate applied inside the MLP.  Default is 0.0.
    act : callable, optional
        Activation function for hidden MLP layers.  Default is :class:`tlx.nn.ReLU`.
    use_sum : bool, optional
        Include global sum pooling.  Default is ``True``.
    use_mean : bool, optional
        Include global mean pooling.  Default is ``True``.
    use_max : bool, optional
        Include global max pooling.  Default is ``True``.

    Shapes
    ------
    Input
        - node features ``x`` : :math:`(N, F_{\text{in}})`
        - batch vector ``batch`` : :math:`(N,)`  (graph id per node)

    Output
        - graph-level predictions : :math:`(B, F_{\text{out}})` where
          :math:`B` is the number of graphs in the batch.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        act=tlx.nn.ReLU(),
        use_sum: bool = True,
        use_mean: bool = True,
        use_max: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.use_sum = use_sum
        self.use_mean = use_mean
        self.use_max = use_max

        num_pools = int(use_sum) + int(use_mean) + int(use_max)
        concat_dim = in_channels * num_pools

        initor = tlx.initializers.he_normal()

        self.fc1 = tlx.layers.Linear(
            out_features=hidden_channels,
            in_features=concat_dim,
            W_init=initor,
        )
        self.fc2 = tlx.layers.Linear(
            out_features=out_channels,
            in_features=hidden_channels,
            W_init=initor,
        )
        self.act = act
        self.dropout = tlx.nn.Dropout(p=dropout)

    def forward(self, x, batch):
        r"""Compute molecular readout.

        Parameters
        ----------
        x : tensor
            Node feature matrix :math:`(N, F_{\text{in}})`.
        batch : tensor
            Batch vector :math:`(N,)` assigning each node to a graph.

        Returns
        -------
        tensor
            Molecular-level predictions :math:`(B, F_{\text{out}})`.
        """
        pools = []
        if self.use_sum:
            pools.append(global_sum_pool(x, batch))
        if self.use_mean:
            pools.append(global_mean_pool(x, batch))
        if self.use_max:
            pools.append(global_max_pool(x, batch))

        h = tlx.concat(pools, axis=-1)

        h = self.fc1(h)
        if self.act is not None:
            h = self.act(h)
        h = self.dropout(h)
        out = self.fc2(h)

        return out
