import tensorlayerx as tlx
from gammagl.layers.conv.molecule_gnn_conv import MoleculeGNNConv
from gammagl.layers.pool.molecule_readout import MoleculeReadout


class MoleculeGNN(tlx.nn.Module):
    r"""End-to-end GNN model for molecular property prediction.

    Stacks multiple :class:`MoleculeGNNConv` layers with residual
    connections and applies :class:`MoleculeReadout` to produce a
    molecular-level prediction.

    This model is designed for **graph-level** tasks such as molecular
    property prediction (e.g. OGB-MolHIV, QM9 regression).

    Parameters
    ----------
    node_in_dim : int
        Input atom-feature dimension.
    hidden_dim : int
        Hidden feature dimension shared across all conv layers.
    out_dim : int
        Output dimension (number of property targets / classes).
    num_layers : int, optional
        Number of message-passing layers.  Default is 5.
    num_bond_types : int, optional
        Number of distinct bond types.  Default is 4
        (SINGLE, DOUBLE, TRIPLE, AROMATIC).
    dropout : float, optional
        Dropout rate.  Default is 0.2.
    use_sum : bool, optional
        Include sum pooling in readout.  Default is ``True``.
    use_mean : bool, optional
        Include mean pooling in readout.  Default is ``True``.
    use_max : bool, optional
        Include max pooling in readout.  Default is ``True``.
    residual : bool, optional
        Add residual connections between conv layers.  Default is ``True``.

    Shapes
    ------
    Input
        - node features ``x`` : :math:`(N, F_{\text{node}})`
        - edge index ``edge_index`` : :math:`(2, E)`
        - edge type ``edge_type`` : :math:`(E,)`
        - batch vector ``batch`` : :math:`(N,)`

    Output
        - predictions : :math:`(B, F_{\text{out}})` where :math:`B` is the
          number of molecules in the batch.
    """

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 5,
        num_bond_types: int = 4,
        dropout: float = 0.2,
        use_sum: bool = True,
        use_mean: bool = True,
        use_max: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.residual = residual

        self.node_encoder = tlx.layers.Linear(
            out_features=hidden_dim,
            in_features=node_in_dim,
            W_init=tlx.initializers.he_normal(),
        )

        self.convs = tlx.nn.ModuleList()
        self.batch_norms = tlx.nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(
                MoleculeGNNConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    num_bond_types=num_bond_types,
                    act=tlx.nn.ReLU(),
                    add_bias=True,
                )
            )
            self.batch_norms.append(tlx.nn.BatchNorm1d())

        self.dropout = tlx.nn.Dropout(p=dropout)

        self.readout = MoleculeReadout(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim * 2,
            out_channels=out_dim,
            dropout=dropout,
            act=tlx.nn.ReLU(),
            use_sum=use_sum,
            use_mean=use_mean,
            use_max=use_max,
        )

    def forward(self, x, edge_index, edge_type, batch):
        r"""Forward pass.

        Parameters
        ----------
        x : tensor
            Atom feature matrix :math:`(N, F_{\text{node}})`.
        edge_index : tensor
            COO edge index :math:`(2, E)`.
        edge_type : tensor
            Bond type per edge :math:`(E,)` with values in ``[0, num_bond_types)``.
        batch : tensor
            Batch vector :math:`(N,)` mapping each atom to its molecule id.

        Returns
        -------
        tensor
            Molecular predictions :math:`(B, F_{\text{out}})`.
        """
        h = self.node_encoder(x)

        for i in range(self.num_layers):
            h_conv = self.convs[i](h, edge_index, edge_type)
            h_conv = self.batch_norms[i](h_conv)

            if self.residual and h_conv.shape == h.shape:
                h = h_conv + h
            else:
                h = h_conv

            if i < self.num_layers - 1:
                h = self.dropout(h)

        out = self.readout(h, batch)
        return out
