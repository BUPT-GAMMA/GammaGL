import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.mpops import unsorted_segment_sum


class MoleculeGNNConv(MessagePassing):
    r"""Graph convolution layer designed for molecular graphs.

    This layer extends GammaGL's MessagePassing to support discrete bond
    (edge) types commonly found in molecular graphs such as SINGLE, DOUBLE,
    TRIPLE, and AROMATIC bonds.  It learns a separate linear transformation
    for each bond type and aggregates messages accordingly, similar in
    spirit to RGCNConv but streamlined for the molecular domain.

    The message for an edge of type :math:`r` from node :math:`j` to node
    :math:`i` is:

    .. math::
        \mathbf{m}_{j \rightarrow i} = \mathbf{W}_r \, \mathbf{x}_j

    and the updated node representation is:

    .. math::
        \mathbf{x}_i^{\prime} = \sigma\!\Big(
            \mathbf{W}_{\text{root}} \mathbf{x}_i
            + \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{j \rightarrow i}
        \Big)

    Parameters
    ----------
    in_channels : int
        Input atom-feature dimension.
    out_channels : int
        Output atom-feature dimension.
    num_bond_types : int, optional
        Number of distinct bond (edge) types.  Default is 4
        (SINGLE, DOUBLE, TRIPLE, AROMATIC).
    act : callable, optional
        Activation function applied after aggregation.  Default is
        :class:`tlx.nn.ReLU`.
    add_bias : bool, optional
        If ``True``, learn an additive bias.  Default is ``True``.

    Shapes
    ------
    Input
        - node features ``x`` : :math:`(N, F_{\text{in}})`
        - edge index ``edge_index`` : :math:`(2, E)`
        - edge type ``edge_type`` : :math:`(E,)`  (integer in ``[0, num_bond_types)``)

    Output
        - node features : :math:`(N, F_{\text{out}})`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bond_types: int = 4,
        act=tlx.nn.ReLU(),
        add_bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_bond_types = num_bond_types
        self.act = act
        self.add_bias = add_bias

        initor = tlx.initializers.he_normal()

        self.w_root = tlx.layers.Linear(
            out_features=out_channels,
            in_features=in_channels,
            W_init=initor,
            b_init=None,
        )

        self.w_bonds = tlx.nn.ModuleList()
        for _ in range(num_bond_types):
            self.w_bonds.append(
                tlx.layers.Linear(
                    out_features=out_channels,
                    in_features=in_channels,
                    W_init=initor,
                    b_init=None,
                )
            )

        if add_bias:
            self.bias = self._get_weights(
                "bias", shape=(1, out_channels), init=tlx.initializers.Zeros()
            )

    def propagate(self, x, edge_index, edge_type, **kwargs):
        """Override propagate to iterate over bond types."""
        num_nodes = x.shape[0]
        out = tlx.zeros((num_nodes, self.out_channels), dtype=tlx.float32)

        for r in range(self.num_bond_types):
            if edge_type is None:
                continue
            mask = edge_type == r
            if tlx.BACKEND == "mindspore":
                idx = tlx.convert_to_tensor(
                    [i for i, v in enumerate(mask) if v], dtype=tlx.int64
                )
                edges_r = tlx.gather(edge_index, idx)
            else:
                edges_r = tlx.transpose(tlx.transpose(edge_index)[mask])

            if edges_r.shape[1] == 0:
                continue

            msg = tlx.gather(x, edges_r[0, :])
            msg = self.w_bonds[r](msg)
            agg = unsorted_segment_sum(msg, edges_r[1, :], num_nodes)
            out = out + agg

        return out

    def forward(self, x, edge_index, edge_type=None, num_nodes=None):
        r"""Forward pass.

        Parameters
        ----------
        x : tensor
            Atom feature matrix :math:`(N, F_{\text{in}})`.
        edge_index : tensor
            COO edge index :math:`(2, E)`.
        edge_type : tensor, optional
            Bond type per edge :math:`(E,)`.  If ``None`` all edges are
            treated as a single type (type 0).
        num_nodes : int, optional
            Number of nodes in the graph.  Inferred from ``x`` if not given.

        Returns
        -------
        tensor
            Updated atom features :math:`(N, F_{\text{out}})`.
        """
        x_root = self.w_root(x)

        if edge_type is not None:
            x_msg = self.propagate(x, edge_index, edge_type)
        else:
            x_msg = self.propagate(x, edge_index, edge_weight=None)

        out = x_root + x_msg

        if self.add_bias:
            out = out + self.bias

        if self.act is not None:
            out = self.act(out)

        return out
