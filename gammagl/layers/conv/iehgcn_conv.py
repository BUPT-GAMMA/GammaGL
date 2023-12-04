import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing, GCNConv
from tensorlayerx.nn import ModuleDict, Linear, Dropout
from tensorlayerx import elu


class ieHGCNConv(MessagePassing):
    r"""
        ie-HGCN from paper `Interpretable and Efficient Heterogeneous Graph Convolutional Network
        <https://arxiv.org/pdf/2005.13183.pdf>`__.

        `Source Code Link <https://github.com/kepsail/ie-HGCN>`_

        The core part of ie-HGCN, the calculating flow of projection, object-level aggregation and type-level aggregation in
        a specific type block.

        Projection

        .. math::
            Y^{Self-\Omega }=H^{\Omega} \cdot W^{Self-\Omega} \quad (1)-1

            Y^{\Gamma - \Omega}=H^{\Gamma} \cdot W^{\Gamma - \Omega} , \Gamma \in N_{\Omega} \quad (1)-2

        Object-level Aggregation

        .. math::
            Z^{ Self - \Omega } = Y^{ Self - \Omega}=H^{\Omega} \cdot W^{Self - \Omega} \quad (2)-1

            Z^{\Gamma - \Omega}=\hat{A}^{\Omega-\Gamma} \cdot Y^{\Gamma - \Omega} = \hat{A}^{\Omega-\Gamma} \cdot H^{\Gamma} \cdot W^{\Gamma - \Omega} \quad (2)-2

        Type-level Aggregation

        .. math::
            Q^{\Omega}=Z^{Self-\Omega} \cdot W_q^{\Omega} \quad (3)-1

            K^{Self-\Omega}=Z^{Self -\Omega} \cdot W_{k}^{\Omega} \quad (3)-2

            K^{\Gamma - \Omega}=Z^{\Gamma - \Omega} \cdot W_{k}^{\Omega}, \quad \Gamma \in N_{\Omega} \quad (3)-3

        .. math::
            e^{Self-\Omega}={ELU} ([K^{ Self-\Omega} \| Q^{\Omega}] \cdot w_{a}^{\Omega}) \quad (4)-1

            e^{\Gamma - \Omega}={ELU} ([K^{\Gamma - \Omega} \| Q^{\Omega}] \cdot w_{a}^{\Omega}), \Gamma \in N_{\Omega} \quad (4)-2

        .. math::
            [a^{Self-\Omega}\|a^{1 - \Omega}\| \ldots . a^{\Gamma - \Omega}\|\ldots\| a^{|N_{\Omega}| - \Omega}] \\
            = {softmax}([e^{Self - \Omega}\|e^{1 - \Omega}\| \ldots\|e^{\Gamma - \Omega}\| \ldots \| e^{|N_{\Omega}| - \Omega}]) \quad (5)

        .. math::
            H_{i,:}^{\Omega \prime}=\sigma(a_{i}^{Self-\Omega} \cdot Z_{i,:}^{Self-\Omega}+\sum_{\Gamma \in N_{\Omega}} a_{i}^{\Gamma - \Omega} \cdot Z_{i,:}^{\Gamma - \Omega}) \quad (6)

        Parameters
        ----------
        in_channels : int, dict
            input feature dimensions of different input nodes
        out_channels : int
            number of the target type node
        attn_channels : int
            the dimension of attention vector
        metadata : tuple[list[str], list[tuple[str, str, str]]]
            The metadata of the heterogeneous graph, *i.e.* its node and edge types given by a list of strings and a list of string triplets, respectively. 
            See :class:`gammagl.data.HeteroGraph.metadata` for more information.
        batchnorm : bool
            whether we need batchnorm
        add_bias : bool
            whether we need bias vector
        activation : nn.module
            the activation function
        dropout_rate : float
            the drop out rate
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            attn_channels,
            metadata,
            batchnorm=False,
            add_bias=False,
            activation=elu,
            dropout_rate=0.0):
        super(ieHGCNConv, self).__init__()

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_channels = attn_channels
        self.metadata = metadata
        self.batchnorm = batchnorm
        self.add_bias = add_bias
        self.dropout_rate = dropout_rate

        self.W_self = ModuleDict()
        self.W_al = ModuleDict()
        self.W_ar = ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.W_self[node_type] = Linear(in_features=in_channels, out_features=self.out_channels,
                                            W_init='xavier_uniform', b_init=None)
            self.W_al[node_type] = Linear(in_features=self.attn_channels, out_features=self.out_channels,
                                          W_init='xavier_uniform', b_init=None)
            self.W_ar[node_type] = Linear(in_features=self.attn_channels, out_features=self.out_channels,
                                          W_init='xavier_uniform', b_init=None)

        self.gcn_dict = ModuleDict({})
        for edge_type in metadata[1]:
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            self.gcn_dict[edge_type] = GCNConv(in_channels=self.in_channels[src_type],
                                               out_channels=self.out_channels,
                                               norm='right')

        self.linear_q = ModuleDict()
        self.linear_k = ModuleDict()
        for node_type, _ in self.in_channels.items():
            self.linear_q[node_type] = Linear(in_features=self.out_channels, out_features=self.attn_channels,
                                              W_init='xavier_uniform', b_init=None)
            self.linear_k[node_type] = Linear(in_features=self.out_channels, out_features=self.attn_channels,
                                              W_init='xavier_uniform', b_init=None)

        self.activation = activation

        if self.batchnorm:
            self.bn = tlx.layers.BatchNorm1d(num_features=out_channels)
        if self.add_bias:
            initor = tlx.initializers.Zeros()
            self.bias = self._get_weights("bias", shape=(1, self.out_channels), init=initor)

        self.dropout = Dropout(self.dropout_rate)

    def forward(self, x_dict, edge_index_dict, num_nodes_dict):
        """
        The forward part of the ieHGCN.

        Parameters
        ----------
        x_dict : dict
            the feature dict of different node types
        edge_index_dict : dict
            the edge_index dict of different metapaths
        num_nodes_dict : dict
            the number of different node types
        Returns
        -------
        dict
            The embeddings after the ieHGCNConv
        """
        dst_dict, out_dict = {}, {}

        # formulas (2)-1
        # Iterate over node-types:
        for node_type, x in x_dict.items():
            dst_dict[node_type] = self.W_self[node_type](x)
            out_dict[node_type] = []

        query = {}
        key = {}
        attn = {}
        attention = {}

        # formulas (3)-1 and (3)-2
        for node_type, _ in x_dict.items():
            query[node_type] = self.linear_q[node_type](dst_dict[node_type])
            key[node_type] = self.linear_k[node_type](dst_dict[node_type])

        # formulas (4)-1
        h_l = {}
        h_r = {}
        for node_type, _ in x_dict.items():
            h_l[node_type] = self.W_al[node_type](key[node_type])
            h_r[node_type] = self.W_ar[node_type](query[node_type])

        for node_type, x in x_dict.items():
            attention[node_type] = elu(h_l[node_type] + h_r[node_type])
            attention[node_type] = tlx.expand_dims(attention[node_type], axis=0)

        # Iterate over edge-types:
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            # formulas (2)-2
            out = self.gcn_dict[edge_type](x_dict[src_type], edge_index, num_nodes=num_nodes_dict[dst_type])
            out_dict[dst_type].append(out)

            # formulas (3)-3
            attn[dst_type] = self.linear_k[dst_type](out)
            # formulas (4)-2
            h_attn = self.W_al[dst_type](attn[dst_type])
            attn.clear()

            edge_attention = elu(h_attn + h_r[dst_type])
            edge_attention = tlx.expand_dims(edge_attention, axis=0)
            attention[dst_type] = tlx.concat([attention[dst_type], edge_attention], axis=0)

        # formulas (5)
        for node_type, _ in x_dict.items():
            attention[node_type] = tlx.softmax(attention[node_type], axis=0)

        # formulas (6)
        rst = {node_type: 0 for node_type, _ in x_dict.items()}
        for node_type, data in out_dict.items():
            data = [dst_dict[node_type]] + data
            if len(data) != 0:
                for i in range(len(data)):
                    aggregation = tlx.multiply(data[i], attention[node_type][i])
                    rst[node_type] = aggregation + rst[node_type]

        def _apply(ntype, h):
            if self.add_bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)
            if self.batchnorm:
                h = self.bn(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, feat) for ntype, feat in rst.items()}