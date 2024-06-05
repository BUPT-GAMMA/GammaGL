import tensorlayerx as tlx
from gammagl.layers.conv import HEATlayer


class HEAT(tlx.nn.Module):
    r"""The heterogeneous edge-enhanced graph attention operator from the`"Heterogeneous Edge-Enhanced Graph Attention Network For Multi-Agent Trajectory Prediction"
       <https://arxiv.org/abs/2106.07161>`_ paper.

    Parameters
    ----------
    in_channels_node: int
        The input feature dimension for nodes.
    in_channels_edge_attr: int
        The input feature dimension for edge attributes.
    in_channels_edge_type: int
        The input feature dimension for edge types.
    edge_attr_emb_size: int
        The embedding size for edge attributes.
    edge_type_emb_size: int
        The embedding size for edge types.
    node_emb_size: int
        The embedding size for nodes.
    out_channels: int
        The output feature dimension.
    heads: int
        The number of attention heads.
    concat: bool
        If set to `True`, the multi-head attentions are concatenated. Otherwise, they are averaged.
    hist_length: int
        The length of the history trajectory used for node features.
    out_length: int
        The length of the future trajectory

    """

    def __init__(self, hist_length, in_channels_node, out_channels, out_length, in_channels_edge_attr,
                 in_channels_edge_type, edge_attr_emb_size, edge_type_emb_size, node_emb_size, heads, concat,
                 dropout_rate, leaky_rate):
        super(HEAT, self).__init__()

        self.hist_length = hist_length
        self.in_channels_node = in_channels_node
        self.out_channels = out_channels
        self.out_length = out_length
        self.in_channels_edge_attr = in_channels_edge_attr
        self.in_channels_edge_type = in_channels_edge_type
        self.edge_attr_emb_size = edge_attr_emb_size
        self.edge_type_emb_size = edge_type_emb_size
        self.node_emb_size = node_emb_size
        self.heads = heads
        self.concat = concat
        self.leaky_rate = leaky_rate
        self.dropout_rate = dropout_rate
        self.lin1 = tlx.nn.Linear(in_features=4 * self.hist_length, out_features=self.in_channels_node, W_init='xavier_uniform')
        self.lin2 = tlx.nn.Linear(in_features=self.out_channels, out_features=self.out_length * 2)
        self.leaky_relu = tlx.nn.LeakyReLU(self.leaky_rate)

        # HEAT layers
        self.heat_conv1 = HEATlayer(in_channels_node=self.in_channels_node,
                                    in_channels_edge_attr=self.in_channels_edge_attr,
                                    in_channels_edge_type=self.in_channels_edge_type,
                                    edge_attr_emb_size=self.edge_attr_emb_size,
                                    edge_type_emb_size=self.edge_type_emb_size,
                                    node_emb_size=self.node_emb_size,
                                    out_channels=self.out_channels,
                                    heads=self.heads,
                                    concat=self.concat
                                    )

        self.heat_conv2 = HEATlayer(in_channels_node=self.out_channels + int(self.concat) * (self.heads - 1) * self.out_channels,
                                    in_channels_edge_attr=self.in_channels_edge_attr,
                                    in_channels_edge_type=self.in_channels_edge_type,
                                    edge_attr_emb_size=self.edge_attr_emb_size,
                                    edge_type_emb_size=self.edge_type_emb_size,
                                    node_emb_size=self.node_emb_size,
                                    out_channels=self.out_channels,
                                    heads=self.heads,
                                    concat=self.concat
                                    )

        # fully connected
        self.fc = tlx.nn.Linear(in_features=int(self.concat) * (self.heads - 1) * self.out_channels + self.out_channels,
                                out_features=self.out_channels)

        self.dropout = tlx.nn.Dropout(p=self.dropout_rate)

    def forward(self, x, edge_index, edge_attr, edge_type):
        node_f = tlx.reshape(x, (x.shape[0], -1))
        node_f = self.lin1(node_f)

        new_node_feature = self.heat_conv1(node_f, edge_index, edge_attr, edge_type.float())
        new_node_feature = self.dropout(new_node_feature)
        new_node_feature = self.heat_conv2(new_node_feature, edge_index, edge_attr, edge_type.float())
        new_node_feature = self.dropout(new_node_feature)
        fut_pre = self.leaky_relu(self.fc(new_node_feature))
        fut_pre = self.lin2(fut_pre)

        return fut_pre

