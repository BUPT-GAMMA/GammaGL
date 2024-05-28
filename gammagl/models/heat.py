import os
import tensorlayerx as tlx
from gammagl.layers.conv import HEATlayer

# os.environ['TL_BACKEND'] = 'torch'


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
    concat: bool, optional
        If set to `True`, the multi-head attentions are concatenated. Otherwise, they are averaged.
    hist_length: int
        The length of the history trajectory used for node features.
    out_length: int
        The length of the future trajectory

    """

    def __init__(self, args):
        super(HEAT, self).__init__()

        self.args = args
        self.op = tlx.nn.Linear(in_features=4 * self.args.hist_length, out_features=self.args.in_channels_node, W_init='xavier_uniform')
        self.op2 = tlx.nn.Linear(in_features=args.out_channels, out_features=self.args.out_length * 2)
        self.leaky_relu = tlx.nn.LeakyReLU(0.1)

        # HEAT layers
        self.heat_conv1 = HEATlayer(in_channels_node=self.args.in_channels_node,
                                    in_channels_edge_attr=self.args.in_channels_edge_attr,
                                    in_channels_edge_type=self.args.in_channels_edge_type,
                                    edge_attr_emb_size=self.args.edge_attr_emb_size,
                                    edge_type_emb_size=self.args.edge_type_emb_size,
                                    node_emb_size=self.args.node_emb_size,
                                    out_channels=self.args.out_channels,
                                    heads=self.args.heads,
                                    concat=self.args.concat
                                    )

        self.heat_conv2 = HEATlayer(in_channels_node=self.args.out_channels + int(self.args.concat) * (
                self.args.heads - 1) * self.args.out_channels,
                                    in_channels_edge_attr=self.args.in_channels_edge_attr,
                                    in_channels_edge_type=self.args.in_channels_edge_type,
                                    edge_attr_emb_size=self.args.edge_attr_emb_size,
                                    edge_type_emb_size=self.args.edge_type_emb_size,
                                    node_emb_size=self.args.node_emb_size,
                                    out_channels=self.args.out_channels,
                                    heads=self.args.heads,
                                    concat=self.args.concat
                                    )

        # fully connected
        self.fc = tlx.nn.Linear(in_features=int(self.args.concat) * ( self.args.heads - 1) * self.args.out_channels + self.args.out_channels,
                                out_features=self.args.out_channels)

        # self.batch_norm = tlx.nn.BatchNorm1d(act=tlx.ReLU, num_features=self.args['heat_out_channels'] * self.args[
        # 'heat_heads'])
        self.dropout = tlx.nn.Dropout(p=0.5)

    def HEAT_enc(self, node_feature, edge_index, edge_attr, edge_type, v_node_mask, p_node_mask):
        # print("HEAT_enc:", node_feature.shape, edge_attr.shape)
        new_node_feature = self.heat_conv1(node_feature, edge_index, edge_attr, edge_type, v_node_mask, p_node_mask)
        # new_node_feature = self.batch_norm(new_node_feature)
        new_node_feature = self.dropout(new_node_feature)
        new_node_feature = self.heat_conv2(new_node_feature, edge_index, edge_attr, edge_type, v_node_mask, p_node_mask)
        # new_node_feature = self.batch_norm(new_node_feature)
        # print("new_node_features", new_node_feature.shape)
        new_node_feature = self.dropout(new_node_feature)
        node_enc = self.leaky_relu(self.fc(new_node_feature))
        return node_enc

    def forward(self, data):
        # print(data.x.shape)

        node_f = data.x.view(data.x.shape[0], -1)
        node_f = self.op(node_f)
        # print("HEAT_forward:", data.x.shape)

        fut_pre = self.HEAT_enc(node_feature=node_f,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                edge_type=data.edge_type.float(),
                                v_node_mask=data.veh_mask,
                                p_node_mask=data.ped_mask
                                )

        # print("fut_pre", fut_pre.shape)
        fut_pre = self.op2(fut_pre)

        return fut_pre

