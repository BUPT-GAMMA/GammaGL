import tensorlayerx as tlx
from tensorlayerx import nn

import os
os.environ['TL_BACKEND'] = 'torch'
from gammagl.layers.conv import MessagePassing
from gammagl.layers.conv import GATConv, HANConv



class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_features=in_size, out_features=hidden_size,act=0),
            nn.Tanh(),
            nn.Linear(in_features = hidden_size, out_features=1,act=0, b_init=False)
        )
    def forward(self, z):

        z = z.permute(1, 0, 2)

        w = self.project(z).mean(0)                    # (M, 1)
        beta = tlx.softmax(w, axis=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)

class HANLayer(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 metadata,
                 heads=1,
                 dropout_rate=0.5):
        super(HANLayer, self).__init__()
        metadata[0].append(metadata[1][0][0])
        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}
        print('in_channels',in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.metadata = metadata

        self.heads = heads
        self.dropout_rate = dropout_rate
        self.gat_dict = nn.ModuleDict({})
        for edge_type in metadata[1]:
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)

            self.gat_dict[edge_type] = GATConv(in_channels = in_channels[src_type], out_channels = out_channels, heads = heads,
                                          dropout_rate = dropout_rate)
        self.sem_att_aggr = SemanticAttention(in_size=out_channels*heads,
                                       hidden_size=out_channels)

    def forward(self, x_dict, edge_index_dict, num_nodes_dict):
        out_dict = {}
        # Iterate over node types:

        x_dict = {next(iter(num_nodes_dict)):x_dict}
        for node_type, x_node in x_dict.items():
            out_dict[node_type] = []

        # node level attention aggregation
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            out = self.gat_dict[edge_type](x_dict[src_type],
                                           edge_index.cuda())#,
                                           #num_nodes = num_nodes_dict[dst_type])
            out = tlx.elu(out)
            out_dict[dst_type].append(out)

        # semantic attention aggregation
        for node_type, outs in out_dict.items():
            outs = tlx.stack(outs)
            out_dict[node_type] = self.sem_att_aggr(outs)

        return out_dict

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'heads={self.heads})')

class model(nn.Module):
    def __init__(self, metadata, in_size, hidden_size, out_size, num_heads,dropout):
        super(model,self).__init__()
        self.han_conv = HANLayer(in_size,
                                hidden_size,
                                metadata,
                                heads=num_heads[0],
                                dropout_rate=dropout)
        self.lin = nn.Linear(in_features=hidden_size * num_heads[0],
                            out_features=out_size)

    def forward(self, data,h):
        x_dict, edge_index_dict,node_type = h, data.edge_index_dict,data['node_types']

        num_nodes_dict = {node_type:None}
        x = self.han_conv(x_dict, edge_index_dict, num_nodes_dict)

        out = {}
        for node_type, _ in num_nodes_dict.items():
            out[node_type] = self.lin(x[node_type])


        return next(iter(out.values()))

