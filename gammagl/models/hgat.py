import tensorlayerx as tlx
from tensorlayerx.nn import ModuleDict
from gammagl.layers.conv import MessagePassing, GCNConv, HINConv, HGATConv

class HGAT(tlx.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 metadata,
                 drop_rate,
                 hidden_channels=128,
                 name=None):
        

        super().__init__(name=name)
        self.hin_conv = HINConv(in_channels,
                                hidden_channels,
                                metadata,
                                drop_rate=drop_rate)
        self.hgat_conv = HGATConv(in_channels=hidden_channels,
                                  out_channels=hidden_channels,
                                  metadata=metadata,
                                  drop_rate=drop_rate)
        self.linear = tlx.nn.Linear(out_features=out_channels,
                                    in_features=hidden_channels)
        self.softmax = tlx.nn.activation.Softmax()

    def forward(self, x_dict, edge_index_dict, num_nodes_dict):
        x = self.hin_conv(x_dict, edge_index_dict, num_nodes_dict)
        out = {}
        x = self.hgat_conv(x, edge_index_dict, num_nodes_dict)
        for node_type, _ in num_nodes_dict.items():
            out[node_type] = self.softmax(self.linear(x[node_type]))
        return out
