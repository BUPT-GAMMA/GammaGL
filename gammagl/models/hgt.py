from gammagl.layers.conv import HGTConv
import tensorlayerx as tlx
from tensorlayerx.nn import Linear, ModuleDict, ModuleList


class HGTModel(tlx.nn.Module):
    r"""The Heterogeneous Graph Transformer (HGT) proposed in `here <https://arxiv.org/pdf/2003.01332>`_ paper.

        Parameters
        ----------
        cfg: configuration of HGT model.

    """

    def __init__(self, data, hidden_channels, out_channels, num_heads, num_layers, target_node_type,
                 drop_rate=0.5):
        super().__init__()

        self.lin_dict = ModuleDict()
        self.target_node_type = target_node_type
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(out_features=hidden_channels, act='tanh')
        self.convs = ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), num_heads, group='sum',
                           dropout_rate=drop_rate)
            self.convs.append(conv)

        self.lin = Linear(in_features=hidden_channels, out_features=out_channels)

    def forward(self, x_dict, edge_index_dict):
        out = {}
        for node_type, x in x_dict.items():
            out[node_type] = self.lin_dict[node_type](x)

        for conv in self.convs:
            out = conv(out, edge_index_dict)

        return self.lin(out[self.target_node_type])
