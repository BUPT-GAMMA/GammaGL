from tensorlayerx.nn import ModuleDict
from gammagl.layers.conv import MessagePassing, GCNConv
import tensorlayerx as tlx

class HINConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 metadata,
                 negative_slope=0.2,
                 drop_rate=0.5):
        super().__init__()
        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.metadata = metadata
        self.negetive_slop = negative_slope
        self.drop_rate = drop_rate
        self.dropout = tlx.layers.Dropout(drop_rate)

        self.gcn_dict = ModuleDict({})
        for node_type in self.metadata[0]:
            self.gcn_dict[node_type] = GCNConv(in_channels=in_channels[node_type],
                                        out_channels=out_channels,add_bias=False
                                        )

    def forward(self, x_dict, edge_index_dict, num_nodes_dict):
        out_dict = {}
        for node_type, x_node in x_dict.items():
            out_dict[node_type] = []
        
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            out = self.gcn_dict[src_type](self.dropout(x_dict[src_type]),
                                            edge_index,
                                            num_nodes = num_nodes_dict[dst_type])
            out = out[0:num_nodes_dict[dst_type],:]
            out_dict[dst_type].append(out)

        for node_type, outs in out_dict.items():
            aggr_out = tlx.reduce_sum(outs,axis=0)
            out_dict[node_type]=tlx.relu(aggr_out)

        return out_dict
