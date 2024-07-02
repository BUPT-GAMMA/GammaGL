from tensorlayerx.nn import ModuleDict
from gammagl.layers.conv import MessagePassing
from gammagl.utils import segment_softmax, to_homograph, to_heterograph, degree
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

        self.heterlinear = ModuleDict({})
        for node_type in self.metadata[0]:
            self.heterlinear[node_type] = tlx.layers.Linear(out_features=out_channels,
                                                 in_features=in_channels[node_type],
                                                 W_init='xavier_uniform')
        

    def forward(self, x_dict, edge_index_dict, num_nodes_dict):
        out_dict = {}
        for node_type, x_node in x_dict.items():
            out_dict[node_type]= self.dropout(self.heterlinear[node_type](x_node))
        return out_dict
        
