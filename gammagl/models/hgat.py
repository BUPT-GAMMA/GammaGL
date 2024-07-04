import tensorlayerx as tlx
from tensorlayerx.nn import ModuleDict
from gammagl.layers.conv import MessagePassing, GCNConv, HGATConv,GATConv
from gammagl.utils import segment_softmax, to_homograph, to_heterograph
from gammagl.mpops import unsorted_segment_sum

class HGATModel(tlx.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 metadata,
                 drop_rate,
                 hidden_channels=128,
                 name=None):
        

        super().__init__(name=name)
        self.hin_conv1 = HINConv(in_channels,
                                hidden_channels,
                                metadata,
                                drop_rate=drop_rate)
        self.hgat_conv1 = HGATConv(in_channels=hidden_channels,
                                  out_channels=hidden_channels,
                                  metadata=metadata,
                                  drop_rate=drop_rate)
        
        self.hin_conv2 = GCNConv(hidden_channels,
                                hidden_channels)
        self.hgat_conv2 = GATConv(in_channels=hidden_channels,
                                  out_channels=hidden_channels,
                                  dropout_rate=drop_rate)
        self.linear = tlx.nn.Linear(out_features=out_channels,
                                    in_features=hidden_channels)
        self.softmax = tlx.nn.activation.Softmax()

    def forward(self, x_dict, edge_index_dict, num_nodes_dict):
        out = self.hin_conv1(x_dict, edge_index_dict, num_nodes_dict)

        out = self.hgat_conv1(out, edge_index_dict, num_nodes_dict)
        out, edge_index, edge_value = to_homograph(out,edge_index_dict,num_nodes_dict,None)
        out = self.hin_conv2(out, edge_index)

        out = self.hgat_conv2(out, edge_index)

        out_dict = to_heterograph(out,edge_index,num_nodes_dict)
        for node_type, _ in x_dict.items():
            out_dict[node_type] = self.softmax(self.linear(out_dict[node_type]))
        
        return out_dict
    
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