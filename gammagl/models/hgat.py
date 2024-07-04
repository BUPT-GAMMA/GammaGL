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

        self.hgat_conv = HGATConv(in_channels=hidden_channels,
                                  out_channels=hidden_channels,
                                  metadata=metadata,
                                  drop_rate=drop_rate)
        
        # This two layers is equal to another HGAT Layer, as I followed the code of the paper.
        # It seems like after the first HGAT layer, the node can be considered as a same type
        # If we use another HGATConv again, there will be error, indicating: Grad is none
        # This may because the design of the HGAT layer, if it is placed in the last layer, some of the parameter won't influence the output
        self.gcn_conv = GCNConv(hidden_channels,
                                hidden_channels)
        self.gat_conv = GATConv(in_channels=hidden_channels,
                                  out_channels=hidden_channels,
                                  dropout_rate=drop_rate)
        self.linear = tlx.nn.Linear(out_features=out_channels,
                                    in_features=hidden_channels)
        self.softmax = tlx.nn.activation.Softmax()

    def forward(self, x_dict, edge_index_dict, num_nodes_dict):
        out = self.hgat_conv(x_dict, edge_index_dict, num_nodes_dict)

        out, edge_index, edge_value = to_homograph(out,edge_index_dict,num_nodes_dict,None)
        out = self.gcn_conv(out, edge_index)

        out = self.gat_conv(out, edge_index)

        out_dict = to_heterograph(out,edge_index,num_nodes_dict)
        for node_type, _ in x_dict.items():
            out_dict[node_type] = self.softmax(self.linear(out_dict[node_type]))
        
        return out_dict
    