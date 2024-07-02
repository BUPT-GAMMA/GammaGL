import tensorlayerx as tlx
from tensorlayerx.nn import ModuleDict
from gammagl.layers.conv import MessagePassing, GCNConv, HINConv, HGATConv,GATConv
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
        #     out_dict={}
        # for node_type, _ in x_dict.items():
        #     out_dict[node_type]=[]
        # for edge_type, edge_index in edge_index_dict.items():
        #     src_type, _, dst_type = edge_type
        #     src = edge_index[0,:]
        #     dst = edge_index[1,:]
        #     message = unsorted_segment_sum(tlx.gather(x_dict[src_type],src),dst,num_nodes_dict[dst_type])
        #     out_dict[dst_type].append(message)
        # for node_type, outs in out_dict.items():
        #     aggr_out = tlx.reduce_sum(outs,axis=0)
        #     out_dict[node_type]=tlx.relu(aggr_out)