import tensorlayerx as tlx
from tensorlayerx.nn import ModuleDict
from gammagl.layers.conv import MessagePassing
from gammagl.utils import segment_softmax
from To_homograph import to_heterograph,to_homograph
from gammagl.mpops import unsorted_segment_sum
class HGATConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 metadata,
                 negative_slope=0.2,
                 drop_rate=0.5):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.metadata = metadata
        self.negetive_slop = negative_slope
        self.dropout = tlx.layers.Dropout(drop_rate)


        self.Linear_dict_l = ModuleDict({})
        self.Linear_dict_r = ModuleDict({})

        for node_type in metadata[0]:
            self.Linear_dict_l[node_type] = tlx.layers.Linear(out_features=1,
                                                          in_features=in_channels,
                                                          W_init='xavier_uniform',
                                                          b_init=None)
            self.Linear_dict_r[node_type]= tlx.layers.Linear(out_features=1,
                                                          in_features=in_channels,
                                                          W_init='xavier_uniform',
                                                          b_init=None)
            
        self.Linear_l = tlx.layers.Linear(out_features=1,
                                          in_features=in_channels,
                                          W_init='xavier_uniform',
                                          b_init=None)
        self.Linear_r = tlx.layers.Linear(out_features=1,
                                    in_features=in_channels,
                                    W_init='xavier_uniform',
                                    b_init=None)
        self.leakyReLu = tlx.LeakyReLU(negative_slope=negative_slope)


    def forward(self, x_dict, edge_index_dict, num_nodes_dict):
        Type_Attention_Value_dict = {}

        for edge_type, edge_index in edge_index_dict.items():
            Type_Attention_Value_dict[edge_type] = []

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            src = edge_index[0,:]
            dst = edge_index[1,:]

            h_l = self.Linear_dict_l[src_type](x_dict[src_type])
            h_r = self.Linear_dict_r[dst_type](x_dict[dst_type])
            h_l = tlx.gather(h_l,src)
            h_r = tlx.gather(h_r,dst)

            Type_Attention_Value = h_l + h_r
            Type_Attention_Value = self.leakyReLu(Type_Attention_Value)
            Type_Attention_Value_dict[edge_type].append(Type_Attention_Value)

        x_value, edge_index, edge_value = to_homograph(x_dict,edge_index_dict,num_nodes_dict, Type_Attention_Value_dict)
        print(edge_value)
        print(edge_index)
        alpha = segment_softmax(edge_value,edge_index[1,:],num_segments=None)
        alpha = self.dropout(alpha)

        src = edge_index[0,:]
        dst = edge_index[1,:]
        h_l = self.Linear_l(x_value)
        h_r = self.Linear_r(x_value)
        h_l = tlx.gather(h_l,src)
        h_r = tlx.gather(h_r,dst)
        Node_Attention_Value = (h_l + h_r)*alpha

        beta = segment_softmax(Node_Attention_Value,edge_index[1,:],num_segments=None)
        out = unsorted_segment_sum(beta*tlx.gather(x_value,edge_index[0,:]),edge_index[0,:])
        out_dict = to_heterograph(self.dropout(out),edge_index,num_nodes_dict)
        return out_dict
