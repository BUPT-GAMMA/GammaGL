import tensorlayerx as tlx
from tensorlayerx.nn import ModuleDict
from gammagl.layers.conv import MessagePassing
from gammagl.utils import segment_softmax
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

        for node_type in self.metadata[0]:
            self.heterlinear[node_type] = tlx.layers.Linear(out_features=out_channels,
                                                 in_features=in_channels[node_type],
                                                 W_init='xavier_uniform')


        self.Linear_dict_l = ModuleDict({})
        self.Linear_dict_r = ModuleDict({})

        l_list = []
        r_list = []

        for edge_type in metadata[1]:
            src_type,_, dst_type = edge_type
            if src_type not in r_list:
                r_list.append(src_type)
            if dst_type not in l_list:
                l_list.append(dst_type)

        for dst_type in l_list:
            print(dst_type)
            self.Linear_dict_l[dst_type] = tlx.layers.Linear(out_features=1,
                                                          in_features=in_channels,
                                                          W_init='xavier_uniform')
        for src_type in r_list:
            print(src_type)
            self.Linear_dict_r[src_type]= tlx.layers.Linear(out_features=1,
                                                          in_features=in_channels,
                                                          W_init='xavier_uniform')
            

        self.nodeAttention = tlx.layers.Linear(out_features=1,
                                               in_features=in_channels*2,
                                               W_init='xavier_uniform')

        
        self.leakyReLu = tlx.LeakyReLU(negative_slope=negative_slope)


    def forward(self, x_dict, edge_index_dict, num_nodes_dict):

        for node_type, x_node in x_dict.items():
            x_dict[node_type]= self.dropout(self.heterlinear[node_type](x_node))

        edge_pattern_dict={}
        for node_type, value in x_dict.items():
            edge_pattern_dict[node_type]={}
            for edge_type, edge_index in edge_index_dict.items():
                src_type, _, dst_type = edge_type
                if(dst_type==node_type):
                    edge_pattern_dict[node_type][edge_type]=edge_index



        # There are several edge_type in each pattern, and their scr_type are the same
        alpha_pattern_dict={}
        beta_pattern_dict = {}
        for node_type, pattern_dict in edge_pattern_dict.items():
            alpha_pattern_dict[node_type]={}
            beta_pattern_dict[node_type]={}
            Attention_value_dict = {}
            for edge_type, edge_index in pattern_dict.items():
                src_type, _, dst_type = edge_type
                src = edge_index[0,:]
                dst = edge_index[1,:]

                message = unsorted_segment_sum(tlx.gather(x_dict[src_type],src),dst,x_dict[dst_type].shape[0])

                h_l = self.Linear_dict_l[dst_type](x_dict[dst_type])
                h_r = self.Linear_dict_r[dst_type](message)
                Type_Attention_Value = h_l + h_r # N values, N equals the number of edges
                Type_Attention_Value = self.leakyReLu(Type_Attention_Value)

                Type_Attention_Value = tlx.exp(Type_Attention_Value)
                Attention_value_dict[edge_type]=Type_Attention_Value
                Attention_value_list = [value for value in Attention_value_dict.values()]

            Summation = tlx.reduce_sum(tlx.stack(Attention_value_list,axis=0),axis=0)


            for edge_type, edge_index in pattern_dict.items():
                alpha_pattern_dict[node_type][edge_type] = Attention_value_dict[edge_type]/Summation # N values, N equals the number of edges

        out_dict={}
        for node_type, pattern_dict in edge_pattern_dict.items():
            message_list= []
            for edge_type, edge_index in pattern_dict.items():
                alpha = alpha_pattern_dict[node_type][edge_type] 
                src_type, _, dst_type = edge_type
                src = edge_index[0,:]
                dst = edge_index[1,:]
                # Use the broadcast mechanism alpha(N,1), followed by (N,hidden_dim*2), where N denotes the number of edges.
                value = self.nodeAttention(tlx.gather(alpha,dst)*tlx.concat([tlx.gather(x_dict[dst_type],dst),tlx.gather(x_dict[src_type],src)],axis=1))
                value = self.leakyReLu(value)
                value = segment_softmax(value,dst,num_segments=None)
                beta_pattern_dict[node_type][edge_type] = value
                message_list.append(unsorted_segment_sum(value*tlx.gather(x_dict[dst_type],dst),dst,x_dict[dst_type].shape[0]))
            out_dict[node_type]=tlx.reduce_sum(tlx.stack(message_list,axis=0),axis=0)

        return out_dict
    
