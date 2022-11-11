import tensorlayerx as tlx
from gammagl.datasets import Planetoid
import numpy as np
from gammagl.layers.conv import MessagePassing
from gammagl.utils.softmax import segment_softmax
from gammagl.mpops import *

def scatter_add(src, index, lenth):
    
    a = np.zeros(lenth)
    for j in range(lenth):
        a[index[j]] += src[j]
    return a

class HypergraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, ea_len, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            W_init = tlx.nn.initializers.XavierUniform(gain=1.0)  
            b_init = tlx.nn.initializers.constant(value=0.1)
            self.lin = tlx.layers.Linear(in_features=in_channels, out_features=out_channels, W_init=W_init, b_init=b_init) 
            self.lin_ea = tlx.layers.Linear(in_features=ea_len, out_features=self.out_channels, W_init=W_init, b_init=b_init)                                                          
            # self.att = tlx.nn.Parameter(data=tlx.zeros(1, heads, 2 * out_channels))
            initor = tlx.initializers.Zeros()
            self.att = self._get_weights('att', init=initor, shape=(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            W_init = tlx.nn.initializers.XavierUniform(gain=1.0)    
            b_init = tlx.nn.initializers.constant(value=0.1)
            self.lin = tlx.layers.Linear(in_features=in_channels, out_features=out_channels, W_init=W_init, b_init=b_init) 
            self.lin_ea = tlx.layers.Linear(in_features=ea_len, out_features=self.out_channels, W_init=W_init, b_init=b_init)       

        if bias and concat:
            # self.bias = tlx.nn.Parameter(data=tlx.zeros(heads * out_channels))
            initor = tlx.initializers.Zeros()
            self.bias = self._get_weights('bias', init=initor, shape=(heads * out_channels,))
        elif bias and not concat:
            # self.bias = tlx.nn.Parameter(data=tlx.zeros(out_channels))
            initor = tlx.initializers.Zeros()
            self.bias = self._get_weights('bias', init=initor, shape=(out_channels,))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot = tlx.nn.initializers.XavierUniform(gain=1.0) 
        zeros = tlx.nn.initializers.Zeros()
        if self.use_attention:
            self.att = glorot(self.att)
        # self.bias = zeros(shape=self.bias.shape, dtype=tlx.float32)

    def forward(self, x, hyperedge_index,hyperedge_weight,hyperedge_attr):
        num_nodes, num_edges = x.shape[0], 0
        if tlx.ops.count_nonzero(hyperedge_index + 1) > 0:
            num_edges = max(hyperedge_index[1]) + 1

        if hyperedge_weight is None:
            hyperedge_weight = tlx.ones(shape=num_edges,dtype=x.dtype)

        x = self.lin(x)
        hyperedge_attr = self.lin_ea(hyperedge_attr)
        

        alpha = None
        # hyperedge_weight = tlx.convert_to_numpy(hyperedge_weight)
        # hyperedge_index = tlx.convert_to_numpy(hyperedge_index)
        # hyperedge_attr = tlx.convert_to_numpy(hyperedge_attr)

        if self.use_attention:
            assert hyperedge_attr is not None
            x = tlx.reshape(x, (-1,self.heads, self.out_channels))
            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = tlx.reshape(hyperedge_attr, (-1,self.heads,self.out_channels))
            # x_i = x[hyperedge_index[0]]
            # x_j = hyperedge_attr[hyperedge_index[1]]
            x_i = tlx.gather(x, hyperedge_index[0])
            x_j = tlx.gather(hyperedge_attr, hyperedge_index[1])
            alpha = tlx.reduce_sum((tlx.ops.concat([x_i, x_j],-1) * self.att), axis=-1)
            alpha = tlx.nn.LeakyReLU(self.negative_slope)(alpha)
            alpha = segment_softmax(alpha, hyperedge_index[0], num_nodes=x.size(0)) 
            alpha = tlx.nn.dropout(self.dropout)(alpha)
        else: 
            assert hyperedge_attr is not None
            # x_j = hyperedge_attr[hyperedge_index[1]]
            x_j = tlx.gather(hyperedge_attr, hyperedge_index[1])
            
            
            
        
        D = scatter_add(hyperedge_weight[hyperedge_index[1]],hyperedge_index[0],num_nodes)
        D = 1.0 / D
        D[D == float("inf")] = 0
        # print('D.shape',D.shape)
        B = scatter_add(np.ones(len(hyperedge_index[0])),hyperedge_index[1],num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0
        # print('B.shape',B.shape)
        # hyperedge_weight = tlx.convert_to_tensor(hyperedge_weight)
        # hyperedge_index = tlx.convert_to_tensor(hyperedge_index)
        # hyperedge_attr = tlx.convert_to_tensor(hyperedge_attr)

        # print('propagate1 start')
        out = self.propagate(x=x, edge_index=hyperedge_index,aggr='sum',
                            norm=B, alpha=alpha,y=hyperedge_attr)
        # print('propagate1 end')
        # print('propagate2 start')
        hyperedge_index_r  = tlx.gather(hyperedge_index, [1, 0], axis = 0)
        # hyperedge_index_r = np.flip(tlx.convert_to_numpy(hyperedge_index),axis=0)
        # print('clear')
        # hyperedge_index_r = tlx.convert_to_tensor(hyperedge_index_r)
        out = self.propagate(x=out, edge_index=hyperedge_index_r ,aggr='sum',
                            norm=D, alpha=alpha, y=None)
        # print('propagate2 end')
        if self.concat is True:
            out = tlx.reshape(out, (-1, self.heads * self.out_channels))
        else:
            out = tlx.ops.reduce_mean(out, axis=1, keepdims=False)

        if self.bias is not None:
            out = out + self.bias

        # print('conv forward done!')
        return out

    # def message(self, x, norm, alpha, y):
    #     print('inside message')
    #     H, F = self.heads, self.out_channels
    #     print('message start')

    #     print('norm_i:',tlx.reshape(norm, (-1, 1, 1)).shape)
    #     print('x:',tlx.reshape(x, (-1, H, F)).shape)
    #     norm = tlx.ops.convert_to_tensor(norm,dtype='float32')
    #     if y is None:
    #         out = tlx.reshape(norm, (-1, 1, 1)) * tlx.reshape(x, (-1, H, F))
    #     elif y is not None:
    #         out = tlx.reshape(norm, (-1, 1, 1)) * tlx.reshape(y, (-1, H, F))
    #     print('message end')
    #     if alpha is not None:
    #         out = tlx.reshape(alpha, (-1, self.heads, 1)) * out
    #     print('propagete out.shape:',out.shape)
    #     return out
    
    # def aggregate(self, msg, edge_index, num_nodes=None, aggr='sum', y):
    #     dst_index = edge_index[1, :]
    #     if y is not None:
            
    #     if aggr == 'sum':
    #         return unsorted_segment_sum(msg, dst_index, num_nodes)
    #     elif aggr == 'mean':
    #         return unsorted_segment_mean(msg, dst_index, num_nodes)
    #     elif aggr == 'max':
    #         return unsorted_segment_max(msg, dst_index, num_nodes)
    #     else:
    #         raise NotImplementedError('Not support for this opearator')
    def message(self, x, edge_index, alpha, y, edge_weight=None):
        # print('inside message')
        if y is None:
            msg = tlx.gather(x, edge_index[0, :])
        else:
            msg = tlx.gather(y, edge_index[0, :])
        # print('gather done')
        if edge_weight is not None:
            edge_weight = tlx.expand_dims(edge_weight, -1)
            return msg * edge_weight
        else:
            return msg
    
    def aggregate(self, msg, edge_index, num_nodes=None, aggr='sum'):
        dst_index = edge_index[1, :]
        # print('inside aggregate')
        # print('msg.shape:',msg.shape)
        # print('dst_index.shape:',dst_index.shape)
        if aggr == 'sum':
            return unsorted_segment_sum(msg, dst_index, num_nodes)
        elif aggr == 'mean':
            return unsorted_segment_mean(msg, dst_index, num_nodes)
        elif aggr == 'max':
            return unsorted_segment_max(msg, dst_index, num_nodes)
        else:
            raise NotImplementedError('Not support for this opearator')