import tensorlayerx as tlx
from gammagl.datasets import Planetoid
import numpy as np
from gammagl.layers.conv import MessagePassing
from gammagl.utils.softmax import segment_softmax
from gammagl.mpops import *

class HypergraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, ea_len, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0., bias=True,
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
            self.lin = tlx.layers.Linear(in_features=in_channels, out_features=out_channels * heads, W_init=W_init, b_init=b_init) 
            self.lin_ea = tlx.layers.Linear(in_features=ea_len, out_features=self.out_channels * heads, W_init=W_init, b_init=b_init)                                                          
            # self.att = tlx.nn.Parameter(data=tlx.zeros(1, heads, 2 * out_channels))
            initor = tlx.initializers.Ones()
            # self.att = tlx.nn.Parameter(data = tlx.ones(shape = (1, heads, 2 * out_channels)), name = "att")
            self.att = self._get_weights('att', init=initor, shape=(1, heads, 2 * out_channels), order = True)
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

        # self.reset_parameters()

    def reset_parameters(self):
        if self.use_attention:
            glorot = tlx.nn.initializers.XavierUniform(gain=1.0) 
            self.att = glorot(self.att)

    def forward(self, x, hyperedge_index, hyperedge_weight, hyperedge_attr):
        x = self.lin(x)
        hyperedge_attr = self.lin_ea(hyperedge_attr)
        

        alpha = None

        if self.use_attention:
            assert hyperedge_attr is not None
            x = tlx.reshape(x, (-1, self.heads, self.out_channels))
            hyperedge_attr = tlx.reshape(hyperedge_attr, (-1, self.heads, self.out_channels))
            x_i = tlx.gather(x, hyperedge_index[0])
            x_j = tlx.gather(hyperedge_attr, hyperedge_index[1])
            
            # self.att = tlx.reshape(self.att, shape=(1, self.heads, 2 * self.out_channels))
            alpha = tlx.reduce_sum(((tlx.ops.concat([x_i, x_j],-1)) * self.att), axis=-1)
            alpha = tlx.nn.LeakyReLU(self.negative_slope)(alpha)
            alpha = segment_softmax(alpha, hyperedge_index[0], num_segments=max(hyperedge_index[0])+1) 
            alpha = tlx.nn.Dropout(self.dropout)(alpha)
        else: 
            assert hyperedge_attr is not None
            x_j = tlx.gather(hyperedge_attr, hyperedge_index[1])

        out = self.propagate(x=x, edge_index=hyperedge_index, aggr='sum',
                            alpha=alpha, y=hyperedge_attr)
        hyperedge_index_r  = tlx.gather(hyperedge_index, tlx.convert_to_tensor([1, 0]), axis = 0)
        out = self.propagate(x=out, edge_index=hyperedge_index_r, aggr='sum',
                            alpha=alpha, y=None)
        
        if self.concat is True:
            out = tlx.reshape(out, (-1, self.heads * self.out_channels))
        else:
            out = tlx.ops.reduce_mean(out, axis=1, keepdims=False)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x, edge_index, alpha, y, edge_weight=None):
        if y is None:
            msg = tlx.gather(x, edge_index[0, :])
        else:
            msg = tlx.gather(y, edge_index[0, :])
        if edge_weight is not None:
            edge_weight = tlx.expand_dims(edge_weight, -1)
            return msg * edge_weight
        else:
            return msg
    
    def aggregate(self, msg, edge_index, num_nodes=None, aggr='sum'):
        dst_index = edge_index[1, :]
        num_segments = max(dst_index) + 1
        if aggr == 'sum':
            return unsorted_segment_sum(msg, dst_index, num_segments)
        elif aggr == 'mean':
            return unsorted_segment_mean(msg, dst_index, num_segments)
        elif aggr == 'max':
            return unsorted_segment_max(msg, dst_index, num_segments)
        else:
            raise NotImplementedError('Not support for this opearator')