import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.layers.conv import MessagePassing
from gammagl.utils import degree

class TransConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_features=in_channels, out_features=out_channels * num_heads)
        self.Wq = nn.Linear(in_features=in_channels, out_features=out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_features=in_channels, out_features=out_channels * num_heads)
        
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def forward(self, query_input, source_input):
        N = query_input.shape[0] 
        

        qs = self.Wq(query_input)  # [N, num_heads * out_channels]
        ks = self.Wk(source_input)  # [N, num_heads * out_channels]
        

        qs = tlx.reshape(qs, (-1, self.num_heads, self.out_channels))  # [N, num_heads, out_channels]
        ks = tlx.reshape(ks, (-1, self.num_heads, self.out_channels))  # [N, num_heads, out_channels]
        
        if self.use_weight:
            vs = self.Wv(source_input)  # [N, num_heads * out_channels]
            vs = tlx.reshape(vs, (-1, self.num_heads, self.out_channels))  # [N, num_heads, out_channels]
        else:
            vs = tlx.reshape(source_input, (-1, 1, source_input.shape[-1]))  # [N, 1, in_channels]

        qs = tlx.ops.l2_normalize(qs, axis=-1)  # [N, num_heads, out_channels]
        ks = tlx.ops.l2_normalize(ks, axis=-1)  # [N, num_heads, out_channels]

        # [num_heads, N, out_channels] x [num_heads, out_channels, N] -> [num_heads, N, N]
        attention = tlx.matmul(qs, tlx.transpose(ks, (0, 2, 1)))
        attention = attention / tlx.sqrt(tlx.convert_to_tensor(self.out_channels, dtype=tlx.float32))

        if self.use_weight:
            out = tlx.matmul(attention, vs)  # [num_heads, N, out_channels]
        else:
            out = tlx.matmul(attention, vs)  # [num_heads, N, in_channels]
            

        out = tlx.reduce_mean(out, axis=1)  # [N, out_channels] or [N, in_channels]
        
        return out

class GraphConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, use_weight=True, use_init=False):
        super().__init__()
        self.use_init = use_init
        self.use_weight = use_weight
        
        in_channels_ = 2 * in_channels if use_init else in_channels
        if use_weight:
            self.linear = nn.Linear(in_features=in_channels_, out_features=out_channels)

    def forward(self, x, edge_index, x0, num_nodes=None):
        if num_nodes is None:
            num_nodes = tlx.get_tensor_shape(x)[0]
            
        device = x.device
        edge_index = tlx.convert_to_tensor(edge_index)
        
        row, col = edge_index[0], edge_index[1]
        deg = degree(col, num_nodes=num_nodes, dtype=tlx.float32)
        deg_inv_sqrt = tlx.pow(deg, -0.5)
        deg_inv_sqrt = tlx.gather(deg_inv_sqrt, row) * tlx.gather(deg_inv_sqrt, col)
        
        deg_inv_sqrt = tlx.convert_to_tensor(deg_inv_sqrt, device=device)
        

        x = self.propagate(x, edge_index, edge_weight=deg_inv_sqrt, size=num_nodes)
        
        if self.use_init:
            x = tlx.concat([x, x0], axis=1)
            
        if self.use_weight:
            x = self.linear(x)
            
        return x
