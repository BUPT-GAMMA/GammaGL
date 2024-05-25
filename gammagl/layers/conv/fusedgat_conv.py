import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import sort_edge_index 
from gammagl.ops.sparse import ind2ptr


class FusedGATConv(MessagePassing):
    r"""The fused graph attention operator from the
    `"Understanding GNN Computational Graph: A Coordinated Computation, IO, and Memory Perspective"
    <https://arxiv.org/abs/2110.09524>`_ paper.

    :class:`FusedGATConv` is an optimized version of
    :class:`gammagl.layers.conv.GATConv` based on the :obj:`dgNN` package
    that fuses message passing computation for accelerated execution and lower
    memory footprint.

    .. note::

        This implementation is based on the :obj:`dgNN` package.
        See `here <https://github.com/dgSPARSE/dgNN>`__ for instructions on how
        to install.
    
    Parameters
    ----------
    in_channels: int or tuple
        Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    out_channels: int
        Size of each output sample.
    heads: int, optional
        Number of multi-head-attentions.
        (default: :obj:`1`)
    concat: bool, optional
        If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated.
        (default: :obj:`True`)
    negative_slope: float, optional
        LeakyReLU angle of the negative
        slope. (default: :obj:`0.2`)
    dropout_rate: float, optional
        Dropout probability of the normalized
        attention coefficients which exposes each node to a stochastically
        sampled neighborhood during training. (default: :obj:`0`)
    add_bias: bool, optional
        If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout_rate=0.,
                 add_bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout_rate = dropout_rate
        self.add_bias = add_bias
        
        from dgNN.operators import GATConvFuse
        self.op = GATConvFuse

        self.linear = tlx.layers.Linear(out_features=self.out_channels * self.heads,
                                        in_features=self.in_channels,
                                        b_init=None)

        initor = tlx.initializers.TruncatedNormal()
        self.att_src = self._get_weights("att_src", shape=(1, self.heads, self.out_channels), init=initor, order=True)
        self.att_dst = self._get_weights("att_dst", shape=(1, self.heads, self.out_channels), init=initor, order=True)

        self.leaky_relu = tlx.layers.LeakyReLU(negative_slope)
        self.dropout = tlx.layers.Dropout(self.dropout_rate)

        if self.add_bias and concat:
            self.bias = self._get_weights("bias", shape=(self.heads * self.out_channels,), init=initor)
        elif self.add_bias and not concat:
            self.bias = self._get_weights("bias", shape=(self.out_channels,), init=initor)

    def forward(self, x, edge_index, num_nodes=None, **kwargs):
        x = tlx.reshape(self.linear(x), shape=(-1, self.heads, self.out_channels))
        
        alpha_dst = tlx.reduce_sum(x * self.att_src, -1) 
        alpha_src = tlx.reduce_sum(x * self.att_dst, -1) 
        
        if 'row_ptr' in kwargs.keys():
            row_ptr = kwargs['row_ptr']
            col_ind = kwargs['col_ind']
            permute = kwargs['permute']
            row_ind = kwargs['row_ind']
            col_ptr = kwargs['col_ptr']

        else:
            num_edges = tlx.ops.get_tensor_shape(edge_index)[1]
            if num_nodes is None:
                num_nodes = int(tlx.reduce_max(edge_index)) + 1
            edge_index = sort_edge_index(edge_index)
            row_ptr = ind2ptr(tlx.convert_to_numpy(edge_index[0, :]), num_nodes)
            col_ind = edge_index[1, :]
            permute = tlx.ops.arange(0, num_edges)
            edge_index, permute = sort_edge_index(edge_index, permute, sort_by_row=False)
            row_ind = edge_index[0, :]
            col_ptr = ind2ptr(tlx.convert_to_numpy(edge_index[1, :]), num_nodes)
            row_ptr = tlx.convert_to_tensor(row_ptr, tlx.int32)
            col_ind = tlx.convert_to_tensor(col_ind, tlx.int32)
            col_ptr = tlx.convert_to_tensor(col_ptr, tlx.int32)
            row_ind = tlx.convert_to_tensor(row_ind, tlx.int32)
            permute = tlx.convert_to_tensor(permute, tlx.int32)

        import torch
        torch.cuda.synchronize()
        x = self.op(alpha_dst, alpha_src, row_ptr, col_ind, col_ptr, row_ind, permute, self.negative_slope, x, self.dropout_rate)
        torch.cuda.synchronize()

        if self.concat:
            x = tlx.reshape(x, (-1, self.heads * self.out_channels))
        else:
            x = tlx.reduce_mean(x, axis=1)
        if self.add_bias:
            x += self.bias
        return x

