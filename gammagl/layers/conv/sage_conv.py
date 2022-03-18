import tensorlayerx as tlx
import tensorflow as tf
from gammagl.layers.conv import MessagePassing


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
       Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

       .. math::
           \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
           \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.

        norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.

        aggr : Aggregator type to use (``mean``).
        add_bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)

    """

    def __init__(self, in_channels, out_channels, aggr="mean", add_bias=False):
        super(SAGEConv, self).__init__()
        self.aggr = aggr
        # relu用he激活
        initor = tlx.initializers.he_normal()
        self.fc_neigh = tlx.nn.layers.Dense(in_channels=in_channels, n_units=out_channels, W_init=initor)
        # tlx.layers.Dense(n_units=out_channels,
        #                             in_channels=in_channels,
        #                             b_init=initor)
        if aggr != 'gcn':
            self.fc_self = tlx.layers.Dense(n_units=out_channels, in_channels=in_channels, W_init=initor)
        if aggr == "lstm":
            # self.lstm 怎么初始化？lstm有点难啊
            self.lstm = tlx.nn.LSTM(input_size=in_channels, hidden_size=in_channels, batch_first=True)

        if aggr == "pool":
            self.pool = tlx.nn.layers.Dense(in_channels=in_channels, n_units=in_channels, W_init=initor)
        self.add_bias = add_bias

        if add_bias:
            initor = tlx.initializers.zeros()
            self.bias = self._get_weights("bias", shape=(1, out_channels), init=initor)
            
    def forward(self, feat, edge):
        r"""

                Description
                -----------
                Compute GraphSAGE layer.

                Parameters
                ----------
                feat : Pair of Tensor
                    The pair must contain two tensors of shape
                    :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
                Returns
                -------
                Tensor
                    The output feature of shape :math:`(N_{dst}, D_{out})`
                    where :math:`N_{dst}` is the number of destination nodes in the input graph,
                    math:`D_{out}` is size of output feature.
        """
        src_feat = feat[0]
        dst_feat = feat[1]
        num_nodes = dst_feat.shape[0]
        if self.aggr == 'mean':
            src_feat = self.fc_neigh(src_feat)
            out = self.propagate(src_feat, edge, edge_weight=None, num_nodes=num_nodes, aggr='mean')
        elif self.aggr =='gcn':
            src_feat = self.fc_neigh(src_feat)
            col, row, weight = calc(edge, num_nodes)
            out = self.propagate(src_feat, tlx.stack([col,row]), edge_weight=weight, num_nodes=num_nodes, aggr='sum')
        elif self.aggr == 'pool':
            src_feat = tlx.nn.ReLU()(self.pool(src_feat))
            out = self.propagate(src_feat, edge, edge_weight=None, num_nodes=num_nodes, aggr='max')
            out = self.fc_neigh(out)
        elif self.aggr == 'lstm':
            pass
        if self.aggr != 'gcn':
            out += self.fc_self(dst_feat)
        if self.add_bias:
            out += self.bias
        return out
    
    # def aggregate(self, msg, edge_index, num_nodes=None, aggr_type='mean'):
    #     dst_index = edge_index[1, :]
    #     if aggr_type == 'mean':
    #         return tlx.ops.unsorted_segment_mean(msg, dst_index, num_nodes)
import numpy as np
import scipy.sparse as sp

def calc(edge, num_nodes):
    weight = np.ones(edge.shape[1])
    sparse_adj = sp.coo_matrix((weight, (edge[0], edge[1])), shape=(2708,2708))
    A = (sparse_adj + sp.eye(num_nodes)).tocoo()
    col, row, weight = A.col, A.row, A.data
    deg = np.array(A.sum(1))
    # deg_inv_sqrt = np.power(deg, -0.5).flatten()
    # return col, row, np.array(deg_inv_sqrt[row] * weight * deg_inv_sqrt[col], dtype=np.float32)
    deg_inv_sqrt = np.power(deg, -1).flatten()
    return col, row, np.array(weight * deg_inv_sqrt[col], dtype=np.float32)


