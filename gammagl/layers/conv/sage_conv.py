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

    def __init__(self, in_channels, out_channels, aggr="mean", add_bias=True):
        super(SAGEConv, self).__init__()
        self.aggr = aggr
        initor = tlx.initializers.XavierUniform()
        self.fc_neigh = tlx.layers.Dense(n_units=out_channels,
                                        in_channels=in_channels,
                                        b_init=initor)
        self.fc_self = tlx.layers.Dense(n_units=out_channels,
                                       in_channels=in_channels,
                                       b_init=initor)
        self.add_bias = add_bias
        if add_bias is True:
            initor = tlx.initializers.Zeros()
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
        src_feat = self.fc_neigh(src_feat)

        self.num_nodes = dst_feat.shape[0]
        out = self.propagate(src_feat, edge)
        dst_feat = self.fc_self(dst_feat)
        out += dst_feat
        # out = tf.tensor_scatter_nd_add(out, tf.reshape(tf.range(0, len(dst_feat)), (-1, 1)), dst_feat)
        if self.add_bias:
            out += self.bias
        return out
    def aggregate(self, x, edge_index):
        if self.aggr == "mean":
            return tf.math.unsorted_segment_mean(tf.gather(x, edge_index[0]), edge_index[1], self.num_nodes)
