import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import segment_softmax


class AGNNConv(MessagePassing):
    r'''The graph attention operator from the `"Attention-based Graph Neural Network for Semi-supervised Learning"
    <http://arxiv.org/abs/1803.03735>`_ paper

    .. math::
        \mathbf{X}^{(i+1)} = \mathbf{P} \mathbf{X}^{(i)}

    where the propagation matrix :math:`\mathbf{P}` is computed as

    .. math::
        P_{i,j} = \frac{\exp( \beta \cdot \cos(\mathbf{x}_i, \mathbf{x}_j))}
        {\sum_{k \in \mathcal{N}(i)\cup \{ i \}} \exp( \beta \cdot
        \cos(\mathbf{x}_i, \mathbf{x}_k))}

    with trainable parameter :math:`\beta`.

    Parameters
    ----------
    in_channels: int
        Size of each input sample.
    out_channels: int
        Size of each output sample.
    edge_index: 2-D tensor
        Shape:(2, num_edges). A element(integer) of dim-1 expresses a node of graph and
        edge_index[0,i] points to edge_index[1,i].
    num_nodes: int
        Number of nodes on the graph.
    require_grad: bool, optional
        If set to :obj:`False`, :math:`\beta`
        will not be trainable. (default: :obj:`True`)

    '''


    def __init__(self,
                in_channels,
                require_grad = True):
        super().__init__()

        self.in_channels = in_channels
        if(require_grad == True):
            initor = tlx.initializers.RandomUniform(0,1)
            self.beta = self._get_weights("beta", shape = [1], init = initor)
        else:
            initor = tlx.initializers.Zeros()
            self.beta = self._get_weights("beta", shape = [1], init = initor, trainable = False)

        
    def message(self, x, edge_index, num_nodes, edge_weight=None):
        node_src = edge_index[0, :]
        node_dst = edge_index[1, :]

        x_src = tlx.gather(x, node_src)
        x_dst = tlx.gather(x, node_dst)
        
        cos = tlx.reduce_sum(
            tlx.l2_normalize(x_src, axis = -1) * tlx.l2_normalize(x_dst, axis = -1), axis = -1)
        unsoftmax_weight = cos * self.beta
        softmax_weight = tlx.expand_dims(segment_softmax(unsoftmax_weight, node_dst, num_nodes), axis = -1)
        
        return softmax_weight * x_src

    def forward(self, x, edge_index, num_nodes):
        return self.propagate(x, edge_index, num_nodes=num_nodes)
