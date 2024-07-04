import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import calc_gcn_norm, add_self_loops
from gammagl.utils.num_nodes import maybe_num_nodes
import math
import numpy as np


class Linear(tlx.nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        super(Linear, self).__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        shape = (groups, in_channels // groups, out_channels // groups)[::-1]
        self.weight = self._get_weights('weights', shape=shape, init=tlx.initializers.he_uniform(a=math.sqrt(5)))

        if bias:
            initor = tlx.initializers.RandomUniform(0, 1)
            self.bias = tlx.nn.Parameter(tlx.random_uniform(shape = (self.out_channels,)))
        else:
            self.bias = None

    def forward(self, src):
        if self.groups > 1:
            src_shape = tlx.get_tensor_shape(src)[:-1]
            src = tlx.reshape(src, (-1, self.groups, self.in_channels // self.groups))
            src = tlx.transpose(src, (1, 0, 2))
            out = tlx.matmul(src, self.weight)
            out = tlx.transpose(out, (1, 0, 2))
            out = tlx.reshape(out, tuple(src_shape) + (self.out_channels,))
        else:
            out = tlx.matmul(src, tlx.squeeze(self.weight, axis=0))

        if self.bias is not None:
            out += self.bias
        
        return out


def restricted_softmax(src, dim: int = -1, margin: float = 0.):
    src_max = tlx.reduce_max(src, axis=dim, keepdims=True)
    src_max = np.clip(tlx.convert_to_numpy(src_max), 0, None)
    src_max = tlx.convert_to_tensor(src_max)
    out = tlx.exp(src - src_max)
    out = out / (tlx.reduce_sum(out, axis=dim, keepdims=True) + tlx.exp(margin - src_max))

    return out
    
class Attention(tlx.nn.Module):
    def __init__(self, dropout=0):
        super(Attention, self).__init__()
        self.dropout = tlx.nn.Dropout(p = dropout)

    def forward(self, query, key, value):
        return self.compute_attention(query, key, value)

    def compute_attention(self, query, key, value):
        # query: [*, query_entries, dim_k]
        # key: [*, key_entries, dim_k]
        # value: [*, key_entries, dim_v]
        # Output: [*, query_entries, dim_v]

        assert query.ndim == key.ndim == value.ndim >= 2
        assert query.shape[-1] == key.shape[-1]
        assert key.shape[-2] == value.shape[-2]

        score = tlx.matmul(query, tlx.ops.transpose(key, (0, 1, 3, 2)))
        score = score / math.sqrt(key.shape[-1])
        score = restricted_softmax(score, dim=-1)
        score = self.dropout(score)

        return tlx.matmul(score, value)
    

class MultiHead(Attention):
    def __init__(self, in_channels, out_channels, heads=1, groups=1, dropout=0, bias=True):
        super().__init__(dropout=dropout)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.groups = groups
        self.bias = bias

        assert in_channels % heads == 0 and out_channels % heads == 0
        assert in_channels % groups == 0 and out_channels % groups == 0
        assert max(groups, heads) % min(groups, heads) == 0


        self.lin_q = Linear(in_channels, out_channels, groups=groups, bias=bias)
        self.lin_k = Linear(in_channels, out_channels, groups=groups, bias=bias)
        self.lin_v = Linear(in_channels, out_channels, groups=groups, bias=bias)

    def forward(self, query, key, value):
        query = self.lin_q(query)
        key = self.lin_k(key)
        value = self.lin_v(value)

        batch_size = tlx.get_tensor_shape(query)[:-2]

        out_channels_per_head = self.out_channels // self.heads

        query_size = tuple(batch_size) + (query.shape[-2], self.heads, out_channels_per_head)
        query = tlx.ops.reshape(query, query_size)
        query = tlx.ops.transpose(query, (0, 2, 1, 3))

        key_size = tuple(batch_size) + (key.shape[-2], self.heads, out_channels_per_head)
        key = tlx.ops.reshape(key, key_size)
        key = tlx.ops.transpose(key, (0, 2, 1, 3))

        value_size = tuple(batch_size) + (value.shape[-2], self.heads, out_channels_per_head)
        value = tlx.ops.reshape(value, value_size)
        value = tlx.ops.transpose(value, (0, 2, 1, 3))

        out = self.compute_attention(query, key, value)

        out = tlx.transpose(out, (0, 2, 1, 3))
        out = tlx.reshape(out, tuple(batch_size) + (query.shape[-2], self.out_channels))
        
        return out
    

class DNAConv(MessagePassing):
    r"""The dynamic neighborhood aggregation operator from the `"Just Jump:
    Towards Dynamic Neighborhood Aggregation in Graph Neural Networks"
    <https://arxiv.org/abs/1904.04849>`_ paper.

    .. math::
        \mathbf{x}_v^{(t)} = h_{\mathbf{\Theta}}^{(t)} \left( \mathbf{x}_{v
        \leftarrow v}^{(t)}, \left\{ \mathbf{x}_{v \leftarrow w}^{(t)} : w \in
        \mathcal{N}(v) \right\} \right)

    based on (multi-head) dot-product attention

    .. math::
        \mathbf{x}_{v \leftarrow w}^{(t)} = \textrm{Attention} \left(
        \mathbf{x}^{(t-1)}_v \, \mathbf{\Theta}_Q^{(t)}, [\mathbf{x}_w^{(1)},
        \ldots, \mathbf{x}_w^{(t-1)}] \, \mathbf{\Theta}_K^{(t)}, \,
        [\mathbf{x}_w^{(1)}, \ldots, \mathbf{x}_w^{(t-1)}] \,
        \mathbf{\Theta}_V^{(t)} \right)

    with :math:`\mathbf{\Theta}_Q^{(t)}, \mathbf{\Theta}_K^{(t)},
    \mathbf{\Theta}_V^{(t)}` denoting (grouped) projection matrices for query,
    key and value information, respectively.
    :math:`h^{(t)}_{\mathbf{\Theta}}` is implemented as a non-trainable
    version of :class:`torch_geometric.nn.conv.GCNConv`.

    .. note::
        In contrast to other layers, this operator expects node features as
        shape :obj:`[num_nodes, num_layers, channels]`.

    Parameters
    ----------
    channels: int
        Size of each input/output sample.
    heads: int, optional
        Number of multi-head-attentions.
        (default: :obj:`1`)
    groups: int, optional
        Number of groups to use for all linear projections.
        (default: :obj:`1`)
    dropout: float, optional
        Dropout probability of attention coefficients.
        (default: :obj:`0.`)
    normalize: bool, optional
        Whether to add self-loops and apply symmetric normalization.
        (default: :obj:`True`)
    add_self_loops: bool, optional
        If set to :obj:`False`, will not add self-loops to the input graph.
        (default: :obj:`True`)
    bias: bool, optional
        If set to :obj:`False`, the layer will not learn an additive bias.
        (default: :obj:`True`)

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, L, F)` where :math:`L` is the
          number of layers,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F)`
    """

    def __init__(self, channels: int, heads: int = 1, groups: int = 1,
                 dropout: float = 0., normalize: bool = True, add_self_loops: bool = True,
                 bias: bool = True):
        super().__init__()

        self.bias = bias
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self.multi_head = MultiHead(channels, channels, heads, groups, dropout, bias)

    def forward(self, x, edge_index, edge_weight = None):
        if self.normalize and edge_weight == None:
            edge_index, edge_weight = add_self_loops(edge_index)
            edge_weight = calc_gcn_norm(edge_index=edge_index, num_nodes=maybe_num_nodes(edge_index), edge_weight=edge_weight)
        else:
            edge_weight = tlx.ones((edge_index.shape[1],))

        return self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight)
    
    def message(self, x, edge_index, edge_weight=None):
        x_i = tlx.gather(x, edge_index[0, :])
        x_j = tlx.gather(x, edge_index[1, :])

        x_i = x_i[:, -1:]
        out = self.multi_head(x_i, x_j, x_j)
        return tlx.reshape(edge_weight, (-1, 1)) * tlx.squeeze(out, axis=1)
